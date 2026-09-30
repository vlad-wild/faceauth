//! `faceauthd`: root daemon that verifies the calling user's face over a Unix socket.
//!
//! Screen lockers run unprivileged and cannot read the root-only model store, so
//! they ask this daemon instead (directly, or through `faceauth-auth` running as
//! the user). The caller's uid comes from `SO_PEERCRED`; a non-root caller can
//! only ever have its own face verified. See `faceauth::daemon` for the protocol.
//!
//! Started by systemd socket activation (`faceauthd.socket`); exits after being
//! idle so the models do not stay in memory forever.

use anyhow::{Context, Result, bail};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::ops::ControlFlow;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use faceauth::authenticate::{AuthOutcome, authenticate};
use faceauth::config::{Config, SYSTEM_CONFIG_PATH};
use faceauth::daemon::{Event, Outcome, RateLimiter, Request, SOCKET_PATH, peer_uid};
use faceauth::gate::{Gate, pre_auth_checks};
use faceauth::logger;
use faceauth::pipeline::{FaceVerdict, Models, Pipeline};
use faceauth::privilege::{is_root, username_for_uid, validate_username};

/// Failed attempts allowed per uid within [`FAILURE_WINDOW`].
const MAX_FAILURES: usize = 5;
const FAILURE_WINDOW: Duration = Duration::from_secs(60);
/// How long a client has to send its request after connecting.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Parser)]
#[command(name = "faceauthd")]
#[command(about = "Face verification daemon for screen lockers (run by systemd)")]
struct Args {
    /// Bind this socket instead of using systemd socket activation
    #[arg(long)]
    socket: Option<PathBuf>,

    /// Exit after this many idle seconds (0 = never)
    #[arg(long, default_value_t = 300)]
    idle_timeout: u64,
}

/// Models kept loaded between attempts, reloaded when the system config changes.
#[derive(Default)]
struct Engine {
    models: Option<Models>,
    config_stamp: Option<SystemTime>,
}

struct Shared {
    /// Held for the whole attempt: there is one camera.
    engine: Mutex<Engine>,
    limiter: Mutex<RateLimiter>,
    active: AtomicUsize,
    last_activity: Mutex<Instant>,
}

fn main() {
    logger::init_from_env();
    if let Err(e) = run(Args::parse()) {
        log::error!("{e:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if !is_root() {
        bail!("faceauthd must run as root (it reads the root-only model store)");
    }
    let (listener, bound_path) = match socket_from_systemd()? {
        Some(listener) => (listener, None),
        None => {
            let path = args.socket.unwrap_or_else(|| PathBuf::from(SOCKET_PATH));
            (bind_socket(&path)?, Some(path))
        }
    };
    listener.set_nonblocking(true)?;
    log::info!("faceauthd ready");

    let shared = Arc::new(Shared {
        engine: Mutex::new(Engine::default()),
        limiter: Mutex::new(RateLimiter::new(MAX_FAILURES, FAILURE_WINDOW)),
        active: AtomicUsize::new(0),
        last_activity: Mutex::new(Instant::now()),
    });
    let idle = Duration::from_secs(args.idle_timeout);

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                *shared.last_activity.lock().unwrap() = Instant::now();
                shared.active.fetch_add(1, Ordering::SeqCst);
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    if let Err(e) = handle(stream, &shared) {
                        log::warn!("Client error: {e:#}");
                    }
                    *shared.last_activity.lock().unwrap() = Instant::now();
                    shared.active.fetch_sub(1, Ordering::SeqCst);
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(200));
                let idle_for = shared.last_activity.lock().unwrap().elapsed();
                if !idle.is_zero() && shared.active.load(Ordering::SeqCst) == 0 && idle_for >= idle
                {
                    log::info!("Idle for {}s; exiting", idle_for.as_secs());
                    break;
                }
            }
            Err(e) => log::warn!("accept failed: {e}"),
        }
    }
    if let Some(path) = bound_path {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

/// The listening socket passed by systemd (`LISTEN_FDS`), if this is socket activation.
fn socket_from_systemd() -> Result<Option<UnixListener>> {
    use std::os::fd::FromRawFd;

    let pid_matches = std::env::var("LISTEN_PID")
        .ok()
        .and_then(|p| p.parse::<u32>().ok())
        == Some(std::process::id());
    let fds: u32 = std::env::var("LISTEN_FDS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    // SAFETY: remove_var before any other thread exists.
    unsafe {
        std::env::remove_var("LISTEN_PID");
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_FDNAMES");
    }
    if !pid_matches || fds == 0 {
        return Ok(None);
    }
    if fds > 1 {
        bail!("Expected one socket from systemd, got {fds}");
    }
    // SAFETY: systemd passes the listening socket as fd 3 (SD_LISTEN_FDS_START)
    // and nothing else in this process owns it.
    Ok(Some(unsafe { UnixListener::from_raw_fd(3) }))
}

/// Bind the socket ourselves (manual runs and tests); any local user may connect.
fn bind_socket(path: &Path) -> Result<UnixListener> {
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if !meta.file_type().is_socket() {
            bail!("{} exists and is not a socket", path.display());
        }
        std::fs::remove_file(path)?;
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let listener =
        UnixListener::bind(path).with_context(|| format!("Cannot bind {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666))?;
    Ok(listener)
}

fn send(stream: &mut UnixStream, event: &Event) -> std::io::Result<()> {
    stream.write_all(event.to_line().as_bytes())
}

fn handle(stream: UnixStream, shared: &Shared) -> Result<()> {
    let uid = peer_uid(&stream)?;
    let mut writer = stream.try_clone()?;
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    let mut reader = BufReader::new(stream);

    let mut line = String::new();
    reader.read_line(&mut line).context("Reading request")?;
    let request: Request = match serde_json::from_str(line.trim()) {
        Ok(r) => r,
        Err(_) => {
            send(
                &mut writer,
                &Event::result(Outcome::Error, Some("bad_request")),
            )?;
            return Ok(());
        }
    };
    let requested_user = match request {
        Request::Verify { user } => user,
        Request::Cancel => {
            send(
                &mut writer,
                &Event::result(Outcome::Error, Some("nothing_to_cancel")),
            )?;
            return Ok(());
        }
    };

    // Non-root callers are always verified as themselves.
    let own_name = username_for_uid(uid)?;
    let user = match requested_user {
        Some(name) if uid == 0 => {
            validate_username(&name)?;
            name
        }
        Some(name) if name != own_name => {
            log::warn!("uid {uid} asked to verify {name}; refused");
            send(
                &mut writer,
                &Event::result(Outcome::Error, Some("not_allowed")),
            )?;
            return Ok(());
        }
        _ => own_name,
    };

    if !shared.limiter.lock().unwrap().allowed(uid, Instant::now()) {
        send(
            &mut writer,
            &Event::result(Outcome::Busy, Some("rate_limited")),
        )?;
        return Ok(());
    }
    let Ok(mut engine) = shared.engine.try_lock() else {
        send(
            &mut writer,
            &Event::result(Outcome::Busy, Some("attempt_in_progress")),
        )?;
        return Ok(());
    };

    // Watch the connection: "cancel" or a closed socket stops the attempt.
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = Arc::clone(&cancel);
        let _ = reader.get_ref().set_read_timeout(None);
        std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if matches!(serde_json::from_str(line.trim()), Ok(Request::Cancel)) {
                            break;
                        }
                    }
                }
            }
            cancel.store(true, Ordering::SeqCst);
        });
    }

    let (outcome, reason) = match attempt(&mut engine, &user, &mut writer, &cancel) {
        Ok(result) => result,
        Err(e) => {
            log::error!("Attempt for {user} failed: {e:#}");
            (Outcome::Error, Some("setup_error".to_string()))
        }
    };
    drop(engine);
    shared
        .limiter
        .lock()
        .unwrap()
        .record(uid, outcome, Instant::now());
    log::info!(
        "Result for {user}: {outcome:?}{}",
        reason
            .as_deref()
            .map(|r| format!(" ({r})"))
            .unwrap_or_default()
    );
    let _ = send(&mut writer, &Event::result(outcome, reason.as_deref()));
    // Unblocks the watcher thread even if the client keeps the socket open.
    let _ = writer.shutdown(std::net::Shutdown::Both);
    Ok(())
}

fn config_stamp() -> Option<SystemTime> {
    std::fs::metadata(SYSTEM_CONFIG_PATH)
        .and_then(|m| m.modified())
        .ok()
}

/// One verification attempt; the camera is open only inside this function.
fn attempt(
    engine: &mut Engine,
    user: &str,
    writer: &mut UnixStream,
    cancel: &AtomicBool,
) -> Result<(Outcome, Option<String>)> {
    let cfg = Config::load_resolved(Path::new(SYSTEM_CONFIG_PATH))
        .with_context(|| format!("Failed to load {SYSTEM_CONFIG_PATH}"))?;

    let model = match pre_auth_checks(&cfg, user, false)? {
        Gate::Proceed(model) => model,
        Gate::Skip(reason) => return Ok((Outcome::Skipped, Some(reason.key().to_string()))),
    };

    // The camera key (or a privacy switch) removes the device: skip without waiting.
    let device = &cfg.video.device_path;
    if device.starts_with("/dev/") && !Path::new(device).exists() {
        return Ok((Outcome::Skipped, Some("camera_unavailable".to_string())));
    }

    let stamp = config_stamp();
    if engine.config_stamp != stamp {
        engine.models = None;
    }
    let models = match engine.models.take() {
        Some(models) => models,
        None => Models::load(&cfg)?,
    };
    engine.config_stamp = stamp;

    let camera = match Pipeline::open_camera(&cfg, device) {
        Ok(camera) => camera,
        Err(e) => {
            engine.models = Some(models);
            return Err(e);
        }
    };
    let mut pipeline = Pipeline::with_models(camera, models);
    if send(writer, &Event::Started).is_err() {
        engine.models = Some(pipeline.into_models());
        return Ok((Outcome::Cancelled, Some("client_gone".to_string())));
    }

    let timeout = Duration::from_secs(cfg.video.timeout as u64);
    let result = authenticate(&mut pipeline, &cfg, &model, timeout, |ev| {
        let event = Event::Frame {
            face: ev.frame.analysis.face.is_some(),
            dark: ev.frame.analysis.verdict == FaceVerdict::TooDark,
            score: ev.score,
            matched: ev.matched,
        };
        if cancel.load(Ordering::SeqCst) || send(writer, &event).is_err() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    // Close the camera, keep the models.
    engine.models = Some(pipeline.into_models());

    let report = result?;
    Ok(match report.outcome {
        AuthOutcome::Success => (Outcome::Success, None),
        AuthOutcome::NoMatch => (
            Outcome::NoMatch,
            Some(format!("best_score={:.3}", report.best_score)),
        ),
        AuthOutcome::AllDark => (Outcome::TooDark, None),
        AuthOutcome::NoFrames => (Outcome::Error, Some("no_frames".to_string())),
        AuthOutcome::Cancelled => (Outcome::Cancelled, None),
    })
}
