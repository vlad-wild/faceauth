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
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::ops::ControlFlow;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use faceauth::authenticate::{AuthOutcome, authenticate};
use faceauth::calibration;
use faceauth::config::{Config, SYSTEM_CONFIG_PATH};
use faceauth::daemon::{
    DaemonStatus, Event, HISTORY_LEN, HistoryEntry, MAX_PASSWORD_ONLY_SECS, Outcome,
    PASSWORD_ONLY_DIR, RateLimiter, Request, SOCKET_PATH, peer_uid, sanitize_service,
};
use faceauth::database::{DISABLED_FLAG, load_user_model};
use faceauth::gate::{Gate, password_only_until, pre_auth_checks, unix_now};
use faceauth::logger;
use faceauth::pipeline::{FaceVerdict, Models, Pipeline};
use faceauth::privilege::{is_root, lookup_user, username_for_uid, validate_username};
use faceauth::session;

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
    /// Recent attempts per uid (memory only).
    history: Mutex<HashMap<u32, VecDeque<HistoryEntry>>>,
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
        history: Mutex::new(HashMap::new()),
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
    let Ok(request) = serde_json::from_str::<Request>(line.trim()) else {
        send(
            &mut writer,
            &Event::result(Outcome::Error, Some("bad_request")),
        )?;
        return Ok(());
    };
    let own_name = username_for_uid(uid)?;

    let (outcome, reason) = match request {
        Request::Verify { user, service } => {
            // Non-root callers are always verified as themselves.
            let user = match user {
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
                _ => own_name.clone(),
            };
            let service = sanitize_service(service.as_deref());
            let (outcome, reason) = verify(shared, uid, &user, reader, &mut writer);
            remember(shared, uid, &service, outcome, reason.as_deref());
            log::info!(
                "Result for {user} ({service}): {outcome:?}{}",
                reason
                    .as_deref()
                    .map(|r| format!(" ({r})"))
                    .unwrap_or_default()
            );
            (outcome, reason)
        }
        Request::Cancel => (Outcome::Error, Some("nothing_to_cancel".to_string())),
        Request::Status => match status(shared, &own_name) {
            Ok(st) => {
                send(
                    &mut writer,
                    &Event::Status {
                        status: Box::new(st),
                    },
                )?;
                (Outcome::Success, None)
            }
            Err(e) => {
                log::error!("Status for {own_name} failed: {e:#}");
                (Outcome::Error, Some("setup_error".to_string()))
            }
        },
        Request::History => {
            let entries = shared
                .history
                .lock()
                .unwrap()
                .get(&uid)
                .map(|h| h.iter().cloned().collect())
                .unwrap_or_default();
            send(&mut writer, &Event::History { entries })?;
            (Outcome::Success, None)
        }
        Request::PasswordOnly { seconds } => match set_password_only(uid, seconds) {
            Ok(()) => (Outcome::Success, None),
            Err(e) => {
                log::error!("password_only for uid {uid} failed: {e:#}");
                (Outcome::Error, Some("setup_error".to_string()))
            }
        },
        Request::Calibrate { frames } => {
            let frames = frames.unwrap_or(30).clamp(5, MAX_CALIBRATION_FRAMES);
            calibrate(shared, &own_name, frames, reader, &mut writer)
        }
    };

    let _ = send(&mut writer, &Event::result(outcome, reason.as_deref()));
    // Unblocks the watcher thread even if the client keeps the socket open.
    let _ = writer.shutdown(std::net::Shutdown::Both);
    Ok(())
}

/// Longest calibration a client may ask for.
const MAX_CALIBRATION_FRAMES: usize = 100;

/// Stop flag set when the client sends `cancel` or closes the connection.
fn watch_for_cancel(mut reader: BufReader<UnixStream>) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
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
        flag.store(true, Ordering::SeqCst);
    });
    cancel
}

fn verify(
    shared: &Shared,
    uid: u32,
    user: &str,
    reader: BufReader<UnixStream>,
    writer: &mut UnixStream,
) -> (Outcome, Option<String>) {
    if !shared.limiter.lock().unwrap().allowed(uid, Instant::now()) {
        return (Outcome::Busy, Some("rate_limited".to_string()));
    }
    let Ok(mut engine) = shared.engine.try_lock() else {
        return (Outcome::Busy, Some("attempt_in_progress".to_string()));
    };
    let cancel = watch_for_cancel(reader);
    let result = match attempt(&mut engine, user, writer, &cancel) {
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
        .record(uid, result.0, Instant::now());
    result
}

fn remember(shared: &Shared, uid: u32, service: &str, outcome: Outcome, reason: Option<&str>) {
    let mut history = shared.history.lock().unwrap();
    let list = history.entry(uid).or_default();
    if list.len() == HISTORY_LEN {
        list.pop_front();
    }
    list.push_back(HistoryEntry {
        time: unix_now(),
        service: service.to_string(),
        outcome,
        reason: reason.map(str::to_string),
    });
}

fn load_system_config() -> Result<Config> {
    Config::load_resolved(Path::new(SYSTEM_CONFIG_PATH))
        .with_context(|| format!("Failed to load {SYSTEM_CONFIG_PATH}"))
}

fn camera_present(device: &str) -> bool {
    !device.starts_with("/dev/") || Path::new(device).exists()
}

fn status(shared: &Shared, user: &str) -> Result<DaemonStatus> {
    let cfg = load_system_config()?;
    let uid = lookup_user(user)?.uid;
    let backend = shared.engine.try_lock().ok().and_then(|e| {
        e.models.as_ref().map(|m| {
            format!(
                "detector: {}, recognizer: {}",
                m.detector.describe(),
                m.recognizer.backend_info()
            )
        })
    });
    Ok(DaemonStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        user: user.to_string(),
        disabled: Path::new(DISABLED_FLAG).exists(),
        lid_closed: session::lid_closed(),
        password_only_until: password_only_until(Path::new(PASSWORD_ONLY_DIR), uid, unix_now()),
        camera_device: cfg.video.device_path.clone(),
        camera_present: camera_present(&cfg.video.device_path),
        ir_mode: cfg.video.ir_mode,
        backend,
        model: load_user_model(user)?.map(|m| m.summary(user)),
        threshold: cfg.recognition.distance_threshold,
        required_matches: cfg.recognition.required_matches,
    })
}

fn set_password_only(uid: u32, seconds: u64) -> Result<()> {
    let dir = Path::new(PASSWORD_ONLY_DIR);
    let path = dir.join(uid.to_string());
    if seconds == 0 {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755))?;
    let until = unix_now() + seconds.min(MAX_PASSWORD_ONLY_SECS) as i64;
    let tmp = dir.join(format!(".{uid}.tmp"));
    std::fs::write(&tmp, format!("{until}\n"))?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
    std::fs::rename(&tmp, &path)?;
    log::info!("password_only for uid {uid} until {until}");
    Ok(())
}

/// Open the camera with cached models; on failure the models are kept.
fn open_pipeline(engine: &mut Engine, cfg: &Config) -> Result<Pipeline> {
    let stamp = config_stamp();
    if engine.config_stamp != stamp {
        engine.models = None;
    }
    let models = match engine.models.take() {
        Some(models) => models,
        None => Models::load(cfg)?,
    };
    engine.config_stamp = stamp;
    match Pipeline::open_camera(cfg, &cfg.video.device_path) {
        Ok(camera) => Ok(Pipeline::with_models(camera, models)),
        Err(e) => {
            engine.models = Some(models);
            Err(e)
        }
    }
}

fn calibrate(
    shared: &Shared,
    user: &str,
    frames: usize,
    reader: BufReader<UnixStream>,
    writer: &mut UnixStream,
) -> (Outcome, Option<String>) {
    let Ok(mut engine) = shared.engine.try_lock() else {
        return (Outcome::Busy, Some("attempt_in_progress".to_string()));
    };
    let cancel = watch_for_cancel(reader);
    let mut run = || -> Result<(Outcome, Option<String>)> {
        let cfg = load_system_config()?;
        let model = match pre_auth_checks(&cfg, user, false)? {
            Gate::Proceed(model) => model,
            Gate::Skip(reason) => return Ok((Outcome::Skipped, Some(reason.key().to_string()))),
        };
        if !camera_present(&cfg.video.device_path) {
            return Ok((Outcome::Skipped, Some("camera_unavailable".to_string())));
        }
        let mut pipeline = open_pipeline(&mut engine, &cfg)?;
        let result = calibration::calibrate(
            &mut pipeline,
            &cfg,
            &model,
            frames,
            Duration::from_secs(60),
            |p| {
                let sent = match p {
                    calibration::Progress::Scored { n, of } => {
                        send(writer, &Event::CalibrationProgress { n, of }).is_ok()
                    }
                    calibration::Progress::Rejected(_) => true,
                };
                if !sent || cancel.load(Ordering::SeqCst) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        engine.models = Some(pipeline.into_models());
        Ok(match result? {
            Some(calibration) => {
                send(writer, &Event::Calibration { calibration })?;
                (Outcome::Success, None)
            }
            None if cancel.load(Ordering::SeqCst) => (Outcome::Cancelled, None),
            None => (Outcome::NoMatch, Some("no_usable_frames".to_string())),
        })
    };
    run().unwrap_or_else(|e| {
        log::error!("Calibration for {user} failed: {e:#}");
        (Outcome::Error, Some("setup_error".to_string()))
    })
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
    let cfg = load_system_config()?;

    let model = match pre_auth_checks(&cfg, user, false)? {
        Gate::Proceed(model) => model,
        Gate::Skip(reason) => return Ok((Outcome::Skipped, Some(reason.key().to_string()))),
    };

    // The camera key (or a privacy switch) removes the device: skip without waiting.
    if !camera_present(&cfg.video.device_path) {
        return Ok((Outcome::Skipped, Some("camera_unavailable".to_string())));
    }
    let mut pipeline = open_pipeline(engine, &cfg)?;
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
