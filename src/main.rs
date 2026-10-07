use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use log::info;
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use faceauth::{
    authenticate::{AuthOutcome, authenticate},
    calibration, camera,
    config::Config,
    database::{
        DISABLED_FLAG, Database, EnrollMerge, Enrollment, ImportPayload, MODELS_DIR, ModelSummary,
        VerifyPayload, VerifyResult, legacy_model_path, load_legacy, load_user_model,
        user_model_path, validate_vectors,
    },
    doctor,
    enroll::{self, EnrollEvent, EnrollParams},
    i18n::{t, tf},
    matching::file_fingerprint,
    pipeline::Pipeline,
    privilege::{authorize_for_user, authorize_global, pkexec_caller},
};

/// Largest JSON payload accepted on stdin by `import` / `verify`.
const MAX_STDIN_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Parser)]
#[command(name = "faceauth")]
#[command(about = "Face authentication system for Linux", long_about = None)]
struct Cli {
    /// Config file (default: /etc/faceauth/config.toml as root, else ./faceauth.toml,
    /// ~/.config/faceauth/config.toml, /etc/faceauth/config.toml). Ignored via pkexec.
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Test camera capture
    TestCamera {
        /// Camera device path or index (default: video.device_path)
        #[arg(short, long)]
        device: Option<String>,
        /// Number of frames to capture
        #[arg(short, long, default_value = "5")]
        frames: usize,
    },
    /// Generate default configuration file
    Config {
        /// Output path for config file
        #[arg(short, long, default_value = "faceauth.toml")]
        output: PathBuf,
    },
    /// Enroll a face for a user (root)
    Add {
        /// Username
        #[arg(short, long)]
        user: String,
        /// Label for this model (optional)
        #[arg(short, long)]
        label: Option<String>,
        /// Number of samples to capture
        #[arg(short, long, default_value = "9")]
        samples: usize,
        /// Camera device path or index (default: video.device_path, same as faceauth-auth)
        #[arg(short, long)]
        device: Option<String>,
        /// IR / low-light: skip darkness filter and relax Haar (same as `ir_mode` in config)
        #[arg(long)]
        ir: bool,
        /// Append new samples instead of replacing
        #[arg(long)]
        append: bool,
        /// Named appearance variant (e.g. glasses). Without `--append`, replaces that variant's samples.
        #[arg(long, value_name = "NAME")]
        variant: Option<String>,
    },
    /// Capture an enrollment as the desktop user and print JSON lines; the final
    /// `payload` event is what `pkexec faceauth import` reads on stdin (no root needed)
    Capture {
        /// Number of samples to capture
        #[arg(short, long, default_value = "9")]
        samples: usize,
        /// Camera device path or index (default: video.device_path)
        #[arg(short, long)]
        device: Option<String>,
        /// IR / low-light mode (same as `ir_mode` in config)
        #[arg(long)]
        ir: bool,
        /// Append to the model instead of replacing it
        #[arg(long)]
        append: bool,
        /// Named appearance variant (e.g. glasses)
        #[arg(long, value_name = "NAME")]
        variant: Option<String>,
        /// Label for the model or variant
        #[arg(short, long)]
        label: Option<String>,
        /// Include small grayscale previews (PGM data URLs, ~5 fps) in frame events
        #[arg(long)]
        preview: bool,
    },
    /// List enrolled face models (root)
    List {
        /// Username (if omitted, list all users)
        user: Option<String>,
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },
    /// Remove a user's model, or one variant of it (root)
    Remove {
        /// Username
        #[arg(short, long)]
        user: String,
        /// Only remove this variant
        #[arg(long, value_name = "NAME")]
        variant: Option<String>,
    },
    /// Rename an appearance variant (root)
    RenameVariant {
        #[arg(short, long)]
        user: String,
        from: String,
        to: String,
    },
    /// Delete all face data of a user (root)
    Clear {
        #[arg(short, long)]
        user: String,
    },
    /// Disable face authentication for everyone (root)
    Disable,
    /// Re-enable face authentication (root)
    Enable,
    /// Run the authentication loop against the enrolled model (root)
    Test {
        #[arg(short, long)]
        user: String,
        /// Override authentication timeout in seconds (without editing config)
        #[arg(long)]
        timeout: Option<u32>,
        /// IR / low-light: same as `ir_mode` in config (for testing against IR enrollment)
        #[arg(long)]
        ir: bool,
    },
    /// Measure your own match scores and suggest a distance threshold (root)
    Calibrate {
        #[arg(short, long)]
        user: String,
        /// Number of good frames to score
        #[arg(long, default_value = "40")]
        frames: usize,
    },
    /// Move a model from ~/.local/share/faceauth (≤ 0.2) into the root store (root)
    Migrate {
        #[arg(short, long)]
        user: String,
        /// Overwrite a model already in the root store
        #[arg(long)]
        force: bool,
    },
    /// Import embeddings captured by faceauth-ui (JSON on stdin; used via pkexec)
    Import {
        #[arg(short, long)]
        user: String,
    },
    /// Score probe embeddings from faceauth-ui (JSON on stdin; used via pkexec)
    Verify {
        #[arg(short, long)]
        user: String,
    },
    /// Check camera, models, OpenVINO/NPU, storage and PAM setup
    Doctor {
        /// User whose model to check (default: SUDO_USER / USER)
        #[arg(short, long)]
        user: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    faceauth::logger::init_from_env();
    let cli = Cli::parse();
    let (cfg, source) = load_config(cli.config.as_deref())?;

    match cli.command {
        Commands::TestCamera { device, frames } => test_camera(
            &cfg,
            device.as_deref().unwrap_or(&cfg.video.device_path),
            frames,
        ),
        Commands::Config { output } => {
            Config::default().save(&output)?;
            info!("Default config saved to {}", output.display());
            Ok(())
        }
        Commands::Add {
            user,
            label,
            samples,
            device,
            ir,
            append,
            variant,
        } => {
            authorize_for_user(&user)?;
            let variant = variant
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            let params = EnrollParams {
                merge: EnrollMerge::from_flags(variant.is_some(), append),
                username: user,
                label,
                samples,
                device,
                ir,
                variant,
            };
            let result = enroll::enroll_user(cfg, params, print_enroll_event);
            eprintln!();
            result
        }
        Commands::Capture {
            samples,
            device,
            ir,
            append,
            variant,
            label,
            preview,
        } => {
            let variant = variant
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            let merge = EnrollMerge::from_flags(variant.is_some(), append);
            capture(cfg, samples, device, ir, merge, variant, label, preview)
        }
        Commands::List { user, json } => list_models(user, json),
        Commands::Remove { user, variant } => remove(&user, variant.as_deref()),
        Commands::RenameVariant { user, from, to } => {
            authorize_for_user(&user)?;
            modify_model(&user, |m| m.rename_variant(&from, &to))
        }
        Commands::Clear { user } => remove(&user, None),
        Commands::Disable => set_disabled(true),
        Commands::Enable => set_disabled(false),
        Commands::Test { user, timeout, ir } => test_auth(cfg, &user, timeout, ir),
        Commands::Calibrate { user, frames } => calibrate(&cfg, &user, frames),
        Commands::Migrate { user, force } => migrate(&user, force),
        Commands::Import { user } => import(&cfg, &user),
        Commands::Verify { user } => verify(&cfg, &user),
        Commands::Doctor { user, json } => {
            let user = user
                .or_else(|| std::env::var("SUDO_USER").ok())
                .or_else(|| std::env::var("USER").ok());
            let checks = doctor::run(&cfg, source.as_deref(), user.as_deref());
            if json {
                println!("{}", serde_json::to_string_pretty(&checks)?);
            } else {
                doctor::print(&checks);
            }
            Ok(())
        }
    }
}

fn load_config(explicit: Option<&Path>) -> Result<(Config, Option<PathBuf>)> {
    match explicit {
        Some(_) if pkexec_caller().is_some() => {
            bail!("--config is not accepted via pkexec")
        }
        Some(p) => Ok((
            Config::load_resolved(p).with_context(|| format!("Failed to load {}", p.display()))?,
            Some(p.to_path_buf()),
        )),
        None => Ok(Config::discover()),
    }
}

fn print_enroll_event(ev: &EnrollEvent) {
    let line = match ev {
        EnrollEvent::Sample { cur, tot } => tf("enroll.progress", &[cur, tot]),
        EnrollEvent::Rejected(v) => t(v.message_key()).to_string(),
        EnrollEvent::Hint(h) => t(h.message_key()).to_string(),
        EnrollEvent::Duplicate => t("hint.duplicate").to_string(),
    };
    eprint!("\r\x1b[2K{line}");
    let _ = std::io::stderr().flush();
}

/// Print one JSON event line for `capture`.
fn emit(value: serde_json::Value) -> Result<()> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}")?;
    out.flush()?;
    Ok(())
}

/// Unprivileged enrollment capture for other front ends (the embeddings never
/// touch the disk: the caller pipes the `payload` event into `faceauth import`).
#[allow(clippy::too_many_arguments)]
fn capture(
    mut cfg: Config,
    samples: usize,
    device: Option<String>,
    ir: bool,
    merge: EnrollMerge,
    variant: Option<String>,
    label: Option<String>,
    preview: bool,
) -> Result<()> {
    if ir {
        cfg.video.ir_mode = true;
    }
    let device = device.unwrap_or_else(|| cfg.video.device_path.clone());
    let mut pipeline = match Pipeline::open(&cfg, &device) {
        Ok(p) => p,
        Err(e) => {
            emit(serde_json::json!({
                "event": "error",
                "reason": "camera",
                "detail": format!("{e:#}"),
            }))?;
            std::process::exit(1);
        }
    };
    let model_id = pipeline.recognizer.model_id().to_string();
    emit(serde_json::json!({
        "event": "started",
        "device": device,
        "ir": cfg.video.ir_mode,
        "backend": pipeline.describe(),
        "samples": samples,
    }))?;

    let pause = Duration::from_millis(cfg.video.frame_interval_ms);
    let preview_every = Duration::from_millis(200);
    let mut last_preview: Option<Instant> = None;
    let mut session = enroll::EnrollSession::new(samples, &cfg);
    while !session.is_done() {
        let step = match session.step(&mut pipeline, &cfg) {
            Ok(step) => step,
            Err(_) => {
                std::thread::sleep(pause);
                continue;
            }
        };
        let mut frame = serde_json::json!({
            "event": "frame",
            "face": step.frame.analysis.face.is_some(),
            "verdict": step.frame.analysis.verdict.key(),
        });
        if preview && last_preview.is_none_or(|t| t.elapsed() >= preview_every) {
            last_preview = Some(Instant::now());
            if let Ok(url) = faceauth::preview::gray_preview(&step.frame.gray) {
                frame["preview"] = serde_json::Value::String(url);
            }
        }
        emit(frame)?;
        match step.event {
            Some(EnrollEvent::Sample { cur, tot }) => {
                emit(serde_json::json!({ "event": "sample", "n": cur, "of": tot }))?
            }
            Some(EnrollEvent::Hint(h)) => emit(serde_json::json!({
                "event": "hint",
                "hint": h.message_key().trim_start_matches("hint."),
            }))?,
            Some(EnrollEvent::Duplicate) => {
                emit(serde_json::json!({ "event": "hint", "hint": "duplicate" }))?
            }
            Some(EnrollEvent::Rejected(_)) | None => {}
        }
    }

    let embeddings = session.into_vectors();
    if embeddings.is_empty() {
        emit(serde_json::json!({ "event": "error", "reason": "no_samples" }))?;
        std::process::exit(1);
    }
    let payload = ImportPayload {
        merge,
        variant,
        label,
        model_id,
        embeddings,
    };
    emit(serde_json::json!({ "event": "payload", "payload": payload }))
}

fn test_camera(cfg: &Config, device: &str, frames: usize) -> Result<()> {
    info!("Opening camera {}...", device);
    let mut cam = camera::Camera::open_configured(device, &cfg.video)?;
    info!("Camera opened: {}x{}", cam.width(), cam.height());

    for i in 0..frames {
        match cam.read_frame() {
            Ok((_color, gray)) => {
                let darkness = camera::darkness(&gray)?;
                info!("Frame {}: darkness={:.2}%", i, darkness);
            }
            Err(e) => {
                log::error!("Failed to read frame {}: {}", i, e);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    info!("Test completed.");
    Ok(())
}

fn list_models(user: Option<String>, json: bool) -> Result<()> {
    let summaries: Vec<ModelSummary> = match user {
        Some(user) => {
            authorize_for_user(&user)?;
            load_user_model(&user)?
                .map(|m| m.summary(&user))
                .into_iter()
                .collect()
        }
        None => {
            authorize_global()?;
            let mut out = Vec::new();
            let entries = match std::fs::read_dir(MODELS_DIR) {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return print_summaries(&[], json);
                }
                Err(e) => return Err(e.into()),
            };
            for entry in entries {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                match Database::load_trusted(&path) {
                    Ok(Some(db)) => out.extend(db.users.iter().map(|(u, m)| m.summary(u))),
                    Ok(None) => {}
                    Err(e) => log::warn!("{e:#}"),
                }
            }
            out.sort_by(|a, b| a.user.cmp(&b.user));
            out
        }
    };
    print_summaries(&summaries, json)
}

fn print_summaries(summaries: &[ModelSummary], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(summaries)?);
        return Ok(());
    }
    if summaries.is_empty() {
        println!("No enrolled face models found");
    }
    for s in summaries {
        let variants = s
            .variants
            .iter()
            .map(|v| format!("{}:{}", v.label, v.samples))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "User {}: label='{}', primary={}, variants=[{}], created_at={}, recognizer={}",
            s.user,
            s.label,
            s.primary_samples,
            if variants.is_empty() {
                "-".into()
            } else {
                variants
            },
            s.created_at,
            s.model_id.as_deref().unwrap_or("unknown (legacy)"),
        );
    }
    Ok(())
}

/// Load, change and save one user's model.
fn modify_model(
    user: &str,
    f: impl FnOnce(&mut faceauth::database::FaceModel) -> Result<()>,
) -> Result<()> {
    let path = user_model_path(user)?;
    let mut db = Database::load_trusted(&path)?.unwrap_or_default();
    let model = db
        .get_user_mut(user)
        .with_context(|| format!("No face model enrolled for {user}"))?;
    f(model)?;
    model.updated_at = Some(chrono::Utc::now());
    db.save_secure(&path)?;
    println!("{}", t("models.done"));
    Ok(())
}

fn remove(user: &str, variant: Option<&str>) -> Result<()> {
    authorize_for_user(user)?;
    if let Some(v) = variant {
        return modify_model(user, |m| m.remove_variant(v));
    }
    let path = user_model_path(user)?;
    match std::fs::remove_file(&path) {
        Ok(()) => println!("Removed {}", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("No face model enrolled for {user}")
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

fn set_disabled(disabled: bool) -> Result<()> {
    authorize_global()?;
    let flag = Path::new(DISABLED_FLAG);
    if disabled {
        if let Some(dir) = flag.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(flag, b"")?;
        println!("Face authentication disabled ({DISABLED_FLAG})");
    } else {
        match std::fs::remove_file(flag) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        println!("Face authentication enabled");
    }
    Ok(())
}

fn test_auth(mut cfg: Config, user: &str, timeout_override: Option<u32>, ir: bool) -> Result<()> {
    authorize_for_user(user)?;
    if ir {
        cfg.video.ir_mode = true;
    }
    let model =
        load_user_model(user)?.with_context(|| format!("No model enrolled for user {user}"))?;
    let mut pipeline = Pipeline::open(&cfg, &cfg.video.device_path)?;
    let timeout = Duration::from_secs(timeout_override.unwrap_or(cfg.video.timeout).max(3) as u64);
    let threshold = cfg.recognition.distance_threshold;
    let report = authenticate(&mut pipeline, &cfg, &model, timeout, |ev| {
        let line = match ev.score {
            Some(s) => format!(
                "score {s:.4} (threshold {threshold:.2}) {}",
                if ev.matched { "✔" } else { "✘" }
            ),
            None => t(ev.frame.analysis.verdict.message_key()).to_string(),
        };
        eprint!("\r\x1b[2K{line}");
        ControlFlow::Continue(())
    })?;
    eprintln!();
    println!("{}", report.summary());
    if report.outcome == AuthOutcome::Success {
        println!("Authentication test PASSED for {user}");
        Ok(())
    } else {
        eprintln!("Authentication test FAILED for {user}");
        std::process::exit(1);
    }
}

fn calibrate(cfg: &Config, user: &str, frames: usize) -> Result<()> {
    authorize_for_user(user)?;
    let model =
        load_user_model(user)?.with_context(|| format!("No model enrolled for user {user}"))?;
    let mut pipeline = Pipeline::open(cfg, &cfg.video.device_path)?;
    let k = cfg.recognition.top_k;
    eprintln!("Look at the camera as you normally would when logging in…");
    let result = calibration::calibrate(
        &mut pipeline,
        cfg,
        &model,
        frames,
        Duration::from_secs(60),
        |p| {
            match p {
                calibration::Progress::Scored { n, of } => eprint!("\r\x1b[2K{n}/{of}"),
                calibration::Progress::Rejected(v) => {
                    eprint!("\r\x1b[2K{}", t(v.message_key()))
                }
            }
            ControlFlow::Continue(())
        },
    )?;
    eprintln!();
    let Some(c) = result else {
        bail!("No usable frames captured");
    };
    println!("Frames scored: {}", c.frames);
    println!(
        "Score (top-{k}): min {:.4}, median {:.4}, p95 {:.4}",
        c.min, c.median, c.p95
    );
    if let Some((mean, sd)) = c.face_brightness {
        println!(
            "Face region brightness: mean {mean:.1}, stddev {sd:.1} (liveness thresholds: {:.1} / {:.1})",
            cfg.liveness.min_face_brightness, cfg.liveness.min_face_stddev
        );
    }
    println!("Current distance_threshold: {:.3}", c.current);
    if c.raw_suggestion < calibration::MIN_SUGGESTED_THRESHOLD {
        println!(
            "Suggested distance_threshold: {:.3} (frames were nearly identical; \
             p95 × 1.15 = {:.3} would reject normal variation)",
            c.suggested, c.raw_suggestion
        );
    } else {
        println!(
            "Suggested distance_threshold: {:.3} (p95 × 1.15)",
            c.suggested
        );
    }
    if c.raw_suggestion > 0.9 {
        println!(
            "⚠ Your own scores are high: re-enroll with more samples or check the camera/lighting before raising the threshold."
        );
    }
    Ok(())
}

fn migrate(user: &str, force: bool) -> Result<()> {
    authorize_for_user(user)?;
    let (legacy, uid) = legacy_model_path(user)?;
    let db = load_legacy(&legacy, uid)?
        .with_context(|| format!("No legacy model at {}", legacy.display()))?;
    let mut model = db
        .users
        .get(user)
        .cloned()
        .with_context(|| format!("{} has no entry for {user}", legacy.display()))?;
    let target = user_model_path(user)?;
    let mut store = Database::load_trusted(&target)?.unwrap_or_default();
    if store.get_user(user).is_some() && !force {
        bail!(
            "{} already has a model; pass --force to overwrite",
            target.display()
        );
    }
    // Recognizer unknown for legacy files: keep None so auth warns until re-enrolled.
    model.model_id = None;
    model.updated_at = Some(chrono::Utc::now());
    store.users.insert(user.to_string(), model);
    store.save_secure(&target)?;
    println!("Migrated {} → {}", legacy.display(), target.display());
    println!(
        "The old file is no longer used; remove it with: rm {}",
        legacy.display()
    );
    Ok(())
}

fn read_stdin_json<T: serde::de::DeserializeOwned>() -> Result<T> {
    let mut buf = String::new();
    std::io::stdin()
        .take(MAX_STDIN_BYTES)
        .read_to_string(&mut buf)?;
    serde_json::from_str(&buf).context("Invalid JSON on stdin")
}

/// Fingerprint of the system recognizer (without loading the network).
fn system_model_id(cfg: &Config) -> Result<String> {
    file_fingerprint(Path::new(&cfg.recognition.model_path))
}

fn import(cfg: &Config, user: &str) -> Result<()> {
    authorize_for_user(user)?;
    let payload: ImportPayload = read_stdin_json()?;
    validate_vectors(&payload.embeddings)?;
    if payload.model_id != system_model_id(cfg)? {
        bail!("faceauth-ui used a different recognition model than /etc/faceauth/config.toml");
    }
    let count = payload.embeddings.len();
    let path = user_model_path(user)?;
    let mut db = Database::load_trusted(&path)?.unwrap_or_default();
    db.apply_enrollment(
        user,
        Enrollment {
            merge: payload.merge,
            variant: payload.variant,
            label: payload.label,
            vectors: payload.embeddings,
            model_id: payload.model_id,
        },
    )?;
    db.save_secure(&path)?;
    println!("{}", serde_json::json!({ "saved": count }));
    Ok(())
}

fn verify(cfg: &Config, user: &str) -> Result<()> {
    authorize_for_user(user)?;
    let payload: VerifyPayload = read_stdin_json()?;
    let dim = validate_vectors(&payload.embeddings)?;
    if payload.model_id != system_model_id(cfg)? {
        bail!("faceauth-ui used a different recognition model than /etc/faceauth/config.toml");
    }
    let model = load_user_model(user)?.with_context(|| format!("No model enrolled for {user}"))?;
    model.check_compatible(&payload.model_id, dim)?;
    let k = cfg.recognition.top_k;
    let result = VerifyResult {
        scores: payload
            .embeddings
            .iter()
            .map(|v| model.match_score(v, k))
            .collect(),
        threshold: cfg.recognition.distance_threshold as f32,
        required_matches: cfg.recognition.required_matches,
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
