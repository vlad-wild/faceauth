use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::{error, info};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use faceauth::{
    camera,
    config,
    database::{Database, get_user_model_path},
    detection::HaarCascadeDetector,
    enroll::{self, EnrollMerge, EnrollParams},
    recognition::FaceRecognizer,
};

#[derive(Parser)]
#[command(name = "faceauth")]
#[command(about = "Face authentication system for Linux", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Test camera capture
    TestCamera {
        /// Camera device path or index
        #[arg(short, long, default_value = "0")]
        device: String,
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
    /// Add a new face model for a user
    Add {
        /// Username
        #[arg(short, long)]
        user: String,
        /// Label for this model (optional)
        #[arg(short, long)]
        label: Option<String>,
        /// Number of samples to capture
        #[arg(short, long, default_value = "5")]
        samples: usize,
        /// Camera device path or index
        #[arg(short, long, default_value = "0")]
        device: String,
        /// IR / low-light: skip darkness filter and relax Haar (same as `ir_mode` in config)
        #[arg(long)]
        ir: bool,
        /// Append new samples to the primary embedding set instead of replacing the whole model
        #[arg(long)]
        append: bool,
        /// Named appearance variant (e.g. glasses). Without `--append`, replaces that variant's samples.
        #[arg(long, value_name = "NAME")]
        variant: Option<String>,
    },
    /// List face models for a user
    List {
        /// Username (if omitted, list all users)
        user: Option<String>,
    },
    /// Remove a specific model or user
    Remove {
        /// Username
        #[arg(short, long)]
        user: String,
        /// Model index (if omitted, remove all models for this user)
        #[arg(short, long)]
        index: Option<usize>,
    },
    /// Clear all face models for a user
    Clear {
        /// Username
        #[arg(short, long)]
        user: String,
    },
    /// Disable or enable face authentication
    Disable {
        /// Disable (true) or enable (false)
        #[arg(short, long)]
        disable: bool,
    },
    /// Test authentication for a user
    Test {
        /// Username
        #[arg(short, long)]
        user: String,
        /// Override authentication timeout in seconds (without editing config)
        #[arg(long)]
        timeout: Option<u32>,
        /// IR / low-light: same as `ir_mode` in config (for testing against IR enrollment)
        #[arg(long)]
        ir: bool,
    },
    /// Authenticate a user (for internal use)
    Auth {
        /// Username
        #[arg(short, long)]
        user: String,
    },
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::TestCamera { device, frames } => {
            test_camera(&device, frames)?;
        }
        Commands::Config { output } => {
            let config = config::Config::default();
            config.save(&output)?;
            info!("Default config saved to {}", output.display());
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
            info!("Adding face model for user {} (samples: {})", user, samples);
            add_model(&user, label, samples, &device, ir, append, variant)?;
        }
        Commands::List { user } => {
            list_models(user)?;
        }
        Commands::Remove { user, index } => {
            info!("Removing model for user {} index {:?}", user, index);
            todo!("Remove command not yet implemented");
        }
        Commands::Clear { user } => {
            info!("Clearing all models for user {}", user);
            todo!("Clear command not yet implemented");
        }
        Commands::Disable { disable } => {
            info!("{} face authentication", if disable { "Disabling" } else { "Enabling" });
            todo!("Disable command not yet implemented");
        }
        Commands::Test { user, timeout, ir } => {
            info!("Testing authentication for user {}", user);
            test_auth(&user, timeout, ir)?;
        }
        Commands::Auth { user } => {
            info!("Authenticating user {} (internal)", user);
            error!("Use faceauth-auth binary for PAM authentication");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn test_camera(device: &str, frames: usize) -> Result<()> {
    info!("Opening camera {}...", device);
    let mut cam = camera::Camera::open(device, 320.0, 0)?;
    info!("Camera opened: {}x{}", cam.width(), cam.height());

    for i in 0..frames {
        match cam.read_frame() {
            Ok((_color, gray)) => {
                let darkness = camera::darkness(&gray)?;
                info!("Frame {}: darkness={:.2}%", i, darkness);
            }
            Err(e) => {
                error!("Failed to read frame {}: {}", i, e);
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    info!("Test completed.");
    Ok(())
}

fn list_models(user: Option<String>) -> Result<()> {
    info!("Listing models for {:?}", user);
    if let Some(user) = user {
        let path = get_user_model_path(&user)?;
        let db = Database::load(&path)?;
        if let Some(model) = db.get_user(&user) {
            let ext = model
                .extensions
                .iter()
                .map(|e| format!("{}:{}", e.label, e.embeddings.len()))
                .collect::<Vec<_>>()
                .join(", ");
            info!(
                "User {}: label='{}', primary={}, variants=[{}], created_at={}",
                user,
                model.label,
                model.embeddings.len(),
                if ext.is_empty() { "-".into() } else { ext },
                model.created_at
            );
        } else {
            info!("No model found for user {}", user);
        }
        return Ok(());
    }

    let models_dir = faceauth::database::get_models_dir()?;
    let mut count = 0usize;
    for entry in std::fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let db = Database::load(&path)?;
        for (username, model) in db.users {
            count += 1;
            let ext = model
                .extensions
                .iter()
                .map(|e| format!("{}:{}", e.label, e.embeddings.len()))
                .collect::<Vec<_>>()
                .join(", ");
            info!(
                "User {}: label='{}', primary={}, variants=[{}], created_at={}",
                username,
                model.label,
                model.embeddings.len(),
                if ext.is_empty() { "-".into() } else { ext },
                model.created_at
            );
        }
    }
    if count == 0 {
        info!("No enrolled face models found");
    }
    Ok(())
}

fn add_model(
    user: &str,
    label: Option<String>,
    samples: usize,
    device: &str,
    ir: bool,
    append: bool,
    variant: Option<String>,
) -> Result<()> {
    let variant_clean = variant.and_then(|v| {
        let t = v.trim().to_string();
        (!t.is_empty()).then_some(t)
    });
    let merge = match (&variant_clean, append) {
        (Some(_), true) => EnrollMerge::AppendVariant,
        (Some(_), false) => EnrollMerge::ReplaceVariant,
        (None, true) => EnrollMerge::AppendPrimary,
        (None, false) => EnrollMerge::ReplaceAll,
    };
    let cfg = enroll::load_cli_config();
    enroll::enroll_user_with_progress(
        cfg,
        EnrollParams {
            username: user.to_string(),
            label,
            samples,
            device: device.to_string(),
            ir,
            merge,
            variant: variant_clean,
        },
        |_, _| {},
    )
}

fn test_auth(user: &str, timeout_override: Option<u32>, ir: bool) -> Result<()> {
    let mut cfg = enroll::load_cli_config();
    if ir {
        cfg.video.ir_mode = true;
    }
    let model_path = get_user_model_path(user)?;
    let db = Database::load(&model_path)?;
    if db.get_user(user).is_none() {
        anyhow::bail!("No model enrolled for user {}", user);
    }

    let mut cam = camera::Camera::open(&cfg.video.device_path, cfg.video.max_height, cfg.video.rotate)?;
    let haar_neighbors = if cfg.video.ir_mode { 2 } else { 3 };
    let mut detector = HaarCascadeDetector::with_min_neighbors(enroll::DEFAULT_HAAR_CASCADE, haar_neighbors)?;
    let mut recognizer = FaceRecognizer::load(&cfg.recognition.model_path)?;
    let timeout_secs = timeout_override.unwrap_or(cfg.video.timeout).max(3);
    let timeout = Duration::from_secs(timeout_secs as u64);
    let started = Instant::now();
    let mut attempts = 0usize;
    let probe = loop {
        if started.elapsed() >= timeout {
            anyhow::bail!(
                "Could not detect a face during test authentication within {}s (attempts: {})",
                timeout.as_secs(),
                attempts
            );
        }
        attempts += 1;
        if let Some(embedding) = enroll::capture_single_embedding(&mut cam, &mut detector, &mut recognizer, &cfg)? {
            break embedding;
        }
        std::thread::sleep(Duration::from_millis(60));
    };
    let threshold = cfg.recognition.distance_threshold as f32;
    let distance = db
        .get_user(user)
        .map(|m| m.best_match_distance(&probe))
        .context("No model enrolled for user")?;
    info!(
        "Verification distance for {}: {:.4} (threshold {:.4}, attempts {})",
        user, distance, threshold, attempts
    );
    let pass = distance < threshold;
    if pass {
        info!("Authentication test PASSED for {}", user);
    } else {
        error!("Authentication test FAILED for {}", user);
        std::process::exit(1);
    }
    Ok(())
}
