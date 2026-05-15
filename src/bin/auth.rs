use anyhow::{Context, Result};
use clap::Parser;
use opencv::prelude::MatTraitConst;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use faceauth::camera;
use faceauth::config::Config;
use faceauth::database::{Database, get_user_model_path};
use faceauth::detection::HaarCascadeDetector;
use faceauth::logger;
use faceauth::recognition::FaceRecognizer;

#[derive(Parser)]
#[command(name = "faceauth-auth")]
#[command(about = "Face authentication daemon for PAM integration")]
struct Args {
    /// Username to authenticate (optional)
    #[arg(short, long)]
    user: Option<String>,

    /// Configuration file path
    #[arg(short, long, default_value = "/etc/faceauth/config.toml")]
    config: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Resolve the account being authenticated. PAM does not set `USER` for `pam_exec` children;
/// Linux-PAM typically sets `PAM_USER`. (`-u` on the command line always wins.)
fn resolve_pam_username(cli_user: Option<String>) -> Result<String, String> {
    if let Some(u) = cli_user {
        let u = u.trim().to_string();
        if !u.is_empty() {
            return Ok(u);
        }
    }
    for key in ["PAM_USER", "USER", "LOGNAME"] {
        if let Ok(val) = env::var(key) {
            let val = val.trim().to_string();
            if !val.is_empty() {
                return Ok(val);
            }
        }
    }
    Err(
        "No username: pass -u on the pam_exec line, or ensure PAM_USER is set (standard for pam_exec). \
         Note: in /etc/pam.d/* the string $USER is NOT expanded by the shell — use -u <name> or omit -u and rely on PAM_USER."
            .to_string(),
    )
}

fn main() -> Result<()> {
    logger::init_from_env();
    let args = Args::parse();
    if args.verbose {
        log::info!("Verbose output enabled");
    }

    let user = match resolve_pam_username(args.user) {
        Ok(u) => u,
        Err(msg) => {
            log::error!("{}", msg);
            std::process::exit(10);
        }
    };

    log::info!("Starting face authentication for user {}", user);

    // Load configuration
    let config = Config::load(&args.config).context("Failed to load config")?;

    // Load database
    let model_path = get_user_model_path(&user)?;
    let db = Database::load(&model_path)?;
    if db.get_user(&user).is_none() {
        log::error!("No face model found for user {}", user);

        std::process::exit(10); // Howdy uses exit code 10 for missing model
    }

    // Initialize camera
    let mut camera = camera::Camera::open(
        &config.video.device_path,
        config.video.max_height,
        config.video.rotate,
    )?;

    if config.video.exposure >= 0 {
        camera.set_exposure(config.video.exposure as f64)?;
    }

    if config.video.ir_mode {
        log::info!("IR mode: darkness filter disabled; use the same IR device for enrollment");
    }

    let haar_neighbors = if config.video.ir_mode { 2 } else { 3 };
    let mut detector = HaarCascadeDetector::with_min_neighbors(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        haar_neighbors,
    )?;

    // Initialize face recognizer
    let mut recognizer = FaceRecognizer::load(&config.recognition.model_path)
        .context("Failed to load recognition model")?;

    // Main authentication loop
    let start = Instant::now();
    let timeout = Duration::from_secs(config.video.timeout as u64);
    let dark_threshold = config.video.dark_threshold;
    let certainty_threshold = config.video.certainty / 10.0; // convert from howdy scale
    let mut valid_frames = 0;
    let mut dark_tries = 0;
    let mut lowest_certainty = f32::INFINITY;

    while start.elapsed() < timeout {
        // Read frame
        let (color, gray) = match camera.read_frame() {
            Ok(frames) => frames,
            Err(e) => {
                log::warn!("Failed to read frame: {}", e);
                continue;
            }
        };

        // Check darkness
        let darkness = match camera::darkness(&gray) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if darkness >= 100.0 {
            continue;
        }
        valid_frames += 1;
        if !config.video.ir_mode && darkness > dark_threshold {
            dark_tries += 1;
            continue;
        }

        // Detect faces
        let faces = match detector.detect(&color) {
            Ok(faces) => faces,
            Err(e) => {
                log::warn!("Detection error: {}", e);
                continue;
            }
        };

        for face in faces {
            if face.confidence < config.detection.confidence_threshold as f32 {
                continue;
            }

            // Crop face
            let crop = color.roi(face.bbox)?.try_clone()?;
            // Extract embedding
            let embedding = match recognizer.extract(&crop) {
                Ok(emb) => emb,
                Err(e) => {
                    log::warn!("Embedding extraction failed: {}", e);
                    continue;
                }
            };

            // Verify against user's model
            let distance = db.verify(&user, &embedding, certainty_threshold as f32);
            if distance {
                log::info!("Authentication successful for {}", user);
                std::process::exit(0);
            } else {
                // Update lowest certainty
                let cert = embedding.euclidean_distance(&embedding); // placeholder
                if cert < lowest_certainty {
                    lowest_certainty = cert;
                }
            }
        }
    }

    // Timeout or no match
    log::error!("Authentication failed for {}", user); // general failure
    if dark_tries == valid_frames {
        log::error!("All frames were too dark"); // "All frames were too dark"
        std::process::exit(13); // exit code 13
    }
    std::process::exit(11); // exit code 11
}
