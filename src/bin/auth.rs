use anyhow::{Context, Result};
use clap::Parser;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use opencv::prelude::MatTraitConst;

use faceauth::camera;
use faceauth::config::Config;
use faceauth::database::{Database, get_user_model_path_for_user};
use faceauth::detection::{create_detector, crop_face};
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
    let model_path = get_user_model_path_for_user(&user)?;
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
    let mut detector = create_detector(
        Some(&config.detection.yunet_path).filter(|p| !p.is_empty()).map(|x| x.as_str()),
        &config.detection.model_path,
        config.detection.confidence_threshold as f32,
        config.detection.nms_threshold as f32,
        config.detection.use_cnn,
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        haar_neighbors,
    )
    .context("Failed to initialize face detector")?;

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

        // Filter faces by size and confidence
        let img_area = color.rows() * color.cols();
        let min_area = (img_area as f64 * config.detection.min_face_size_ratio).max(1.0) as i32;
        let max_area = (img_area as f64 * config.detection.max_face_size_ratio).max(1.0) as i32;

        let valid_faces: Vec<_> = faces
            .into_iter()
            .filter(|f| {
                let area = f.bbox.width * f.bbox.height;
                area >= min_area && area <= max_area
                    && f.confidence >= config.detection.confidence_threshold as f32
            })
            .collect();

        for face in valid_faces {
            // Crop face with padding (adds context like ears/chin for better recognition)
            let crop = match crop_face(&color, &face.bbox, config.detection.face_padding) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("Failed to crop face: {}", e);
                    continue;
                }
            };

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
                // Update lowest distance for logging
                if let Some(model) = db.get_user(&user) {
                    let dist = model.best_match_distance(&embedding);
                    if dist < lowest_certainty {
                        lowest_certainty = dist;
                    }
                }
            }
        }
    }

    // Timeout or no match
    log::error!("Authentication failed for {}", user);
    if dark_tries == valid_frames {
        log::error!("All frames were too dark");
        std::process::exit(13); // exit code 13
    }
    std::process::exit(11); // exit code 11
}
