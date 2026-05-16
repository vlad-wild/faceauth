//! Face enrollment (CLI and GUI).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use log::{debug, info};
use opencv::prelude::MatTraitConst;
use std::time::{Duration, Instant};

use crate::camera;
use crate::config::Config;
use crate::database::{Database, FaceModel, get_user_model_path};
use crate::detection::{create_detector, crop_face, Detector};
use crate::recognition::{FaceEmbedding, FaceRecognizer};

pub const DEFAULT_HAAR_CASCADE: &str =
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

/// Same as legacy CLI: only `./faceauth.toml` in current directory.
pub fn load_cli_config() -> Config {
    let local = PathBuf::from("faceauth.toml");
    match Config::load(&local) {
        Ok(c) => c,
        Err(_) => Config::default(),
    }
}

/// Try common config locations (for GUI / desktop). Returns loaded path for resolving relative model paths.
pub fn load_enrollment_config() -> (Config, Option<PathBuf>) {
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from("faceauth.toml")];
    if let Some(cfg_dir) = dirs::config_dir() {
        candidates.push(cfg_dir.join("faceauth").join("config.toml"));
    }
    candidates.push(Path::new("/etc/faceauth/config.toml").to_path_buf());

    for path in candidates {
        if !path.exists() {
            continue;
        }
        if let Ok(mut cfg) = Config::load(&path) {
            resolve_relative_model_paths(&mut cfg, path.parent());
            return (cfg, Some(path));
        }
    }

    (Config::default(), None)
}

fn resolve_relative_model_paths(cfg: &mut Config, base: Option<&Path>) {
    let Some(base) = base else { return; };
    let rec = Path::new(&cfg.recognition.model_path);
    if !rec.is_absolute() {
        cfg.recognition.model_path = base.join(rec).to_string_lossy().into_owned();
    }
    let det = Path::new(&cfg.detection.model_path);
    if !det.is_absolute() {
        cfg.detection.model_path = base.join(det).to_string_lossy().into_owned();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnrollMerge {
    /// Drop previous model for this user; save only this capture (clears extensions).
    #[default]
    ReplaceAll,
    /// Append new vectors to primary `embeddings`; keep extensions and primary label.
    AppendPrimary,
    /// Replace or create named extension (`--variant`); primary unchanged unless new user.
    ReplaceVariant,
    /// Append into named extension (`--variant` + `--append`).
    AppendVariant,
}

#[derive(Clone)]
pub struct EnrollParams {
    pub username: String,
    pub label: Option<String>,
    pub samples: usize,
    pub device: String,
    pub ir: bool,
    pub merge: EnrollMerge,
    /// Required when merge is ReplaceVariant or AppendVariant.
    pub variant: Option<String>,
}

/// Save face model for user (blocking).
pub fn enroll_user(cfg: Config, params: EnrollParams) -> Result<()> {
    enroll_user_with_progress(cfg, params, |_, _| {})
}

pub fn enroll_user_with_progress(
    mut cfg: Config,
    params: EnrollParams,
    mut on_sample: impl FnMut(usize, usize),
) -> Result<()> {
    let EnrollParams {
        username,
        label,
        samples,
        device,
        ir,
        merge,
        variant,
    } = params;
    let samples = samples.max(1);
    if ir {
        cfg.video.ir_mode = true;
    }

    let mut cam = camera::Camera::open(&device, cfg.video.max_height, cfg.video.rotate)
        .with_context(|| format!("Failed to open camera {}", device))?;
    if cfg.video.exposure >= 0 {
        let _ = cam.set_exposure(cfg.video.exposure as f64);
    }
    let haar_neighbors = if cfg.video.ir_mode { 2 } else { 3 };
    let mut detector = create_detector(
        &cfg.detection.model_path,
        cfg.detection.confidence_threshold as f32,
        cfg.detection.nms_threshold as f32,
        cfg.detection.use_cnn,
        DEFAULT_HAAR_CASCADE,
        haar_neighbors,
    )?;
    let mut recognizer = FaceRecognizer::load(&cfg.recognition.model_path)?;

    if cfg.video.ir_mode {
        info!("IR mode: darkness filter disabled; enroll on the same IR device as faceauth-auth");
    }
    info!("Collecting {} face samples for {}", samples, username);
    let vectors = capture_embeddings_with_progress(
        &mut cam,
        &mut detector,
        &mut recognizer,
        samples,
        &cfg,
        &mut on_sample,
    )?;
    if vectors.is_empty() {
        anyhow::bail!("Could not collect any valid face samples");
    }

    let model_path = get_user_model_path(&username)?;
    let mut db = Database::load(&model_path)?;

    match merge {
        EnrollMerge::ReplaceAll => {
            let model_label = label.unwrap_or_else(|| format!("{}-default", username));
            let model = FaceModel::new(model_label, vectors);
            db.add_model(username.clone(), model);
        }
        EnrollMerge::AppendPrimary => {
            if let Some(m) = db.users.get_mut(&username) {
                m.embeddings.extend(vectors);
                info!(
                    "Appended {} samples to primary embeddings (total {})",
                    samples,
                    m.embeddings.len()
                );
            } else {
                let model_label = label.unwrap_or_else(|| format!("{}-default", username));
                db.add_model(username.clone(), FaceModel::new(model_label, vectors));
                info!("Created new model with {} primary samples", samples);
            }
        }
        EnrollMerge::ReplaceVariant | EnrollMerge::AppendVariant => {
            let vname = variant.as_ref().map(|s| s.trim().to_string()).unwrap_or_default();
            if vname.is_empty() {
                anyhow::bail!("variant name is empty");
            }
            let append = merge == EnrollMerge::AppendVariant;
            if let Some(m) = db.users.get_mut(&username) {
                m.upsert_extension(vname.clone(), vectors, append);
                info!(
                    "Updated variant {:?} (append={}), extensions={}",
                    vname,
                    append,
                    m.extensions.len()
                );
            } else {
                let model_label = label.unwrap_or_else(|| format!("{}-default", username));
                let mut m = FaceModel::new(model_label, Vec::new());
                m.upsert_extension(vname.clone(), vectors, append);
                db.add_model(username.clone(), m);
                info!("Created user model with only variant {:?}", vname);
            }
        }
    }

    db.save(&model_path)?;
    info!("Saved model to {}", model_path.display());
    Ok(())
}

pub fn capture_embeddings(
    cam: &mut camera::Camera,
    detector: &mut Detector,
    recognizer: &mut FaceRecognizer,
    target_samples: usize,
    cfg: &Config,
) -> Result<Vec<Vec<f32>>> {
    let mut noop = |_: usize, _: usize| {};
    capture_embeddings_with_progress(
        cam,
        detector,
        recognizer,
        target_samples,
        cfg,
        &mut noop,
    )
}

pub fn capture_embeddings_with_progress(
    cam: &mut camera::Camera,
    detector: &mut Detector,
    recognizer: &mut FaceRecognizer,
    target_samples: usize,
    cfg: &Config,
    on_sample: &mut impl FnMut(usize, usize),
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(target_samples);
    let started = Instant::now();
    let timeout = Duration::from_secs((cfg.video.timeout as u64).saturating_mul(3).max(6));

    while out.len() < target_samples && started.elapsed() < timeout {
        if let Some(embedding) = capture_single_embedding(cam, detector, recognizer, cfg)? {
            out.push(embedding.vector);
            info!("Captured sample {}/{}", out.len(), target_samples);
            on_sample(out.len(), target_samples);
            std::thread::sleep(Duration::from_millis(220));
        } else {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    Ok(out)
}

pub fn capture_single_embedding(
    cam: &mut camera::Camera,
    detector: &mut Detector,
    recognizer: &mut FaceRecognizer,
    cfg: &Config,
) -> Result<Option<FaceEmbedding>> {
    let (color, gray) = match cam.read_frame() {
        Ok(frames) => frames,
        Err(e) => {
            debug!("skip frame: read_frame failed: {e}");
            return Ok(None);
        }
    };

    let darkness = camera::darkness(&gray)?;
    if !cfg.video.ir_mode && darkness > cfg.video.dark_threshold {
        debug!(
            "skip frame: darkness {:.1}% > threshold {:.1}%",
            darkness, cfg.video.dark_threshold
        );
        return Ok(None);
    }

    let faces = match detector.detect(&color) {
        Ok(faces) => faces,
        Err(e) => {
            debug!("skip frame: detect failed: {e}");
            return Ok(None);
        }
    };
    if faces.is_empty() {
        debug!("skip frame: no faces detected (frontal, good light helps)");
        return Ok(None);
    }

    // Filter faces by size ratio and pick the largest valid one
    let img_area = color.rows() * color.cols();
    let min_area = (img_area as f64 * cfg.detection.min_face_size_ratio).max(1.0) as i32;
    let max_area = (img_area as f64 * cfg.detection.max_face_size_ratio).max(1.0) as i32;

    let valid_faces: Vec<_> = faces
        .into_iter()
        .filter(|f| {
            let area = f.bbox.width * f.bbox.height;
            area >= min_area && area <= max_area
        })
        .collect();

    if valid_faces.is_empty() {
        debug!("skip frame: no faces within size constraints");
        return Ok(None);
    }

    // Select the largest face (most likely the intended subject)
    let biggest = valid_faces
        .into_iter()
        .max_by_key(|f| f.bbox.width * f.bbox.height)
        .context("Failed to select face candidate")?;

    if biggest.confidence < cfg.detection.confidence_threshold as f32 {
        debug!(
            "skip frame: confidence {:.3} < threshold {:.3}",
            biggest.confidence, cfg.detection.confidence_threshold
        );
        return Ok(None);
    }

    // Crop face with padding (adds context like ears/chin for better recognition)
    let crop = crop_face(&color, &biggest.bbox, cfg.detection.face_padding)?;
    let emb = recognizer.extract(&crop)?;
    Ok(Some(emb))
}
