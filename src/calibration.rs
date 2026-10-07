//! Score the enrolled user's own face over many frames and suggest a threshold
//! (`faceauth calibrate`, and the daemon's `calibrate` request).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use crate::camera;
use crate::config::Config;
use crate::database::FaceModel;
use crate::matching::score_stats;
use crate::pipeline::{AnalysisMode, FaceVerdict, Pipeline};

/// Never suggest a threshold below this (identical frames give p95 ≈ 0).
pub const MIN_SUGGESTED_THRESHOLD: f32 = 0.3;
/// Suggested threshold = p95 of the user's own scores × this margin.
pub const THRESHOLD_MARGIN: f32 = 1.15;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub frames: usize,
    pub min: f32,
    pub median: f32,
    pub p95: f32,
    /// p95 × margin, before clamping to [`MIN_SUGGESTED_THRESHOLD`].
    pub raw_suggestion: f32,
    pub suggested: f32,
    pub current: f32,
    /// Mean brightness and stddev of the face region (for liveness tuning).
    pub face_brightness: Option<(f64, f64)>,
}

/// Progress while collecting scores.
pub enum Progress {
    Scored { n: usize, of: usize },
    Rejected(FaceVerdict),
}

/// Collect up to `frames` scores within `limit`; `None` if no frame was usable.
pub fn calibrate(
    pipeline: &mut Pipeline,
    cfg: &Config,
    model: &FaceModel,
    frames: usize,
    limit: Duration,
    mut on_progress: impl FnMut(Progress) -> ControlFlow<()>,
) -> Result<Option<Calibration>> {
    let k = cfg.recognition.top_k;
    let deadline = Instant::now() + limit;
    let mut scores = Vec::with_capacity(frames);
    let mut face_stats = Vec::new();
    while scores.len() < frames && Instant::now() < deadline {
        let frame = match pipeline.capture(cfg, AnalysisMode::Auth) {
            Ok(f) => f,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(cfg.video.frame_interval_ms));
                continue;
            }
        };
        if let Some(face) = &frame.analysis.face
            && let Ok(s) = camera::roi_stats(&frame.gray, face.bbox)
        {
            face_stats.push(s);
        }
        let flow = if let Some(emb) = pipeline.embed(&frame, cfg)? {
            model.check_compatible(pipeline.recognizer.model_id(), emb.vector.len())?;
            scores.push(model.match_score(&emb.vector, k));
            on_progress(Progress::Scored {
                n: scores.len(),
                of: frames,
            })
        } else {
            on_progress(Progress::Rejected(frame.analysis.verdict))
        };
        if flow.is_break() {
            return Ok(None);
        }
    }
    let Some((min, median, p95)) = score_stats(&scores) else {
        return Ok(None);
    };
    let raw_suggestion = p95 * THRESHOLD_MARGIN;
    let face_brightness = (!face_stats.is_empty()).then(|| {
        let n = face_stats.len() as f64;
        (
            face_stats.iter().map(|s| s.0).sum::<f64>() / n,
            face_stats.iter().map(|s| s.1).sum::<f64>() / n,
        )
    });
    Ok(Some(Calibration {
        frames: scores.len(),
        min,
        median,
        p95,
        raw_suggestion,
        suggested: raw_suggestion.max(MIN_SUGGESTED_THRESHOLD),
        current: cfg.recognition.distance_threshold as f32,
        face_brightness,
    }))
}
