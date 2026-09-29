//! Face enrollment (CLI and GUI).
//!
//! [`EnrollSession`] is stepped one frame at a time so the GUI can show each
//! frame; [`enroll_user`] drives it for the CLI and writes the root-only store.

use anyhow::{Context, Result};
use log::info;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::database::{Database, EnrollMerge, Enrollment, user_model_path};
use crate::matching::{PoseCollector, PoseHint, SampleDecision};
use crate::pipeline::{AnalysisMode, FaceVerdict, Frame, Pipeline};

#[derive(Clone, Debug)]
pub struct EnrollParams {
    pub username: String,
    pub label: Option<String>,
    pub samples: usize,
    /// Capture device; defaults to `video.device_path` so enrollment and
    /// authentication use the same camera.
    pub device: Option<String>,
    pub ir: bool,
    pub merge: EnrollMerge,
    /// Required when merge is ReplaceVariant or AppendVariant.
    pub variant: Option<String>,
}

/// What happened on one enrollment step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnrollEvent {
    /// A sample was accepted.
    Sample {
        cur: usize,
        tot: usize,
    },
    /// The frame was rejected by the quality gates.
    Rejected(FaceVerdict),
    /// Good frame, but this pose is covered / it duplicates an earlier sample.
    Hint(PoseHint),
    Duplicate,
}

pub struct EnrollStep {
    pub frame: Frame,
    pub event: Option<EnrollEvent>,
}

/// Collects quality-checked samples spread over head poses.
pub struct EnrollSession {
    collector: PoseCollector,
    started: Instant,
    timeout: Duration,
}

impl EnrollSession {
    pub fn new(samples: usize, cfg: &Config) -> Self {
        let samples = samples.max(1);
        // At least 2 s per sample so pose guidance has time to work.
        let base = (cfg.video.timeout as u64).saturating_mul(3).max(6);
        let timeout = Duration::from_secs(base.max(2 * samples as u64));
        Self {
            collector: PoseCollector::new(samples),
            started: Instant::now(),
            timeout,
        }
    }

    pub fn is_done(&self) -> bool {
        self.collector.is_complete() || self.started.elapsed() >= self.timeout
    }

    pub fn collected(&self) -> usize {
        self.collector.len()
    }

    pub fn target(&self) -> usize {
        self.collector.target()
    }

    pub fn hint(&self) -> Option<PoseHint> {
        self.collector.missing_hint()
    }

    /// Capture and evaluate one frame.
    pub fn step(&mut self, pipeline: &mut Pipeline, cfg: &Config) -> Result<EnrollStep> {
        let frame = pipeline.capture(cfg, AnalysisMode::Enroll)?;
        if frame.analysis.verdict != FaceVerdict::Ok {
            let event = Some(EnrollEvent::Rejected(frame.analysis.verdict));
            return Ok(EnrollStep { frame, event });
        }
        let Some(embedding) = pipeline.embed(&frame, cfg)? else {
            return Ok(EnrollStep { frame, event: None });
        };
        let fraction =
            (self.started.elapsed().as_secs_f32() / self.timeout.as_secs_f32()).clamp(0.0, 1.0);
        let event = match self
            .collector
            .offer(embedding.vector, frame.analysis.yaw, fraction)
        {
            SampleDecision::Accepted => EnrollEvent::Sample {
                cur: self.collector.len(),
                tot: self.collector.target(),
            },
            SampleDecision::Duplicate => EnrollEvent::Duplicate,
            SampleDecision::PoseFull(h) => EnrollEvent::Hint(h),
        };
        Ok(EnrollStep {
            frame,
            event: Some(event),
        })
    }

    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.collector.into_vectors()
    }
}

/// CLI enrollment: capture, merge and save into the root store (caller checks privileges).
pub fn enroll_user(
    mut cfg: Config,
    params: EnrollParams,
    mut on_event: impl FnMut(&EnrollEvent),
) -> Result<()> {
    if params.ir {
        cfg.video.ir_mode = true;
    }
    let device = params
        .device
        .clone()
        .unwrap_or_else(|| cfg.video.device_path.clone());
    let mut pipeline = Pipeline::open(&cfg, &device)?;
    info!(
        "Enrolling {} on {} ({})",
        params.username,
        device,
        pipeline.describe()
    );
    if cfg.video.ir_mode {
        info!("IR mode: darkness filter disabled; enroll on the same IR device as faceauth-auth");
    }

    let mut session = EnrollSession::new(params.samples, &cfg);
    while !session.is_done() {
        match session.step(&mut pipeline, &cfg) {
            Ok(step) => {
                if let Some(ev) = &step.event {
                    on_event(ev);
                }
            }
            Err(e) => {
                log::debug!("Enrollment frame failed: {e}");
                std::thread::sleep(Duration::from_millis(cfg.video.frame_interval_ms));
            }
        }
    }
    let collected = session.collected();
    if collected < params.samples.max(1) {
        log::warn!(
            "Collected only {collected}/{} samples before timeout",
            params.samples
        );
    }
    let vectors = session.into_vectors();
    if vectors.is_empty() {
        anyhow::bail!("Could not collect any valid face samples");
    }

    let model_path = user_model_path(&params.username)?;
    let mut db = Database::load_trusted(&model_path)?.unwrap_or_default();
    db.apply_enrollment(
        &params.username,
        Enrollment {
            merge: params.merge,
            variant: params.variant,
            label: params.label,
            vectors,
            model_id: pipeline.recognizer.model_id().to_string(),
        },
    )?;
    db.save_secure(&model_path)
        .with_context(|| format!("Failed to save {}", model_path.display()))?;
    info!("Saved {collected} samples to {}", model_path.display());
    Ok(())
}
