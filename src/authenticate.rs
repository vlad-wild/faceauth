//! The authentication loop shared by `faceauth-auth` (PAM) and `faceauth test`.

use anyhow::Result;
use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::database::FaceModel;
use crate::matching::MatchTracker;
use crate::pipeline::{AnalysisMode, FaceVerdict, Frame, Pipeline};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthOutcome {
    /// `required_matches` consecutive frames matched.
    Success,
    /// Timed out without enough consecutive matches.
    NoMatch,
    /// Every frame read was too dark.
    AllDark,
    /// Not a single frame could be read.
    NoFrames,
    /// The caller stopped the attempt from its frame callback.
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct AuthReport {
    pub outcome: AuthOutcome,
    pub frames_read: u32,
    pub dark_frames: u32,
    pub face_frames: u32,
    pub verdicts: BTreeMap<&'static str, u32>,
    pub best_score: f32,
    pub consecutive: u32,
    pub required: u32,
    pub elapsed: Duration,
    pub backend: String,
}

impl AuthReport {
    pub fn summary(&self) -> String {
        let verdicts = self
            .verdicts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "outcome={:?} frames={} dark={} with_face={} best_score={:.4} streak={}/{} elapsed={:.2}s [{}] ({})",
            self.outcome,
            self.frames_read,
            self.dark_frames,
            self.face_frames,
            self.best_score,
            self.consecutive,
            self.required,
            self.elapsed.as_secs_f32(),
            verdicts,
            self.backend,
        )
    }
}

/// Per-frame observation handed to the caller (CLI progress, daemon events).
/// Returning [`ControlFlow::Break`] from the callback ends the attempt as
/// [`AuthOutcome::Cancelled`].
pub struct FrameEvent<'a> {
    pub frame: &'a Frame,
    pub score: Option<f32>,
    pub matched: bool,
}

/// Run the capture loop until `required_matches` consecutive frames match
/// `model` or `timeout` passes.
pub fn authenticate(
    pipeline: &mut Pipeline,
    cfg: &Config,
    model: &FaceModel,
    timeout: Duration,
    mut on_frame: impl FnMut(FrameEvent<'_>) -> ControlFlow<()>,
) -> Result<AuthReport> {
    let started = Instant::now();
    let pause = Duration::from_millis(cfg.video.frame_interval_ms);
    let threshold = cfg.recognition.distance_threshold as f32;
    let k = cfg.recognition.top_k.max(1);
    let mut tracker = MatchTracker::new(cfg.recognition.required_matches);
    let mut report = AuthReport {
        outcome: AuthOutcome::NoMatch,
        frames_read: 0,
        dark_frames: 0,
        face_frames: 0,
        verdicts: BTreeMap::new(),
        best_score: f32::INFINITY,
        consecutive: 0,
        required: tracker.required(),
        elapsed: Duration::ZERO,
        backend: pipeline.describe(),
    };
    let mut compat_checked = false;

    while started.elapsed() < timeout {
        let frame = match pipeline.capture(cfg, AnalysisMode::Auth) {
            Ok(f) => f,
            Err(e) => {
                log::debug!("Frame capture failed: {e}");
                std::thread::sleep(pause);
                continue;
            }
        };
        report.frames_read += 1;
        *report
            .verdicts
            .entry(frame.analysis.verdict.key())
            .or_default() += 1;
        match frame.analysis.verdict {
            FaceVerdict::TooDark => {
                report.dark_frames += 1;
                if on_frame(FrameEvent {
                    frame: &frame,
                    score: None,
                    matched: false,
                })
                .is_break()
                {
                    report.outcome = AuthOutcome::Cancelled;
                    break;
                }
                std::thread::sleep(pause);
                continue;
            }
            FaceVerdict::Ok => {}
            FaceVerdict::NotLive => {
                // A face was there but failed liveness: it breaks the streak.
                report.face_frames += 1;
                tracker.observe(f32::INFINITY, false);
                if on_frame(FrameEvent {
                    frame: &frame,
                    score: None,
                    matched: false,
                })
                .is_break()
                {
                    report.outcome = AuthOutcome::Cancelled;
                    break;
                }
                continue;
            }
            _ => {
                if on_frame(FrameEvent {
                    frame: &frame,
                    score: None,
                    matched: false,
                })
                .is_break()
                {
                    report.outcome = AuthOutcome::Cancelled;
                    break;
                }
                continue;
            }
        }
        report.face_frames += 1;

        let embedding = match pipeline.embed(&frame, cfg) {
            Ok(Some(e)) => e,
            Ok(None) => continue,
            Err(e) => {
                log::warn!("Embedding extraction failed: {e}");
                continue;
            }
        };
        if !compat_checked {
            model.check_compatible(pipeline.recognizer.model_id(), embedding.vector.len())?;
            compat_checked = true;
        }

        let score = model.match_score(&embedding.vector, k);
        let matched = score < threshold;
        let done = tracker.observe(score, matched);
        let flow = on_frame(FrameEvent {
            frame: &frame,
            score: Some(score),
            matched,
        });
        if !done && flow.is_break() {
            report.outcome = AuthOutcome::Cancelled;
            break;
        }
        if done {
            report.outcome = AuthOutcome::Success;
            break;
        }
    }

    report.best_score = tracker.best_score();
    report.consecutive = tracker.consecutive();
    report.elapsed = started.elapsed();
    if !matches!(
        report.outcome,
        AuthOutcome::Success | AuthOutcome::Cancelled
    ) {
        report.outcome = if report.frames_read == 0 {
            AuthOutcome::NoFrames
        } else if report.dark_frames == report.frames_read {
            AuthOutcome::AllDark
        } else {
            AuthOutcome::NoMatch
        };
    }
    Ok(report)
}
