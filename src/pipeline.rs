//! Shared capture → detect → analyze → embed pipeline used by `faceauth-auth`,
//! the CLI and the GUI, so every entry point applies the same filters.

use anyhow::Result;
use opencv::core::Mat;
use opencv::prelude::MatTraitConst;

use crate::camera::{self, Camera};
use crate::config::{Config, DetectionConfig};
use crate::detection::{
    Detector, Face, clip_rect, create_detector, crop_face, estimate_pitch, estimate_yaw,
};
use crate::recognition::{FaceEmbedding, FaceRecognizer, align_face};

pub use crate::verdict::FaceVerdict;

/// Enrollment additionally rejects blurry and strongly turned faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    Auth,
    Enroll,
}

#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// The selected face (set whenever one passed detection filters, even if the
    /// verdict is not `Ok`, so the overlay can highlight it).
    pub face: Option<Face>,
    pub verdict: FaceVerdict,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub darkness: f64,
}

/// Pick the largest confident face whose area is within the configured ratios.
pub fn select_face(
    faces: &[Face],
    cols: i32,
    rows: i32,
    cfg: &DetectionConfig,
) -> std::result::Result<Face, FaceVerdict> {
    if faces.is_empty() {
        return Err(FaceVerdict::NoFace);
    }
    let img_area = (cols as f64 * rows as f64).max(1.0);
    let min_area = img_area * cfg.min_face_size_ratio;
    let max_area = img_area * cfg.max_face_size_ratio;
    let confident: Vec<Face> = faces
        .iter()
        .filter(|f| f.confidence >= cfg.confidence_threshold as f32)
        .filter_map(|f| {
            let bbox = clip_rect(f.bbox, cols, rows)?;
            Some(Face { bbox, ..f.clone() })
        })
        .collect();
    if confident.is_empty() {
        return Err(FaceVerdict::LowConfidence);
    }
    let area = |f: &Face| f.bbox.width as f64 * f.bbox.height as f64;
    if let Some(best) = confident
        .iter()
        .filter(|f| (min_area..=max_area).contains(&area(f)))
        .max_by(|a, b| area(a).total_cmp(&area(b)))
    {
        return Ok(best.clone());
    }
    let largest = confident.iter().map(area).fold(0.0, f64::max);
    Err(if largest < min_area {
        FaceVerdict::TooSmall
    } else {
        FaceVerdict::TooLarge
    })
}

/// Apply all per-frame gates and pick the face to use.
pub fn analyze_frame(
    color: &Mat,
    gray: &Mat,
    faces: &[Face],
    cfg: &Config,
    mode: AnalysisMode,
) -> Result<FrameAnalysis> {
    let darkness = camera::darkness(gray)?;
    let mut out = FrameAnalysis {
        face: None,
        verdict: FaceVerdict::Ok,
        yaw: None,
        pitch: None,
        darkness,
    };
    if darkness >= 99.5 || (!cfg.video.ir_mode && darkness > cfg.video.dark_threshold) {
        out.verdict = FaceVerdict::TooDark;
        return Ok(out);
    }
    let face = match select_face(faces, color.cols(), color.rows(), &cfg.detection) {
        Ok(f) => f,
        Err(v) => {
            out.verdict = v;
            return Ok(out);
        }
    };
    out.yaw = estimate_yaw(&face);
    out.pitch = estimate_pitch(&face);
    out.verdict = face_verdict(gray, &face, out.yaw, out.pitch, cfg, mode)?;
    out.face = Some(face);
    Ok(out)
}

fn face_verdict(
    gray: &Mat,
    face: &Face,
    yaw: Option<f32>,
    pitch: Option<f32>,
    cfg: &Config,
    mode: AnalysisMode,
) -> Result<FaceVerdict> {
    if cfg.video.ir_mode && cfg.liveness.ir_check {
        let (mean, stddev) = camera::roi_stats(gray, face.bbox)?;
        if mean < cfg.liveness.min_face_brightness || stddev < cfg.liveness.min_face_stddev {
            log::debug!("IR liveness rejected face: mean {mean:.1}, stddev {stddev:.1}");
            return Ok(FaceVerdict::NotLive);
        }
    }
    if mode == AnalysisMode::Enroll {
        let too_turned = yaw.is_some_and(|y| y.abs() as f64 > cfg.enroll.max_abs_yaw)
            || pitch.is_some_and(|p| p.abs() as f64 > cfg.enroll.max_abs_pitch);
        if too_turned {
            return Ok(FaceVerdict::LookStraight);
        }
        if camera::sharpness(gray, face.bbox)? < cfg.enroll.min_sharpness {
            return Ok(FaceVerdict::Blurry);
        }
    }
    Ok(FaceVerdict::Ok)
}

/// Aligned 112×112 face when landmarks exist (YuNet), padded crop otherwise.
pub fn face_crop(color: &Mat, face: &Face, cfg: &DetectionConfig) -> Result<Mat> {
    if face.landmarks.len() >= 2 {
        align_face(color, &face.landmarks, 112)
    } else {
        crop_face(color, &face.bbox, cfg.face_padding)
    }
}

/// One captured and analyzed frame.
pub struct Frame {
    pub color: Mat,
    pub gray: Mat,
    /// All detections (for drawing); `analysis.face` is the selected one.
    pub faces: Vec<Face>,
    pub analysis: FrameAnalysis,
}

pub struct Pipeline {
    pub camera: Camera,
    pub detector: Detector,
    pub recognizer: FaceRecognizer,
}

impl Pipeline {
    /// Open the camera and load both models (fails if the recognizer is missing).
    pub fn open(cfg: &Config, device: &str) -> Result<Self> {
        let camera = Camera::open_configured(device, &cfg.video)
            .map_err(|e| e.context(format!("Failed to open camera {device}")))?;
        let detector = create_detector(&cfg.detection, &cfg.openvino, cfg.video.ir_mode)?;
        let recognizer = FaceRecognizer::from_config(&cfg.recognition, &cfg.openvino)?;
        Ok(Self {
            camera,
            detector,
            recognizer,
        })
    }

    /// Read, detect and analyze one frame.
    pub fn capture(&mut self, cfg: &Config, mode: AnalysisMode) -> Result<Frame> {
        let (color, gray) = self.camera.read_frame()?;
        let faces = self.detector.detect(&color).unwrap_or_else(|e| {
            log::debug!("Detection error: {e}");
            Vec::new()
        });
        let analysis = analyze_frame(&color, &gray, &faces, cfg, mode)?;
        Ok(Frame {
            color,
            gray,
            faces,
            analysis,
        })
    }

    /// Embedding of the selected face, if the frame's verdict is `Ok`.
    pub fn embed(&mut self, frame: &Frame, cfg: &Config) -> Result<Option<FaceEmbedding>> {
        match (&frame.analysis.face, frame.analysis.verdict) {
            (Some(face), FaceVerdict::Ok) => {
                let crop = face_crop(&frame.color, face, &cfg.detection)?;
                Ok(Some(self.recognizer.extract(&crop)?))
            }
            _ => Ok(None),
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "detector: {}, recognizer: {}",
            self.detector.describe(),
            self.recognizer.backend_info()
        )
    }
}
