use anyhow::{Context, Result};
use log::warn;
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Mat, Point2f, Rect, Size, Vector};
use opencv::prelude::{
    CascadeClassifierTrait, CascadeClassifierTraitConst, FaceDetectorYNTrait, MatTraitConst,
    MatTraitConstManual,
};
use std::path::Path;
use tract_onnx::prelude::*;

use crate::config::{DetectionConfig, OpenVinoConfig};

/// OpenCV Haar cascade used when no CNN detector is available.
pub const DEFAULT_HAAR_CASCADE: &str = HAAR_CASCADES[0];

/// Cascade locations for the installed OpenCV major version first.
#[cfg(opencv5)]
const HAAR_CASCADES: [&str; 2] = [
    "/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
];
#[cfg(not(opencv5))]
const HAAR_CASCADES: [&str; 2] = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml",
];

// OpenCV 5 moved the cascade classifier from objdetect to xobjdetect.
#[cfg(not(opencv5))]
use opencv::objdetect as cascade;
#[cfg(opencv5)]
use opencv::xobjdetect as cascade;

/// Detected face with optional landmarks
#[derive(Debug, Clone)]
pub struct Face {
    pub bbox: Rect,
    pub confidence: f32,
    /// 5 landmarks from YuNet: [right_eye, left_eye, nose, right_mouth, left_mouth]
    pub landmarks: Vec<Point2f>,
}

impl Face {
    fn new(bbox: Rect, confidence: f32) -> Self {
        Self {
            bbox,
            confidence,
            landmarks: Vec::new(),
        }
    }

    fn with_landmarks(bbox: Rect, confidence: f32, landmarks: Vec<Point2f>) -> Self {
        Self {
            bbox,
            confidence,
            landmarks,
        }
    }
}

/// Approximate head yaw angle based on eye-nose asymmetry.
/// Positive = facing left (user's left), negative = facing right.
/// Returns None if landmarks are not available.
pub fn estimate_yaw(face: &Face) -> Option<f32> {
    if face.landmarks.len() < 3 {
        return None;
    }
    let re = &face.landmarks[0]; // right eye (from viewer perspective)
    let le = &face.landmarks[1]; // left eye
    let nose = &face.landmarks[2];

    let d_re_nose = (re.x - nose.x).hypot(re.y - nose.y);
    let d_le_nose = (le.x - nose.x).hypot(le.y - nose.y);

    // Inter-eye distance as normalization factor
    let d_eyes = (re.x - le.x).hypot(re.y - le.y).max(1.0);

    // Asymmetry ratio; 0 = perfectly frontal
    let asym = (d_le_nose - d_re_nose) / d_eyes;
    Some(asym.atan().to_degrees())
}

/// Rough head pitch from where the nose sits between the eye line and the mouth
/// line (≈0.49 of the way down on a frontal face). Positive = looking down.
/// Returns None without 5 landmarks. Only meant for "look straight" hints.
pub fn estimate_pitch(face: &Face) -> Option<f32> {
    if face.landmarks.len() < 5 {
        return None;
    }
    let l = &face.landmarks;
    let eyes_y = (l[0].y + l[1].y) / 2.0;
    let mouth_y = (l[3].y + l[4].y) / 2.0;
    let span = mouth_y - eyes_y;
    if span.abs() < 1.0 {
        return None;
    }
    let ratio = (l[2].y - eyes_y) / span;
    Some((ratio - FRONTAL_NOSE_RATIO) * 120.0)
}

/// Nose position between eye and mouth lines on the canonical 112×112 face.
const FRONTAL_NOSE_RATIO: f32 = 0.494;

/// Intersect `r` with a `cols × rows` image; None if nothing is left.
pub fn clip_rect(r: Rect, cols: i32, rows: i32) -> Option<Rect> {
    let x1 = r.x.max(0);
    let y1 = r.y.max(0);
    let x2 = (r.x + r.width).min(cols);
    let y2 = (r.y + r.height).min(rows);
    (x2 > x1 && y2 > y1).then(|| Rect::new(x1, y1, x2 - x1, y2 - y1))
}

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

enum UltraLightBackend {
    #[cfg(feature = "openvino")]
    OpenVino(crate::openvino_backend::OpenVinoSession),
    Tract(OnnxModel),
}

/// Ultra-Light face detector using ONNX
pub struct UltraLightDetector {
    backend: UltraLightBackend,
    width: usize,
    height: usize,
    prob_threshold: f32,
    nms_threshold: f32,
}

impl UltraLightDetector {
    pub fn load(
        model_path: &str,
        prob_threshold: f32,
        nms_threshold: f32,
        use_openvino: bool,
        ov: &OpenVinoConfig,
    ) -> Result<Self> {
        #[cfg(not(feature = "openvino"))]
        let _ = (use_openvino, ov);
        let path = Path::new(model_path);
        if !path.exists() {
            anyhow::bail!("Model not found: {}", model_path);
        }

        #[cfg(feature = "openvino")]
        if use_openvino {
            match crate::openvino_backend::OpenVinoSession::from_onnx(model_path, ov) {
                Ok(session) => {
                    log::info!(
                        "Ultra-Light detector loaded via OpenVINO on {}",
                        session.device()
                    );
                    let shape = &session.input_shape;
                    let height = shape
                        .get(2)
                        .copied()
                        .context("Cannot get input height from OpenVINO model")?
                        as usize;
                    let width = shape
                        .get(3)
                        .copied()
                        .context("Cannot get input width from OpenVINO model")?
                        as usize;
                    return Ok(Self {
                        backend: UltraLightBackend::OpenVino(session),
                        width,
                        height,
                        prob_threshold,
                        nms_threshold,
                    });
                }
                Err(e) => {
                    warn!("OpenVINO detector init failed: {e}. Falling back to tract-onnx.");
                }
            }
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to read ONNX model")?
            .into_optimized()
            .context("Failed to optimize ONNX model")?
            .into_runnable()
            .context("Failed to create ONNX runnable model")?;
        log::info!("Ultra-Light detector loaded via tract-onnx (CPU)");

        let input_fact = model.model().input_fact(0)?;
        let shape = input_fact.shape.to_tvec();
        let height = shape
            .get(2)
            .and_then(|d| d.as_i64())
            .context("Cannot get input height")? as usize;
        let width = shape
            .get(3)
            .and_then(|d| d.as_i64())
            .context("Cannot get input width")? as usize;

        Ok(Self {
            backend: UltraLightBackend::Tract(model),
            width,
            height,
            prob_threshold,
            nms_threshold,
        })
    }

    pub fn detect(&mut self, image: &Mat) -> Result<Vec<Face>> {
        let orig_h = image.rows() as f32;
        let orig_w = image.cols() as f32;

        let input = self.preprocess(image)?;

        let (scores_data, boxes_data) = match &mut self.backend {
            #[cfg(feature = "openvino")]
            UltraLightBackend::OpenVino(session) => {
                let outputs = session.run(input).context("OpenVINO inference failed")?;
                if outputs.len() < 2 {
                    anyhow::bail!("Ultra-Light OpenVINO model returned fewer than 2 outputs");
                }
                let scores = outputs[0].1.clone();
                let boxes = outputs[1].1.clone();
                (scores, boxes)
            }
            UltraLightBackend::Tract(model) => {
                let outputs = model.run(tvec!(input.into_tensor().into()))?;
                if outputs.len() < 2 {
                    anyhow::bail!("Ultra-Light tract model returned fewer than 2 outputs");
                }
                let scores_view = outputs[0].to_array_view::<f32>()?;
                let boxes_view = outputs[1].to_array_view::<f32>()?;
                let scores = scores_view.iter().copied().collect::<Vec<f32>>();
                let boxes = boxes_view.iter().copied().collect::<Vec<f32>>();
                (scores, boxes)
            }
        };

        let mut faces = Vec::new();

        // Ultra-Light: outputs[0] = scores [1,N,2], outputs[1] = boxes [1,N,4]
        // Tract shapes are known from its own model; for OpenVINO we also receive flat arrays.
        // We can reconstruct based on length ratios: boxes has 4x the elements of scores (per anchor).
        let num_scores = scores_data.len();
        let num_boxes = boxes_data.len();
        // typical: scores [1,N,2] -> 2*N elements, boxes [1,N,4] -> 4*N elements
        let num_anchors = num_scores / 2;
        if num_anchors == 0 || num_boxes != num_anchors * 4 {
            anyhow::bail!(
                "Ultra-Light output shape mismatch: scores={}, boxes={}",
                num_scores,
                num_boxes
            );
        }

        // Boxes are normalized to [0, 1]; clamp to the frame before computing size.
        for i in 0..num_anchors {
            let score = scores_data[i * 2 + 1]; // class 1 = face
            if score < self.prob_threshold {
                continue;
            }
            let x1 = (boxes_data[i * 4] * orig_w).clamp(0.0, orig_w);
            let y1 = (boxes_data[i * 4 + 1] * orig_h).clamp(0.0, orig_h);
            let x2 = (boxes_data[i * 4 + 2] * orig_w).clamp(0.0, orig_w);
            let y2 = (boxes_data[i * 4 + 3] * orig_h).clamp(0.0, orig_h);
            if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
                continue;
            }
            faces.push(Face::new(
                Rect::new(x1 as i32, y1 as i32, (x2 - x1) as i32, (y2 - y1) as i32),
                score,
            ));
        }

        Ok(nms(faces, self.nms_threshold))
    }

    fn preprocess(&self, image: &Mat) -> Result<Array4<f32>> {
        let mut resized = Mat::default();
        opencv::imgproc::resize(
            image,
            &mut resized,
            Size::new(self.width as i32, self.height as i32),
            0.0,
            0.0,
            opencv::imgproc::INTER_AREA,
        )?;

        let mut rgb = Mat::default();
        opencv::imgproc::cvt_color(
            &resized,
            &mut rgb,
            opencv::imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        if !rgb.is_continuous() {
            rgb = rgb.try_clone()?;
        }

        let pixels = rgb.data_bytes()?;
        let mut input = Array4::<f32>::zeros((1, 3, self.height, self.width));

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * 3;
                let r = pixels[idx] as f32;
                let g = pixels[idx + 1] as f32;
                let b = pixels[idx + 2] as f32;

                input[[0, 0, y, x]] = (r - 127.0) / 128.0;
                input[[0, 1, y, x]] = (g - 127.0) / 128.0;
                input[[0, 2, y, x]] = (b - 127.0) / 128.0;
            }
        }

        Ok(input)
    }

    pub fn backend_info(&self) -> String {
        match &self.backend {
            #[cfg(feature = "openvino")]
            UltraLightBackend::OpenVino(session) => format!("OpenVINO ({})", session.device()),
            UltraLightBackend::Tract(_) => "tract-onnx (CPU)".to_string(),
        }
    }
}

/// YuNet face detector with built-in landmarks.
///
/// The network input is resized to each frame's own size (`setInputSize`), so
/// non-square frames such as 640×360 IR are not stretched and landmarks keep
/// their geometry.
pub struct YuNetDetector {
    model: opencv::core::Ptr<opencv::objdetect::FaceDetectorYN>,
    input_size: Size,
    nms_threshold: f32,
}

impl YuNetDetector {
    pub fn load(model_path: &str, prob_threshold: f32, nms_threshold: f32) -> Result<Self> {
        let input_size = Size::new(320, 320);
        let detector = opencv::objdetect::FaceDetectorYN::create(
            model_path,
            "",
            input_size,
            prob_threshold,
            nms_threshold,
            5000,
            0,
            0,
        )?;
        Ok(Self {
            model: detector,
            input_size,
            nms_threshold,
        })
    }

    pub fn detect(&mut self, image: &Mat) -> Result<Vec<Face>> {
        if image.cols() != self.input_size.width || image.rows() != self.input_size.height {
            self.input_size = Size::new(image.cols(), image.rows());
            self.model.set_input_size(self.input_size)?;
        }

        let mut faces_mat = Mat::default();
        self.model.detect(image, &mut faces_mat)?;

        let mut faces = Vec::new();
        if !faces_mat.empty() {
            for i in 0..faces_mat.rows() {
                let data = faces_mat.at_row::<f32>(i)?;
                let x = data[0] as i32;
                let y = data[1] as i32;
                let w = data[2] as i32;
                let h = data[3] as i32;
                let conf = data[14];

                let landmarks = vec![
                    Point2f::new(data[4], data[5]),
                    Point2f::new(data[6], data[7]),
                    Point2f::new(data[8], data[9]),
                    Point2f::new(data[10], data[11]),
                    Point2f::new(data[12], data[13]),
                ];

                faces.push(Face::with_landmarks(Rect::new(x, y, w, h), conf, landmarks));
            }
        }

        Ok(nms(faces, self.nms_threshold))
    }
}

/// Fallback face detector using OpenCV Haar cascades
pub struct HaarCascadeDetector {
    classifier: cascade::CascadeClassifier,
    min_neighbors: i32,
}

impl HaarCascadeDetector {
    pub fn new(cascade_path: &str) -> Result<Self> {
        Self::with_min_neighbors(cascade_path, 3)
    }

    pub fn with_min_neighbors(cascade_path: &str, min_neighbors: i32) -> Result<Self> {
        let classifier = cascade::CascadeClassifier::new(cascade_path)?;
        if classifier.empty()? {
            anyhow::bail!("Failed to load cascade classifier from {}", cascade_path);
        }
        let min_neighbors = min_neighbors.max(2);
        Ok(Self {
            classifier,
            min_neighbors,
        })
    }

    pub fn detect(&mut self, image: &opencv::core::Mat) -> Result<Vec<Face>> {
        let mut gray = opencv::core::Mat::default();
        opencv::imgproc::cvt_color(
            image,
            &mut gray,
            opencv::imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut faces: Vector<Rect> = Vector::new();
        self.classifier.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            self.min_neighbors,
            cascade::CASCADE_SCALE_IMAGE,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;
        let faces = faces.into_iter().map(|rect| Face::new(rect, 1.0)).collect();
        Ok(faces)
    }
}

/// Unified detector enum
pub enum Detector {
    Haar(HaarCascadeDetector),
    Cnn(Box<UltraLightDetector>),
    YuNet(YuNetDetector),
}

impl Detector {
    pub fn describe(&self) -> String {
        match self {
            Detector::Haar(_) => "Haar cascade (CPU)".to_string(),
            Detector::Cnn(d) => format!("Ultra-Light ({})", d.backend_info()),
            Detector::YuNet(_) => "YuNet (OpenCV DNN)".to_string(),
        }
    }

    pub fn detect(&mut self, image: &Mat) -> Result<Vec<Face>> {
        match self {
            Detector::Haar(d) => d.detect(image),
            Detector::Cnn(d) => d.detect(image),
            Detector::YuNet(d) => d.detect(image),
        }
    }
}

/// Pick a detector: YuNet whenever `yunet_path` is set and loads (independent of
/// `use_cnn`), then Ultra-Light if `use_cnn`, then the Haar cascade.
/// `ir_mode` relaxes Haar `minNeighbors` from 3 to 2.
pub fn create_detector(
    cfg: &DetectionConfig,
    ov: &OpenVinoConfig,
    ir_mode: bool,
) -> Result<Detector> {
    let confidence = cfg.confidence_threshold as f32;
    let nms_threshold = cfg.nms_threshold as f32;

    if !cfg.yunet_path.is_empty() {
        for path in yunet_candidates(&cfg.yunet_path) {
            if !Path::new(&path).exists() {
                continue;
            }
            match YuNetDetector::load(&path, confidence, nms_threshold) {
                Ok(d) => {
                    log::info!("Using YuNet face detector with landmarks ({path})");
                    return Ok(Detector::YuNet(d));
                }
                Err(e) => warn!("YuNet {path} failed to load ({e})"),
            }
        }
    }

    if cfg.use_cnn {
        match UltraLightDetector::load(
            &cfg.model_path,
            confidence,
            nms_threshold,
            cfg.use_openvino,
            ov,
        ) {
            Ok(d) => {
                log::info!(
                    "Using CNN face detector (Ultra-Light, {})",
                    d.backend_info()
                );
                return Ok(Detector::Cnn(Box::new(d)));
            }
            Err(e) => {
                warn!(
                    "CNN detector failed to load ({}). Falling back to Haar cascade.",
                    e
                );
            }
        }
    }
    log::info!("Using Haar cascade face detector (CPU, no landmarks)");
    let neighbors = if ir_mode { 2 } else { 3 };
    let path = HAAR_CASCADES
        .iter()
        .copied()
        .find(|p| Path::new(p).exists())
        .unwrap_or(DEFAULT_HAAR_CASCADE);
    Ok(Detector::Haar(HaarCascadeDetector::with_min_neighbors(
        path, neighbors,
    )?))
}

/// The configured YuNet model, then the 2023mar export next to it: the 2026may
/// export (dynamic input shape) targets OpenCV 5 and may not load on OpenCV 4.x.
fn yunet_candidates(configured: &str) -> Vec<String> {
    let mut out = vec![configured.to_string()];
    let p = Path::new(configured);
    if let (Some(dir), Some(name)) = (p.parent(), p.file_name().and_then(|n| n.to_str()))
        && name.contains("2026may")
    {
        out.push(
            dir.join(name.replace("2026may", "2023mar"))
                .to_string_lossy()
                .into_owned(),
        );
    }
    out
}

/// Crop a face from image with optional padding (ratio relative to face size),
/// clipped to the frame.
pub fn crop_face(image: &Mat, bbox: &Rect, padding_ratio: f64) -> Result<Mat> {
    let padding_ratio = padding_ratio.max(0.0);
    let pad_x = (bbox.width as f64 * padding_ratio) as i32;
    let pad_y = (bbox.height as f64 * padding_ratio) as i32;
    let padded = Rect::new(
        bbox.x - pad_x,
        bbox.y - pad_y,
        bbox.width + pad_x * 2,
        bbox.height + pad_y * 2,
    );
    let clipped =
        clip_rect(padded, image.cols(), image.rows()).context("Face box lies outside the frame")?;
    Ok(image.roi(clipped)?.try_clone()?)
}

fn nms(mut faces: Vec<Face>, threshold: f32) -> Vec<Face> {
    if faces.is_empty() {
        return faces;
    }

    // Ascending, so `pop` yields the most confident remaining face.
    faces.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));

    let mut keep = Vec::new();
    while let Some(first) = faces.pop() {
        faces.retain(|face| iou(&first.bbox, &face.bbox) < threshold);
        keep.push(first);
    }

    keep
}

fn iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    let inter_width = (x2 - x1).max(0);
    let inter_height = (y2 - y1).max(0);
    let inter_area = (inter_width * inter_height) as f32;

    let area_a = (a.width * a.height) as f32;
    let area_b = (b.width * b.height) as f32;

    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }

    inter_area / union_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recognition::CANONICAL_LANDMARKS_112;

    fn canonical_face() -> Face {
        let lm = CANONICAL_LANDMARKS_112
            .iter()
            .map(|&(x, y)| Point2f::new(x, y))
            .collect();
        Face::with_landmarks(Rect::new(20, 30, 70, 80), 0.9, lm)
    }

    #[test]
    fn clip() {
        assert_eq!(
            clip_rect(Rect::new(-10, -5, 30, 20), 100, 100),
            Some(Rect::new(0, 0, 20, 15))
        );
        assert_eq!(
            clip_rect(Rect::new(90, 90, 30, 30), 100, 100),
            Some(Rect::new(90, 90, 10, 10))
        );
        assert_eq!(clip_rect(Rect::new(120, 0, 10, 10), 100, 100), None);
        assert_eq!(clip_rect(Rect::new(10, 10, 0, 5), 100, 100), None);
    }

    #[test]
    fn iou_and_nms() {
        let a = Rect::new(0, 0, 10, 10);
        assert_eq!(iou(&a, &a), 1.0);
        assert_eq!(iou(&a, &Rect::new(20, 20, 5, 5)), 0.0);
        assert!((iou(&a, &Rect::new(5, 0, 10, 10)) - 50.0 / 150.0).abs() < 1e-6);

        let faces = vec![
            Face::new(Rect::new(0, 0, 10, 10), 0.6),
            Face::new(Rect::new(1, 1, 10, 10), 0.9),
            Face::new(Rect::new(50, 50, 10, 10), 0.8),
        ];
        let kept = nms(faces, 0.5);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|f| f.confidence == 0.9));
        assert!(kept.iter().all(|f| f.confidence != 0.6));
    }

    #[test]
    fn frontal_pose_is_near_zero() {
        let f = canonical_face();
        assert!(estimate_yaw(&f).unwrap().abs() < 3.0);
        assert!(estimate_pitch(&f).unwrap().abs() < 1.0);
    }

    #[test]
    fn pitch_sign() {
        let mut f = canonical_face();
        f.landmarks[2].y += 8.0; // nose closer to mouth → looking down
        assert!(estimate_pitch(&f).unwrap() > 10.0);
        f.landmarks[2].y -= 16.0;
        assert!(estimate_pitch(&f).unwrap() < -10.0);
        assert!(estimate_pitch(&Face::new(Rect::new(0, 0, 1, 1), 1.0)).is_none());
    }

    #[test]
    fn yunet_fallback_candidates() {
        let c = yunet_candidates("/m/face_detection_yunet_2026may.onnx");
        assert_eq!(c.len(), 2);
        assert!(c[1].ends_with("face_detection_yunet_2023mar.onnx"));
        assert_eq!(
            yunet_candidates("/m/face_detection_yunet_2023mar.onnx").len(),
            1
        );
    }
}
