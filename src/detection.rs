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

type OnnxModel = Arc<TypedRunnableModel>;

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
            // The whole OpenVINO attempt (including reading the static input
            // shape) must fail soft: a dynamic export falls back to tract-onnx.
            let attempt = || -> Result<Self> {
                let session = crate::openvino_backend::OpenVinoSession::from_onnx(model_path, ov)?;
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
                Ok(Self {
                    backend: UltraLightBackend::OpenVino(session),
                    width,
                    height,
                    prob_threshold,
                    nms_threshold,
                })
            };
            match attempt() {
                Ok(detector) => return Ok(detector),
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
                let outputs = model.run(tvec!(input.into_tvalue()))?;
                if outputs.len() < 2 {
                    anyhow::bail!("Ultra-Light tract model returned fewer than 2 outputs");
                }
                let scores_view = outputs[0].to_plain_array_view::<f32>()?;
                let boxes_view = outputs[1].to_plain_array_view::<f32>()?;
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

/// SCRFD face detector (InsightFace): 3 FPN levels (strides 8/16/32), 2 anchors
/// per cell and optional 5-point landmarks. Beats YuNet on WIDER FACE at a
/// comparable size and, unlike Haar, is IR-friendly.
///
/// The network takes a square letterboxed input (`detection.scrfd_input_size`,
/// default 640), so 16:9 IR frames keep their geometry; detections and
/// landmarks are mapped back through the letterbox scale. Preprocessing follows
/// the reference implementation: RGB, `(p − 127.5) / 128`, black padding.
pub struct ScrfdDetector {
    backend: ScrfdBackend,
    input_size: i32,
    prob_threshold: f32,
    nms_threshold: f32,
}

enum ScrfdBackend {
    #[cfg(feature = "openvino")]
    OpenVino(crate::openvino_backend::OpenVinoSession),
    Tract(OnnxModel),
}

impl ScrfdDetector {
    pub fn load(
        model_path: &str,
        input_size: i32,
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
        // The decoder only understands strides 8/16/32, so round up to /32.
        let input_size = scrfd_input_size(input_size);

        #[cfg(feature = "openvino")]
        if use_openvino {
            let side = i64::from(input_size);
            match crate::openvino_backend::OpenVinoSession::from_onnx_static_input(
                model_path, ov, side, side,
            ) {
                Ok(session) => {
                    log::info!("SCRFD detector loaded via OpenVINO on {}", session.device());
                    return Ok(Self {
                        backend: ScrfdBackend::OpenVino(session),
                        input_size,
                        prob_threshold,
                        nms_threshold,
                    });
                }
                Err(e) => warn!("OpenVINO SCRFD init failed: {e}. Falling back to tract-onnx."),
            }
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to read ONNX model")?
            .with_input_fact(0, f32::fact([1, 3, input_size, input_size]).into())
            .context("Failed to set SCRFD input fact")?
            .into_optimized()
            .context("Failed to optimize ONNX model")?
            .into_runnable()
            .context("Failed to create ONNX runnable model")?;
        log::info!("SCRFD detector loaded via tract-onnx (CPU)");
        Ok(Self {
            backend: ScrfdBackend::Tract(model),
            input_size,
            prob_threshold,
            nms_threshold,
        })
    }

    pub fn detect(&mut self, image: &Mat) -> Result<Vec<Face>> {
        let (cols, rows) = (image.cols(), image.rows());
        if cols <= 0 || rows <= 0 {
            anyhow::bail!("Empty frame");
        }
        let (new_w, new_h) = letterbox_fit(cols, rows, self.input_size);
        let input = self.preprocess(image, new_w, new_h)?;

        let outputs = match &mut self.backend {
            #[cfg(feature = "openvino")]
            ScrfdBackend::OpenVino(session) => {
                session.run(input).context("OpenVINO inference failed")?
            }
            ScrfdBackend::Tract(model) => {
                let tensors = model.run(tvec!(input.into_tvalue()))?;
                tensors
                    .iter()
                    .map(|t| {
                        let view = t.to_plain_array_view::<f32>()?;
                        let dims = view.shape().iter().map(|&d| d as i64).collect();
                        Ok((dims, view.iter().copied().collect::<Vec<f32>>()))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };

        // Undo the letterbox: canvas x = orig x * sx, canvas y = orig y * sy.
        let (sx, sy) = (new_w as f32 / cols as f32, new_h as f32 / rows as f32);
        let faces = scrfd_decode(
            &outputs,
            self.input_size,
            self.prob_threshold,
            sx,
            sy,
            cols,
            rows,
        )?;
        Ok(nms(faces, self.nms_threshold))
    }

    fn preprocess(&self, image: &Mat, new_w: i32, new_h: i32) -> Result<Array4<f32>> {
        let mut resized = Mat::default();
        opencv::imgproc::resize(
            image,
            &mut resized,
            Size::new(new_w, new_h),
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
        if pixels.len() < (new_w * new_h * 3) as usize {
            anyhow::bail!("Unexpected pixel buffer size after SCRFD preprocessing");
        }

        let size = self.input_size as usize;
        // Black padding in *pixel* space, exactly like the reference code.
        let pad = -127.5f32 / 128.0;
        let mut input = Array4::<f32>::from_elem((1, 3, size, size), pad);
        let (nw, nh) = (new_w as usize, new_h as usize);
        for y in 0..nh {
            for x in 0..nw {
                let idx = (y * nw + x) * 3;
                input[[0, 0, y, x]] = (pixels[idx] as f32 - 127.5) / 128.0;
                input[[0, 1, y, x]] = (pixels[idx + 1] as f32 - 127.5) / 128.0;
                input[[0, 2, y, x]] = (pixels[idx + 2] as f32 - 127.5) / 128.0;
            }
        }
        Ok(input)
    }

    pub fn backend_info(&self) -> String {
        match &self.backend {
            #[cfg(feature = "openvino")]
            ScrfdBackend::OpenVino(session) => format!("OpenVINO ({})", session.device()),
            ScrfdBackend::Tract(_) => "tract-onnx (CPU)".to_string(),
        }
    }
}

/// SCRFD input side: at least 64 and divisible by the smallest stride (32).
fn scrfd_input_size(requested: i32) -> i32 {
    let requested = requested.max(64);
    (requested + 31) / 32 * 32
}

/// Aspect-preserving fit of `cols × rows` into a `size × size` canvas.
/// Returns the canvas size (top-left aligned; the rest stays black).
fn letterbox_fit(cols: i32, rows: i32, size: i32) -> (i32, i32) {
    let scale = (size as f32 / cols as f32).min(size as f32 / rows as f32);
    let new_w = ((cols as f32 * scale).round() as i32).clamp(1, size);
    let new_h = ((rows as f32 * scale).round() as i32).clamp(1, size);
    (new_w, new_h)
}

/// One SCRFD feature level: `rows = (input_size / stride)² × anchors`
/// predictions of `width` values each.
struct ScrfdLevel<'a> {
    stride: i32,
    anchors: usize,
    rows: usize,
    data: &'a [f32],
}

/// Map a level's prediction count back to `(stride, anchors)`.
fn scrfd_level(rows: usize, input_size: i32) -> Option<(i32, usize)> {
    [8, 16, 32, 64, 128].into_iter().find_map(|stride| {
        let cells = (input_size / stride) as usize;
        let cells = cells * cells;
        [1usize, 2]
            .into_iter()
            .find(|anchors| cells * *anchors == rows)
            .map(|anchors| (stride, anchors))
    })
}

/// Group flat model outputs into SCRFD levels by their trailing dimension:
/// 1 = scores, 4 = bbox distances, 10 = 5 landmarks, sorted by stride.
/// Unknown trailing dimensions are ignored so a re-export with extra heads
/// still works.
fn scrfd_levels<'a>(
    outputs: &'a [(Vec<i64>, Vec<f32>)],
    input_size: i32,
    width: usize,
) -> Result<Vec<ScrfdLevel<'a>>> {
    let mut levels = Vec::new();
    for (dims, data) in outputs {
        if dims.last().copied() != Some(width as i64) || data.is_empty() {
            continue;
        }
        let rows = data.len() / width;
        let (stride, anchors) = scrfd_level(rows, input_size).with_context(|| {
            format!("Unexpected SCRFD output of {rows} rows for a {input_size}px input")
        })?;
        levels.push(ScrfdLevel {
            stride,
            anchors,
            rows,
            data,
        });
    }
    levels.sort_by_key(|l| l.stride);
    Ok(levels)
}

/// Decode SCRFD outputs to faces in original-frame coordinates.
///
/// `sx`/`sy` map canvas → frame (`new / original`), `clip_w`/`clip_h` bound the
/// result. Follows `insightface.model_zoo.scrfd`: anchor centers at multiples of
/// the stride (two anchors per cell, cell-major), distances scaled by the stride.
fn scrfd_decode(
    outputs: &[(Vec<i64>, Vec<f32>)],
    input_size: i32,
    prob_threshold: f32,
    sx: f32,
    sy: f32,
    clip_w: i32,
    clip_h: i32,
) -> Result<Vec<Face>> {
    let scores = scrfd_levels(outputs, input_size, 1)?;
    let boxes = scrfd_levels(outputs, input_size, 4)?;
    let kps = scrfd_levels(outputs, input_size, 10)?;
    if scores.is_empty() {
        anyhow::bail!("SCRFD model returned no score outputs");
    }
    if boxes.len() != scores.len() {
        anyhow::bail!(
            "SCRFD output mismatch: {} score levels, {} bbox levels",
            scores.len(),
            boxes.len()
        );
    }
    if !kps.is_empty() && kps.len() != scores.len() {
        anyhow::bail!(
            "SCRFD output mismatch: {} score levels, {} landmark levels",
            scores.len(),
            kps.len()
        );
    }

    let mut faces = Vec::new();
    for (i, score) in scores.iter().enumerate() {
        let bbox = &boxes[i];
        if bbox.data.len() < score.rows * 4 {
            anyhow::bail!("SCRFD bbox level has too few values");
        }
        let landmarks_level = kps.get(i);
        if let Some(kps) = landmarks_level
            && kps.data.len() < score.rows * 10
        {
            anyhow::bail!("SCRFD landmark level has too few values");
        }

        let stride = score.stride as f32;
        let grid = (input_size / score.stride) as usize;
        for row in 0..score.rows {
            let conf = score.data[row];
            if conf < prob_threshold {
                continue;
            }
            let cell = row / score.anchors;
            // Anchor centers sit at multiples of the stride (reference decode).
            let cx = ((cell % grid) as f32) * stride;
            let cy = ((cell / grid) as f32) * stride;

            let d = &bbox.data[row * 4..row * 4 + 4];
            let x1 = ((cx - d[0] * stride) / sx).clamp(0.0, clip_w as f32);
            let y1 = ((cy - d[1] * stride) / sy).clamp(0.0, clip_h as f32);
            let x2 = ((cx + d[2] * stride) / sx).clamp(0.0, clip_w as f32);
            let y2 = ((cy + d[3] * stride) / sy).clamp(0.0, clip_h as f32);
            let (x1, y1, x2, y2) = (
                x1.round() as i32,
                y1.round() as i32,
                x2.round() as i32,
                y2.round() as i32,
            );
            if x2 - x1 < 1 || y2 - y1 < 1 {
                continue;
            }

            let landmarks = landmarks_level.map(|kps| {
                (0..5)
                    .map(|j| {
                        let off = row * 10 + j * 2;
                        Point2f::new(
                            (cx + kps.data[off] * stride) / sx,
                            (cy + kps.data[off + 1] * stride) / sy,
                        )
                    })
                    .collect()
            });

            faces.push(Face::with_landmarks(
                Rect::new(x1, y1, x2 - x1, y2 - y1),
                conf,
                landmarks.unwrap_or_default(),
            ));
        }
    }
    Ok(faces)
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
    Scrfd(ScrfdDetector),
}

impl Detector {
    pub fn describe(&self) -> String {
        match self {
            Detector::Haar(_) => "Haar cascade (CPU)".to_string(),
            Detector::Cnn(d) => format!("Ultra-Light ({})", d.backend_info()),
            Detector::YuNet(_) => "YuNet (OpenCV DNN)".to_string(),
            Detector::Scrfd(d) => format!("SCRFD ({})", d.backend_info()),
        }
    }

    pub fn detect(&mut self, image: &Mat) -> Result<Vec<Face>> {
        match self {
            Detector::Haar(d) => d.detect(image),
            Detector::Cnn(d) => d.detect(image),
            Detector::YuNet(d) => d.detect(image),
            Detector::Scrfd(d) => d.detect(image),
        }
    }
}

/// Pick a detector: SCRFD whenever `scrfd_path` is set and loads, then YuNet,
/// then Ultra-Light if `use_cnn`, then the Haar cascade.
/// `ir_mode` relaxes Haar `minNeighbors` from 3 to 2.
pub fn create_detector(
    cfg: &DetectionConfig,
    ov: &OpenVinoConfig,
    ir_mode: bool,
) -> Result<Detector> {
    let confidence = cfg.confidence_threshold as f32;
    let nms_threshold = cfg.nms_threshold as f32;

    if !cfg.scrfd_path.is_empty() {
        if !Path::new(&cfg.scrfd_path).exists() {
            warn!("SCRFD model {} not found; trying YuNet", cfg.scrfd_path);
        } else {
            match ScrfdDetector::load(
                &cfg.scrfd_path,
                cfg.scrfd_input_size,
                confidence,
                nms_threshold,
                cfg.use_openvino,
                ov,
            ) {
                Ok(d) => {
                    log::info!(
                        "Using SCRFD face detector with landmarks ({}, {})",
                        cfg.scrfd_path,
                        d.backend_info()
                    );
                    return Ok(Detector::Scrfd(d));
                }
                Err(e) => warn!("SCRFD {} failed to load ({e})", cfg.scrfd_path),
            }
        }
    }

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

    /// Build a flat SCRFD output set with a single detection at a known cell.
    fn scrfd_outputs(input_size: i32, stride: i32, conf: f32) -> Vec<(Vec<i64>, Vec<f32>)> {
        let grid = (input_size / stride) as usize;
        let anchors = 2usize;
        let rows = grid * grid * anchors;
        let cell = grid * grid / 2 + grid / 2;
        let row = cell * anchors; // first anchor of the middle cell
        let cx = (cell % grid) as f32 * stride as f32;
        let cy = (cell / grid) as f32 * stride as f32;
        let _ = (cx, cy);

        let mut scores = vec![0.0f32; rows];
        scores[row] = conf;
        let mut bboxes = vec![0.0f32; rows * 4];
        bboxes[row * 4..row * 4 + 4].copy_from_slice(&[2.0, 2.0, 2.0, 2.0]);
        let mut kps = vec![0.0f32; rows * 10];
        for (j, v) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
            kps[row * 10 + j * 2] = *v;
            kps[row * 10 + j * 2 + 1] = *v;
        }
        vec![
            (vec![rows as i64, 1], scores),
            (vec![rows as i64, 4], bboxes),
            (vec![rows as i64, 10], kps),
        ]
    }

    #[test]
    fn scrfd_input_size_rounds_up_to_stride() {
        assert_eq!(scrfd_input_size(640), 640);
        assert_eq!(scrfd_input_size(641), 672);
        assert_eq!(scrfd_input_size(10), 64);
        assert_eq!(scrfd_input_size(0), 64);
    }

    #[test]
    fn letterbox_preserves_aspect() {
        assert_eq!(letterbox_fit(1644, 2052, 640), (513, 640));
        assert_eq!(letterbox_fit(2052, 1644, 640), (640, 513));
        assert_eq!(letterbox_fit(640, 640, 640), (640, 640));
        let (w, h) = letterbox_fit(100, 1000, 640);
        assert_eq!((w, h), (64, 640));
    }

    #[test]
    fn scrfd_levels_group_by_stride() {
        let outputs = scrfd_outputs(64, 8, 0.9);
        let scores = scrfd_levels(&outputs, 64, 1).unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(
            (scores[0].stride, scores[0].anchors, scores[0].rows),
            (8, 2, 128)
        );
        // The 4-wide and 10-wide heads group separately.
        assert_eq!(scrfd_levels(&outputs, 64, 4).unwrap().len(), 1);
        assert_eq!(scrfd_levels(&outputs, 64, 10).unwrap().len(), 1);
        // An output width the model does not produce is ignored.
        assert!(scrfd_levels(&outputs, 64, 3).unwrap().is_empty());
        // Row counts that fit no known (stride, anchors) pair are an error.
        let bad = vec![(vec![7_i64, 1], vec![0.5f32; 7])];
        assert!(scrfd_levels(&bad, 64, 1).is_err());
    }

    #[test]
    fn scrfd_decodes_box_and_landmarks() {
        let outputs = scrfd_outputs(64, 8, 0.9);
        let faces = scrfd_decode(&outputs, 64, 0.5, 1.0, 1.0, 64, 64).unwrap();
        assert_eq!(faces.len(), 1);
        let f = &faces[0];
        assert!((f.confidence - 0.9).abs() < 1e-6);
        // middle cell of an 8×8 grid: center (32, 32), distance 2 × stride 8
        assert_eq!(f.bbox, Rect::new(16, 16, 32, 32));
        assert_eq!(f.landmarks.len(), 5);
        assert!((f.landmarks[0].x - 16.0).abs() < 1e-4);
        assert!((f.landmarks[4].x - 48.0).abs() < 1e-4);
    }

    #[test]
    fn scrfd_respects_threshold_and_scaling() {
        let outputs = scrfd_outputs(64, 8, 0.4);
        assert!(
            scrfd_decode(&outputs, 64, 0.5, 1.0, 1.0, 64, 64)
                .unwrap()
                .is_empty()
        );

        // sx/sy map canvas → frame: halving them doubles the reported box.
        let outputs = scrfd_outputs(64, 8, 0.9);
        let faces = scrfd_decode(&outputs, 64, 0.5, 0.5, 0.5, 128, 128).unwrap();
        assert_eq!(faces[0].bbox, Rect::new(32, 32, 64, 64));
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
