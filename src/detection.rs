use anyhow::{Context, Result};
use log::warn;
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Mat, Rect, Size, Vector};
use opencv::prelude::{CascadeClassifierTrait, CascadeClassifierTraitConst, MatTraitConst, MatTraitConstManual};
use std::path::Path;
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct Face {
    pub bbox: Rect,
    pub confidence: f32,
}

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Ultra-Light face detector using ONNX
pub struct UltraLightDetector {
    model: OnnxModel,
    width: usize,
    height: usize,
    prob_threshold: f32,
    nms_threshold: f32,
}

impl UltraLightDetector {
    pub fn load(model_path: &str, prob_threshold: f32, nms_threshold: f32) -> Result<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            anyhow::bail!("Model not found: {}", model_path);
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to read ONNX model")?
            .into_optimized()
            .context("Failed to optimize ONNX model")?
            .into_runnable()
            .context("Failed to create ONNX runnable model")?;

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
            model,
            width,
            height,
            prob_threshold,
            nms_threshold,
        })
    }

    pub fn detect(&self, image: &Mat) -> Result<Vec<Face>> {
        let orig_h = image.rows() as f32;
        let orig_w = image.cols() as f32;

        let input = self.preprocess(image)?;
        let outputs = self.model.run(tvec!(input.into_tensor().into()))?;

        log::debug!("Ultra-Light outputs count: {}", outputs.len());
        for (i, o) in outputs.iter().enumerate() {
            if let Ok(view) = o.to_array_view::<f32>() {
                log::debug!("Output {} shape: {:?}", i, view.shape());
            }
        }

        let mut faces = Vec::new();

        // Ultra-Light: outputs[0] = scores [1,N,2], outputs[1] = boxes [1,N,4]
        if outputs.len() >= 2 {
            let scores_view = outputs[0].to_array_view::<f32>()?;
            let boxes_view = outputs[1].to_array_view::<f32>()?;

            if boxes_view.ndim() >= 3 && scores_view.ndim() >= 3 {
                let num_anchors = boxes_view.shape()[1];
                for i in 0..num_anchors {
                    let score = scores_view[[0, i, 1]]; // class 1 = face
                    if score < self.prob_threshold {
                        continue;
                    }
                    // Normalized coords [x1, y1, x2, y2] relative to model input
                    let x1_f = boxes_view[[0, i, 0]] * self.width as f32;
                    let y1_f = boxes_view[[0, i, 1]] * self.height as f32;
                    let x2_f = boxes_view[[0, i, 2]] * self.width as f32;
                    let y2_f = boxes_view[[0, i, 3]] * self.height as f32;

                    let x1 = x1_f.max(0.0) as i32;
                    let y1 = y1_f.max(0.0) as i32;
                    let width = (x2_f - x1_f).max(1.0) as i32;
                    let height = (y2_f - y1_f).max(1.0) as i32;

                    faces.push(Face {
                        bbox: Rect::new(x1, y1, width, height),
                        confidence: score,
                    });
                }
            }
        }

        // Scale back to original image size
        let scale_x = orig_w / self.width as f32;
        let scale_y = orig_h / self.height as f32;
        for face in &mut faces {
            face.bbox.x = (face.bbox.x as f32 * scale_x) as i32;
            face.bbox.y = (face.bbox.y as f32 * scale_y) as i32;
            face.bbox.width = (face.bbox.width as f32 * scale_x) as i32;
            face.bbox.height = (face.bbox.height as f32 * scale_y) as i32;
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
}

/// Fallback face detector using OpenCV Haar cascades
pub struct HaarCascadeDetector {
    classifier: opencv::objdetect::CascadeClassifier,
    min_neighbors: i32,
}

impl HaarCascadeDetector {
    pub fn new(cascade_path: &str) -> Result<Self> {
        Self::with_min_neighbors(cascade_path, 3)
    }

    /// `min_neighbors`: OpenCV Haar `minNeighbors` (lower = more detections, more false positives; try 2 for IR).
    pub fn with_min_neighbors(cascade_path: &str, min_neighbors: i32) -> Result<Self> {
        let classifier = opencv::objdetect::CascadeClassifier::new(cascade_path)?;
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
            opencv::objdetect::CASCADE_SCALE_IMAGE,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;
        let faces = faces
            .into_iter()
            .map(|rect| Face {
                bbox: rect,
                confidence: 1.0, // Haar doesn't provide confidence
            })
            .collect();
        Ok(faces)
    }
}

/// Unified detector enum (CNN or Haar) with automatic fallback.
pub enum Detector {
    Haar(HaarCascadeDetector),
    Cnn(UltraLightDetector),
}

impl Detector {
    pub fn detect(&mut self, image: &Mat) -> Result<Vec<Face>> {
        match self {
            Detector::Haar(d) => d.detect(image),
            Detector::Cnn(d) => d.detect(image),
        }
    }
}

/// Try to load CNN detector, fall back to Haar cascade on failure.
pub fn create_detector(
    cnn_model_path: &str,
    confidence_threshold: f32,
    nms_threshold: f32,
    use_cnn: bool,
    haar_path: &str,
    haar_neighbors: i32,
) -> Result<Detector> {
    if use_cnn {
        match UltraLightDetector::load(cnn_model_path, confidence_threshold, nms_threshold) {
            Ok(d) => {
                log::info!("Using CNN face detector (Ultra-Light)");
                return Ok(Detector::Cnn(d));
            }
            Err(e) => {
                warn!("CNN detector failed to load ({}). Falling back to Haar cascade.", e);
            }
        }
    }
    Ok(Detector::Haar(HaarCascadeDetector::with_min_neighbors(
        haar_path, haar_neighbors,
    )?))
}

/// Crop a face from image with optional padding (ratio relative to face size).
pub fn crop_face(image: &Mat, bbox: &Rect, padding_ratio: f64) -> Result<Mat> {
    if padding_ratio <= 0.0 {
        return Ok(image.roi(*bbox)?.try_clone()?);
    }
    let pad_x = (bbox.width as f64 * padding_ratio) as i32;
    let pad_y = (bbox.height as f64 * padding_ratio) as i32;

    let x = (bbox.x - pad_x).max(0);
    let y = (bbox.y - pad_y).max(0);
    let width = (bbox.width + pad_x * 2).min(image.cols() - x);
    let height = (bbox.height + pad_y * 2).min(image.rows() - y);

    let padded = Rect::new(x, y, width, height);
    Ok(image.roi(padded)?.try_clone()?)
}

fn nms(mut faces: Vec<Face>, threshold: f32) -> Vec<Face> {
    if faces.is_empty() {
        return faces;
    }

    faces.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
