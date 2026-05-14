use anyhow::Result;
use opencv::core::{Rect, Size, Vector, AlgorithmHint};
use opencv::prelude::{CascadeClassifierTrait, CascadeClassifierTraitConst};
use log::warn;

#[derive(Debug, Clone)]
pub struct Face {
    pub bbox: Rect,          // bounding box (x, y, width, height)
    pub confidence: f32,     // detection confidence
}

pub struct FaceDetector {
    // Placeholder for future ONNX model
}

impl FaceDetector {
    /// Load ONNX model from file (stub)
    pub fn load(_model_path: &str, _confidence_threshold: f32) -> Result<Self> {
        warn!("FaceDetector::load is a stub, using Haar cascade instead");
        Ok(Self {})
    }

    /// Detect faces in an image (BGR format) - stub
    pub fn detect(&mut self, _image: &opencv::core::Mat) -> Result<Vec<Face>> {
        warn!("FaceDetector::detect is a stub, returning empty list");
        Ok(Vec::new())
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
        let faces = faces.into_iter()
            .map(|rect| Face {
                bbox: rect,
                confidence: 1.0, // Haar doesn't provide confidence
            })
            .collect();
        Ok(faces)
    }
}