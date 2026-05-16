use anyhow::{Context, Result};
use log::{info, warn};
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Mat, Size};
use opencv::prelude::{MatTraitConst, MatTraitConstManual};
use std::path::Path;
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct FaceEmbedding {
    pub vector: Vec<f32>, // embedding vector
    pub norm: f32,        // L2 norm for faster comparison
}

impl FaceEmbedding {
    pub fn new(mut vector: Vec<f32>) -> Self {
        let mut norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut vector {
                *v /= norm;
            }
            norm = 1.0;
        }
        Self { vector, norm }
    }

    /// Cosine similarity between two embeddings
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.vector.iter().zip(&other.vector).map(|(a, b)| a * b).sum();
        dot / (self.norm * other.norm)
    }

    /// Euclidean distance
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        self.vector.iter().zip(&other.vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

enum Backend {
    #[cfg(feature = "openvino")]
    OpenVino(crate::openvino_backend::OpenVinoSession),
    Onnx(OnnxModel),
    Fallback,
}

pub struct FaceRecognizer {
    backend: Backend,
}

impl FaceRecognizer {
    /// Load model. If `use_openvino` is true and the feature is compiled in,
    /// attempt OpenVINO first, then tract-onnx, then deterministic fallback.
    /// If `use_openvino` is false, skip OpenVINO.
    pub fn load(model_path: &str, use_openvino: bool) -> Result<Self> {
        let _ = use_openvino;
        let path = Path::new(model_path);
        if !path.exists() {
            warn!(
                "Recognition model not found at {}, using deterministic fallback",
                model_path
            );
            info!("Recognition loaded via deterministic fallback (CPU)");
            return Ok(Self {
                backend: Backend::Fallback,
            });
        }

        #[cfg(feature = "openvino")]
        if use_openvino {
            match crate::openvino_backend::OpenVinoSession::from_onnx(model_path) {
                Ok(session) => {
                    info!(
                        "Recognition loaded via OpenVINO on {}",
                        session.device()
                    );
                    return Ok(Self {
                        backend: Backend::OpenVino(session),
                    });
                }
                Err(e) => {
                    warn!("OpenVINO backend init failed: {e}. Trying tract-onnx fallback");
                }
            }
        }

        match tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to read ONNX model")?
            .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())
            .context("Failed to set model input fact")?
            .into_optimized()
            .context("Failed to optimize ONNX model")?
            .into_runnable()
            .context("Failed to create ONNX runnable model")
        {
            Ok(model) => {
                info!("Recognition loaded via tract-onnx (CPU)");
                Ok(Self {
                    backend: Backend::Onnx(model),
                })
            },
            Err(e) => {
                warn!("tract-onnx backend init failed: {e}. Using deterministic fallback");
                info!("Recognition loaded via deterministic fallback (CPU)");
                Ok(Self {
                    backend: Backend::Fallback,
                })
            }
        }
    }

    pub fn backend_info(&self) -> String {
        match &self.backend {
            #[cfg(feature = "openvino")]
            Backend::OpenVino(session) => format!("OpenVINO ({})", session.device()),
            Backend::Onnx(_) => "tract-onnx (CPU)".to_string(),
            Backend::Fallback => "deterministic fallback (CPU)".to_string(),
        }
    }

    /// Extract embedding from a face crop, using best available backend.
    pub fn extract(&mut self, face_image: &opencv::core::Mat) -> Result<FaceEmbedding> {
        #[cfg(feature = "openvino")]
        if let Backend::OpenVino(session) = &mut self.backend {
            return match extract_openvino_embedding(session, face_image) {
                Ok(emb) => Ok(emb),
                Err(e) => {
                    warn!("OpenVINO inference failed, falling back: {e}");
                    extract_fallback_embedding(face_image)
                }
            };
        }

        if let Backend::Onnx(model) = &self.backend {
            match extract_onnx_embedding(model, face_image) {
                Ok(embedding) => return Ok(embedding),
                Err(e) => {
                    warn!("ONNX inference failed, using deterministic fallback: {e}");
                }
            }
        }

        extract_fallback_embedding(face_image)
    }
}

#[cfg(feature = "openvino")]
fn extract_openvino_embedding(
    session: &mut crate::openvino_backend::OpenVinoSession,
    face_image: &opencv::core::Mat,
) -> Result<FaceEmbedding> {
    let input = preprocess_for_mobilefacenet(face_image)?;
    let outputs = session.run(input).context("OpenVINO inference failed")?;
    let (_, data) = outputs
        .into_iter()
        .next()
        .context("OpenVINO model returned no outputs")?;
    if data.is_empty() {
        anyhow::bail!("OpenVINO output embedding is empty");
    }
    Ok(FaceEmbedding::new(data))
}

fn extract_onnx_embedding(model: &OnnxModel, face_image: &opencv::core::Mat) -> Result<FaceEmbedding> {
    let input = preprocess_for_mobilefacenet(face_image)?;
    let outputs = model
        .run(tvec!(input.into_tensor().into()))
        .context("ONNX run failed")?;
    let output = outputs
        .first()
        .context("ONNX model returned no outputs")?;
    let view = output
        .to_array_view::<f32>()
        .context("ONNX output is not f32 tensor")?;
    let vec = view.iter().copied().collect::<Vec<f32>>();
    if vec.is_empty() {
        anyhow::bail!("ONNX output embedding is empty");
    }
    Ok(FaceEmbedding::new(vec))
}

fn preprocess_for_mobilefacenet(face_image: &opencv::core::Mat) -> Result<Array4<f32>> {
    let mut rgb = Mat::default();
    opencv::imgproc::cvt_color(
        face_image,
        &mut rgb,
        opencv::imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut resized = Mat::default();
    opencv::imgproc::resize(
        &rgb,
        &mut resized,
        Size::new(112, 112),
        0.0,
        0.0,
        opencv::imgproc::INTER_AREA,
    )?;

    if !resized.is_continuous() {
        resized = resized.try_clone()?;
    }
    let pixels = resized.data_bytes()?;
    if pixels.len() < 112 * 112 * 3 {
        anyhow::bail!("Unexpected pixel buffer size after face preprocessing");
    }

    let mut input = Array4::<f32>::zeros((1, 3, 112, 112));
    for y in 0..112usize {
        for x in 0..112usize {
            let idx = (y * 112 + x) * 3;
            let r = pixels[idx] as f32;
            let g = pixels[idx + 1] as f32;
            let b = pixels[idx + 2] as f32;

            // Typical MobileFaceNet normalization to [-1, 1]
            input[[0, 0, y, x]] = (r - 127.5) / 128.0;
            input[[0, 1, y, x]] = (g - 127.5) / 128.0;
            input[[0, 2, y, x]] = (b - 127.5) / 128.0;
        }
    }
    Ok(input)
}

/// Deterministic fallback embedding extractor.
fn extract_fallback_embedding(face_image: &opencv::core::Mat) -> Result<FaceEmbedding> {
        let mut gray = Mat::default();
        opencv::imgproc::cvt_color(
            face_image,
            &mut gray,
            opencv::imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut resized = Mat::default();
        opencv::imgproc::resize(
            &gray,
            &mut resized,
            Size::new(16, 8),
            0.0,
            0.0,
            opencv::imgproc::INTER_AREA,
        )?;

        let pixels = resized.data_typed::<u8>()?;
        let mut vec = Vec::with_capacity(128);
        for &p in pixels.iter().take(128) {
            vec.push((p as f32 / 255.0) - 0.5);
        }
        while vec.len() < 128 {
            vec.push(0.0);
        }

        // Zero-center to reduce sensitivity to global illumination changes.
        let mean = vec.iter().sum::<f32>() / vec.len() as f32;
        for v in &mut vec {
            *v -= mean;
        }

        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut vec {
                *v /= norm;
            }
        }

        Ok(FaceEmbedding::new(vec))
}

/// Utility to align face using landmarks (simplified)
pub fn align_face(_image: &opencv::core::Mat, _landmarks: &[opencv::core::Point]) -> Result<opencv::core::Mat> {
    // This is a placeholder; real alignment would involve affine transformation.
    // For now, just return the original image.
    Ok(_image.clone())
}
