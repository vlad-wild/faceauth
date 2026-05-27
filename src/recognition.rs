use anyhow::{Context, Result};
use log::{info, warn};
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Mat, Point2f, Scalar, Size};
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

/// Align a face using landmarks from YuNet (or any 5-point detector).
///
/// **5 landmarks** — computes a full least-squares affine transform (6 DOF) that
/// maps all five detected points to InsightFace canonical positions. This
/// partially corrects for yaw, pitch, and roll, improving recognition at
/// non-frontal angles.
///
/// **2 landmarks** — falls back to a similarity transform (4 DOF) that corrects
/// in-plane rotation and scale only.
///
/// `image` — full BGR frame.
/// `landmarks` — points from YuNet: [right_eye, left_eye, nose, right_mouth, left_mouth].
/// `output_size` — width/height of the output square (e.g. 112 for MobileFaceNet).
///
/// Returns an `output_size × output_size` aligned BGR face.
pub fn align_face(
    image: &Mat,
    landmarks: &[Point2f],
    output_size: i32,
) -> Result<Mat> {
    if landmarks.len() < 2 {
        anyhow::bail!("Need at least 2 eye landmarks for alignment");
    }

    let out_f = output_size as f64;
    let scale = out_f / 112.0;

    // Compute the affine matrix (different methods depending on available points)
    let affine = if landmarks.len() >= 5 {
        // — Full 5-point least-squares affine (6 DOF) —
        // Canonical InsightFace positions in 112×112 output:
        let dst = [
            Point2f::new((38.2946 * scale) as f32, (51.6963 * scale) as f32),  // YuNet[0]
            Point2f::new((73.5318 * scale) as f32, (51.5014 * scale) as f32),  // YuNet[1]
            Point2f::new((56.0252 * scale) as f32, (71.7366 * scale) as f32),  // YuNet[2]
            Point2f::new((41.5493 * scale) as f32, (92.3655 * scale) as f32),  // YuNet[3]
            Point2f::new((70.7299 * scale) as f32, (92.2041 * scale) as f32),  // YuNet[4]
        ];
        compute_affine_5pt(&landmarks[..5], &dst)?
    } else {
        // — 2-point similarity (4 DOF) fallback —
        let src_rx = landmarks[0].x as f64;
        let src_ry = landmarks[0].y as f64;
        let src_lx = landmarks[1].x as f64;
        let src_ly = landmarks[1].y as f64;

        let dst_rx = 38.2946 * scale;
        let dst_ry = 51.6963 * scale;
        let dst_lx = 73.5318 * scale;
        let dst_ly = 51.5014 * scale;

        let dx = src_lx - src_rx;
        let dy = src_ly - src_ry;
        let du = dst_lx - dst_rx;
        let dv = dst_ly - dst_ry;

        let denom = dx * dx + dy * dy;
        if denom < f64::EPSILON {
            anyhow::bail!("Eye landmarks are too close to compute alignment");
        }

        let c = (du * dx + dv * dy) / denom;
        let d = (-du * dy + dv * dx) / denom;
        let tx = dst_rx - c * src_rx + d * src_ry;
        let ty = dst_ry - d * src_rx - c * src_ry;

        vec![c, -d, tx, d, c, ty]
    };

    let affine_mat = Mat::from_slice(&affine)?;
    let affine_mat = affine_mat.reshape(1, 2)?;

    let mut aligned = Mat::default();
    opencv::imgproc::warp_affine(
        image,
        &mut aligned,
        &affine_mat,
        Size::new(output_size, output_size),
        opencv::imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    Ok(aligned)
}

/// Compute a least-squares full affine transform from 5 point correspondences.
/// Solves the overdetermined system via normal equations (A^T·A·x = A^T·b).
/// Returns the 6-element vector [a00, a01, a02, a10, a11, a12] forming
/// the 2×3 affine matrix [[a00, a01, a02], [a10, a11, a12]].
fn compute_affine_5pt(src: &[Point2f], dst: &[Point2f]) -> Result<Vec<f64>> {
    let n = src.len().min(dst.len());
    if n < 3 {
        anyhow::bail!("Need at least 3 points for full affine");
    }

    // Build normal equations: A^T * A (6×6) and A^T * b (6)
    let mut ata = [[0.0; 6]; 6];
    let mut atb = [0.0; 6];

    for i in 0..n {
        let sx = src[i].x as f64;
        let sy = src[i].y as f64;
        let dx = dst[i].x as f64;
        let dy = dst[i].y as f64;

        // Row for x': [sx, sy, 1,  0,  0, 0]
        let row_x = [sx, sy, 1.0, 0.0, 0.0, 0.0];
        for r in 0..6 {
            for c in 0..6 {
                ata[r][c] += row_x[r] * row_x[c];
            }
            atb[r] += row_x[r] * dx;
        }

        // Row for y': [ 0,  0, 0, sx, sy, 1]
        let row_y = [0.0, 0.0, 0.0, sx, sy, 1.0];
        for r in 0..6 {
            for c in 0..6 {
                ata[r][c] += row_y[r] * row_y[c];
            }
            atb[r] += row_y[r] * dy;
        }
    }

    // Solve 6×6 system using Gaussian elimination with partial pivoting
    let x = solve_6x6(ata, atb)?;
    Ok(x.to_vec())
}

/// Solve a 6×6 linear system Ax = b via Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve_6x6(a: [[f64; 6]; 6], b: [f64; 6]) -> Result<[f64; 6]> {
    // Augmented matrix [A | b] – 6 rows × 7 columns
    let mut m = [[0.0; 7]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = a[i][j];
        }
        m[i][6] = b[i];
    }

    // Forward elimination
    for col in 0..6 {
        // Partial pivoting
        let mut best = col;
        for row in (col + 1)..6 {
            if m[row][col].abs() > m[best][col].abs() {
                best = row;
            }
        }
        m.swap(col, best);

        if m[col][col].abs() < f64::EPSILON {
            anyhow::bail!("Singular matrix in affine least-squares solve");
        }

        let pivot = m[col][col];
        for row in (col + 1)..6 {
            let factor = m[row][col] / pivot;
            for j in col..=6 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut sum = m[i][6];
        for j in (i + 1)..6 {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }

    Ok(x)
}
