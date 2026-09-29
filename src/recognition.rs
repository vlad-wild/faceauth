use anyhow::{Context, Result};
use log::info;
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Mat, Point2f, Scalar, Size};
use opencv::prelude::{MatTraitConst, MatTraitConstManual};
use std::path::Path;
use tract_onnx::prelude::*;

use crate::config::{OpenVinoConfig, RecognitionConfig};
use crate::matching::{file_fingerprint, l2_distance, l2_normalize};

#[derive(Debug, Clone)]
pub struct FaceEmbedding {
    /// L2-normalized embedding vector.
    pub vector: Vec<f32>,
}

impl FaceEmbedding {
    pub fn new(mut vector: Vec<f32>) -> Self {
        l2_normalize(&mut vector);
        Self { vector }
    }

    /// Cosine similarity (vectors are unit length, so this is the dot product).
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        self.vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean distance; infinite for vectors of different length.
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        l2_distance(&self.vector, &other.vector)
    }
}

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

enum Backend {
    #[cfg(feature = "openvino")]
    OpenVino(crate::openvino_backend::OpenVinoSession),
    Onnx(OnnxModel),
}

/// Face embedding extractor. Fails closed: there is no substitute embedding when
/// the model is missing or inference fails, because authentication must never
/// compare vectors that did not come from the enrolled recognizer.
pub struct FaceRecognizer {
    backend: Backend,
    model_id: String,
}

impl FaceRecognizer {
    pub fn from_config(rc: &RecognitionConfig, ov: &OpenVinoConfig) -> Result<Self> {
        Self::load(&rc.model_path, rc.use_openvino, ov)
    }

    /// Load model. With `use_openvino` (and the feature compiled in) OpenVINO is
    /// tried first; tract-onnx runs the same model on CPU otherwise.
    pub fn load(model_path: &str, use_openvino: bool, ov: &OpenVinoConfig) -> Result<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            anyhow::bail!("Recognition model not found: {model_path}");
        }
        let model_id = file_fingerprint(path)?;

        #[cfg(feature = "openvino")]
        if use_openvino {
            match crate::openvino_backend::OpenVinoSession::from_onnx(model_path, ov) {
                Ok(session) => {
                    info!("Recognition loaded via OpenVINO on {}", session.device());
                    return Ok(Self {
                        backend: Backend::OpenVino(session),
                        model_id,
                    });
                }
                Err(e) => {
                    log::warn!("OpenVINO backend init failed: {e}. Trying tract-onnx");
                }
            }
        }
        #[cfg(not(feature = "openvino"))]
        let _ = (use_openvino, ov);

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to read ONNX model")?
            .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())
            .context("Failed to set model input fact")?
            .into_optimized()
            .context("Failed to optimize ONNX model")?
            .into_runnable()
            .context("Failed to create ONNX runnable model")?;
        info!("Recognition loaded via tract-onnx (CPU)");
        Ok(Self {
            backend: Backend::Onnx(model),
            model_id,
        })
    }

    /// Fingerprint of the loaded ONNX file (see [`file_fingerprint`]).
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn backend_info(&self) -> String {
        match &self.backend {
            #[cfg(feature = "openvino")]
            Backend::OpenVino(session) => format!("OpenVINO ({})", session.device()),
            Backend::Onnx(_) => "tract-onnx (CPU)".to_string(),
        }
    }

    /// Extract an embedding from an aligned / cropped face.
    pub fn extract(&mut self, face_image: &opencv::core::Mat) -> Result<FaceEmbedding> {
        match &mut self.backend {
            #[cfg(feature = "openvino")]
            Backend::OpenVino(session) => extract_openvino_embedding(session, face_image),
            Backend::Onnx(model) => extract_onnx_embedding(model, face_image),
        }
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

fn extract_onnx_embedding(
    model: &OnnxModel,
    face_image: &opencv::core::Mat,
) -> Result<FaceEmbedding> {
    let input = preprocess_for_mobilefacenet(face_image)?;
    let outputs = model
        .run(tvec!(input.into_tensor().into()))
        .context("ONNX run failed")?;
    let output = outputs.first().context("ONNX model returned no outputs")?;
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
pub fn align_face(image: &Mat, landmarks: &[Point2f], output_size: i32) -> Result<Mat> {
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
            Point2f::new((38.2946 * scale) as f32, (51.6963 * scale) as f32), // YuNet[0]
            Point2f::new((73.5318 * scale) as f32, (51.5014 * scale) as f32), // YuNet[1]
            Point2f::new((56.0252 * scale) as f32, (71.7366 * scale) as f32), // YuNet[2]
            Point2f::new((41.5493 * scale) as f32, (92.3655 * scale) as f32), // YuNet[3]
            Point2f::new((70.7299 * scale) as f32, (92.2041 * scale) as f32), // YuNet[4]
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
    let size = Size::new(output_size, output_size);
    let (flags, border) = (opencv::imgproc::INTER_LINEAR, opencv::core::BORDER_CONSTANT);
    // OpenCV 5 added an AlgorithmHint argument.
    #[cfg(opencv5)]
    opencv::imgproc::warp_affine(
        image,
        &mut aligned,
        &affine_mat,
        size,
        flags,
        border,
        Scalar::all(0.0),
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    #[cfg(not(opencv5))]
    opencv::imgproc::warp_affine(
        image,
        &mut aligned,
        &affine_mat,
        size,
        flags,
        border,
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

/// InsightFace canonical landmark positions in a 112×112 crop
/// (right eye, left eye, nose, right mouth corner, left mouth corner).
pub const CANONICAL_LANDMARKS_112: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical() -> Vec<Point2f> {
        CANONICAL_LANDMARKS_112
            .iter()
            .map(|&(x, y)| Point2f::new(x, y))
            .collect()
    }

    #[test]
    fn affine_recovers_known_transform() {
        // x' = 0.8x - 0.1y + 5, y' = 0.2x + 1.1y - 3
        let src = canonical();
        let dst: Vec<Point2f> = src
            .iter()
            .map(|p| Point2f::new(0.8 * p.x - 0.1 * p.y + 5.0, 0.2 * p.x + 1.1 * p.y - 3.0))
            .collect();
        let a = compute_affine_5pt(&src, &dst).unwrap();
        let expected = [0.8, -0.1, 5.0, 0.2, 1.1, -3.0];
        for (got, want) in a.iter().zip(expected) {
            assert!((got - want).abs() < 1e-3, "{a:?}");
        }
    }

    #[test]
    fn canonical_points_give_identity() {
        let src = canonical();
        let a = compute_affine_5pt(&src, &src).unwrap();
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (got, want) in a.iter().zip(expected) {
            assert!((got - want).abs() < 1e-4, "{a:?}");
        }
    }

    #[test]
    fn embedding_is_normalized() {
        let e = FaceEmbedding::new(vec![3.0, 4.0]);
        assert!((e.vector[0] - 0.6).abs() < 1e-6);
        assert!((e.cosine_similarity(&e) - 1.0).abs() < 1e-6);
        assert_eq!(
            e.euclidean_distance(&FaceEmbedding::new(vec![1.0])),
            f32::INFINITY
        );
    }
}
