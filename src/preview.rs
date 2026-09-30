//! Tiny grayscale previews for JSON clients (`faceauth capture --preview`).
//!
//! Frames are sent as `data:` URLs of binary PGM images: every image toolkit
//! (Qt included) decodes PGM, and it needs no image codec in OpenCV.

use anyhow::Result;
use opencv::core::{Mat, Size};
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, MatTraitConstManual};

/// Width of preview images; height keeps the aspect ratio.
pub const PREVIEW_WIDTH: i32 = 160;

/// Downscale an 8-bit gray frame to [`PREVIEW_WIDTH`] and encode it.
pub fn gray_preview(gray: &Mat) -> Result<String> {
    let (w, h) = (gray.cols().max(1), gray.rows().max(1));
    let height = (PREVIEW_WIDTH * h / w).max(1);
    let mut small = Mat::default();
    imgproc::resize(
        gray,
        &mut small,
        Size::new(PREVIEW_WIDTH, height),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    let bytes = small.data_bytes()?;
    Ok(pgm_data_url(PREVIEW_WIDTH as usize, height as usize, bytes))
}

/// Encode 8-bit gray pixels (row-major, `width × height`) as a PGM data URL.
pub fn pgm_data_url(width: usize, height: usize, pixels: &[u8]) -> String {
    let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
    pgm.extend_from_slice(&pixels[..pixels.len().min(width * height)]);
    format!("data:image/x-portable-graymap;base64,{}", base64(&pgm))
}

/// Standard base64 with padding.
pub fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn pgm_header() {
        let url = pgm_data_url(2, 1, &[0, 255]);
        assert_eq!(
            url,
            format!(
                "data:image/x-portable-graymap;base64,{}",
                base64(b"P5\n2 1\n255\n\x00\xff")
            )
        );
    }
}
