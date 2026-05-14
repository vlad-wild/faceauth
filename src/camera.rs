use anyhow::Result;
use opencv::{
    prelude::*,
    videoio,
    core,
    imgproc,
};

/// Camera capture abstraction
pub struct Camera {
    cap: videoio::VideoCapture,
    width: i32,
    height: i32,
    rotate: i32,
    max_height: f64,
}

impl Camera {
    /// Open camera by device path (e.g., "/dev/video0") or index (0, 1, ...)
    pub fn open(device: &str, max_height: f64, rotate: i32) -> Result<Self> {
        let cap = if let Ok(index) = device.parse::<i32>() {
            videoio::VideoCapture::new(index, videoio::CAP_V4L2)?
        } else if let Some(index) = parse_v4l2_device_index(device) {
            videoio::VideoCapture::new(index, videoio::CAP_V4L2)?
        } else {
            videoio::VideoCapture::from_file(device, videoio::CAP_ANY)?
        };

        if !cap.is_opened()? {
            anyhow::bail!("Failed to open camera {}", device);
        }

        let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

        Ok(Self {
            cap,
            width,
            height,
            rotate,
            max_height,
        })
    }

    /// Read a frame from the camera, applying rotation and scaling if needed.
    /// Returns (color_frame, grayscale_frame).
    pub fn read_frame(&mut self) -> Result<(Mat, Mat)> {
        let mut frame = Mat::default();
        self.cap.read(&mut frame)?;
        if frame.empty() {
            anyhow::bail!("Empty frame captured");
        }

        // Apply rotation
        let frame = self.apply_rotation(frame)?;

        // Scale down if height exceeds max_height
        let frame = self.apply_scaling(frame)?;

        // Support both RGB/BGR sensors and monochrome IR sensors.
        let channels = frame.channels();
        if channels == 1 {
            let gray = frame.try_clone()?;
            let mut bgr = Mat::default();
            imgproc::cvt_color(
                &gray,
                &mut bgr,
                imgproc::COLOR_GRAY2BGR,
                0,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
            return Ok((bgr, gray));
        }

        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        Ok((frame, gray))
    }

    fn apply_rotation(&self, frame: Mat) -> Result<Mat> {
        let mut rotated = Mat::default();
        match self.rotate {
            1 => {
                // Rotate 90 degrees counter-clockwise
                opencv::core::rotate(&frame, &mut rotated, opencv::core::ROTATE_90_COUNTERCLOCKWISE)?;
                Ok(rotated)
            }
            2 => {
                // Rotate 90 degrees clockwise
                opencv::core::rotate(&frame, &mut rotated, opencv::core::ROTATE_90_CLOCKWISE)?;
                Ok(rotated)
            }
            _ => Ok(frame),
        }
    }

    fn apply_scaling(&self, frame: Mat) -> Result<Mat> {
        let height = frame.rows() as f64;
        if height <= self.max_height {
            return Ok(frame);
        }
        let scaling_factor = self.max_height / height;
        let new_width = (frame.cols() as f64 * scaling_factor) as i32;
        let new_height = self.max_height as i32;
        let mut resized = Mat::default();
        imgproc::resize(
            &frame,
            &mut resized,
            core::Size::new(new_width, new_height),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;
        Ok(resized)
    }

    /// Set manual exposure if supported
    pub fn set_exposure(&mut self, exposure: f64) -> Result<()> {
        self.cap.set(videoio::CAP_PROP_AUTO_EXPOSURE, 1.0)?; // manual
        self.cap.set(videoio::CAP_PROP_EXPOSURE, exposure)?;
        Ok(())
    }

    /// Get camera properties
    pub fn width(&self) -> i32 { self.width }
    pub fn height(&self) -> i32 { self.height }
}

fn parse_v4l2_device_index(device: &str) -> Option<i32> {
    let prefix = "/dev/video";
    if !device.starts_with(prefix) {
        return None;
    }
    let suffix = &device[prefix.len()..];
    suffix.parse::<i32>().ok()
}

/// Calculate darkness of a grayscale frame (percentage of pixels near black)
pub fn darkness(frame: &Mat) -> Result<f64> {
    let mean = core::mean(frame, &Mat::default())?;
    let brightness = mean.0[0].clamp(0.0, 255.0);
    Ok(100.0 - (brightness / 255.0 * 100.0))
}