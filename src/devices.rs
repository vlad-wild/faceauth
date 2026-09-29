//! V4L2 device discovery via sysfs (no OpenCV needed).

/// A V4L2 capture node as listed in sysfs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoDevice {
    pub path: String,
    pub name: String,
    /// Heuristic from the device name ("IR", "Infrared"); confirmed by reading a frame.
    pub likely_ir: bool,
}

impl std::fmt::Display for VideoDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} — {}{}",
            self.path,
            self.name,
            if self.likely_ir { " (IR)" } else { "" }
        )
    }
}

/// List `/dev/videoN` nodes with their driver-reported names.
pub fn list_devices() -> Vec<VideoDevice> {
    let Ok(entries) = std::fs::read_dir("/sys/class/video4linux") else {
        return Vec::new();
    };
    let mut devices: Vec<VideoDevice> = entries
        .flatten()
        .filter_map(|e| {
            let node = e.file_name().to_string_lossy().into_owned();
            if !node.starts_with("video") {
                return None;
            }
            let name = std::fs::read_to_string(e.path().join("name"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            Some(VideoDevice {
                path: format!("/dev/{node}"),
                likely_ir: name_suggests_ir(&name),
                name,
            })
        })
        .collect();
    devices.sort_by_key(|d| parse_v4l2_device_index(&d.path).unwrap_or(i32::MAX));
    devices
}

fn name_suggests_ir(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("infrared")
        || n.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w == "ir")
}

pub fn parse_v4l2_device_index(device: &str) -> Option<i32> {
    let prefix = "/dev/video";
    if !device.starts_with(prefix) {
        return None;
    }
    let suffix = &device[prefix.len()..];
    suffix.parse::<i32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_name_heuristic() {
        assert!(name_suggests_ir("USB2.0 FHD UVC WebCam: IR Camera"));
        assert!(name_suggests_ir("Integrated_Webcam_Infrared"));
        assert!(!name_suggests_ir("Integrated Camera: Integrated C"));
        assert!(!name_suggests_ir("Mirror cam"));
    }

    #[test]
    fn device_index() {
        assert_eq!(parse_v4l2_device_index("/dev/video2"), Some(2));
        assert_eq!(parse_v4l2_device_index("/dev/foo"), None);
    }
}
