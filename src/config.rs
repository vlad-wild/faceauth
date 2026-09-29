//! TOML configuration.
//!
//! Every section and field has a default, so old or partial config files keep
//! deserializing when new options are added. Unknown keys (e.g. removed fields
//! such as `certainty`) are ignored by serde.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// System-wide config used by `faceauth-auth` and by every command run as root.
pub const SYSTEM_CONFIG_PATH: &str = "/etc/faceauth/config.toml";
/// Where packaged ONNX models live.
pub const SYSTEM_MODELS_DIR: &str = "/usr/share/faceauth/models";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub video: VideoConfig,
    pub detection: DetectionConfig,
    pub recognition: RecognitionConfig,
    pub liveness: LivenessConfig,
    pub openvino: OpenVinoConfig,
    pub auth: AuthConfig,
    pub enroll: EnrollConfig,
    pub debug: DebugConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VideoConfig {
    pub device_path: String,
    /// Authentication timeout in seconds.
    pub timeout: u32,
    pub dark_threshold: f64,
    pub max_height: f64,
    pub rotate: i32,
    pub exposure: i32,
    /// Use IR / low-light capture: skip brightness gating and relax Haar (enroll and auth on the same IR device).
    pub ir_mode: bool,
    /// Pause after a failed / black frame so a broken camera does not spin the CPU.
    pub frame_interval_ms: u64,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video0".to_string(),
            timeout: 4,
            dark_threshold: 85.0,
            max_height: 320.0,
            rotate: 0,
            exposure: -1,
            ir_mode: false,
            frame_interval_ms: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionConfig {
    pub model_path: String,
    /// YuNet model. When set and loadable, YuNet is used regardless of `use_cnn`.
    pub yunet_path: String,
    pub use_cnn: bool,
    pub confidence_threshold: f64,
    pub nms_threshold: f64,
    pub face_padding: f64,
    pub min_face_size_ratio: f64,
    pub max_face_size_ratio: f64,
    pub use_openvino: bool,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            model_path: format!("{SYSTEM_MODELS_DIR}/ultra_light_640.onnx"),
            yunet_path: format!("{SYSTEM_MODELS_DIR}/face_detection_yunet_2023mar.onnx"),
            use_cnn: false,
            confidence_threshold: 0.7,
            nms_threshold: 0.5,
            face_padding: 0.15,
            min_face_size_ratio: 0.05,
            max_face_size_ratio: 0.75,
            use_openvino: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecognitionConfig {
    pub model_path: String,
    pub distance_threshold: f64,
    pub use_openvino: bool,
    /// Consecutive matching frames required for a successful authentication.
    pub required_matches: u32,
    /// Score = mean of the `top_k` smallest distances within a sample set.
    pub top_k: usize,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            model_path: format!("{SYSTEM_MODELS_DIR}/MobileFaceNet.onnx"),
            distance_threshold: 0.6,
            use_openvino: true,
            required_matches: 3,
            top_k: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LivenessConfig {
    /// In `ir_mode`, reject faces whose IR region is too dark / too flat
    /// (phone and laptop screens emit almost no IR). Does not stop printed photos.
    pub ir_check: bool,
    pub min_face_brightness: f64,
    pub min_face_stddev: f64,
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            ir_check: true,
            min_face_brightness: 25.0,
            min_face_stddev: 8.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenVinoConfig {
    /// `AUTO` (default: starts on CPU while NPU/GPU compile), `NPU`, `GPU` or `CPU`.
    pub device: String,
    /// Compiled-model cache. Falls back to the user cache dir when not writable.
    pub cache_dir: String,
}

impl Default for OpenVinoConfig {
    fn default() -> Self {
        Self {
            device: "AUTO".to_string(),
            cache_dir: "/var/cache/faceauth/openvino".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Skip face auth for remote sessions (e.g. `sudo` over SSH), so whoever sits
    /// in front of the laptop cannot approve a remote command.
    pub skip_remote: bool,
    /// Skip face auth when the laptop lid is closed.
    pub skip_lid_closed: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            skip_remote: true,
            skip_lid_closed: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnrollConfig {
    /// Minimum variance of the Laplacian over the face region (blur filter).
    pub min_sharpness: f64,
    pub max_abs_yaw: f64,
    pub max_abs_pitch: f64,
}

impl Default for EnrollConfig {
    fn default() -> Self {
        Self {
            min_sharpness: 60.0,
            max_abs_yaw: 25.0,
            max_abs_pitch: 20.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Log a summary (frames, verdicts, best score, device) after each authentication.
    pub end_report: bool,
}

impl Config {
    /// Parse a config file as-is (no path resolution).
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Parse a config file and resolve model paths (relative to the file, then
    /// falling back to [`SYSTEM_MODELS_DIR`]).
    pub fn load_resolved(path: &Path) -> anyhow::Result<Self> {
        let mut cfg = Self::load(path)?;
        cfg.resolve_model_paths(path.parent());
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Find the config to use and where it came from.
    ///
    /// As root only [`SYSTEM_CONFIG_PATH`] is trusted: a `./faceauth.toml` in the
    /// caller's working directory must not steer privileged commands. Otherwise:
    /// `./faceauth.toml` → `~/.config/faceauth/config.toml` → `/etc/faceauth/config.toml`.
    pub fn discover() -> (Self, Option<PathBuf>) {
        let candidates = if crate::privilege::is_root() {
            vec![PathBuf::from(SYSTEM_CONFIG_PATH)]
        } else {
            Self::user_candidates()
        };
        for path in candidates {
            if !path.exists() {
                continue;
            }
            match Self::load_resolved(&path) {
                Ok(cfg) => return (cfg, Some(path)),
                Err(e) => log::warn!("Ignoring config {}: {e}", path.display()),
            }
        }
        let mut cfg = Self::default();
        cfg.resolve_model_paths(None);
        (cfg, None)
    }

    fn user_candidates() -> Vec<PathBuf> {
        let mut candidates = vec![PathBuf::from("faceauth.toml")];
        if let Some(cfg_dir) = dirs::config_dir() {
            candidates.push(cfg_dir.join("faceauth").join("config.toml"));
        }
        candidates.push(PathBuf::from(SYSTEM_CONFIG_PATH));
        candidates
    }

    fn resolve_model_paths(&mut self, base: Option<&Path>) {
        resolve_model_path(&mut self.recognition.model_path, base);
        resolve_model_path(&mut self.detection.model_path, base);
        if !self.detection.yunet_path.is_empty() {
            resolve_model_path(&mut self.detection.yunet_path, base);
        }
    }
}

/// Make `path` absolute relative to `base`; if the result does not exist but a
/// file with the same name ships in [`SYSTEM_MODELS_DIR`], use that instead
/// (older installs kept models in `/etc/faceauth/models`).
fn resolve_model_path(path: &mut String, base: Option<&Path>) {
    let p = Path::new(path.as_str());
    let resolved = match base {
        Some(base) if !p.is_absolute() => base.join(p),
        _ => p.to_path_buf(),
    };
    if !resolved.exists()
        && let Some(name) = resolved.file_name()
    {
        let packaged = Path::new(SYSTEM_MODELS_DIR).join(name);
        if packaged.exists() {
            log::warn!(
                "Model {} not found, using {}",
                resolved.display(),
                packaged.display()
            );
            *path = packaged.to_string_lossy().into_owned();
            return;
        }
    }
    *path = resolved.to_string_lossy().into_owned();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_config_still_parses() {
        let legacy = r#"
[video]
device_path = "/dev/video2"
timeout = 8
dark_threshold = 99.0
certainty = 3.5
max_height = 480.0
rotate = 0
exposure = -1
ir_mode = true

[detection]
model_path = "/etc/faceauth/models/ultra_light_640.onnx"
use_cnn = true
confidence_threshold = 0.7

[recognition]
model_path = "/etc/faceauth/models/MobileFaceNet.onnx"
embedding_size = 128
distance_threshold = 0.6

[debug]
end_report = false
save_failed = false
save_successful = false
"#;
        let cfg: Config = toml::from_str(legacy).unwrap();
        assert_eq!(cfg.video.device_path, "/dev/video2");
        assert!(cfg.video.ir_mode);
        assert_eq!(cfg.video.frame_interval_ms, 30);
        assert_eq!(cfg.recognition.required_matches, 3);
        assert_eq!(cfg.detection.nms_threshold, 0.5);
        assert!(cfg.auth.skip_remote);
        assert_eq!(cfg.openvino.device, "AUTO");
    }

    #[test]
    fn minimal_config_parses() {
        let cfg: Config = toml::from_str("[video]\ndevice_path = \"/dev/video3\"\n").unwrap();
        assert_eq!(cfg.video.device_path, "/dev/video3");
        assert_eq!(cfg.video.timeout, 4);
        assert_eq!(cfg.recognition.distance_threshold, 0.6);
    }

    #[test]
    fn packaged_config_parses() {
        let cfg: Config = toml::from_str(include_str!("../packaging/config.toml")).unwrap();
        assert!(cfg.recognition.model_path.starts_with(SYSTEM_MODELS_DIR));
        assert_eq!(cfg.openvino.device, "AUTO");
        assert_eq!(cfg.recognition.required_matches, 3);
    }

    #[test]
    fn empty_config_parses() {
        let cfg: Config = toml::from_str("").unwrap();
        assert_eq!(cfg.video.device_path, "/dev/video0");
    }

    #[test]
    fn default_roundtrips() {
        let s = toml::to_string_pretty(&Config::default()).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(back.recognition.top_k, 3);
        assert_eq!(back.liveness.min_face_stddev, 8.0);
    }

    #[test]
    fn relative_model_path_is_resolved_against_base() {
        let mut p = "models/does-not-exist.onnx".to_string();
        resolve_model_path(&mut p, Some(Path::new("/opt/fa")));
        assert_eq!(
            Path::new(&p),
            Path::new("/opt/fa").join("models/does-not-exist.onnx")
        );
    }
}
