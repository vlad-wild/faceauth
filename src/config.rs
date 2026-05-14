use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub video: VideoConfig,
    pub detection: DetectionConfig,
    pub recognition: RecognitionConfig,
    pub debug: DebugConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    pub device_path: String,
    pub timeout: u32,
    pub dark_threshold: f64,
    pub certainty: f64,
    pub max_height: f64,
    pub rotate: i32,
    pub exposure: i32,
    /// Use IR / low-light capture: skip brightness gating and relax Haar (enroll and auth on the same IR device).
    #[serde(default)]
    pub ir_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub model_path: String,
    pub use_cnn: bool,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    pub model_path: String,
    pub embedding_size: usize,
    pub distance_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    pub end_report: bool,
    pub save_failed: bool,
    pub save_successful: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            video: VideoConfig {
                device_path: "/dev/video0".to_string(),
                timeout: 4,
                dark_threshold: 50.0,
                certainty: 3.5,
                max_height: 320.0,
                rotate: 0,
                exposure: -1,
                ir_mode: false,
            },
            detection: DetectionConfig {
                model_path: "models/ultra_light_640.onnx".to_string(),
                use_cnn: false,
                confidence_threshold: 0.7,
            },
            recognition: RecognitionConfig {
                model_path: "models/mobilefacenet.onnx".to_string(),
                embedding_size: 128,
                distance_threshold: 0.6,
            },
            debug: DebugConfig {
                end_report: false,
                save_failed: false,
                save_successful: false,
            },
        }
    }
}

impl Config {
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}