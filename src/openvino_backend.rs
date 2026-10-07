use anyhow::{Context, Result};
use log::{info, warn};
use ndarray::Array4;
use openvino::{
    Core, DeviceType, ElementType, PartialShape, PropertyKey, RwPropertyKey, Shape, Tensor,
};
use std::path::PathBuf;

use crate::config::OpenVinoConfig;

pub struct OpenVinoSession {
    // Kept alive for as long as the infer request that was created from it.
    _compiled: openvino::CompiledModel,
    request: openvino::InferRequest,
    input_name: String,
    output_count: usize,
    device: String,
    pub input_shape: Vec<i64>,
}

impl OpenVinoSession {
    pub fn from_onnx(model_path: &str, cfg: &OpenVinoConfig) -> Result<Self> {
        Self::with_static_input(model_path, cfg, None)
    }

    /// Like [`Self::from_onnx`], but first reshapes the single input to a fixed
    /// `[1, 3, height, width]`. Needed for models exported with dynamic spatial
    /// dimensions (SCRFD): a static graph compiles to a stable pipeline instead
    /// of re-specializing on every new frame size.
    pub fn from_onnx_static_input(
        model_path: &str,
        cfg: &OpenVinoConfig,
        height: i64,
        width: i64,
    ) -> Result<Self> {
        Self::with_static_input(model_path, cfg, Some((height, width)))
    }

    fn with_static_input(
        model_path: &str,
        cfg: &OpenVinoConfig,
        static_hw: Option<(i64, i64)>,
    ) -> Result<Self> {
        let mut core = Core::new().context("Failed to initialize OpenVINO core")?;
        let onnx_data = std::fs::read(model_path)
            .with_context(|| format!("Failed to read ONNX model {}", model_path))?;
        let mut model = core
            .read_model_from_buffer(&onnx_data, None)
            .context("Failed to read ONNX model into OpenVINO")?;

        if let Some((height, width)) = static_hw {
            let shape = PartialShape::new_static(4, &[1, 3, height, width])
                .context("Failed to build the static input shape")?;
            model
                .reshape_single_input(&shape)
                .context("Failed to reshape the model input")?;
        }

        let available: Vec<String> = core
            .available_devices()
            .context("Failed to query OpenVINO devices")?
            .iter()
            .map(|d| d.to_string())
            .collect();
        let (device_str, priorities) = choose_device(&cfg.device, &available);
        let device = DeviceType::from(device_str.as_str());

        if let Some(dir) = writable_cache_dir(&cfg.cache_dir) {
            let dir = dir.to_string_lossy().into_owned();
            // The cache is a per-plugin property; AUTO forwards it to the device it compiles for.
            for name in available
                .iter()
                .map(String::as_str)
                .chain([device_str.as_str()])
            {
                let target = DeviceType::from(name);
                if let Err(e) = core.set_property(&target, &RwPropertyKey::CacheDir, &dir) {
                    warn!("OpenVINO: cannot set CACHE_DIR for {name}: {e}");
                }
            }
        }
        if let Some(p) = &priorities
            && let Err(e) = core.set_property(&device, &RwPropertyKey::DevicePriorities, p)
        {
            warn!("OpenVINO: cannot set AUTO device priorities: {e}");
        }
        if let Err(e) = core.set_property(&device, &RwPropertyKey::HintPerformanceMode, "LATENCY") {
            warn!("OpenVINO: cannot set LATENCY hint on {device_str}: {e}");
        }

        info!("OpenVINO: compiling {} for {}", model_path, device_str);
        let mut compiled = core
            .compile_model(&model, device)
            .with_context(|| format!("Failed to compile model for {}", device_str))?;

        let device_label =
            match compiled.get_property(&PropertyKey::Other("EXECUTION_DEVICES".into())) {
                Ok(exec) if !exec.trim().is_empty() && device_str == "AUTO" => {
                    format!("AUTO:{}", exec.trim())
                }
                _ => device_str.clone(),
            };
        info!("OpenVINO: model ready on {}", device_label);

        let input_node = compiled
            .get_input_by_index(0)
            .context("Failed to get model input")?;
        let input_name = input_node.get_name().context("Failed to get input name")?;
        // Dynamic dimensions (e.g. an exported batch axis) have no static shape:
        // report an empty one instead of failing the whole session, callers that
        // need it must validate.
        let input_shape = match input_node.get_shape() {
            Ok(shape) => shape.get_dimensions().to_vec(),
            Err(e) => {
                warn!("OpenVINO input shape is dynamic ({e}); not reporting a static shape");
                Vec::new()
            }
        };

        let output_count = compiled
            .get_output_size()
            .context("Failed to get output count")?;
        let request = compiled
            .create_infer_request()
            .context("Failed to create OpenVINO infer request")?;

        Ok(Self {
            _compiled: compiled,
            request,
            input_name,
            output_count,
            device: device_label,
            input_shape,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run inference and return (shape_dims, data) for each output.
    pub fn run(&mut self, input: Array4<f32>) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
        let shape_dims: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        let ov_shape = Shape::new(&shape_dims).context("Failed to create OpenVINO shape")?;
        let mut tensor =
            Tensor::new(ElementType::F32, &ov_shape).context("Failed to create input tensor")?;
        {
            let slice = input.as_slice().context("Input array is not contiguous")?;
            let data = tensor
                .get_data_mut::<f32>()
                .context("Failed to get tensor data")?;
            data.copy_from_slice(slice);
        }

        self.request
            .set_tensor(&self.input_name, &tensor)
            .context("Failed to set input tensor")?;
        self.request.infer().context("OpenVINO inference failed")?;

        let mut outputs = Vec::with_capacity(self.output_count);
        for idx in 0..self.output_count {
            let tensor = self
                .request
                .get_output_tensor_by_index(idx)
                .with_context(|| format!("Failed to get output tensor {}", idx))?;
            let shape = tensor
                .get_shape()
                .with_context(|| format!("Failed to get output {} shape", idx))?;
            let dims = shape.get_dimensions().to_vec();
            let data = tensor
                .get_data::<f32>()
                .with_context(|| format!("Failed to read output {} data", idx))?
                .to_vec();
            outputs.push((dims, data));
        }
        Ok(outputs)
    }
}

/// Map the configured device to one OpenVINO can use, plus AUTO priorities.
///
/// `AUTO` with an accelerator present lets OpenVINO run the first inferences on
/// CPU while the NPU/GPU compile finishes — the main cost for a PAM helper that
/// starts fresh on every `sudo`. Without accelerators plain `CPU` avoids the
/// AUTO overhead. An explicitly requested but missing device falls back to CPU.
pub fn choose_device(wanted: &str, available: &[String]) -> (String, Option<String>) {
    let has = |prefix: &str| available.iter().any(|d| d.starts_with(prefix));
    match wanted.trim().to_ascii_uppercase().as_str() {
        "AUTO" | "" => {
            let prio: Vec<&str> = ["NPU", "GPU"].into_iter().filter(|d| has(d)).collect();
            if prio.is_empty() {
                ("CPU".to_string(), None)
            } else {
                let mut p = prio.join(",");
                p.push_str(",CPU");
                ("AUTO".to_string(), Some(p))
            }
        }
        dev @ ("NPU" | "GPU" | "CPU") if has(dev) => (dev.to_string(), None),
        other => {
            warn!("OpenVINO device {other:?} not available ({available:?}); using CPU");
            ("CPU".to_string(), None)
        }
    }
}

/// `preferred` if it can be created and written, else the user cache dir, else none.
fn writable_cache_dir(preferred: &str) -> Option<PathBuf> {
    let candidates = [
        (!preferred.trim().is_empty()).then(|| PathBuf::from(preferred)),
        dirs::cache_dir().map(|d| d.join("faceauth").join("openvino")),
    ];
    candidates.into_iter().flatten().find(|dir| {
        std::fs::create_dir_all(dir).is_ok() && {
            let probe = dir.join(".write-test");
            let ok = std::fs::write(&probe, b"").is_ok();
            let _ = std::fs::remove_file(&probe);
            ok
        }
    })
}

#[cfg(test)]
mod tests {
    use super::choose_device;

    fn devs(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn auto_prefers_accelerators() {
        assert_eq!(
            choose_device("AUTO", &devs(&["CPU", "GPU.0", "NPU"])),
            ("AUTO".into(), Some("NPU,GPU,CPU".into()))
        );
        assert_eq!(choose_device("auto", &devs(&["CPU"])), ("CPU".into(), None));
        assert_eq!(
            choose_device("NPU", &devs(&["CPU", "NPU"])),
            ("NPU".into(), None)
        );
        assert_eq!(choose_device("NPU", &devs(&["CPU"])), ("CPU".into(), None));
        assert_eq!(choose_device("TPU", &devs(&["CPU"])), ("CPU".into(), None));
    }
}
