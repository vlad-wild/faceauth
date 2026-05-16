use anyhow::{Context, Result};
use log::info;
use ndarray::Array4;
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};

pub struct OpenVinoSession {
    compiled: openvino::CompiledModel,
    input_name: String,
    output_count: usize,
    device: String,
    pub input_shape: Vec<i64>,
}

impl OpenVinoSession {
    pub fn from_onnx(model_path: &str) -> Result<Self> {
        let mut core = Core::new().context("Failed to initialize OpenVINO core")?;
        let onnx_data = std::fs::read(model_path)
            .with_context(|| format!("Failed to read ONNX model {}", model_path))?;
        let model = core
            .read_model_from_buffer(&onnx_data, None)
            .context("Failed to read ONNX model into OpenVINO")?;

        let devices = core
            .available_devices()
            .context("Failed to query OpenVINO devices")?;
        let device_str: String;
        let device = if devices.iter().any(|d| matches!(d, DeviceType::NPU)) {
            device_str = "NPU".to_string();
            info!("OpenVINO: NPU detected, compiling model for NPU");
            DeviceType::NPU
        } else if devices.iter().any(|d| matches!(d, DeviceType::GPU)) {
            device_str = "GPU".to_string();
            info!("OpenVINO: GPU detected, compiling model for GPU");
            DeviceType::GPU
        } else {
            device_str = "CPU".to_string();
            info!("OpenVINO: using CPU");
            DeviceType::CPU
        };

        let compiled = core
            .compile_model(&model, device)
            .with_context(|| format!("Failed to compile model for {}", device_str))?;

        let input_node = compiled
            .get_input_by_index(0)
            .context("Failed to get model input")?;
        let input_name = input_node
            .get_name()
            .context("Failed to get input name")?;
        let input_shape = input_node
            .get_shape()
            .context("Failed to get input shape")?
            .get_dimensions()
            .to_vec();

        let output_count = compiled
            .get_output_size()
            .context("Failed to get output count")?;

        Ok(Self {
            compiled,
            input_name,
            output_count,
            device: device_str,
            input_shape,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run inference and return (shape_dims, data) for each output.
    pub fn run(&mut self, input: Array4<f32>) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
        let mut infer_request = self
            .compiled
            .create_infer_request()
            .context("Failed to create OpenVINO infer request")?;

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

        infer_request
            .set_tensor(&self.input_name, &tensor)
            .context("Failed to set input tensor")?;
        infer_request
            .infer()
            .context("OpenVINO inference failed")?;

        let mut outputs = Vec::with_capacity(self.output_count);
        for idx in 0..self.output_count {
            let tensor = infer_request
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
