pub mod camera;
pub mod config;
pub mod database;
pub mod detection;
pub mod enroll;
pub mod recognition;
pub mod logger;

#[cfg(feature = "openvino")]
pub mod openvino_backend;
