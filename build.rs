//! Detects the installed OpenCV major version so the code can follow API moves
//! between 4.x and 5.x (e.g. `CascadeClassifier` moved to `xobjdetect`,
//! `warpAffine` gained an `AlgorithmHint`). Emits `cfg(opencv5)`.

use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(opencv5)");
    println!("cargo::rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo::rerun-if-env-changed=FACEAUTH_OPENCV_MAJOR");

    let major = std::env::var("FACEAUTH_OPENCV_MAJOR").ok().or_else(|| {
        let exists = |pc: &str| {
            Command::new("pkg-config")
                .args(["--exists", pc])
                .status()
                .is_ok_and(|s| s.success())
        };
        if exists("opencv5") {
            Some("5".to_string())
        } else if exists("opencv4") {
            Some("4".to_string())
        } else {
            None
        }
    });
    if major.as_deref() == Some("5") {
        println!("cargo::rustc-cfg=opencv5");
    }
}
