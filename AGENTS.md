# Faceauth — Agent Guide

## Overview
`faceauth` is a face-authentication system for Linux, written in Rust. It captures a face from a webcam, extracts an embedding with an ONNX neural net (MobileFaceNet), and verifies it against stored per-user JSON models.

## Architecture
- **CLI** (`src/main.rs`) – `faceauth` binary for enrolling, testing, and listing models.
- **PAM helper** (`src/bin/auth.rs`) – `faceauth-auth` binary invoked by `pam_exec` for login / sudo authentication.
- **GUI** (`src/bin/faceauth-ui.rs`) – `faceauth-ui` (Iced) for enrolling via a desktop window.

### Key modules
| Module | Purpose |
|--------|---------|
| `camera.rs` | V4L2 capture, rotation, optional downscaling |
| `openvino_backend.rs` | OpenVINO ONNX inference wrapper (auto NPU → GPU → CPU) |
| `detection.rs` | Face detection: **YuNet ONNX** (preferred, provides landmarks), **Ultra-Light ONNX/OpenVINO**, **OpenCV Haar cascade** fallback |
| `recognition.rs` | Face embedding via MobileFaceNet ONNX (input 1×3×112×112); **face alignment via 5-point least-squares affine** |
| `enroll.rs` | `enroll_user_with_progress`, `capture_single_embedding`, `process_face_sample`, size filtering & crop padding |
| `database.rs` | JSON storage of `FaceModel` per user in `~/.local/share/faceauth/models/<user>.json` |
| `config.rs` | TOML config (`faceauth.toml`) – video, detection, recognition, debug |

## Build
```bash
cargo build --release
```

## Tests
```bash
cargo test
```
(There are currently no unit tests in this repo; testing is manual via `faceauth test -u <user>`.)

## Code style
- Standard `cargo fmt` / `cargo clippy`.
- Prefer `anyhow::Result` in binary code.
- OpenCV image format is **BGR**; conversion to RGB is done before feeding ONNX models.
- Use `opencv::prelude::*` traits (`MatTraitConst`, `CascadeClassifierTrait`, etc.) when calling OpenCV methods.

## Key implementation notes for agents

### 1. Face detector selection
`detection.rs` exposes a unified `Detector` enum (`Haar` / `Cnn` / `YuNet`).
- `create_detector(...)` tries YuNet first, then Ultra-Light (with OpenVINO if `use_openvino=true`), then falls back to Haar.
- If `use_cnn=false` in config, Haar is used directly.
- OpenVINO backend auto-selects device priority: **NPU → GPU → CPU**.
- Haar path is hard-coded: `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`.

### 2. Face crop padding and alignment
Two strategies depending on detector:
- **YuNet (preferred)**: `align_face()` in `recognition.rs` computes a **5-point least-squares affine transform** mapping all five detected landmarks (eyes, nose, mouth) to InsightFace canonical positions. This corrects for in-plane rotation (roll) AND partially for yaw/pitch, improving recognition at non-frontal angles. Falls back to 2-point similarity when <5 landmarks.
- **Haar / Ultra-Light**: `crop_face()` in `detection.rs` adds configurable padding around the detected bounding box (`face_padding` ratio, default **0.15**). This gives the recognition net more context (ears, chin).

### 3. Size filtering
Faces are filtered by min/max area ratios relative to the whole frame:
- `min_face_size_ratio` default **0.05**
- `max_face_size_ratio` default **0.75**
This prevents tiny / huge false positives from being passed to the recognizer.

### 4. Model input shapes
- **Ultra-Light detector**: automatically inspects ONNX (or OpenVINO) input shape at load time (usually `[1, 3, 480, 640]`). Coordinates are normalized to `[0,1]` inside the net and scaled back to original image dimensions after inference.
- **MobileFaceNet recognizer**: hard-coded `[1, 3, 112, 112]` in `recognition.rs`. Pre-processing normalizes pixels to `[-1, 1]` via `(pixel − 127.5) / 128.0`.

### 5. Config file locations (in order of precedence)
1. `./faceauth.toml`
2. `~/.config/faceauth/config.toml`
3. `/etc/faceauth/config.toml`

When editing config, **add `#[serde(default = ...)]`** on new fields so old user configs do not break deserialization.
- `detection.use_openvino` (bool, default **true**) — enables OpenVINO backend for the Ultra-Light detector when available.
- `recognition.use_openvino` (bool, default **true**) — enables OpenVINO backend for MobileFaceNet when available.

### 6. Database format
Each user has a single JSON file (`<user>.json`) containing a `HashMap<String, FaceModel>`. A `FaceModel` stores:
- `label` – human-readable tag
- `embeddings` – primary embedding vectors
- `extensions` – optional named variant sets (e.g. `glasses`, `hat`)

Authentication succeeds if the probe embedding matches **any** stored set (primary or extension) with Euclidean distance `< distance_threshold`.

### 7. IR / low-light mode
`ir_mode=true` disables the darkness gate and lowers Haar `minNeighbors` from 3 → 2. Enrollment and authentication must use the **same IR device**.

### 8. Typical manual workflow for improving accuracy
1. Re-enroll with more samples: `faceauth add -u <user> -s 10`
2. Enable CNN detector in config: `use_cnn = true`
3. Increase `max_height` (e.g. `720.0`) so the camera frame is not downscaled as much.
4. Adjust `distance_threshold` (lower = stricter, higher = more lenient).
5. Add appearance variants: `faceauth add -u <user> --variant glasses -s 5`
6. If you have an Intel NPU, ensure `use_openvino = true` in both `[detection]` and `[recognition]` sections.

## Files agents often touch
- `src/detection.rs` – detector logic / NMS / crop / OpenVINO wiring
- `src/recognition.rs` – ONNX/OpenVINO embedding extractor
- `src/enroll.rs` – enrollment pipeline
- `src/config.rs` – config schema & defaults
- `src/openvino_backend.rs` – OpenVINO session wrapper
- `faceauth.toml` – local config for quick experiments

## Common pitfalls
- Forgetting to import `MatTraitConst` / `CascadeClassifierTrait` when calling OpenCV methods on `Mat` or `CascadeClassifier`.
- Changing the CNN detector output parsing without checking the actual ONNX output shapes (Ultra-Light may output `boxes`/`scores` in different order depending on the ONNX export).
- Not adding `#[serde(default)]` on new config fields → breaks existing user configs.
- Forgetting that OpenVINO feature is gated behind `default = ["openvino"]` — builds with `--no-default-features` will omit the OpenVINO backend entirely.
- Mismatched `use_openvino` settings between enrollment and authentication do not matter (the model files are the same), but performance and accuracy may differ slightly between CPU and NPU/GPU backends.

---

## Hardware Human Presence Detection (HPD) — Research Notes

This section documents an investigation into adding **walk-away lock / adaptive dimming** (hardware HPD) to complement faceauth. This is **not yet implemented**; the hardware path is blocked.

### What was attempted
- Intel ISH (`hid-ishtp`) sensor hub at `/dev/hidraw5` (VID:PID `8087:0AC2`) exposes a `Fused_HuP` (Human Presence) HID sensor (`HID-SENSOR-200001`).
- The ISH firmware (`ish_lnlm.bin.zst`) contains `HUMAN_PRESENCE` and `RADAR_HUMAN_DETECTION` strings, confirming firmware support.
- Exhaustive attempts to activate it under Linux failed:
  - sysfs `HID-SENSOR-200001.1.auto` has no sensor attributes or `enable_sensor` writable interface.
  - `hid_sensor_custom` cannot bind to the device (`ENODEV`).
  - Only report ID `01` (56 bytes) is emitted; no presence data observed.
  - Feature Report 5 (containing `LUID:0011000`) is readable/writable but does not switch the device into presence-reporting mode.
- The `Jappan-SV/ish-presence-linux` project was evaluated — its HID report structure is incompatible with this ASUS device (expects report ID `02 02 06`, ASUS emits report ID `01`).

### Root cause
- ASUS MyASUS implements HPD via **Intel Wi-Fi Sensing** (802.11bf / CSI-based), **not** the ISH HID sensor.
- Intel Wi-Fi Sensing is a **proprietary firmware feature** with no public Linux API or `iwlwifi` driver support.
- The Intel Context Sensing Technology (CST) user-space service (`IntelCstService`) is a Windows-only component.
- Windows driver reverse-engineering (`HumanPresenceProvider.dll`, `IshHidMini.sys`, etc.) confirmed no straightforward HID activation sequence exists — the HPD path goes through Wi-Fi PHY/firmware, not raw HID reports.

### Alternative: Software HPD via IR camera
- The existing IR camera (`/dev/video2`, `GREY 640x360`) already used by `faceauth` can support walk-away lock.
- A future `faceauth-guard` daemon could:
  1. Poll the IR stream every 200–500 ms.
  2. Run lightweight face detection on each frame.
  3. If **no face is detected for N seconds** → run `loginctl lock-session` (or Hyprland-equivalent lock).
  4. If a **face re-appears after absence** → trigger the existing `faceauth-auth` unlock pipeline.
- This avoids all proprietary dependencies and works natively in Linux + Hyprland.

### External references evaluated
- `ruvnet/RuView` (Wi-Fi DensePose / CSI sensing on ESP32) — interesting, but requires extra ESP32-S3 hardware and does not activate the built-in ASUS HPD stack.
- `Jappan-SV/ish-presence-linux` — works on some Lenovo models, incompatible with ASUS report structure.

### Decision
- **Hardware HPD is blocked** until Intel publishes a Linux Wi-Fi Sensing API or ASUS open-sources the ISH HID activation sequence.
- **Recommended next step** if implementing walk-away lock: build `faceauth-guard` software daemon using the IR camera pipeline.
