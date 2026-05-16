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
| `detection.rs` | Face detection: **Ultra-Light ONNX** (`use_cnn=true`) or **OpenCV Haar cascade** fallback |
| `recognition.rs` | Face embedding via MobileFaceNet ONNX (input 1×3×112×112) |
| `enroll.rs` | `enroll_user_with_progress`, `capture_single_embedding`, size filtering & crop padding |
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
`detection.rs` exposes a unified `Detector` enum (`Haar` / `Cnn`).
- `create_detector(...)` tries CNN first and **falls back to Haar** if loading fails.
- If `use_cnn=false` in config, Haar is used directly.
- Haar path is hard-coded: `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`.

### 2. Face crop padding
`crop_face()` in `detection.rs` adds configurable padding around the detected bounding box (`face_padding` ratio, default **0.15**). This gives the recognition net more context (ears, chin) and usually improves accuracy.

### 3. Size filtering
Faces are filtered by min/max area ratios relative to the whole frame:
- `min_face_size_ratio` default **0.05**
- `max_face_size_ratio` default **0.75**
This prevents tiny / huge false positives from being passed to the recognizer.

### 4. Model input shapes
- **Ultra-Light detector**: automatically inspects ONNX input fact at load time (usually `[1, 3, 480, 640]`). Coordinates are normalized to `[0,1]` inside the net and scaled back to original image dimensions after inference.
- **MobileFaceNet recognizer**: hard-coded `[1, 3, 112, 112]` in `recognition.rs`. Pre-processing normalizes pixels to `[-1, 1]` via `(pixel − 127.5) / 128.0`.

### 5. Config file locations (in order of precedence)
1. `./faceauth.toml`
2. `~/.config/faceauth/config.toml`
3. `/etc/faceauth/config.toml`

When editing config, **add `#[serde(default = ...)]`** on new fields so old user configs do not break deserialization.

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

## Files agents often touch
- `src/detection.rs` – detector logic / NMS / crop
- `src/recognition.rs` – ONNX embedding extractor
- `src/enroll.rs` – enrollment pipeline
- `src/config.rs` – config schema & defaults
- `faceauth.toml` – local config for quick experiments

## Common pitfalls
- Forgetting to import `MatTraitConst` / `CascadeClassifierTrait` when calling OpenCV methods on `Mat` or `CascadeClassifier`.
- Changing the CNN detector output parsing without checking the actual ONNX output shapes (Ultra-Light may output `boxes`/`scores` in different order depending on the ONNX export).
- Not adding `#[serde(default)]` on new config fields → breaks existing user configs.
