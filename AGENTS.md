# Faceauth — Agent Guide

## Overview
`faceauth` is a face-authentication system for Linux, written in Rust. It captures a face from a webcam, extracts an embedding with an ONNX neural net (MobileFaceNet), and verifies it against stored per-user JSON models.

## Architecture
- **CLI** (`src/main.rs`) – `faceauth` binary: enroll, list/remove/rename, test, calibrate, migrate, doctor, and the `import`/`verify` helpers used by the GUI through `pkexec`.
- **PAM helper** (`src/bin/auth.rs`) – `faceauth-auth`, run by `pam_exec`. As root it runs the pipeline itself; unprivileged (screen lockers) it asks `faceauthd` and only if `PAM_USER` is the calling user. Exit codes: 0 match, 10 skipped, 11 no match, 12 setup error, 13 too dark.
- **Daemon** (`src/bin/faceauthd.rs`) – `faceauthd`, root, socket-activated (`packaging/faceauthd.{socket,service}`). Verifies the caller's own face (uid from `SO_PEERCRED`), one attempt at a time, rate-limited, keeps `Models` loaded and opens the camera per attempt, exits when idle. Protocol and client in `daemon.rs`.
- **GUI** (`src/bin/ui/main.rs` + `src/bin/ui/worker.rs`) – `faceauth-ui` (Iced 0.14). A single worker thread owns the camera and pipeline; all store access goes through `pkexec faceauth …`.

### Key modules
| Module | Purpose |
|--------|---------|
| `pipeline.rs` | `Pipeline` (camera + detector + recognizer), `analyze_frame` → `FrameAnalysis`/`FaceVerdict`, `select_face`, `face_crop`, IR liveness |
| `authenticate.rs` | Shared auth loop (PAM helper, daemon, `faceauth test`): consecutive matches, top-k scoring, report; the frame callback returns `ControlFlow` (`Break` → `Cancelled`) |
| `gate.rs` | `pre_auth_checks`: disabled flag, remote session, lid, enrolled model — run before the camera opens |
| `daemon.rs` | `faceauthd` protocol (`Request`/`Event`/`Outcome`, JSON lines), `RateLimiter`, `peer_uid`, client `verify()` |
| `enroll.rs` | Step-wise `EnrollSession` (quality gates, pose buckets, duplicates) and CLI `enroll_user` |
| `matching.rs` | `l2_distance`, `set_score` (top-k), `MatchTracker`, `PoseCollector`, `score_stats`, `file_fingerprint` (SHA-256 of the ONNX) — no OpenCV |
| `database.rs` | `FaceModel`/`Database`, `apply_enrollment` (only merge implementation), trusted root-only storage, GUI payload types — no OpenCV |
| `privilege.rs` | `require_root`, `authorize_for_user` / `authorize_global` (a pkexec caller may only touch their own model), passwd lookup |
| `session.rs` | Remote-session (`PAM_RHOST`/`PAM_TTY`) and lid checks |
| `camera.rs` | V4L2 capture, rotation, downscaling, `roi_stats`, `sharpness` |
| `detection.rs` | YuNet (per-frame `set_input_size`), Ultra-Light, Haar; `clip_rect`, NMS, `estimate_yaw`/`estimate_pitch` |
| `recognition.rs` | MobileFaceNet (OpenVINO → tract), 5-point alignment; **fails closed** (no fallback embedding) |
| `openvino_backend.rs` | OpenVINO session: `AUTO`/NPU/GPU/CPU (`choose_device`), `CACHE_DIR`, `LATENCY` hint, one reused infer request |
| `config.rs` | TOML schema with struct-level defaults, `Config::discover`, model path resolution |
| `doctor.rs` / `diagnostics.rs` | `faceauth doctor` checks / serializable result types |
| `i18n.rs` | ru/en message table (`t`, `tf`) |
| `devices.rs`, `verdict.rs` | OpenCV-free device listing and verdict enum (usable from the GUI) |

## Build
```bash
cargo build --release
```

## Tests
```bash
cargo test
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
```
Unit tests cover config compatibility, merge logic, storage permission policy, payload validation, top-k scoring, the streak tracker, pose collection, remote-session detection, alignment math, NMS/clipping, pose estimates, OpenVINO device choice and i18n. Camera behaviour is still tested manually (`sudo faceauth test -u <user>`, `sudo faceauth doctor`). CI (`.github/workflows/ci.yml`) runs on Arch Linux (OpenCV 5.0) because Ubuntu's OpenCV is older than 4.11.

## Code style
- Standard `cargo fmt` / `cargo clippy` (edition 2024, let-chains are fine).
- Prefer `anyhow::Result` in binary code.
- OpenCV image format is **BGR**; conversion to RGB is done before feeding ONNX models.
- Use `opencv::prelude::*` traits (`MatTraitConst`, `CascadeClassifierTrait`, etc.) when calling OpenCV methods.
- Keep logic that does not need OpenCV in the OpenCV-free modules above so it stays unit-testable.

## Key implementation notes for agents

### 1. Face detector selection
`create_detector(&DetectionConfig, &OpenVinoConfig, ir_mode)`:
- YuNet whenever `yunet_path` is set and loads (independent of `use_cnn`); a configured `…2026may.onnx` falls back to the sibling `…2023mar.onnx`.
- Then Ultra-Light if `use_cnn = true`, then the Haar cascade (`DEFAULT_HAAR_CASCADE`, `minNeighbors` 2 in IR mode, else 3).
- YuNet calls `set_input_size(frame size)` so frames are never stretched.

### 2. Face crop padding and alignment
- **YuNet**: `align_face()` maps the five landmarks to `CANONICAL_LANDMARKS_112` with a least-squares affine transform (2-point similarity with fewer landmarks).
- **Haar / Ultra-Light**: `crop_face()` pads by `face_padding` (default **0.15**) and clips to the frame.

### 3. Frame gates (`analyze_frame`)
Darkness (skipped in `ir_mode` except fully black frames) → confidence → face area within `min_face_size_ratio`..`max_face_size_ratio` (largest valid face wins) → IR liveness (`[liveness]`, IR only) → in enrollment also yaw/pitch limits and Laplacian sharpness. The resulting `FaceVerdict` feeds logs, CLI hints and the GUI overlay.

### 4. Model input shapes
- **Ultra-Light**: input shape read from the model; boxes are normalized, clamped, then scaled to the frame.
- **MobileFaceNet**: `[1, 3, 112, 112]`, pixels normalized to `[-1, 1]` via `(pixel − 127.5) / 128.0`.

### 5. Config
- Discovery (CLI/GUI as a normal user): `./faceauth.toml` → `~/.config/faceauth/config.toml` → `/etc/faceauth/config.toml`. **As root only `/etc/faceauth/config.toml`** (or an explicit `--config`, refused under pkexec). `faceauth-auth` takes `--config` (default `/etc/faceauth/config.toml`).
- Every struct has `#[serde(default)]` + `impl Default`: add new fields there; old configs keep parsing, removed keys are ignored.
- Relative model paths resolve against the config file; missing paths fall back to `/usr/share/faceauth/models/<file name>`.
- Sections: `[video]`, `[detection]`, `[recognition]` (`distance_threshold`, `required_matches`, `top_k`), `[liveness]`, `[openvino]` (`device`, `cache_dir`), `[auth]` (`skip_remote`, `skip_lid_closed`), `[enroll]`, `[debug]` (`end_report`). Reference: `packaging/config.toml`.

### 6. Database format and storage
- Store: `/var/lib/faceauth/models/<user>.json` (dir 0700 root, files 0600 root). `Database::load_trusted` refuses symlinks, non-root owners and group/other permissions; `save_secure` writes atomically.
- File content: `{"users": {<user>: FaceModel}}`. `FaceModel`: `label`, `embeddings` (primary), `extensions` (named variants), `created_at`, optional `updated_at`, `model_id` (recognizer fingerprint), `embedding_dim`.
- Scoring: per set, mean of the `top_k` smallest L2 distances; best set wins; a match needs `score < distance_threshold` on `required_matches` consecutive frames.
- `check_compatible` rejects a different `model_id` or dimension; legacy/migrated models (`model_id: None`) only warn.
- All merges go through `Database::apply_enrollment`.

### 7. IR / low-light mode
`ir_mode=true` disables the darkness gate and lowers Haar `minNeighbors` from 3 → 2. Enrollment and authentication must use the **same IR device** (`faceauth add` defaults to `video.device_path`).

### 8. Typical manual workflow for improving accuracy
1. `sudo faceauth doctor` — camera, IR emitter, models, NPU.
2. Re-enroll with pose guidance: `sudo faceauth add -u <user> -s 12`.
3. `sudo faceauth calibrate -u <user>` and set `distance_threshold` from the suggestion.
4. Increase `max_height` (e.g. `720.0`) so frames are not downscaled as much.
5. Add appearance variants: `sudo faceauth add -u <user> --variant glasses`.
6. With an Intel NPU keep `[openvino] device = "AUTO"` and a writable `cache_dir`.

## Files agents often touch
- `src/pipeline.rs`, `src/authenticate.rs`, `src/enroll.rs` – capture/auth/enroll flow
- `src/detection.rs` – detector logic / NMS / crop / OpenVINO wiring
- `src/recognition.rs` – ONNX/OpenVINO embedding extractor
- `src/database.rs`, `src/privilege.rs` – storage and privilege rules
- `src/config.rs` – config schema & defaults (keep `packaging/config.toml` in sync)
- `src/openvino_backend.rs` – OpenVINO session wrapper
- `src/i18n.rs` – add ru + en text for every new user-facing message
- `faceauth.toml` – local config for quick experiments (use `--config ./faceauth.toml` with sudo)

## Common pitfalls
- Forgetting to import `MatTraitConst` / `CascadeClassifierTrait` when calling OpenCV methods on `Mat` or `CascadeClassifier`.
- Changing the CNN detector output parsing without checking the actual ONNX output shapes (Ultra-Light may output `boxes`/`scores` in different order depending on the ONNX export).
- Adding a config struct without `#[serde(default)]` + `impl Default` → breaks existing user configs.
- Reading models from anywhere but the root store, or adding a substitute embedding path: `faceauth-auth` must stay fail-closed.
- Letting a `pkexec` caller act on another user's model: use `authorize_for_user` / `authorize_global`.
- Forgetting that OpenVINO feature is gated behind `default = ["openvino"]` — builds with `--no-default-features` will omit the OpenVINO backend entirely.
- Mismatched `use_openvino` / `[openvino] device` between enrollment and authentication do not matter (same model file, same `model_id`), but scores may differ slightly between CPU and NPU/GPU.
- OpenCV 4.11+ or 5.x is required (`AlgorithmHint` in `cvt_color`); Ubuntu LTS packages are too old. Arch ships OpenCV 5.0: `build.rs` emits `cfg(opencv5)` (pkg-config `opencv5`, or `FACEAUTH_OPENCV_MAJOR`). Under it `CascadeClassifier` comes from `opencv::xobjdetect`, `warp_affine` takes an extra `AlgorithmHint`, and Haar cascades live in `/usr/share/opencv5/haarcascades`. Guard any other version-specific call the same way.
- `dark_threshold` is `100 − mean brightness %` (not Howdy's histogram metric); keep the default high (85).

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
