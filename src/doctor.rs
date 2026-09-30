//! `faceauth doctor`: environment diagnostics with actionable hints.

use std::path::{Path, PathBuf};

pub use crate::diagnostics::{Check, PAM_LINE, Status, print};

use crate::camera::{self, Camera};
use crate::config::Config;
use crate::daemon::SOCKET_PATH;
use crate::database::DISABLED_FLAG;
use crate::matching::file_fingerprint;

/// Run every check. `user` selects whose enrolled model to look for.
pub fn run(cfg: &Config, config_source: Option<&Path>, user: Option<&str>) -> Vec<Check> {
    let mut checks = vec![check_config(config_source)];
    checks.extend(check_camera(cfg));
    checks.extend(check_models(cfg));
    checks.extend(check_openvino(cfg));
    checks.extend(check_storage(user));
    checks.extend(check_pam());
    checks.push(check_daemon());
    if Path::new(DISABLED_FLAG).exists() {
        checks.push(
            Check::new("disabled", Status::Warn, format!("{DISABLED_FLAG} exists"))
                .hint("sudo faceauth enable"),
        );
    }
    checks
}

/// Screen lockers run unprivileged and reach the model store only through faceauthd.
fn check_daemon() -> Check {
    use std::os::unix::fs::FileTypeExt;

    match std::fs::metadata(SOCKET_PATH) {
        Ok(m) if m.file_type().is_socket() => Check::new("faceauthd", Status::Ok, SOCKET_PATH),
        _ => Check::new(
            "faceauthd",
            Status::Warn,
            format!("{SOCKET_PATH} not found (face unlock in screen lockers is unavailable)"),
        )
        .hint("sudo systemctl enable --now faceauthd.socket"),
    }
}

fn check_config(source: Option<&Path>) -> Check {
    match source {
        Some(p) => Check::new("config", Status::Ok, p.display().to_string()),
        None => Check::new(
            "config",
            Status::Warn,
            "no config file found, using defaults",
        )
        .hint("sudo cp /usr/share/doc/faceauth/config.toml /etc/faceauth/config.toml"),
    }
}

fn check_camera(cfg: &Config) -> Vec<Check> {
    let mut out = Vec::new();
    let devices = camera::list_devices();
    let listing = devices
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("; ");
    out.push(if devices.is_empty() {
        Check::new("video devices", Status::Fail, "no /dev/video* devices")
    } else {
        Check::new("video devices", Status::Ok, listing)
    });

    let dev = &cfg.video.device_path;
    let mut cam = match Camera::open_configured(dev, &cfg.video) {
        Ok(c) => c,
        Err(e) => {
            out.push(
                Check::new("camera", Status::Fail, format!("{dev}: {e}"))
                    .hint("check video.device_path and that you are in the `video` group"),
            );
            return out;
        }
    };
    let mut dark_sum = 0.0;
    let mut mono = true;
    let mut n = 0;
    for _ in 0..10 {
        if let Ok((color, gray)) = cam.read_frame()
            && let (Ok(d), Ok(m)) = (
                camera::darkness(&gray),
                opencv::core::mean(&color, &opencv::core::Mat::default()),
            )
        {
            dark_sum += d;
            mono &= (m.0[0] - m.0[1]).abs() < 0.5 && (m.0[1] - m.0[2]).abs() < 0.5;
            n += 1;
        }
    }
    if n == 0 {
        out.push(Check::new(
            "camera",
            Status::Fail,
            format!("{dev}: opened but no frames"),
        ));
        return out;
    }
    let dark = dark_sum / n as f64;
    let kind = if mono { "monochrome (IR?)" } else { "color" };
    let detail = format!(
        "{dev}: {}x{}, {kind}, darkness {dark:.0}%",
        cam.width(),
        cam.height()
    );
    let mut c = Check::new("camera", Status::Ok, detail);
    if dark > 97.0 {
        c.status = Status::Warn;
        c = c.hint(if mono {
            "frames are black: the IR emitter is probably off — try linux-enable-ir-emitter"
        } else {
            "frames are black: check the lens cover / lighting"
        });
    } else if mono != cfg.video.ir_mode {
        c.status = Status::Warn;
        c = c.hint(format!("set video.ir_mode = {mono} for this camera"));
    }
    out.push(c);
    out
}

fn check_models(cfg: &Config) -> Vec<Check> {
    let mut out = Vec::new();
    let mut model = |name: &str, path: &str, required: bool| {
        if path.is_empty() {
            return;
        }
        let c = match file_fingerprint(Path::new(path)) {
            Ok(id) => Check::new(name, Status::Ok, format!("{path} [{id}]")),
            Err(_) if !required => Check::new(name, Status::Warn, format!("{path}: missing")),
            Err(e) => Check::new(name, Status::Fail, format!("{path}: {e}"))
                .hint("install the model or fix recognition.model_path"),
        };
        out.push(c);
    };
    model("recognizer", &cfg.recognition.model_path, true);
    model("yunet", &cfg.detection.yunet_path, false);
    if cfg.detection.use_cnn {
        model("ultra-light", &cfg.detection.model_path, false);
    }
    out
}

#[cfg(feature = "openvino")]
fn check_openvino(cfg: &Config) -> Vec<Check> {
    use crate::config::OpenVinoConfig;
    use crate::openvino_backend::OpenVinoSession;
    use std::time::Instant;

    let mut out = Vec::new();
    let accel = Path::new("/dev/accel/accel0");
    if accel.exists() {
        let rw = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(accel)
            .is_ok();
        out.push(if rw {
            Check::new("npu device", Status::Ok, "/dev/accel/accel0 accessible")
        } else {
            Check::new(
                "npu device",
                Status::Warn,
                "/dev/accel/accel0 not accessible by this user",
            )
            .hint("add the user to the `render` group (root is unaffected)")
        });
    } else {
        out.push(
            Check::new("npu device", Status::Warn, "no /dev/accel/accel0").hint(
                "install intel-npu-driver (≥ 1.30 for Lunar Lake) if this machine has an NPU",
            ),
        );
    }

    let model = &cfg.recognition.model_path;
    if !cfg.recognition.use_openvino || !Path::new(model).exists() {
        return out;
    }
    let time = |ov: &OpenVinoConfig| -> Result<(String, f64, f64), String> {
        let t = Instant::now();
        let mut s = OpenVinoSession::from_onnx(model, ov).map_err(|e| format!("{e:#}"))?;
        let load = t.elapsed().as_secs_f64() * 1000.0;
        let input = ndarray::Array4::<f32>::zeros((1, 3, 112, 112));
        let t = Instant::now();
        s.run(input).map_err(|e| format!("{e:#}"))?;
        Ok((
            s.device().to_string(),
            load,
            t.elapsed().as_secs_f64() * 1000.0,
        ))
    };
    match time(&cfg.openvino) {
        Ok((dev, load, infer)) => out.push(Check::new(
            "openvino",
            Status::Ok,
            format!(
                "{dev}: load {load:.0} ms, first inference {infer:.1} ms (cache: {})",
                cfg.openvino.cache_dir
            ),
        )),
        Err(e) => out.push(
            Check::new("openvino", Status::Warn, e).hint("tract-onnx (CPU) will be used instead"),
        ),
    }
    if !cfg.openvino.device.eq_ignore_ascii_case("CPU") {
        let cpu = OpenVinoConfig {
            device: "CPU".into(),
            ..cfg.openvino.clone()
        };
        if let Ok((_, load, infer)) = time(&cpu) {
            out.push(Check::new(
                "openvino cpu",
                Status::Ok,
                format!("CPU for comparison: load {load:.0} ms, first inference {infer:.1} ms"),
            ));
        }
    }
    out
}

#[cfg(not(feature = "openvino"))]
fn check_openvino(_: &Config) -> Vec<Check> {
    vec![Check::new(
        "openvino",
        Status::Warn,
        "built without the openvino feature (CPU via tract)",
    )]
}

fn check_storage(user: Option<&str>) -> Vec<Check> {
    let mut out = Vec::new();
    #[cfg(unix)]
    {
        use crate::database::MODELS_DIR;
        use std::os::unix::fs::MetadataExt;
        match std::fs::symlink_metadata(MODELS_DIR) {
            Ok(m) if m.is_dir() && m.uid() == 0 && m.mode() & 0o077 == 0 => out.push(Check::new(
                "store",
                Status::Ok,
                format!("{MODELS_DIR} (root, 0700)"),
            )),
            Ok(m) => out.push(
                Check::new(
                    "store",
                    Status::Fail,
                    format!("{MODELS_DIR}: uid {} mode {:o}", m.uid(), m.mode() & 0o777),
                )
                .hint(format!(
                    "sudo chown root:root {MODELS_DIR} && sudo chmod 700 {MODELS_DIR}"
                )),
            ),
            Err(_) => out.push(
                Check::new(
                    "store",
                    Status::Warn,
                    format!("{MODELS_DIR} does not exist yet"),
                )
                .hint("it is created on the first `sudo faceauth add`"),
            ),
        }
    }
    let Some(user) = user else { return out };
    if !crate::privilege::is_root() {
        out.push(
            Check::new(
                "face model",
                Status::Warn,
                "cannot read the root-only store",
            )
            .hint("run `sudo faceauth doctor` to check the enrolled model"),
        );
    } else {
        out.push(match crate::database::load_user_model(user) {
            Ok(Some(m)) => Check::new(
                "face model",
                Status::Ok,
                format!(
                    "{user}: {} samples, {} variants, recognizer {}",
                    m.sample_count(),
                    m.extensions.len(),
                    m.model_id.as_deref().unwrap_or("unknown (legacy)")
                ),
            ),
            Ok(None) => Check::new(
                "face model",
                Status::Warn,
                format!("{user}: nothing enrolled"),
            )
            .hint(format!("sudo faceauth add -u {user}")),
            Err(e) => Check::new("face model", Status::Fail, format!("{e:#}")),
        });
        if let Ok((legacy, _)) = crate::database::legacy_model_path(user)
            && legacy.exists()
        {
            out.push(
                Check::new("legacy model", Status::Warn, legacy.display().to_string()).hint(
                    format!("sudo faceauth migrate -u {user}, then delete the old file"),
                ),
            );
        }
    }
    out
}

fn check_pam() -> Vec<Check> {
    let files = [
        "sudo",
        "system-auth",
        "polkit-1",
        "login",
        "sddm",
        "gdm-password",
        "hyprlock",
    ];
    let mut hits: Vec<PathBuf> = Vec::new();
    let mut out = Vec::new();
    for f in files {
        let path = Path::new("/etc/pam.d").join(f);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
            if line.contains("faceauth-auth") {
                hits.push(path.clone());
                if line.contains("$USER") {
                    out.push(
                        Check::new(
                            "pam",
                            Status::Fail,
                            format!("{}: uses $USER", path.display()),
                        )
                        .hint("PAM does not expand $USER; drop `-u $USER` (PAM_USER is used)"),
                    );
                }
            }
        }
    }
    if hits.is_empty() {
        out.push(
            Check::new("pam", Status::Warn, "faceauth-auth is not in /etc/pam.d").hint(PAM_LINE),
        );
    } else {
        hits.dedup();
        let list = hits
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.push(Check::new("pam", Status::Ok, list));
    }
    out
}
