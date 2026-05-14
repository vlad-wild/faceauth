//! GUI for enrolling a face model (same logic as `faceauth add`).

use std::sync::mpsc;

use eframe::egui;
use opencv::core::AlgorithmHint;
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, MatTraitConstManual};

use faceauth::camera::Camera;
use faceauth::config::Config;
use faceauth::enroll::{self, EnrollMerge, EnrollParams};

#[derive(Clone, Copy, PartialEq, Eq)]
enum UiEnrollMode {
    ReplaceAll,
    AppendPrimary,
    VariantReplace,
    VariantAppend,
}

fn mat_bgr_to_egui(mat: &opencv::core::Mat) -> anyhow::Result<egui::ColorImage> {
    let mut rgb = opencv::core::Mat::default();
    imgproc::cvt_color(
        mat,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let w = rgb.cols() as usize;
    let h = rgb.rows() as usize;
    let bytes = rgb.data_bytes()?;
    let n = w * h * 3;
    if bytes.len() < n {
        anyhow::bail!("buffer too short");
    }
    Ok(egui::ColorImage::from_rgb([w, h], &bytes[..n]))
}

enum EnrollMsg {
    Sample(usize, usize),
    Done(Result<(), String>),
}

struct FaceauthUiApp {
    base_config: Config,
    config_source: Option<std::path::PathBuf>,
    username: String,
    device: String,
    label: String,
    samples: u32,
    ir: bool,
    preview_on: bool,
    cam: Option<Camera>,
    preview_tex: Option<egui::TextureHandle>,
    enrolling: bool,
    enroll_rx: Option<mpsc::Receiver<EnrollMsg>>,
    enroll_cur: usize,
    enroll_tot: usize,
    status: String,
    enroll_mode: UiEnrollMode,
    variant_name: String,
}

impl FaceauthUiApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let _ = env_logger::try_init();
        let (base_config, config_source) = enroll::load_enrollment_config();
        let username = std::env::var("USER").unwrap_or_default();
        let device = base_config.video.device_path.clone();
        let ir = base_config.video.ir_mode;
        Self {
            base_config,
            config_source,
            username,
            device,
            label: String::new(),
            samples: 8,
            ir,
            preview_on: false,
            cam: None,
            preview_tex: None,
            enrolling: false,
            enroll_rx: None,
            enroll_cur: 0,
            enroll_tot: 0,
            status: String::new(),
            enroll_mode: UiEnrollMode::ReplaceAll,
            variant_name: String::new(),
        }
    }

    fn open_camera(&mut self) {
        self.cam = None;
        self.preview_tex = None;
        match Camera::open(
            &self.device,
            self.base_config.video.max_height,
            self.base_config.video.rotate,
        ) {
            Ok(mut c) => {
                if self.base_config.video.exposure >= 0 {
                    let _ = c.set_exposure(self.base_config.video.exposure as f64);
                }
                self.cam = Some(c);
                self.status = "Camera is opened.".to_string();
            }
            Err(e) => {
                self.status = format!("Couldn't open camera: {e}");
            }
        }
    }

    fn close_camera(&mut self) {
        self.cam = None;
        self.preview_tex = None;
    }
}

impl eframe::App for FaceauthUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut enroll_finished: Option<Result<(), String>> = None;
        if let Some(rx) = &self.enroll_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    EnrollMsg::Sample(cur, tot) => {
                        self.enroll_cur = cur;
                        self.enroll_tot = tot;
                        self.status = format!("Shot {cur}/{tot}…");
                    }
                    EnrollMsg::Done(r) => {
                        enroll_finished = Some(r);
                        break;
                    }
                }
            }
        }
        if let Some(r) = enroll_finished {
            self.enroll_rx = None;
            self.enrolling = false;
            match r {
                Ok(()) => {
                    self.status = "Done: model saved.".to_string();
                }
                Err(e) => {
                    self.status = format!("Error: {e}");
                }
            }
        }

        if self.preview_on {
            if let Some(ref mut cam) = self.cam {
                match cam.read_frame() {
                    Ok((color, _)) => {
                        if let Ok(img) = mat_bgr_to_egui(&color) {
                            self.preview_tex = Some(ctx.load_texture(
                                "faceauth_preview",
                                img,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                    Err(e) => {
                        self.status = format!("Frame error: {e}");
                    }
                }
                ctx.request_repaint();
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Faceauth — face recording");
            if let Some(ref p) = self.config_source {
                ui.label(format!("Config: {}", p.display()));
            } else {
                ui.label("Config: default values (put faceauth.toml or ~/.config/faceauth/config.toml)");
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("User:");
                ui.add(egui::TextEdit::singleline(&mut self.username).desired_width(200.0));
            });
            ui.horizontal(|ui| {
                ui.label("Camera:");
                ui.add(egui::TextEdit::singleline(&mut self.device).desired_width(240.0));
            });
            ui.horizontal(|ui| {
                ui.label("Mark (optional):");
                ui.add(egui::TextEdit::singleline(&mut self.label).desired_width(200.0));
            });
            ui.horizontal(|ui| {
                ui.label("Shots:");
                ui.add(egui::DragValue::new(&mut self.samples).range(1..=30));
            });
            ui.checkbox(&mut self.ir, "IR / low light (ir_mode)");

            ui.separator();
            ui.label("How to save shots:");
            ui.radio_value(&mut self.enroll_mode, UiEnrollMode::ReplaceAll, "Заменить всю модель (как раньше)");
            ui.radio_value(
                &mut self.enroll_mode,
                UiEnrollMode::AppendPrimary,
                "Complete the main model (more shots without glasses, etc.)",
            );
            ui.radio_value(
                &mut self.enroll_mode,
                UiEnrollMode::VariantReplace,
                "Appearance option: new pictures will replace the set with this name",
            );
            ui.radio_value(
                &mut self.enroll_mode,
                UiEnrollMode::VariantAppend,
                "Option: Add snapshots to an existing option",
            );
            if matches!(
                self.enroll_mode,
                UiEnrollMode::VariantReplace | UiEnrollMode::VariantAppend
            ) {
                ui.horizontal(|ui| {
                    ui.label("The name of the variant (e.g. glasses):");
                    ui.add(egui::TextEdit::singleline(&mut self.variant_name).desired_width(180.0));
                });
            }

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.enrolling, egui::Button::new("Preview"))
                    .clicked()
                {
                    self.preview_on = !self.preview_on;
                    if self.preview_on {
                        self.open_camera();
                    } else {
                        self.close_camera();
                    }
                }

                let can_start = !self.enrolling && !self.username.trim().is_empty();
                if ui
                    .add_enabled(can_start, egui::Button::new("Record a face"))
                    .clicked()
                {
                    let enroll_plan = match self.enroll_mode {
                        UiEnrollMode::ReplaceAll => Some((EnrollMerge::ReplaceAll, None)),
                        UiEnrollMode::AppendPrimary => Some((EnrollMerge::AppendPrimary, None)),
                        UiEnrollMode::VariantReplace | UiEnrollMode::VariantAppend => {
                            let v = self.variant_name.trim();
                            if v.is_empty() {
                                self.status = "Specify the name of the option (for example, glasses) or select another mode.".to_string();
                                None
                            } else {
                                let merge = if self.enroll_mode == UiEnrollMode::VariantAppend {
                                    EnrollMerge::AppendVariant
                                } else {
                                    EnrollMerge::ReplaceVariant
                                };
                                Some((merge, Some(v.to_string())))
                            }
                        }
                    };

                    if let Some((merge, variant_opt)) = enroll_plan {
                        self.preview_on = false;
                        self.close_camera();
                        let (tx, rx) = mpsc::channel();
                        let mut cfg = self.base_config.clone();
                        if self.ir {
                            cfg.video.ir_mode = true;
                        }
                        let params = EnrollParams {
                            username: self.username.trim().to_string(),
                            label: if self.label.trim().is_empty() {
                                None
                            } else {
                                Some(self.label.trim().to_string())
                            },
                            samples: self.samples as usize,
                            device: self.device.clone(),
                            ir: self.ir,
                            merge,
                            variant: variant_opt,
                        };
                        self.enrolling = true;
                        self.enroll_cur = 0;
                        self.enroll_tot = params.samples;
                        self.status = "Shooting... look at the camera.".to_string();
                        self.enroll_rx = Some(rx);
                        std::thread::spawn(move || {
                            let txp = tx.clone();
                            let r = enroll::enroll_user_with_progress(cfg, params, move |c, t| {
                                let _ = txp.send(EnrollMsg::Sample(c, t));
                            });
                            let _ = tx.send(EnrollMsg::Done(r.map_err(|e| e.to_string())));
                        });
                    }
                }
            });

            if let Some(tex) = &self.preview_tex {
                ui.image((tex.id(), tex.size_vec2()));
            } else if self.preview_on {
                ui.label("There is no frame (check the camera device).");
            }

            ui.separator();
            ui.label(format!("Статус: {}", self.status));
            if self.enrolling && self.enroll_tot > 0 {
                let p = self.enroll_cur as f32 / self.enroll_tot as f32;
                ui.add(egui::ProgressBar::new(p).show_percentage());
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([540.0, 680.0])
            .with_title("Faceauth — face recording"),
        ..Default::default()
    };
    eframe::run_native(
        "Faceauth",
        options,
        Box::new(|cc| Ok(Box::new(FaceauthUiApp::new(cc)) as Box<dyn eframe::App>)),
    )
}
