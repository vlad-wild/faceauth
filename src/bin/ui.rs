use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use iced::{
    futures::StreamExt,
    widget::{button, checkbox, column, container, image, progress_bar, radio, row, slider, text, text_input},
    Alignment, ContentFit, Element, Length, Subscription, Task,
};
use opencv::core::{AlgorithmHint, Point, Rect, Scalar};
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, MatTraitConstManual};

use faceauth::camera::Camera;
use faceauth::config::Config;
use faceauth::database::{Database, get_user_model_path};
use faceauth::detection::create_detector;
use faceauth::enroll::{self, EnrollMerge, EnrollParams};
use faceauth::logger;
use faceauth::recognition::FaceRecognizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UiEnrollMode {
    ReplaceAll,
    AppendPrimary,
    VariantReplace,
    VariantAppend,
}

#[derive(Clone, Debug)]
enum Message {
    UsernameChanged(String),
    DeviceChanged(String),
    LabelChanged(String),
    SamplesChanged(u32),
    IrToggled(bool),
    ModeChanged(UiEnrollMode),
    VariantChanged(String),
    TogglePreview,
    PreviewFrame(u32, u32, Vec<u8>),
    PreviewError(String),
    EnrollPressed,
    EnrollSample(usize, usize),
    EnrollDone(Result<(), String>),
    TestPressed,
    TestResult(Result<bool, String>),
}

#[derive(Clone)]
struct PreviewParams {
    device: String,
    max_height: f64,
    rotate: i32,
    exposure: i32,
    yunet_path: String,
    model_path: String,
    confidence_threshold: f32,
    nms_threshold: f32,
    use_cnn: bool,
    use_openvino: bool,
    haar_neighbors: i32,
}

impl Hash for PreviewParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.device.hash(state);
        self.max_height.to_bits().hash(state);
        self.rotate.hash(state);
        self.exposure.hash(state);
        self.yunet_path.hash(state);
        self.model_path.hash(state);
        self.confidence_threshold.to_bits().hash(state);
        self.nms_threshold.to_bits().hash(state);
        self.use_cnn.hash(state);
        self.use_openvino.hash(state);
        self.haar_neighbors.hash(state);
    }
}

#[derive(Clone)]
struct EnrollJob {
    id: u64,
    cfg: Config,
    params: EnrollParams,
}

impl Hash for EnrollJob {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for EnrollJob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for EnrollJob {}

#[derive(Clone)]
struct TestJob {
    id: u64,
    cfg: Config,
    username: String,
}

impl Hash for TestJob {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for TestJob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TestJob {}

struct FaceauthUi {
    base_config: Config,
    config_source: Option<std::path::PathBuf>,
    username: String,
    device: String,
    label: String,
    samples: u32,
    ir: bool,
    preview_on: bool,
    preview_params: Option<PreviewParams>,
    preview_handle: Option<image::Handle>,
    enrolling: bool,
    enroll_job: Option<EnrollJob>,
    enroll_cur: usize,
    enroll_tot: usize,
    testing: bool,
    test_job: Option<TestJob>,
    status: String,
    enroll_mode: UiEnrollMode,
    variant_name: String,
}

impl FaceauthUi {
    fn new() -> (Self, Task<Message>) {
        let _ = logger::try_init_from_env();
        let (base_config, config_source) = enroll::load_enrollment_config();
        let username = std::env::var("USER").unwrap_or_default();
        let device = base_config.video.device_path.clone();
        let ir = base_config.video.ir_mode;
        (
            Self {
                base_config,
                config_source,
                username,
                device,
                label: String::new(),
                samples: 8,
                ir,
                preview_on: false,
                preview_params: None,
                preview_handle: None,
                enrolling: false,
                enroll_job: None,
                enroll_cur: 0,
                enroll_tot: 0,
                testing: false,
                test_job: None,
                status: String::new(),
                enroll_mode: UiEnrollMode::ReplaceAll,
                variant_name: String::new(),
            },
            Task::none(),
        )
    }
}

fn mat_bgr_to_rgba(mat: &opencv::core::Mat) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let mut rgba = opencv::core::Mat::default();
    imgproc::cvt_color(
        mat,
        &mut rgba,
        imgproc::COLOR_BGR2RGBA,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let w = rgba.cols() as u32;
    let h = rgba.rows() as u32;
    let bytes = rgba.data_bytes()?;
    Ok((w, h, bytes.to_vec()))
}

fn preview_worker(params: &PreviewParams) -> iced::futures::stream::BoxStream<'static, Message> {
    let params = params.clone();
    iced::stream::channel(16, move |mut output: iced::futures::channel::mpsc::Sender<Message>| async move {
        std::thread::spawn(move || {
            let mut cam = match Camera::open(&params.device, params.max_height, params.rotate) {
                Ok(c) => c,
                Err(e) => {
                    let _ = output.try_send(Message::PreviewError(format!("Couldn't open camera: {e}")));
                    return;
                }
            };
            if params.exposure >= 0 {
                let _ = cam.set_exposure(params.exposure as f64);
            }
            let mut detector = match create_detector(
                Some(&params.yunet_path).filter(|p| !p.is_empty()).map(|x| x.as_str()),
                &params.model_path,
                params.confidence_threshold,
                params.nms_threshold,
                params.use_cnn,
                enroll::DEFAULT_HAAR_CASCADE,
                params.haar_neighbors,
                params.use_openvino,
            ) {
                Ok(d) => d,
                Err(e) => {
                    let _ = output.try_send(Message::PreviewError(format!("Detector init failed: {e}")));
                    return;
                }
            };
            loop {
                match cam.read_frame() {
                    Ok((mut color, _)) => {
                        if let Ok(faces) = detector.detect(&color) {
                            for face in faces {
                                let bbox = face.bbox;
                                let _ = imgproc::rectangle(
                                    &mut color,
                                    Rect::new(bbox.x, bbox.y, bbox.width, bbox.height),
                                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                                    2,
                                    imgproc::LINE_8,
                                    0,
                                );
                                for (i, pt) in face.landmarks.iter().enumerate() {
                                    let color_scalar = match i {
                                        0 | 1 => Scalar::new(0.0, 0.0, 255.0, 0.0), // eyes - red
                                        2 => Scalar::new(0.0, 255.0, 0.0, 0.0),     // nose - green
                                        3 | 4 => Scalar::new(255.0, 0.0, 0.0, 0.0), // mouth corners - blue
                                        _ => Scalar::new(255.0, 255.0, 255.0, 0.0),
                                    };
                                    let _ = imgproc::circle(
                                        &mut color,
                                        Point::new(pt.x as i32, pt.y as i32),
                                        3,
                                        color_scalar,
                                        -1,
                                        imgproc::LINE_8,
                                        0,
                                    );
                                }
                            }
                        }
                        let (w, h, rgba) = match mat_bgr_to_rgba(&color) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if output.try_send(Message::PreviewFrame(w, h, rgba)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = output.try_send(Message::PreviewError(format!("Frame error: {e}")));
                        break;
                    }
                }
            }
        });
    })
    .boxed()
}

fn enroll_worker(job: &EnrollJob) -> iced::futures::stream::BoxStream<'static, Message> {
    let job = job.clone();
    iced::stream::channel(16, move |mut output: iced::futures::channel::mpsc::Sender<Message>| async move {
        std::thread::spawn(move || {
            let EnrollJob { cfg, params, .. } = job;
            let mut output2 = output.clone();
            let r = enroll::enroll_user_with_progress(cfg, params, move |c, t| {
                let _ = output2.try_send(Message::EnrollSample(c, t));
            });
            let _ = output.try_send(Message::EnrollDone(r.map_err(|e| e.to_string())));
        });
    })
    .boxed()
}

fn test_worker(job: &TestJob) -> iced::futures::stream::BoxStream<'static, Message> {
    let job = job.clone();
    iced::stream::channel(16, move |mut output: iced::futures::channel::mpsc::Sender<Message>| async move {
        std::thread::spawn(move || {
            let TestJob { cfg, username, .. } = job;
            let result = (|| -> anyhow::Result<bool> {
                let model_path = get_user_model_path(&username)?;
                let db = Database::load(&model_path)?;
                if db.get_user(&username).is_none() {
                    anyhow::bail!("No model enrolled for user {}", username);
                }

                let mut cam = Camera::open(&cfg.video.device_path, cfg.video.max_height, cfg.video.rotate)
                    .map_err(|e| anyhow::anyhow!("Failed to open camera: {e}"))?;
                if cfg.video.exposure >= 0 {
                    let _ = cam.set_exposure(cfg.video.exposure as f64);
                }

                let haar_neighbors = if cfg.video.ir_mode { 2 } else { 3 };
                let mut detector = create_detector(
                    Some(&cfg.detection.yunet_path).filter(|p| !p.is_empty()).map(|x| x.as_str()),
                    &cfg.detection.model_path,
                    cfg.detection.confidence_threshold as f32,
                    cfg.detection.nms_threshold as f32,
                    cfg.detection.use_cnn,
                    enroll::DEFAULT_HAAR_CASCADE,
                    haar_neighbors,
                    cfg.detection.use_openvino,
                ).map_err(|e| anyhow::anyhow!("Detector init failed: {e}"))?;

                let mut recognizer = FaceRecognizer::load(&cfg.recognition.model_path, cfg.recognition.use_openvino)
                    .map_err(|e| anyhow::anyhow!("Recognizer load failed: {e}"))?;

                let timeout = Duration::from_secs(cfg.video.timeout as u64).max(Duration::from_secs(3));
                let started = Instant::now();
                let probe = loop {
                    if started.elapsed() >= timeout {
                        anyhow::bail!("Timeout: no face detected during test");
                    }
                    match enroll::capture_single_embedding(&mut cam, &mut detector, &mut recognizer, &cfg) {
                        Ok(Some(emb)) => break emb,
                        Ok(None) => {
                            std::thread::sleep(Duration::from_millis(60));
                            continue;
                        }
                        Err(e) => anyhow::bail!("Capture error: {e}"),
                    }
                };

                let threshold = cfg.recognition.distance_threshold as f32;
                let distance = db
                    .get_user(&username)
                    .map(|m| m.best_match_distance(&probe))
                    .ok_or_else(|| anyhow::anyhow!("User model disappeared"))?;

                Ok(distance < threshold)
            })();

            let msg = match result {
                Ok(true) => Message::TestResult(Ok(true)),
                Ok(false) => Message::TestResult(Ok(false)),
                Err(e) => Message::TestResult(Err(e.to_string())),
            };
            let _ = output.try_send(msg);
        });
    })
    .boxed()
}

fn update(state: &mut FaceauthUi, message: Message) -> Task<Message> {
    match message {
        Message::UsernameChanged(s) => state.username = s,
        Message::DeviceChanged(s) => state.device = s,
        Message::LabelChanged(s) => state.label = s,
        Message::SamplesChanged(v) => state.samples = v.clamp(1, 30),
        Message::IrToggled(v) => state.ir = v,
        Message::ModeChanged(m) => state.enroll_mode = m,
        Message::VariantChanged(s) => state.variant_name = s,
        Message::TogglePreview => {
            state.preview_on = !state.preview_on;
            if state.preview_on {
                state.preview_params = Some(PreviewParams {
                    device: state.device.clone(),
                    max_height: state.base_config.video.max_height,
                    rotate: state.base_config.video.rotate,
                    exposure: state.base_config.video.exposure,
                    yunet_path: state.base_config.detection.yunet_path.clone(),
                    model_path: state.base_config.detection.model_path.clone(),
                    confidence_threshold: state.base_config.detection.confidence_threshold as f32,
                    nms_threshold: state.base_config.detection.nms_threshold as f32,
                    use_cnn: state.base_config.detection.use_cnn,
                    use_openvino: state.base_config.detection.use_openvino,
                    haar_neighbors: if state.ir { 2 } else { 3 },
                });
                state.status = "Opening camera...".to_string();
            } else {
                state.preview_params = None;
                state.preview_handle = None;
                state.status = "Camera closed.".to_string();
            }
        }
        Message::PreviewFrame(w, h, bytes) => {
            state.preview_handle = Some(image::Handle::from_rgba(w, h, bytes));
            state.status = "Camera is opened.".to_string();
        }
        Message::PreviewError(e) => {
            state.preview_on = false;
            state.preview_params = None;
            state.preview_handle = None;
            state.status = e;
        }
        Message::EnrollPressed => {
            if !state.enrolling && !state.username.trim().is_empty() {
                let plan = match state.enroll_mode {
                    UiEnrollMode::ReplaceAll => Some((EnrollMerge::ReplaceAll, None)),
                    UiEnrollMode::AppendPrimary => Some((EnrollMerge::AppendPrimary, None)),
                    UiEnrollMode::VariantReplace | UiEnrollMode::VariantAppend => {
                        let v = state.variant_name.trim();
                        if v.is_empty() {
                            state.status = "Specify the name of the variant (for example, glasses) or select another mode.".to_string();
                            None
                        } else {
                            let merge = if state.enroll_mode == UiEnrollMode::VariantAppend {
                                EnrollMerge::AppendVariant
                            } else {
                                EnrollMerge::ReplaceVariant
                            };
                            Some((merge, Some(v.to_string())))
                        }
                    }
                };

                if let Some((merge, variant_opt)) = plan {
                    state.preview_on = false;
                    state.preview_params = None;
                    state.preview_handle = None;
                    let mut cfg = state.base_config.clone();
                    if state.ir {
                        cfg.video.ir_mode = true;
                    }
                    let params = EnrollParams {
                        username: state.username.trim().to_string(),
                        label: if state.label.trim().is_empty() {
                            None
                        } else {
                            Some(state.label.trim().to_string())
                        },
                        samples: state.samples as usize,
                        device: state.device.clone(),
                        ir: state.ir,
                        merge,
                        variant: variant_opt,
                    };
                    state.enrolling = true;
                    state.enroll_cur = 0;
                    state.enroll_tot = params.samples;
                    state.status = "Shooting... look at the camera.".to_string();
                    state.enroll_job = Some(EnrollJob {
                        id: rand::random::<u64>(),
                        cfg,
                        params,
                    });
                }
            }
        }
        Message::EnrollSample(cur, tot) => {
            state.enroll_cur = cur;
            state.enroll_tot = tot;
            state.status = format!("Shot {cur}/{tot}…");
        }
        Message::EnrollDone(r) => {
            state.enrolling = false;
            state.enroll_job = None;
            state.enroll_cur = 0;
            match r {
                Ok(()) => state.status = "Done: model saved.".to_string(),
                Err(e) => state.status = format!("Error: {e}"),
            }
        }
        Message::TestPressed => {
            if !state.testing && !state.enrolling && !state.username.trim().is_empty() {
                state.preview_on = false;
                state.preview_params = None;
                state.preview_handle = None;
                let mut cfg = state.base_config.clone();
                if state.ir {
                    cfg.video.ir_mode = true;
                }
                state.testing = true;
                state.status = "Testing authentication...".to_string();
                state.test_job = Some(TestJob {
                    id: rand::random::<u64>(),
                    cfg,
                    username: state.username.trim().to_string(),
                });
            }
        }
        Message::TestResult(r) => {
            state.testing = false;
            state.test_job = None;
            state.status = match r {
                Ok(true) => "Test PASSED".to_string(),
                Ok(false) => "Test FAILED (distance too high)".to_string(),
                Err(e) => format!("Test error: {e}"),
            };
        }
    }
    Task::none()
}

fn subscription(state: &FaceauthUi) -> Subscription<Message> {
    let mut subs = Vec::new();
    if let Some(params) = &state.preview_params {
        subs.push(Subscription::run_with(params.clone(), preview_worker));
    }
    if let Some(job) = &state.enroll_job {
        subs.push(Subscription::run_with(job.clone(), enroll_worker));
    }
    if let Some(job) = &state.test_job {
        subs.push(Subscription::run_with(job.clone(), test_worker));
    }
    Subscription::batch(subs)
}

fn view(state: &FaceauthUi) -> Element<'_, Message> {
    let config_text = if let Some(ref p) = state.config_source {
        format!("Config: {}", p.display())
    } else {
        "Config: default values (put faceauth.toml or ~/.config/faceauth/config.toml)".to_string()
    };

    let preview: Element<'_, Message> = if let Some(handle) = &state.preview_handle {
        container(
            image(handle.clone())
                .width(Length::Fixed(280.0))
                .height(Length::Fixed(280.0))
                .content_fit(ContentFit::Cover),
        )
        .width(Length::Fixed(280.0))
        .height(Length::Fixed(280.0))
        .style(|_theme| iced::widget::container::Style {
            border: iced::border::rounded(140.0),
            ..iced::widget::container::Style::default()
        })
        .clip(true)
        .center_x(Length::Fill)
        .into()
    } else if state.preview_on {
        container(text("There is no frame (check the camera device)."))
            .center_x(Length::Fill)
            .into()
    } else {
        text("").into()
    };

    let progress: Element<'_, Message> = if state.enrolling && state.enroll_tot > 0 {
        let p = state.enroll_cur as f32 / state.enroll_tot as f32;
        progress_bar(0.0..=1.0, p).into()
    } else {
        text("").into()
    };

    let mode_buttons = column![
        radio("Replace the entire model", UiEnrollMode::ReplaceAll, Some(state.enroll_mode), Message::ModeChanged),
        radio("Complete the main model (more shots without glasses, etc.)", UiEnrollMode::AppendPrimary, Some(state.enroll_mode), Message::ModeChanged),
        radio("Appearance option: new pictures will replace the set with this name", UiEnrollMode::VariantReplace, Some(state.enroll_mode), Message::ModeChanged),
        radio("Option: Add snapshots to an existing option", UiEnrollMode::VariantAppend, Some(state.enroll_mode), Message::ModeChanged),
    ]
    .spacing(5);

    let variant_input: Element<'_, Message> = if matches!(
        state.enroll_mode,
        UiEnrollMode::VariantReplace | UiEnrollMode::VariantAppend
    ) {
        row![
            text("The name of the variant (e.g. glasses):"),
            text_input("", &state.variant_name)
                .on_input(Message::VariantChanged)
                .width(Length::Fixed(180.0)),
        ]
        .spacing(10)
        .align_y(Alignment::Center)
        .into()
    } else {
        text("").into()
    };

    let busy = state.enrolling || state.testing;

    let content = column![
        text("Faceauth — face recording").size(24),
        text(config_text).size(14),
        row![
            text("User:"),
            text_input("", &state.username)
                .on_input(Message::UsernameChanged)
                .width(Length::Fixed(200.0)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        row![
            text("Camera:"),
            text_input("", &state.device)
                .on_input(Message::DeviceChanged)
                .width(Length::Fixed(240.0)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        row![
            text("Mark (optional):"),
            text_input("", &state.label)
                .on_input(Message::LabelChanged)
                .width(Length::Fixed(200.0)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        row![
            text("Shots:"),
            slider(1..=30, state.samples, Message::SamplesChanged).width(Length::Fixed(200.0)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        checkbox(state.ir)
            .label("IR / low light (ir_mode)")
            .on_toggle(Message::IrToggled),
        text("How to save shots:").size(16),
        mode_buttons,
        variant_input,
        row![
            button("Preview").on_press_maybe(if !busy { Some(Message::TogglePreview) } else { None }),
            button("Record a face").on_press_maybe(if !busy && !state.username.trim().is_empty() { Some(Message::EnrollPressed) } else { None }),
            button("Test auth").on_press_maybe(if !busy && !state.username.trim().is_empty() { Some(Message::TestPressed) } else { None }),
        ]
        .spacing(10),
        preview,
        text(format!("Status: {}", state.status)),
        progress,
    ]
    .spacing(10)
    .padding(10);

    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn main() -> iced::Result {
    iced::application(FaceauthUi::new, update, view)
        .subscription(subscription)
        .window(iced::window::Settings {
            size: iced::Size::new(540.0, 680.0),
            ..Default::default()
        })
        .title("Faceauth — face recording")
        .run()
}
