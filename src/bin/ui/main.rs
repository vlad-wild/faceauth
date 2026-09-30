//! `faceauth-ui`: enrollment, live test, model management and setup checks.
//!
//! The window runs as the desktop user: it captures faces itself, but every
//! read or write of the root-only model store goes through
//! `pkexec faceauth import | verify | list | remove | rename-variant | clear`.

mod style;
mod worker;

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use iced::widget::button::Button;
use iced::widget::{
    button, checkbox, column, container, image, pick_list, progress_bar, radio, row, scrollable,
    slider, text, text_input,
};
use iced::{Alignment, ContentFit, Element, Font, Length, Padding, Subscription, Task};

use faceauth::config::Config;
use faceauth::database::{EnrollMerge, ImportPayload, ModelSummary, VerifyPayload, VerifyResult};
use faceauth::devices::{VideoDevice, list_devices};
use faceauth::diagnostics::{Check, PAM_LINE, Status};
use faceauth::i18n::{t, tf};
use faceauth::logger;
use faceauth::verdict::FaceVerdict;

use worker::{CmdTx, WorkerCmd, WorkerEvent, WorkerJob};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Enroll,
    Test,
    Models,
    Setup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnrollMode {
    New,
    Add,
}

/// Where enrolled samples go.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Primary,
    Variant(String),
    NewVariant,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Target::Primary => f.write_str(t("enroll.primary")),
            Target::Variant(name) => f.write_str(name),
            Target::NewVariant => f.write_str(t("enroll.new_variant")),
        }
    }
}

/// A command to send once the camera worker is ready.
#[derive(Debug, Clone, Copy)]
enum Pending {
    Enroll(usize),
    Test,
}

#[derive(Debug, Clone)]
enum Message {
    TabSelected(Tab),
    UsernameChanged(String),
    DeviceSelected(VideoDevice),
    IrToggled(bool),
    ToggleCamera,
    Worker(WorkerEvent),
    SamplesChanged(u32),
    LabelChanged(String),
    ModeChanged(EnrollMode),
    TargetSelected(Target),
    NewVariantChanged(String),
    StartEnroll,
    ImportDone(Result<String, String>),
    StartTest,
    StopTest,
    VerifyDone(Result<VerifyResult, String>),
    LoadModels,
    ModelsLoaded(Result<Vec<ModelSummary>, String>),
    RenameInput(String, String),
    RenameVariant(String),
    RemoveVariant(String),
    ClearModel,
    ActionDone(Result<String, String>),
    RunDoctor,
    DoctorDone(Result<Vec<Check>, String>),
    CopyPam,
    /// Re-read the desktop palette if it changed.
    PaletteTick,
}

struct App {
    cfg: Config,
    config_source: Option<PathBuf>,
    tab: Tab,
    username: String,
    devices: Vec<VideoDevice>,
    device: String,
    ir: bool,

    job: Option<WorkerJob>,
    next_job: u64,
    cmd: Option<CmdTx>,
    pending: Option<Pending>,
    backend: String,
    preview: Option<image::Handle>,
    verdict: Option<FaceVerdict>,
    status: String,

    samples: u32,
    label: String,
    mode: EnrollMode,
    target: Target,
    new_variant: String,
    enrolling: bool,
    enroll_cur: usize,
    enroll_tot: usize,
    hint: Option<&'static str>,
    import_plan: Option<(EnrollMerge, Option<String>, Option<String>)>,

    testing: bool,
    verify_in_flight: bool,
    last_score: Option<f32>,
    threshold: f32,
    required: u32,
    streak: u32,

    models: Option<Vec<ModelSummary>>,
    renames: HashMap<String, String>,
    busy: bool,

    checks: Option<Vec<Check>>,

    /// Desktop palette (nothing-rice), if one is installed.
    palette: Option<style::Palette>,
    palette_stamp: Option<std::time::SystemTime>,
}

impl App {
    fn new() -> (Self, Task<Message>) {
        let _ = logger::try_init_from_env();
        let (cfg, config_source) = Config::discover();
        let devices = list_devices();
        let device = cfg.video.device_path.clone();
        let ir = cfg.video.ir_mode;
        let threshold = cfg.recognition.distance_threshold as f32;
        let required = cfg.recognition.required_matches;
        let (palette, palette_stamp) = style::load();
        let app = Self {
            cfg,
            config_source,
            tab: Tab::Enroll,
            username: std::env::var("USER").unwrap_or_default(),
            devices,
            device,
            ir,
            job: None,
            next_job: 1,
            cmd: None,
            pending: None,
            backend: String::new(),
            preview: None,
            verdict: None,
            status: t("camera.off").to_string(),
            samples: 9,
            label: String::new(),
            mode: EnrollMode::New,
            target: Target::Primary,
            new_variant: String::new(),
            enrolling: false,
            enroll_cur: 0,
            enroll_tot: 0,
            hint: None,
            import_plan: None,
            testing: false,
            verify_in_flight: false,
            last_score: None,
            threshold,
            required,
            streak: 0,
            models: None,
            renames: HashMap::new(),
            busy: false,
            checks: None,
            palette,
            palette_stamp,
        };
        (app, Task::none())
    }

    fn user(&self) -> String {
        self.username.trim().to_string()
    }

    fn selected_device(&self) -> Option<VideoDevice> {
        self.devices.iter().find(|d| d.path == self.device).cloned()
    }

    /// (Re)start the worker. Keeps the enroll/test state; a running test resumes.
    fn start_camera(&mut self) {
        let mut cfg = self.cfg.clone();
        cfg.video.ir_mode = self.ir;
        self.shutdown_worker();
        if self.testing {
            self.pending = Some(Pending::Test);
        }
        self.job = Some(WorkerJob {
            id: self.next_job,
            cfg,
            device: self.device.clone(),
        });
        self.next_job += 1;
        self.status = t("camera.opening").to_string();
    }

    fn shutdown_worker(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            cmd.send(WorkerCmd::Stop);
        }
        self.job = None;
        self.preview = None;
        self.verdict = None;
    }

    /// Stop the camera and abandon any enrollment / test in progress.
    fn stop_camera(&mut self) {
        self.shutdown_worker();
        self.pending = None;
        self.import_plan = None;
        self.enrolling = false;
        self.testing = false;
    }

    fn restart_if_running(&mut self) {
        if self.job.is_some() {
            self.start_camera();
        }
    }

    /// Send now if the worker is up, otherwise start the camera and queue it.
    fn dispatch(&mut self, p: Pending) {
        match &self.cmd {
            Some(cmd) => send_pending(cmd, p),
            None => {
                if self.job.is_none() {
                    self.start_camera();
                }
                self.pending = Some(p);
            }
        }
    }

    fn variant_names(&self) -> Vec<String> {
        let user = self.user();
        self.models
            .iter()
            .flatten()
            .filter(|m| m.user == user)
            .flat_map(|m| m.variants.iter().map(|v| v.label.clone()))
            .collect()
    }
}

fn send_pending(cmd: &CmdTx, p: Pending) {
    cmd.send(match p {
        Pending::Enroll(samples) => WorkerCmd::Enroll { samples },
        Pending::Test => WorkerCmd::Test,
    });
}

/// `faceauth` next to this executable (packaged: /usr/bin/faceauth).
fn faceauth_bin() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("faceauth")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("/usr/bin/faceauth"))
}

/// Run `faceauth <args>` (via pkexec when `privileged`), feeding `stdin`, returning stdout.
fn run_faceauth(
    args: Vec<String>,
    stdin: Option<String>,
    privileged: bool,
) -> Result<String, String> {
    let bin = faceauth_bin();
    let mut command = if privileged {
        let mut c = Command::new("pkexec");
        c.arg(&bin);
        c
    } else {
        Command::new(&bin)
    };
    let mut child = command
        .args(&args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{}: {e}", bin.display()))?;
    if let (Some(data), Some(mut pipe)) = (stdin, child.stdin.take()) {
        pipe.write_all(data.as_bytes()).map_err(|e| e.to_string())?;
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        let last = err
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("failed");
        Err(match out.status.code() {
            // pkexec: 126 = dialog dismissed / not authorized, 127 = auth failed
            Some(126) | Some(127) if privileged => "not authorized".to_string(),
            _ => last.trim().to_string(),
        })
    }
}

/// Run blocking work on a thread and deliver its result as a message.
fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
    done: impl FnOnce(Result<T, String>) -> Message + Send + 'static,
) -> Task<Message> {
    Task::perform(
        async move {
            let (tx, rx) = iced::futures::channel::oneshot::channel();
            std::thread::spawn(move || {
                let _ = tx.send(work());
            });
            rx.await
                .unwrap_or_else(|_| Err("background task failed".to_string()))
        },
        done,
    )
}

fn parse_json<T: serde::de::DeserializeOwned>(s: String) -> Result<T, String> {
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

fn update(app: &mut App, message: Message) -> Task<Message> {
    match message {
        Message::TabSelected(tab) => {
            app.tab = tab;
            if tab != Tab::Test && app.testing {
                return update(app, Message::StopTest);
            }
        }
        Message::UsernameChanged(s) => {
            app.username = s;
            app.models = None;
        }
        Message::DeviceSelected(d) => {
            if app.enrolling {
                return Task::none();
            }
            app.ir = d.likely_ir || app.ir && app.device == d.path;
            app.device = d.path;
            app.restart_if_running();
        }
        Message::IrToggled(v) => {
            app.ir = v;
            app.restart_if_running();
        }
        Message::ToggleCamera => {
            if app.job.is_some() {
                app.stop_camera();
                app.status = t("camera.off").to_string();
            } else {
                app.start_camera();
            }
        }
        Message::Worker(ev) => return on_worker(app, ev),
        Message::SamplesChanged(v) => app.samples = v.clamp(1, 30),
        Message::LabelChanged(s) => app.label = s,
        Message::ModeChanged(m) => app.mode = m,
        Message::TargetSelected(t) => app.target = t,
        Message::NewVariantChanged(s) => app.new_variant = s,
        Message::StartEnroll => {
            if app.user().is_empty() {
                app.status = t("status.need_user").to_string();
                return Task::none();
            }
            let variant = match &app.target {
                Target::Primary => None,
                Target::Variant(v) => Some(v.clone()),
                Target::NewVariant => {
                    let v = app.new_variant.trim().to_string();
                    if v.is_empty() {
                        app.status = t("enroll.need_variant").to_string();
                        return Task::none();
                    }
                    Some(v)
                }
            };
            let merge = EnrollMerge::from_flags(variant.is_some(), app.mode == EnrollMode::Add);
            let label = Some(app.label.trim().to_string()).filter(|l| !l.is_empty());
            app.import_plan = Some((merge, variant, label));
            app.enrolling = true;
            app.enroll_cur = 0;
            app.enroll_tot = app.samples as usize;
            app.hint = None;
            app.dispatch(Pending::Enroll(app.samples as usize));
        }
        Message::ImportDone(r) => {
            app.busy = false;
            app.status = match r {
                Ok(_) => {
                    app.models = None;
                    tf("enroll.saved", &[&app.enroll_cur])
                }
                Err(e) => tf("status.error", &[&e]),
            };
        }
        Message::StartTest => {
            if app.user().is_empty() {
                app.status = t("status.need_user").to_string();
                return Task::none();
            }
            app.testing = true;
            app.streak = 0;
            app.last_score = None;
            app.status = t("test.waiting").to_string();
            app.dispatch(Pending::Test);
        }
        Message::StopTest => {
            app.testing = false;
            if let Some(cmd) = &app.cmd {
                cmd.send(WorkerCmd::Preview);
            }
        }
        Message::VerifyDone(r) => {
            app.verify_in_flight = false;
            match r {
                Ok(res) => {
                    app.threshold = res.threshold;
                    app.required = res.required_matches;
                    for s in res.scores {
                        app.streak = if s < res.threshold { app.streak + 1 } else { 0 };
                        app.last_score = Some(s);
                    }
                }
                Err(e) => {
                    app.status = tf("status.error", &[&e]);
                    return update(app, Message::StopTest);
                }
            }
        }
        Message::LoadModels => {
            let user = app.user();
            if user.is_empty() {
                app.status = t("status.need_user").to_string();
                return Task::none();
            }
            app.busy = true;
            let args = vec!["list".into(), user, "--json".into()];
            return blocking(
                move || run_faceauth(args, None, true).and_then(parse_json),
                Message::ModelsLoaded,
            );
        }
        Message::ModelsLoaded(r) => {
            app.busy = false;
            match r {
                Ok(m) => app.models = Some(m),
                Err(e) => app.status = tf("status.error", &[&e]),
            }
        }
        Message::RenameInput(variant, name) => {
            app.renames.insert(variant, name);
        }
        Message::RenameVariant(from) => {
            let to = app.renames.get(&from).cloned().unwrap_or_default();
            if to.trim().is_empty() {
                return Task::none();
            }
            let args = vec![
                "rename-variant".into(),
                "-u".into(),
                app.user(),
                from,
                to.trim().to_string(),
            ];
            return model_action(app, args);
        }
        Message::RemoveVariant(v) => {
            let args = vec![
                "remove".into(),
                "-u".into(),
                app.user(),
                "--variant".into(),
                v,
            ];
            return model_action(app, args);
        }
        Message::ClearModel => {
            let args = vec!["clear".into(), "-u".into(), app.user()];
            return model_action(app, args);
        }
        Message::ActionDone(r) => {
            app.busy = false;
            match r {
                Ok(_) => {
                    app.status = t("models.done").to_string();
                    app.renames.clear();
                    return update(app, Message::LoadModels);
                }
                Err(e) => app.status = tf("status.error", &[&e]),
            }
        }
        Message::RunDoctor => {
            app.busy = true;
            let args = vec!["doctor".into(), "-u".into(), app.user(), "--json".into()];
            return blocking(
                move || run_faceauth(args, None, false).and_then(parse_json),
                Message::DoctorDone,
            );
        }
        Message::DoctorDone(r) => {
            app.busy = false;
            match r {
                Ok(c) => app.checks = Some(c),
                Err(e) => app.status = tf("status.error", &[&e]),
            }
        }
        Message::CopyPam => {
            app.status = t("setup.copied").to_string();
            return iced::clipboard::write(PAM_LINE.to_string());
        }
        Message::PaletteTick => {
            if style::modified() != app.palette_stamp {
                (app.palette, app.palette_stamp) = style::load();
            }
        }
    }
    Task::none()
}

fn model_action(app: &mut App, args: Vec<String>) -> Task<Message> {
    app.busy = true;
    app.status = t("status.busy").to_string();
    blocking(move || run_faceauth(args, None, true), Message::ActionDone)
}

fn on_worker(app: &mut App, ev: WorkerEvent) -> Task<Message> {
    match ev {
        WorkerEvent::Ready(cmd, backend) => {
            if let Some(p) = app.pending.take() {
                send_pending(&cmd, p);
            }
            app.cmd = Some(cmd);
            app.status = backend.clone();
            app.backend = backend;
        }
        WorkerEvent::Frame {
            width,
            height,
            rgba,
            verdict,
        } => {
            app.preview = Some(image::Handle::from_rgba(width, height, rgba));
            app.verdict = Some(verdict);
        }
        WorkerEvent::EnrollProgress { cur, tot } => {
            app.enroll_cur = cur;
            app.enroll_tot = tot;
        }
        WorkerEvent::EnrollHint(key) => app.hint = Some(key),
        WorkerEvent::EnrollDone { vectors, model_id } => {
            app.enrolling = false;
            app.hint = None;
            app.enroll_cur = vectors.len();
            let Some((merge, variant, label)) = app.import_plan.take() else {
                return Task::none();
            };
            if vectors.is_empty() {
                app.status = t("enroll.none").to_string();
                return Task::none();
            }
            let payload = ImportPayload {
                merge,
                variant,
                label,
                model_id,
                embeddings: vectors,
            };
            let json = match serde_json::to_string(&payload) {
                Ok(j) => j,
                Err(e) => {
                    app.status = tf("status.error", &[&e]);
                    return Task::none();
                }
            };
            app.busy = true;
            app.status = t("enroll.saving").to_string();
            let args = vec!["import".into(), "-u".into(), app.user()];
            return blocking(
                move || run_faceauth(args, Some(json), true),
                Message::ImportDone,
            );
        }
        WorkerEvent::Probes { vectors, model_id } => {
            if !app.testing || app.verify_in_flight {
                return Task::none();
            }
            let json = match serde_json::to_string(&VerifyPayload {
                model_id,
                embeddings: vectors,
            }) {
                Ok(j) => j,
                Err(_) => return Task::none(),
            };
            app.verify_in_flight = true;
            let args = vec!["verify".into(), "-u".into(), app.user()];
            return blocking(
                move || run_faceauth(args, Some(json), true).and_then(parse_json),
                Message::VerifyDone,
            );
        }
        WorkerEvent::Error(e) => {
            app.stop_camera();
            app.status = tf("status.error", &[&e]);
        }
    }
    Task::none()
}

fn subscription(app: &App) -> Subscription<Message> {
    let camera = match &app.job {
        Some(job) => Subscription::run_with(job.clone(), worker::run).map(Message::Worker),
        None => Subscription::none(),
    };
    let palette =
        iced::time::every(std::time::Duration::from_secs(2)).map(|_| Message::PaletteTick);
    Subscription::batch([camera, palette])
}

/// A button in the desktop style when a palette is present.
fn btn<'a>(app: &App, label: &'a str, primary: bool) -> Button<'a, Message> {
    let padding = if app.palette.is_some() {
        Padding::from([8, 18])
    } else {
        iced::widget::button::DEFAULT_PADDING
    };
    button(text(label))
        .padding(padding)
        .style(style::button_style(app.palette, primary))
}

fn labeled<'a>(
    label: &'static str,
    widget: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    row![text(t(label)), widget.into()]
        .spacing(10)
        .align_y(Alignment::Center)
        .into()
}

fn view(app: &App) -> Element<'_, Message> {
    let config_text = match &app.config_source {
        Some(p) => tf("config.source", &[&p.display()]),
        None => t("config.defaults").to_string(),
    };

    let camera_picker: Element<'_, Message> = if app.devices.is_empty() {
        text_input("/dev/video0", &app.device)
            .style(style::input_style(app.palette))
            .on_input(|s| {
                Message::DeviceSelected(VideoDevice {
                    path: s,
                    name: String::new(),
                    likely_ir: false,
                })
            })
            .width(Length::Fixed(220.0))
            .into()
    } else {
        pick_list(
            app.devices.as_slice(),
            app.selected_device(),
            Message::DeviceSelected,
        )
        .style(style::pick_style(app.palette))
        .menu_style(style::menu_style(app.palette))
        .width(Length::Fixed(320.0))
        .into()
    };
    let busy = app.busy || app.enrolling;
    let camera_label = if app.job.is_some() {
        t("camera.stop")
    } else {
        t("camera.start")
    };
    let camera_button =
        btn(app, camera_label, true).on_press_maybe((!busy).then_some(Message::ToggleCamera));

    let title = if app.palette.is_some() {
        text(t("app.title").to_uppercase())
            .size(28)
            .font(Font::with_name("Matrix Sans Print"))
    } else {
        text(t("app.title")).size(24)
    };

    let header = column![
        title,
        text(config_text).size(13),
        row![
            labeled(
                "field.user",
                text_input("", &app.username)
                    .style(style::input_style(app.palette))
                    .on_input(Message::UsernameChanged)
                    .width(Length::Fixed(160.0)),
            ),
            labeled("field.camera", camera_picker),
        ]
        .spacing(20)
        .align_y(Alignment::Center),
        row![
            checkbox(app.ir)
                .label(t("field.ir"))
                .on_toggle_maybe((!busy).then_some(Message::IrToggled)),
            camera_button,
        ]
        .spacing(20)
        .align_y(Alignment::Center),
    ]
    .spacing(8);

    let tab_button = |tab: Tab, key: &'static str| {
        let selected = app.tab == tab;
        let b = button(text(t(key))).style(style::tab_style(app.palette, selected));
        if app.palette.is_some() {
            // Styled tabs stay clickable so the selected one is not drawn as disabled.
            b.padding(Padding::from([8, 18]))
                .on_press(Message::TabSelected(tab))
        } else {
            b.on_press_maybe((!selected).then_some(Message::TabSelected(tab)))
        }
    };
    let tabs = row![
        tab_button(Tab::Enroll, "tab.enroll"),
        tab_button(Tab::Test, "tab.test"),
        tab_button(Tab::Models, "tab.models"),
        tab_button(Tab::Setup, "tab.setup"),
    ]
    .spacing(6);

    let body: Element<'_, Message> = match app.tab {
        Tab::Enroll => view_enroll(app),
        Tab::Test => view_test(app),
        Tab::Models => view_models(app),
        Tab::Setup => view_setup(app),
    };

    let preview: Element<'_, Message> = match &app.preview {
        Some(handle) => image(handle.clone())
            .width(Length::Fixed(420.0))
            .content_fit(ContentFit::Contain)
            .into(),
        None => container(text(t("camera.off")))
            .width(Length::Fixed(420.0))
            .center_x(Length::Fixed(420.0))
            .into(),
    };
    let matched = app.testing && app.streak >= app.required;
    let preview: Element<'_, Message> = if app.palette.is_some() {
        container(preview)
            .padding(10)
            .style(style::preview_frame(app.palette, matched))
            .into()
    } else {
        preview
    };
    let verdict_text = app.verdict.map(|v| t(v.message_key())).unwrap_or("");
    let preview_col = column![preview, text(verdict_text).size(16)].spacing(6);

    let body = container(scrollable(body))
        .padding(if app.palette.is_some() { 20 } else { 0 })
        .width(Length::Fill)
        .style(style::card(app.palette));
    let content = column![
        header,
        tabs,
        row![body, preview_col].spacing(16),
        text(app.status.clone()).size(14),
    ]
    .spacing(12)
    .padding(12);

    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn view_enroll(app: &App) -> Element<'_, Message> {
    let mut targets = vec![Target::Primary];
    targets.extend(app.variant_names().into_iter().map(Target::Variant));
    targets.push(Target::NewVariant);

    let mut col = column![
        labeled(
            "field.samples",
            row![
                slider(1..=30, app.samples, Message::SamplesChanged).width(Length::Fixed(160.0)),
                text(app.samples.to_string()),
            ]
            .spacing(8),
        ),
        labeled(
            "field.label",
            text_input("", &app.label)
                .style(style::input_style(app.palette))
                .on_input(Message::LabelChanged)
                .width(Length::Fixed(180.0)),
        ),
        radio(
            t("enroll.mode_new"),
            EnrollMode::New,
            Some(app.mode),
            Message::ModeChanged
        ),
        radio(
            t("enroll.mode_add"),
            EnrollMode::Add,
            Some(app.mode),
            Message::ModeChanged
        ),
        labeled(
            "field.target",
            pick_list(targets, Some(app.target.clone()), Message::TargetSelected)
                .style(style::pick_style(app.palette))
                .menu_style(style::menu_style(app.palette)),
        ),
    ]
    .spacing(10);

    if app.target == Target::NewVariant {
        col = col.push(labeled(
            "field.new_variant",
            text_input("glasses", &app.new_variant)
                .style(style::input_style(app.palette))
                .on_input(Message::NewVariantChanged)
                .width(Length::Fixed(160.0)),
        ));
    }
    let can_start = !app.enrolling && !app.busy && !app.user().is_empty();
    col = col.push(
        btn(app, t("enroll.start"), true).on_press_maybe(can_start.then_some(Message::StartEnroll)),
    );

    if app.enrolling {
        let p = app.enroll_cur as f32 / app.enroll_tot.max(1) as f32;
        let meter: Element<'_, Message> = match &app.palette {
            Some(pal) => style::dot_meter(pal, app.enroll_tot.max(1), app.enroll_cur, None),
            None => progress_bar(0.0..=1.0, p).into(),
        };
        col = col.push(meter).push(text(tf(
            "enroll.progress",
            &[&app.enroll_cur, &app.enroll_tot],
        )));
        if let Some(h) = app.hint {
            col = col.push(text(t(h)).size(18));
        }
    }
    col.into()
}

fn view_test(app: &App) -> Element<'_, Message> {
    let mut col = column![
        row![
            btn(app, t("test.start"), true)
                .on_press_maybe((!app.testing && !app.enrolling).then_some(Message::StartTest)),
            btn(app, t("test.stop"), false)
                .on_press_maybe(app.testing.then_some(Message::StopTest)),
        ]
        .spacing(10)
    ]
    .spacing(10);

    if app.testing || app.last_score.is_some() {
        match app.last_score {
            Some(score) => {
                // Bar fills as the score approaches 0 (a perfect match).
                let range = (app.threshold * 2.0).max(0.1);
                let fill = (1.0 - score / range).clamp(0.0, 1.0);
                let meter: Element<'_, Message> = match &app.palette {
                    // 24 dots; the threshold sits in the middle (score == threshold).
                    Some(pal) => {
                        style::dot_meter(pal, 24, (fill * 24.0).round() as usize, Some(12))
                    }
                    None => progress_bar(0.0..=1.0, fill).into(),
                };
                col = col.push(meter).push(text(tf(
                    "test.score",
                    &[&format!("{score:.3}"), &format!("{:.3}", app.threshold)],
                )));
            }
            None => col = col.push(text(t("test.waiting"))),
        }
        col = col.push(text(tf(
            "test.streak",
            &[&app.streak.min(app.required), &app.required],
        )));
        if let Some(pal) = &app.palette {
            col = col.push(style::dot_meter(
                pal,
                app.required as usize,
                app.streak.min(app.required) as usize,
                None,
            ));
        }
        if app.streak >= app.required {
            col = col.push(text(t("test.pass")).size(20));
        }
    }
    col.into()
}

fn view_models(app: &App) -> Element<'_, Message> {
    let mut col = column![
        btn(app, t("models.load"), true)
            .on_press_maybe((!app.busy && !app.user().is_empty()).then_some(Message::LoadModels)),
    ]
    .spacing(10);

    if let Some(models) = &app.models {
        let user = app.user();
        match models.iter().find(|m| m.user == user) {
            None => col = col.push(text(t("models.none"))),
            Some(m) => {
                col = col.push(text(tf("models.primary", &[&m.primary_samples])));
                if let Some(u) = m.updated_at {
                    col = col.push(text(tf("models.updated", &[&u.format("%Y-%m-%d %H:%M")])));
                }
                for v in &m.variants {
                    let label = v.label.clone();
                    let new_name = app.renames.get(&v.label).cloned().unwrap_or_default();
                    let for_input = label.clone();
                    col = col.push(
                        row![
                            text(tf("models.variant", &[&v.label, &v.samples])),
                            text_input(&v.label, &new_name)
                                .style(style::input_style(app.palette))
                                .on_input(move |s| Message::RenameInput(for_input.clone(), s))
                                .width(Length::Fixed(120.0)),
                            btn(app, t("models.rename"), false).on_press_maybe(
                                (!app.busy).then_some(Message::RenameVariant(label.clone()))
                            ),
                            btn(app, t("models.delete"), false).on_press_maybe(
                                (!app.busy).then_some(Message::RemoveVariant(label))
                            ),
                        ]
                        .spacing(8)
                        .align_y(Alignment::Center),
                    );
                }
                col = col.push(
                    btn(app, t("models.clear"), false)
                        .on_press_maybe((!app.busy).then_some(Message::ClearModel)),
                );
            }
        }
    }
    col.push(text(t("models.disable_hint")).size(13)).into()
}

fn view_setup(app: &App) -> Element<'_, Message> {
    let mut col = column![
        text(t("setup.steps")),
        btn(app, t("setup.run"), true).on_press_maybe((!app.busy).then_some(Message::RunDoctor)),
    ]
    .spacing(10);

    if let Some(checks) = &app.checks {
        for c in checks {
            let line = text(format!("{}: {}", c.name, c.detail)).size(14);
            col = col.push(match &app.palette {
                Some(pal) => Element::from(
                    row![style::status_dot(pal, c.status), line]
                        .spacing(10)
                        .align_y(Alignment::Center),
                ),
                None => {
                    let mark = match c.status {
                        Status::Ok => "✔",
                        Status::Warn => "⚠",
                        Status::Fail => "✘",
                    };
                    text(format!("{mark} {}: {}", c.name, c.detail))
                        .size(14)
                        .into()
                }
            });
            if let Some(h) = &c.hint {
                col = col.push(text(format!("    → {h}")).size(13));
            }
        }
    }
    if !app.backend.is_empty() {
        col = col.push(text(app.backend.clone()).size(13));
    }
    col.push(text(t("setup.pam")))
        .push(
            row![
                text(PAM_LINE).size(13),
                btn(app, t("setup.copy"), false).on_press(Message::CopyPam),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        )
        .into()
}

fn main() -> iced::Result {
    // Interface font of the desktop, when its palette is installed.
    let font = if style::load().0.is_some() {
        Font::with_name("Space Grotesk")
    } else {
        Font::DEFAULT
    };
    iced::application(App::new, update, view)
        .subscription(subscription)
        .theme(|app: &App| style::theme(app.palette.as_ref()))
        .default_font(font)
        .window(iced::window::Settings {
            size: iced::Size::new(980.0, 720.0),
            ..Default::default()
        })
        .title("Faceauth")
        .run()
}
