//! Camera worker: the only owner of the camera and the pipeline in the GUI.
//!
//! It runs on its own thread, streams annotated preview frames, and switches
//! between preview, enrollment and live-test modes on command, so the device is
//! never reopened between modes.

use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use iced::futures::StreamExt;
use iced::futures::channel::mpsc::Sender;
use opencv::core::{AlgorithmHint, Mat, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::{MatTraitConst, MatTraitConstManual};

use faceauth::config::Config;
use faceauth::enroll::{EnrollEvent, EnrollSession};
use faceauth::pipeline::{AnalysisMode, FaceVerdict, Frame, Pipeline};

pub enum WorkerCmd {
    Preview,
    Enroll { samples: usize },
    Test,
    Stop,
}

/// Command channel handed to the UI once the camera is open.
#[derive(Clone)]
pub struct CmdTx(mpsc::Sender<WorkerCmd>);

impl CmdTx {
    pub fn send(&self, cmd: WorkerCmd) {
        let _ = self.0.send(cmd);
    }
}

impl std::fmt::Debug for CmdTx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CmdTx")
    }
}

#[derive(Debug, Clone)]
pub enum WorkerEvent {
    Ready(CmdTx, String),
    Frame {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        verdict: FaceVerdict,
    },
    EnrollProgress {
        cur: usize,
        tot: usize,
    },
    /// i18n key of a hint for the user.
    EnrollHint(&'static str),
    EnrollDone {
        vectors: Vec<Vec<f32>>,
        model_id: String,
    },
    Probes {
        vectors: Vec<Vec<f32>>,
        model_id: String,
    },
    Error(String),
}

#[derive(Clone)]
pub struct WorkerJob {
    pub id: u64,
    pub cfg: Config,
    pub device: String,
}

impl Hash for WorkerJob {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub fn run(job: &WorkerJob) -> iced::futures::stream::BoxStream<'static, WorkerEvent> {
    let job = job.clone();
    iced::stream::channel(32, async move |output: Sender<WorkerEvent>| {
        std::thread::spawn(move || worker_loop(job, output));
    })
    .boxed()
}

enum Mode {
    Preview,
    Enroll(EnrollSession),
    Test,
}

/// How often probe embeddings are sent for verification in test mode.
const PROBE_INTERVAL: Duration = Duration::from_millis(500);

fn worker_loop(job: WorkerJob, mut out: Sender<WorkerEvent>) {
    let WorkerJob { cfg, device, .. } = job;
    let mut pipeline = match Pipeline::open(&cfg, &device) {
        Ok(p) => p,
        Err(e) => {
            send_reliable(&mut out, WorkerEvent::Error(format!("{e:#}")));
            return;
        }
    };
    let (tx, rx) = mpsc::channel();
    let model_id = pipeline.recognizer.model_id().to_string();
    if !send_reliable(&mut out, WorkerEvent::Ready(CmdTx(tx), pipeline.describe())) {
        return;
    }

    let mut mode = Mode::Preview;
    let mut probes: Vec<Vec<f32>> = Vec::new();
    let mut last_probe = Instant::now();
    let pause = Duration::from_millis(cfg.video.frame_interval_ms);

    loop {
        while let Ok(cmd) = rx.try_recv() {
            mode = match cmd {
                WorkerCmd::Preview => Mode::Preview,
                WorkerCmd::Enroll { samples } => Mode::Enroll(EnrollSession::new(samples, &cfg)),
                WorkerCmd::Test => {
                    probes.clear();
                    last_probe = Instant::now();
                    Mode::Test
                }
                WorkerCmd::Stop => return,
            };
        }

        let (frame, event) = match &mut mode {
            Mode::Enroll(session) => match session.step(&mut pipeline, &cfg) {
                Ok(step) => (step.frame, step.event),
                Err(_) => {
                    std::thread::sleep(pause);
                    continue;
                }
            },
            Mode::Preview | Mode::Test => match pipeline.capture(&cfg, AnalysisMode::Auth) {
                Ok(f) => (f, None),
                Err(_) => {
                    std::thread::sleep(pause);
                    continue;
                }
            },
        };

        if !send_frame(&mut out, &frame) {
            return;
        }

        match &mut mode {
            Mode::Enroll(session) => {
                let msg = match event {
                    Some(EnrollEvent::Sample { cur, tot }) => {
                        Some(WorkerEvent::EnrollProgress { cur, tot })
                    }
                    Some(EnrollEvent::Hint(h)) => Some(WorkerEvent::EnrollHint(h.message_key())),
                    Some(EnrollEvent::Duplicate) => Some(WorkerEvent::EnrollHint("hint.duplicate")),
                    Some(EnrollEvent::Rejected(v)) => {
                        Some(WorkerEvent::EnrollHint(v.message_key()))
                    }
                    None => None,
                };
                if let Some(m) = msg {
                    let _ = out.try_send(m);
                }
                if session.is_done() {
                    let Mode::Enroll(session) = std::mem::replace(&mut mode, Mode::Preview) else {
                        unreachable!()
                    };
                    let done = WorkerEvent::EnrollDone {
                        vectors: session.into_vectors(),
                        model_id: model_id.clone(),
                    };
                    if !send_reliable(&mut out, done) {
                        return;
                    }
                }
            }
            Mode::Test => {
                if let Ok(Some(emb)) = pipeline.embed(&frame, &cfg) {
                    probes.push(emb.vector);
                    if probes.len() > 3 {
                        probes.remove(0);
                    }
                }
                if !probes.is_empty() && last_probe.elapsed() >= PROBE_INTERVAL {
                    last_probe = Instant::now();
                    let _ = out.try_send(WorkerEvent::Probes {
                        vectors: std::mem::take(&mut probes),
                        model_id: model_id.clone(),
                    });
                }
            }
            Mode::Preview => {}
        }
    }
}

/// Deliver an event that must not be dropped; false once the UI has gone away.
fn send_reliable(out: &mut Sender<WorkerEvent>, mut ev: WorkerEvent) -> bool {
    loop {
        match out.try_send(ev) {
            Ok(()) => return true,
            Err(e) if e.is_disconnected() => return false,
            Err(e) => {
                ev = e.into_inner();
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// Frames may be dropped when the UI is busy; false once the UI has gone away.
fn send_frame(out: &mut Sender<WorkerEvent>, frame: &Frame) -> bool {
    let Ok(mut color) = frame.color.try_clone() else {
        return true;
    };
    draw_overlay(&mut color, frame);
    let Ok((width, height, rgba)) = mat_bgr_to_rgba(&color) else {
        return true;
    };
    let sent = out.try_send(WorkerEvent::Frame {
        width,
        height,
        rgba,
        verdict: frame.analysis.verdict,
    });
    !matches!(sent, Err(e) if e.is_disconnected())
}

/// Oval target, the selected face (green when usable) and its landmarks.
fn draw_overlay(img: &mut Mat, frame: &Frame) {
    let (w, h) = (img.cols(), img.rows());
    let guide = Scalar::new(200.0, 200.0, 200.0, 0.0);
    let _ = imgproc::ellipse(
        img,
        Point::new(w / 2, h / 2),
        Size::new((h as f64 * 0.26) as i32, (h as f64 * 0.36) as i32),
        0.0,
        0.0,
        360.0,
        guide,
        1,
        imgproc::LINE_AA,
        0,
    );
    let ok = frame.analysis.verdict == FaceVerdict::Ok;
    let color = if ok {
        Scalar::new(0.0, 200.0, 0.0, 0.0)
    } else {
        Scalar::new(0.0, 140.0, 255.0, 0.0)
    };
    // The selected face if any passed the filters, otherwise every detection.
    let faces: Vec<&faceauth::detection::Face> = match &frame.analysis.face {
        Some(f) => vec![f],
        None => frame.faces.iter().collect(),
    };
    for face in faces {
        let _ = imgproc::rectangle(img, face.bbox, color, 2, imgproc::LINE_8, 0);
        for pt in &face.landmarks {
            let _ = imgproc::circle(
                img,
                Point::new(pt.x as i32, pt.y as i32),
                3,
                color,
                -1,
                imgproc::LINE_8,
                0,
            );
        }
    }
}

fn mat_bgr_to_rgba(mat: &Mat) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let mut rgba = Mat::default();
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
