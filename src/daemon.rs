//! Protocol and client for `faceauthd`, the root face-verification daemon.
//!
//! Screen lockers run as the desktop user and cannot read the root-only model
//! store. They connect to [`SOCKET_PATH`] instead; the daemon learns the caller's
//! uid from `SO_PEERCRED` and only ever verifies that user's face. The locker
//! decides what a successful match unlocks.
//!
//! Wire format: one JSON object per line in each direction. The client sends a
//! [`Request`]; the daemon answers with [`Event`]s and ends with `Event::Result`.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::{Duration, Instant};

/// Socket created by `faceauthd.socket` (systemd socket activation).
pub const SOCKET_PATH: &str = "/run/faceauth/faceauthd.sock";

/// Client → daemon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Verify the caller's face. Only root callers may name another user.
    Verify {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        /// Who is asking, for the attempt history (e.g. `lock`, a PAM service name).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        service: Option<String>,
    },
    /// Stop the attempt in progress on this connection.
    Cancel,
    /// Camera, backend and enrolled-model summary for the caller.
    Status,
    /// Score the caller's own face over `frames` frames and suggest a threshold.
    Calibrate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        frames: Option<usize>,
    },
    /// The caller's recent attempts (kept in memory only).
    History,
    /// Do not offer face authentication to the caller for `seconds` (max 600),
    /// so the next polkit prompt asks for the password. `0` clears it.
    PasswordOnly { seconds: u64 },
}

/// Longest `password_only` period a client may request.
pub const MAX_PASSWORD_ONLY_SECS: u64 = 600;

/// Directory of per-uid `password_only` flags (content: expiry, Unix seconds).
pub const PASSWORD_ONLY_DIR: &str = "/run/faceauth/password-only";

/// Attempts remembered per uid for `history`.
pub const HISTORY_LEN: usize = 20;

/// One remembered attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Unix seconds.
    pub time: i64,
    pub service: String,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Answer to `status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub version: String,
    pub user: String,
    /// Face authentication disabled for everyone.
    pub disabled: bool,
    pub lid_closed: bool,
    /// Unix seconds until which `password_only` is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password_only_until: Option<i64>,
    pub camera_device: String,
    pub camera_present: bool,
    pub ir_mode: bool,
    /// Detector / recognizer backends, once the models have been loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<crate::database::ModelSummary>,
    pub threshold: f64,
    pub required_matches: u32,
}

/// Keep client-supplied service labels short and printable.
pub fn sanitize_service(service: Option<&str>) -> String {
    let cleaned: String = service
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .take(32)
        .collect();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned
    }
}

/// How an attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Success,
    NoMatch,
    TooDark,
    /// Not attempted: disabled, no model, lid closed, camera unavailable…
    Skipped,
    Cancelled,
    /// Another attempt is using the camera, or too many recent failures.
    Busy,
    Error,
}

impl Outcome {
    /// Exit code `faceauth-auth` reports for this outcome (anything but 0 lets
    /// PAM fall through to the password).
    pub fn exit_code(self) -> i32 {
        match self {
            Outcome::Success => 0,
            Outcome::Skipped => 10,
            Outcome::NoMatch | Outcome::Cancelled => 11,
            Outcome::Busy | Outcome::Error => 12,
            Outcome::TooDark => 13,
        }
    }
}

/// Daemon → client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The camera is open and frames are being analyzed.
    Started,
    /// One analyzed frame (no image data ever leaves the daemon).
    Frame {
        face: bool,
        dark: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        score: Option<f32>,
        matched: bool,
    },
    /// Answer to `status`.
    Status { status: Box<DaemonStatus> },
    /// Answer to `history`, oldest first.
    History { entries: Vec<HistoryEntry> },
    /// `calibrate` progress.
    CalibrationProgress { n: usize, of: usize },
    /// `calibrate` result.
    Calibration {
        calibration: crate::calibration::Calibration,
    },
    /// Final event on every connection.
    Result {
        outcome: Outcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl Event {
    pub fn result(outcome: Outcome, reason: Option<&str>) -> Self {
        Event::Result {
            outcome,
            reason: reason.map(str::to_string),
        }
    }

    /// Serialize as one protocol line (with the trailing newline).
    pub fn to_line(&self) -> String {
        let mut line = serde_json::to_string(self).expect("protocol events always serialize");
        line.push('\n');
        line
    }
}

/// Sliding-window limit on failed attempts per uid, so a process in the user's
/// session cannot keep the camera busy or grind through attempts.
pub struct RateLimiter {
    max_failures: usize,
    window: Duration,
    failures: HashMap<u32, VecDeque<Instant>>,
}

impl RateLimiter {
    pub fn new(max_failures: usize, window: Duration) -> Self {
        Self {
            max_failures,
            window,
            failures: HashMap::new(),
        }
    }

    /// Whether `uid` may start another attempt at `now`.
    pub fn allowed(&mut self, uid: u32, now: Instant) -> bool {
        let Some(list) = self.failures.get_mut(&uid) else {
            return true;
        };
        while list
            .front()
            .is_some_and(|t| now.duration_since(*t) >= self.window)
        {
            list.pop_front();
        }
        list.len() < self.max_failures
    }

    /// Count a finished attempt: failures add up, a success clears the record.
    pub fn record(&mut self, uid: u32, outcome: Outcome, now: Instant) {
        match outcome {
            Outcome::Success => {
                self.failures.remove(&uid);
            }
            Outcome::NoMatch | Outcome::TooDark => {
                self.failures.entry(uid).or_default().push_back(now);
            }
            _ => {}
        }
    }
}

/// Uid of the process on the other end of a Unix socket.
#[cfg(target_os = "linux")]
pub fn peer_uid(stream: &std::os::unix::net::UnixStream) -> Result<u32> {
    use std::os::fd::AsRawFd;

    let mut cred = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: `cred` and `len` are valid for writes of the sizes passed.
    let rc = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut cred as *mut libc::ucred).cast(),
            &mut len,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("SO_PEERCRED failed");
    }
    Ok(cred.uid)
}

/// Ask the daemon to verify the calling user's face, reporting every event.
#[cfg(unix)]
pub fn verify(
    socket: &Path,
    service: Option<&str>,
    mut on_event: impl FnMut(&Event),
) -> Result<(Outcome, Option<String>)> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("Cannot connect to faceauthd at {}", socket.display()))?;
    // Longer than any configured timeout: the daemon always ends with a result.
    stream.set_read_timeout(Some(Duration::from_secs(90)))?;
    let mut request = serde_json::to_string(&Request::Verify {
        user: None,
        service: service.map(str::to_string),
    })?;
    request.push('\n');
    stream.write_all(request.as_bytes())?;

    for line in BufReader::new(stream).lines() {
        let line = line.context("Reading from faceauthd")?;
        if line.trim().is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(&line)
            .with_context(|| format!("Invalid faceauthd event: {line}"))?;
        on_event(&event);
        if let Event::Result { outcome, reason } = event {
            return Ok((outcome, reason));
        }
    }
    bail!("faceauthd closed the connection without a result")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_parse() {
        let r: Request = serde_json::from_str(r#"{"op":"verify"}"#).unwrap();
        assert_eq!(
            r,
            Request::Verify {
                user: None,
                service: None
            }
        );
        let r: Request =
            serde_json::from_str(r#"{"op":"verify","user":"vlad","service":"lock"}"#).unwrap();
        assert_eq!(
            r,
            Request::Verify {
                user: Some("vlad".into()),
                service: Some("lock".into())
            }
        );
        let r: Request = serde_json::from_str(r#"{"op":"password_only","seconds":60}"#).unwrap();
        assert_eq!(r, Request::PasswordOnly { seconds: 60 });
        let r: Request = serde_json::from_str(r#"{"op":"calibrate"}"#).unwrap();
        assert_eq!(r, Request::Calibrate { frames: None });
        for op in ["status", "history"] {
            assert!(serde_json::from_str::<Request>(&format!(r#"{{"op":"{op}"}}"#)).is_ok());
        }
        let r: Request = serde_json::from_str(r#"{"op":"cancel"}"#).unwrap();
        assert_eq!(r, Request::Cancel);
        assert!(serde_json::from_str::<Request>(r#"{"op":"enroll"}"#).is_err());
    }

    #[test]
    fn events_serialize() {
        assert_eq!(Event::Started.to_line(), "{\"event\":\"started\"}\n");
        let frame = Event::Frame {
            face: true,
            dark: false,
            score: Some(0.5),
            matched: true,
        };
        assert_eq!(
            frame.to_line(),
            "{\"event\":\"frame\",\"face\":true,\"dark\":false,\"score\":0.5,\"matched\":true}\n"
        );
        assert_eq!(
            Event::result(Outcome::Skipped, Some("no_model")).to_line(),
            "{\"event\":\"result\",\"outcome\":\"skipped\",\"reason\":\"no_model\"}\n"
        );
        let back: Event =
            serde_json::from_str(r#"{"event":"result","outcome":"too_dark"}"#).unwrap();
        assert_eq!(back, Event::result(Outcome::TooDark, None));
    }

    #[test]
    fn service_labels_are_sanitized() {
        assert_eq!(sanitize_service(None), "unknown");
        assert_eq!(sanitize_service(Some("sudo")), "sudo");
        assert_eq!(sanitize_service(Some("qs-lock")), "qs-lock");
        assert_eq!(sanitize_service(Some("a b{c}")), "abc");
        assert_eq!(sanitize_service(Some(&"x".repeat(50))).len(), 32);
    }

    #[test]
    fn exit_codes_match_pam_helper() {
        assert_eq!(Outcome::Success.exit_code(), 0);
        assert_eq!(Outcome::Skipped.exit_code(), 10);
        assert_eq!(Outcome::NoMatch.exit_code(), 11);
        assert_eq!(Outcome::Cancelled.exit_code(), 11);
        assert_eq!(Outcome::Busy.exit_code(), 12);
        assert_eq!(Outcome::Error.exit_code(), 12);
        assert_eq!(Outcome::TooDark.exit_code(), 13);
    }

    #[test]
    fn rate_limiter_window() {
        let mut rl = RateLimiter::new(2, Duration::from_secs(60));
        let t0 = Instant::now();
        assert!(rl.allowed(1000, t0));
        rl.record(1000, Outcome::NoMatch, t0);
        rl.record(1000, Outcome::Cancelled, t0);
        assert!(rl.allowed(1000, t0));
        rl.record(1000, Outcome::TooDark, t0);
        assert!(!rl.allowed(1000, t0));
        // Other users are unaffected.
        assert!(rl.allowed(1001, t0));
        // Failures age out of the window.
        assert!(rl.allowed(1000, t0 + Duration::from_secs(61)));
        // A success clears the record.
        rl.record(1000, Outcome::NoMatch, t0);
        rl.record(1000, Outcome::NoMatch, t0);
        assert!(!rl.allowed(1000, t0));
        rl.record(1000, Outcome::Success, t0);
        assert!(rl.allowed(1000, t0));
    }
}
