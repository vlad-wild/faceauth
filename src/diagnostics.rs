//! Result types of `faceauth doctor` (OpenCV-free: the GUI parses `doctor --json`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl Check {
    pub fn new(name: &str, status: Status, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status,
            detail: detail.into(),
            hint: None,
        }
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

/// Suggested PAM line (see `pam/faceauth`).
pub const PAM_LINE: &str = "auth sufficient pam_exec.so quiet /usr/bin/faceauth-auth";

pub fn print(checks: &[Check]) {
    for c in checks {
        let mark = match c.status {
            Status::Ok => "✔",
            Status::Warn => "⚠",
            Status::Fail => "✘",
        };
        println!("{mark} {:<14} {}", c.name, c.detail);
        if let Some(h) = &c.hint {
            println!("  → {h}");
        }
    }
}
