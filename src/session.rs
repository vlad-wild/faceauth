//! Session context checks used before touching the camera in `faceauth-auth`.

use std::path::Path;

/// True when PAM describes a remote session: a non-local `PAM_RHOST`, or a tty
/// set by sshd (`PAM_TTY=ssh`). Face auth there would let whoever sits in front
/// of the laptop approve a command typed remotely.
pub fn is_remote_session(rhost: Option<&str>, tty: Option<&str>) -> bool {
    let rhost = rhost.map(str::trim).unwrap_or("");
    let local_host = matches!(rhost, "" | "localhost" | "127.0.0.1" | "::1");
    let ssh_tty = tty.map(str::trim).is_some_and(|t| t.starts_with("ssh"));
    !local_host || ssh_tty
}

/// Read the PAM environment passed by `pam_exec`.
pub fn is_remote_pam_session() -> bool {
    let rhost = std::env::var("PAM_RHOST").ok();
    let tty = std::env::var("PAM_TTY").ok();
    is_remote_session(rhost.as_deref(), tty.as_deref())
}

/// True when every ACPI lid reports `closed`. Machines without a lid report false.
pub fn lid_closed() -> bool {
    lid_closed_in(Path::new("/proc/acpi/button/lid"))
}

fn lid_closed_in(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut seen = false;
    for entry in entries.flatten() {
        let Ok(state) = std::fs::read_to_string(entry.path().join("state")) else {
            continue;
        };
        seen = true;
        if !state.contains("closed") {
            return false;
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_detection() {
        assert!(!is_remote_session(None, None));
        assert!(!is_remote_session(Some(""), Some("/dev/pts/3")));
        assert!(!is_remote_session(Some("localhost"), Some(":0")));
        assert!(!is_remote_session(Some("::1"), None));
        assert!(is_remote_session(Some("192.168.1.5"), Some("/dev/pts/1")));
        assert!(is_remote_session(None, Some("ssh")));
        assert!(is_remote_session(Some(""), Some("ssh ")));
    }

    #[test]
    fn lid_state() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!lid_closed_in(&tmp.path().join("missing")));
        let lid = tmp.path().join("LID0");
        std::fs::create_dir(&lid).unwrap();
        std::fs::write(lid.join("state"), "state:      open\n").unwrap();
        assert!(!lid_closed_in(tmp.path()));
        std::fs::write(lid.join("state"), "state:      closed\n").unwrap();
        assert!(lid_closed_in(tmp.path()));
    }
}
