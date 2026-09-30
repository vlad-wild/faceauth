//! Checks run before the camera is opened, shared by `faceauth-auth` and `faceauthd`.

use anyhow::Result;
use std::path::Path;

use crate::config::Config;
use crate::daemon::PASSWORD_ONLY_DIR;
use crate::database::{DISABLED_FLAG, FaceModel, load_user_model};
use crate::privilege::lookup_user;
use crate::session;

/// Why face authentication was not attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Face authentication is disabled for everyone ([`DISABLED_FLAG`]).
    Disabled,
    /// The request comes from a remote (SSH) session.
    Remote,
    /// Every lid is closed, so the camera cannot see anyone.
    LidClosed,
    /// The user has no enrolled model.
    NoModel,
    /// The user asked for the password for a while (`password_only` request).
    PasswordOnly,
}

impl SkipReason {
    /// Stable identifier used in logs and the daemon protocol.
    pub fn key(self) -> &'static str {
        match self {
            SkipReason::Disabled => "disabled",
            SkipReason::Remote => "remote_session",
            SkipReason::LidClosed => "lid_closed",
            SkipReason::NoModel => "no_model",
            SkipReason::PasswordOnly => "password_only",
        }
    }
}

/// Result of [`pre_auth_checks`].
pub enum Gate {
    /// Go ahead with this model.
    Proceed(FaceModel),
    /// Do not touch the camera.
    Skip(SkipReason),
}

/// Decide whether to attempt face authentication for `user`.
///
/// `remote` is the caller's view of the session: `faceauth-auth` reads the PAM
/// environment, the daemon only serves local sockets.
pub fn pre_auth_checks(cfg: &Config, user: &str, remote: bool) -> Result<Gate> {
    if Path::new(DISABLED_FLAG).exists() {
        return Ok(Gate::Skip(SkipReason::Disabled));
    }
    if cfg.auth.skip_remote && remote {
        return Ok(Gate::Skip(SkipReason::Remote));
    }
    if cfg.auth.skip_lid_closed && session::lid_closed() {
        return Ok(Gate::Skip(SkipReason::LidClosed));
    }
    if let Ok(info) = lookup_user(user)
        && password_only_until(Path::new(PASSWORD_ONLY_DIR), info.uid, unix_now()).is_some()
    {
        return Ok(Gate::Skip(SkipReason::PasswordOnly));
    }
    match load_user_model(user)? {
        Some(model) => Ok(Gate::Proceed(model)),
        None => Ok(Gate::Skip(SkipReason::NoModel)),
    }
}

/// Current time in Unix seconds.
pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Expiry of an active `password_only` flag for `uid` in `dir`, if any. Only
/// root-owned flag files count (the daemon writes them).
pub fn password_only_until(dir: &Path, uid: u32, now: i64) -> Option<i64> {
    use std::os::unix::fs::MetadataExt;

    let path = dir.join(uid.to_string());
    let meta = std::fs::symlink_metadata(&path).ok()?;
    if !meta.is_file() || (meta.uid() != 0 && crate::privilege::is_root()) {
        return None;
    }
    let until: i64 = std::fs::read_to_string(&path).ok()?.trim().parse().ok()?;
    (until > now).then_some(until)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_only_flag() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(password_only_until(tmp.path(), 1000, 100), None);
        std::fs::write(tmp.path().join("1000"), "150\n").unwrap();
        assert_eq!(password_only_until(tmp.path(), 1000, 100), Some(150));
        assert_eq!(password_only_until(tmp.path(), 1000, 150), None);
        assert_eq!(password_only_until(tmp.path(), 1001, 100), None);
        std::fs::write(tmp.path().join("1000"), "garbage").unwrap();
        assert_eq!(password_only_until(tmp.path(), 1000, 100), None);
    }

    #[test]
    fn skip_keys_are_stable() {
        assert_eq!(SkipReason::Disabled.key(), "disabled");
        assert_eq!(SkipReason::Remote.key(), "remote_session");
        assert_eq!(SkipReason::LidClosed.key(), "lid_closed");
        assert_eq!(SkipReason::NoModel.key(), "no_model");
        assert_eq!(SkipReason::PasswordOnly.key(), "password_only");
    }
}
