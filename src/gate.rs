//! Checks run before the camera is opened, shared by `faceauth-auth` and `faceauthd`.

use anyhow::Result;
use std::path::Path;

use crate::config::Config;
use crate::database::{DISABLED_FLAG, FaceModel, load_user_model};
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
}

impl SkipReason {
    /// Stable identifier used in logs and the daemon protocol.
    pub fn key(self) -> &'static str {
        match self {
            SkipReason::Disabled => "disabled",
            SkipReason::Remote => "remote_session",
            SkipReason::LidClosed => "lid_closed",
            SkipReason::NoModel => "no_model",
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
    match load_user_model(user)? {
        Some(model) => Ok(Gate::Proceed(model)),
        None => Ok(Gate::Skip(SkipReason::NoModel)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_keys_are_stable() {
        assert_eq!(SkipReason::Disabled.key(), "disabled");
        assert_eq!(SkipReason::Remote.key(), "remote_session");
        assert_eq!(SkipReason::LidClosed.key(), "lid_closed");
        assert_eq!(SkipReason::NoModel.key(), "no_model");
    }
}
