//! Privilege checks for commands that read or modify the root-owned model store.

use anyhow::{Result, bail};
use std::path::PathBuf;

/// Account information from the passwd database.
#[derive(Debug, Clone)]
pub struct UserInfo {
    pub uid: u32,
    pub home: PathBuf,
}

/// Linux-style account name: `[a-z_][a-z0-9_-]{0,31}` with an optional trailing `$`.
/// Also keeps `/`, `.` and friends out of model file paths.
pub fn validate_username(name: &str) -> Result<()> {
    let body = name.strip_suffix('$').unwrap_or(name);
    let mut chars = body.chars();
    let valid = match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c == '_' => {
            chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
        }
        _ => false,
    };
    if !valid || body.len() > 32 {
        bail!("Invalid user name {name:?}");
    }
    Ok(())
}

#[cfg(unix)]
pub fn is_root() -> bool {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() == 0 }
}

#[cfg(not(unix))]
pub fn is_root() -> bool {
    false
}

pub fn require_root() -> Result<()> {
    if !is_root() {
        bail!("This command needs root: models live in a root-only store. Run it with sudo.");
    }
    Ok(())
}

/// Look up an account in the passwd database.
#[cfg(unix)]
pub fn lookup_user(name: &str) -> Result<UserInfo> {
    use std::ffi::{CStr, CString};

    validate_username(name)?;
    let cname = CString::new(name)?;
    // SAFETY: `getpwnam` returns NULL or a pointer to static storage valid until the
    // next passwd call; fields are copied out before returning.
    unsafe {
        let pw = libc::getpwnam(cname.as_ptr());
        if pw.is_null() {
            bail!("User {name} not found in passwd database");
        }
        let home = CStr::from_ptr((*pw).pw_dir).to_string_lossy().into_owned();
        Ok(UserInfo {
            uid: (*pw).pw_uid,
            home: PathBuf::from(home),
        })
    }
}

#[cfg(not(unix))]
pub fn lookup_user(name: &str) -> Result<UserInfo> {
    validate_username(name)?;
    bail!("User lookup is only supported on Unix")
}

/// The unprivileged user who asked for this command through `pkexec`, if any.
/// `pkexec` sets `PKEXEC_UID` and clears the rest of the environment.
pub fn pkexec_caller() -> Option<u32> {
    std::env::var("PKEXEC_UID").ok()?.trim().parse().ok()
}

/// Commands launched through `pkexec` (polkit `auth_self`) may only touch the
/// caller's own model; root and `sudo` users (already administrators) may touch any.
pub fn authorize_for_user(target: &str) -> Result<()> {
    require_root()?;
    if let Some(caller) = pkexec_caller()
        && caller != 0
        && lookup_user(target)?.uid != caller
    {
        bail!("Via pkexec you can only manage your own face model");
    }
    Ok(())
}

/// For commands that affect every user (disable/enable, list all, migrate others).
pub fn authorize_global() -> Result<()> {
    require_root()?;
    if matches!(pkexec_caller(), Some(uid) if uid != 0) {
        bail!("This command is not available via pkexec; use sudo");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usernames() {
        for ok in ["vlad", "_apt", "user-1", "a_b", "machine$", "x"] {
            assert!(validate_username(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "Root",
            "1abc",
            "../etc",
            "a/b",
            "a.b",
            "a b",
            "$",
            &"a".repeat(33),
        ] {
            assert!(validate_username(bad).is_err(), "{bad}");
        }
    }
}
