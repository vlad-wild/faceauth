//! `faceauth-auth`: PAM helper run by `pam_exec`.
//!
//! As root (sudo, polkit, login managers) it opens the camera itself. Run by an
//! unprivileged screen locker it cannot read the root-only model store, so it asks
//! `faceauthd` to verify the calling user instead, and only when PAM is
//! authenticating that same user.
//!
//! Exit codes (anything but 0 lets PAM fall through to the password):
//! * 0  — face matched
//! * 10 — skipped: face auth disabled, remote session, lid closed, or no model enrolled
//! * 11 — no match before the timeout
//! * 12 — setup error (config, camera, models, untrusted model file)
//! * 13 — every frame was too dark

use anyhow::{Context, Result};
use clap::Parser;
use std::env;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Duration;

use faceauth::authenticate::{AuthOutcome, authenticate};
use faceauth::config::{Config, SYSTEM_CONFIG_PATH};
use faceauth::daemon::{self, SOCKET_PATH};
use faceauth::gate::{Gate, pre_auth_checks};
use faceauth::logger;
use faceauth::pipeline::Pipeline;
use faceauth::privilege::{is_root, username_for_uid};
use faceauth::session;

const EXIT_SUCCESS: i32 = 0;
const EXIT_SKIPPED: i32 = 10;
const EXIT_NO_MATCH: i32 = 11;
const EXIT_SETUP_ERROR: i32 = 12;
const EXIT_TOO_DARK: i32 = 13;

#[derive(Parser)]
#[command(name = "faceauth-auth")]
#[command(about = "Face authentication helper for PAM (pam_exec)")]
struct Args {
    /// Username to authenticate (default: PAM_USER)
    #[arg(short, long)]
    user: Option<String>,

    /// Configuration file path
    #[arg(short, long, default_value = SYSTEM_CONFIG_PATH)]
    config: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Resolve the account being authenticated. PAM does not set `USER` for `pam_exec` children;
/// Linux-PAM typically sets `PAM_USER`. (`-u` on the command line always wins.)
fn resolve_pam_username(cli_user: Option<String>) -> Result<String, String> {
    if let Some(u) = cli_user {
        let u = u.trim().to_string();
        if !u.is_empty() {
            return Ok(u);
        }
    }
    for key in ["PAM_USER", "USER", "LOGNAME"] {
        if let Ok(val) = env::var(key) {
            let val = val.trim().to_string();
            if !val.is_empty() {
                return Ok(val);
            }
        }
    }
    Err(
        "No username: pass -u on the pam_exec line, or ensure PAM_USER is set (standard for pam_exec). \
         Note: in /etc/pam.d/* the string $USER is NOT expanded by the shell — use -u <name> or omit -u and rely on PAM_USER."
            .to_string(),
    )
}

fn main() {
    logger::init_from_env();
    let args = Args::parse();
    let code = match run(args) {
        Ok(code) => code,
        Err(e) => {
            log::error!("{e:#}");
            EXIT_SETUP_ERROR
        }
    };
    std::process::exit(code);
}

/// Unprivileged mode: let `faceauthd` verify the calling user.
fn run_via_daemon(user: &str) -> i32 {
    if session::is_remote_pam_session() {
        log::info!("Remote session (PAM_RHOST/PAM_TTY); skipping face authentication");
        return EXIT_SKIPPED;
    }
    // The daemon verifies whoever owns this process. Only accept that result
    // when PAM is authenticating the same account.
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    match username_for_uid(uid) {
        Ok(own) if own == user => {}
        Ok(own) => {
            log::warn!("PAM user {user} is not the calling user {own}; skipping");
            return EXIT_SKIPPED;
        }
        Err(e) => {
            log::error!("{e:#}");
            return EXIT_SETUP_ERROR;
        }
    }
    match daemon::verify(
        Path::new(SOCKET_PATH),
        env::var("PAM_SERVICE").ok().as_deref(),
        |_| {},
    ) {
        Ok((outcome, reason)) => {
            log::info!(
                "faceauthd result for {user}: {outcome:?}{}",
                reason.map(|r| format!(" ({r})")).unwrap_or_default()
            );
            outcome.exit_code()
        }
        Err(e) => {
            log::error!("{e:#}");
            EXIT_SETUP_ERROR
        }
    }
}

fn run(args: Args) -> Result<i32> {
    if args.verbose {
        log::info!("Verbose output enabled");
    }
    let user = match resolve_pam_username(args.user) {
        Ok(u) => u,
        Err(msg) => {
            log::error!("{}", msg);
            return Ok(EXIT_SKIPPED);
        }
    };

    if !is_root() {
        return Ok(run_via_daemon(&user));
    }

    let config = Config::load_resolved(&args.config)
        .with_context(|| format!("Failed to load config {}", args.config.display()))?;

    let model = match pre_auth_checks(&config, &user, session::is_remote_pam_session())? {
        Gate::Proceed(model) => model,
        Gate::Skip(reason) => {
            log::info!("Skipping face authentication for {user}: {}", reason.key());
            return Ok(EXIT_SKIPPED);
        }
    };

    log::info!("Starting face authentication for user {}", user);
    if config.video.ir_mode {
        log::info!("IR mode: darkness filter disabled; use the same IR device for enrollment");
    }
    let mut pipeline = Pipeline::open(&config, &config.video.device_path)?;
    let timeout = Duration::from_secs(config.video.timeout as u64);
    let report = authenticate(&mut pipeline, &config, &model, timeout, |_| {
        ControlFlow::Continue(())
    })?;

    if config.debug.end_report {
        log::info!("Report for {}: {}", user, report.summary());
    }
    Ok(match report.outcome {
        AuthOutcome::Success => {
            log::info!(
                "Authentication successful for {} (score {:.4})",
                user,
                report.best_score
            );
            EXIT_SUCCESS
        }
        AuthOutcome::NoMatch => {
            log::warn!(
                "Authentication failed for {} (best score {:.4}, threshold {:.4})",
                user,
                report.best_score,
                config.recognition.distance_threshold
            );
            EXIT_NO_MATCH
        }
        AuthOutcome::AllDark => {
            log::error!(
                "Authentication failed for {}: all frames were too dark",
                user
            );
            EXIT_TOO_DARK
        }
        AuthOutcome::Cancelled => {
            log::warn!("Authentication for {} was cancelled", user);
            EXIT_NO_MATCH
        }
        AuthOutcome::NoFrames => {
            log::error!(
                "Authentication failed for {}: camera delivered no frames",
                user
            );
            EXIT_SETUP_ERROR
        }
    })
}
