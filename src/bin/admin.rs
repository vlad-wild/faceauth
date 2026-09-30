//! `faceauth-admin`: system-wide faceauth settings for graphical front ends.
//!
//! Installed in `/usr/lib/faceauth/` and meant to be run through `pkexec`, which
//! uses its own polkit action (`org.faceauth.configure`, administrator password).
//! That keeps it apart from `pkexec faceauth …` (`org.faceauth.manage-own-model`,
//! the user's own password), which may only touch the caller's face model.
//!
//! Output is JSON on stdout; errors go to stderr with a non-zero exit code.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use faceauth::config::{Config, SYSTEM_CONFIG_PATH};
use faceauth::database::DISABLED_FLAG;
use faceauth::privilege::require_root;
use faceauth::settings;

#[derive(Parser)]
#[command(name = "faceauth-admin")]
#[command(about = "Change system-wide faceauth settings (run through pkexec)")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the changeable settings and whether face authentication is disabled
    Get,
    /// Set one setting (see `get` for the keys)
    Set { key: String, value: String },
    /// Disable face authentication for everyone
    Disable,
    /// Re-enable face authentication
    Enable,
}

fn main() {
    faceauth::logger::init_from_env();
    if let Err(e) = run(Args::parse()) {
        eprintln!("{e:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Get => {
            let cfg = Config::load_resolved(Path::new(SYSTEM_CONFIG_PATH))?;
            let mut out = settings::get_all(&cfg)?;
            out.insert("disabled".into(), Path::new(DISABLED_FLAG).exists().into());
            println!("{}", serde_json::Value::Object(out));
        }
        Command::Set { key, value } => {
            require_root()?;
            let path = Path::new(SYSTEM_CONFIG_PATH);
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("Reading {SYSTEM_CONFIG_PATH}"))?;
            let updated = settings::set_in_toml(&text, &key, &value)?;
            // Write next to the file and rename, so a crash never leaves half a config.
            let tmp = path.with_extension("toml.new");
            std::fs::write(&tmp, &updated)?;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
            std::fs::rename(&tmp, path)?;
            log::info!(
                "{key} set to {value} (by uid {})",
                std::env::var("PKEXEC_UID").unwrap_or_else(|_| "0".into())
            );
            println!("{}", serde_json::json!({ "ok": true, "key": key }));
        }
        Command::Disable | Command::Enable => {
            require_root()?;
            let disable = matches!(args.command, Command::Disable);
            if disable {
                if let Some(dir) = Path::new(DISABLED_FLAG).parent() {
                    std::fs::create_dir_all(dir)?;
                }
                std::fs::write(DISABLED_FLAG, b"")?;
            } else {
                match std::fs::remove_file(DISABLED_FLAG) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
            println!("{}", serde_json::json!({ "ok": true, "disabled": disable }));
        }
    }
    Ok(())
}
