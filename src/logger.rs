use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use chrono::Local;
use log::{Level, LevelFilter, Metadata, Record};

/// Dual-output logger: console (with colors) + file (plain text).
pub struct FaceAuthLogger {
    file: Mutex<Option<File>>,
    max_level: LevelFilter,
    use_colors: bool,
}

impl FaceAuthLogger {
    pub fn new(log_file_path: Option<&str>, max_level: LevelFilter, use_colors: bool) -> Self {
        let file = if let Some(path) = log_file_path {
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(file) => Mutex::new(Some(file)),
                Err(e) => {
                    eprintln!("[logger] Failed to open log file {}: {}", path, e);
                    Mutex::new(None)
                }
            }
        } else {
            Mutex::new(None)
        };

        Self {
            file,
            max_level,
            use_colors,
        }
    }

    /// Format one log record for console output (with colors if enabled).
    fn format_console(&self, record: &Record) -> String {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let level = record.level();
        let target = record.target();
        let args = record.args();

        if self.use_colors {
            let level_color = match level {
                Level::Error => "\x1b[31m", // red
                Level::Warn  => "\x1b[33m", // yellow
                Level::Info  => "\x1b[32m", // green
                Level::Debug => "\x1b[36m", // cyan
                Level::Trace => "\x1b[35m", // magenta
            };
            let reset = "\x1b[0m";
            format!(
                "[{}] {}{}{} [{}] {}\n",
                timestamp, level_color, level, reset, target, args
            )
        } else {
            format!(
                "[{}] [{}] [{}] {}\n",
                timestamp, level, target, args
            )
        }
    }

    /// Format one log record for file output (always plain text, no colors).
    fn format_file(&self, record: &Record) -> String {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        format!(
            "[{}] [{}] [{}] {}\n",
            timestamp,
            record.level(),
            record.target(),
            record.args()
        )
    }

    fn write_file(&self, line: &str) {
        if let Ok(mut guard) = self.file.lock() {
            if let Some(file) = guard.as_mut() {
                let _ = file.write_all(line.as_bytes());
                let _ = file.flush();
            }
        }
    }
}

impl log::Log for FaceAuthLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let console_line = self.format_console(record);
        let file_line = self.format_file(record);

        // Output to console: errors and warnings go to stderr, the rest to stdout
        if record.level() <= Level::Warn {
            let _ = io::stderr().write_all(console_line.as_bytes());
        } else {
            let _ = io::stdout().write_all(console_line.as_bytes());
        }

        // Output to file
        self.write_file(&file_line);
    }

    fn flush(&self) {
        if let Ok(mut guard) = self.file.lock() {
            if let Some(file) = guard.as_mut() {
                let _ = file.flush();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Initialization helpers
// ---------------------------------------------------------------------------

static LOGGER_INIT: std::sync::Once = std::sync::Once::new();

fn parse_level(s: &str) -> LevelFilter {
    match s.to_ascii_lowercase().as_str() {
        "error" | "err"     => LevelFilter::Error,
        "warn" | "warning"  => LevelFilter::Warn,
        "info"              => LevelFilter::Info,
        "debug"             => LevelFilter::Debug,
        "trace"             => LevelFilter::Trace,
        _                   => LevelFilter::Info,
    }
}

fn level_from_env() -> LevelFilter {
    if let Ok(val) = std::env::var("FACEAUTH_LOG") {
        return parse_level(&val);
    }
    if let Ok(val) = std::env::var("RUST_LOG") {
        return parse_level(&val);
    }
    LevelFilter::Info
}

/// Initialize the global logger once.
///
/// * `max_level` — maximum level to log (e.g. `LevelFilter::Info`).
/// * `log_file`  — optional path to the log file. If `None`, only console output is used.
pub fn init(max_level: LevelFilter, log_file: Option<&str>) {
    LOGGER_INIT.call_once(|| {
        let use_colors = io::stderr().is_terminal();
        let logger = Box::new(FaceAuthLogger::new(log_file, max_level, use_colors));
        let leaked: &'static FaceAuthLogger = Box::leak(logger);
        let _ = log::set_logger(leaked).map(|()| log::set_max_level(max_level));
    });
}

/// Initialize with `LevelFilter::Info` and default log file `/var/log/faceauth.log`.
pub fn init_default() {
    init(LevelFilter::Info, Some("/var/log/faceauth.log"));
}

/// Initialize reading level from `FACEAUTH_LOG` or `RUST_LOG` env vars.
/// Falls back to `LevelFilter::Info` and default log file.
pub fn init_from_env() {
    let level = level_from_env();
    init(level, Some("/var/log/faceauth.log"));
}

/// Convenience: `init_default()` wrapped in `Result` so it can be used with `?`.
pub fn try_init_default() -> Result<(), log::SetLoggerError> {
    init_default();
    Ok(())
}

/// Try to init from env, returning an error only if the logger was already set.
pub fn try_init_from_env() -> Result<(), log::SetLoggerError> {
    init_from_env();
    Ok(())
}
