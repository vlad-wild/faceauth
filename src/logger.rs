use std::fs::{OpenOptions, File};
use std::io::Write;
use std::sync::Mutex;
use chrono::Local;

pub struct FaceAuthLogger {
    file_logger: Mutex<Option<File>>,
}

impl FaceAuthLogger {
    pub fn new(log_file_path: Option<&str>) -> Self {
        let file_logger = if let Some(path) = log_file_path {
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                Ok(file) => Mutex::new(Some(file)),
                Err(e) => {
                    eprintln!("Failed to open log file {}: {}", path, e);
                    Mutex::new(None)
                }
            }
        } else {
            Mutex::new(None)
        };

        Self {
            file_logger,
        }
    }

    fn log(&self, level: &str, message: &str) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let log_line = format!("[{}] [{}] {}\n", timestamp, level, message);

        // Пишем в файл
        if let Ok(mut file_guard) = self.file_logger.lock() {
            if let Some(file) = file_guard.as_mut() {
                let _ = file.write_all(log_line.as_bytes());
                let _ = file.flush();
            }
        }

        // Пишем в stderr (PAM его перехватывает в journalctl)
        eprint!("{}", log_line);
    }

    pub fn info(&self, message: &str) {
        self.log("INFO", message);
    }

    pub fn warning(&self, message: &str) {
        self.log("WARNING", message);
    }

    pub fn error(&self, message: &str) {
        self.log("ERROR", message);
    }
}

lazy_static::lazy_static! {
    static ref LOGGER: FaceAuthLogger = FaceAuthLogger::new(Some("/var/log/faceauth.log"));
}

pub fn log_info(msg: &str) {
    LOGGER.info(msg);
}

pub fn log_warning(msg: &str) {
    LOGGER.warning(msg);
}

pub fn log_error(msg: &str) {
    LOGGER.error(msg);
}