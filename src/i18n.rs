//! Minimal ru/en message table for the GUI and CLI hints (no extra dependency).
//!
//! The language comes from `FACEAUTH_LANG`, then `LC_ALL` / `LC_MESSAGES` / `LANG`.
//! `{0}`, `{1}` … in messages are filled by [`tf`].

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Ru,
}

pub fn lang() -> Lang {
    static LANG: OnceLock<Lang> = OnceLock::new();
    *LANG.get_or_init(|| {
        let var = ["FACEAUTH_LANG", "LC_ALL", "LC_MESSAGES", "LANG"]
            .iter()
            .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
            .unwrap_or_default();
        lang_from(&var)
    })
}

fn lang_from(locale: &str) -> Lang {
    if locale.to_ascii_lowercase().starts_with("ru") {
        Lang::Ru
    } else {
        Lang::En
    }
}

/// (key, English, Russian)
const MESSAGES: &[(&str, &str, &str)] = &[
    // Frame verdicts
    ("verdict.ok", "Face OK", "Лицо в кадре"),
    (
        "verdict.no_face",
        "No face — look at the camera",
        "Лицо не найдено — посмотрите в камеру",
    ),
    ("verdict.too_dark", "Too dark", "Слишком темно"),
    (
        "verdict.low_confidence",
        "Face unclear — look at the camera",
        "Лицо нечёткое — посмотрите в камеру",
    ),
    ("verdict.too_small", "Move closer", "Подвиньтесь ближе"),
    (
        "verdict.too_large",
        "Move back a little",
        "Отодвиньтесь немного",
    ),
    ("verdict.blurry", "Hold still", "Не двигайтесь"),
    (
        "verdict.look_straight",
        "Look straight at the camera",
        "Смотрите прямо в камеру",
    ),
    (
        "verdict.not_live",
        "Face rejected by the IR liveness check",
        "Лицо не прошло IR-проверку живости",
    ),
    // Pose hints
    (
        "hint.turn_left",
        "Turn your head slightly left",
        "Слегка поверните голову влево",
    ),
    (
        "hint.turn_right",
        "Turn your head slightly right",
        "Слегка поверните голову вправо",
    ),
    (
        "hint.look_straight",
        "Look straight at the camera",
        "Смотрите прямо в камеру",
    ),
    (
        "hint.duplicate",
        "Move a little — this pose is already captured",
        "Немного смените положение — такой кадр уже есть",
    ),
    // Window
    ("app.title", "Faceauth", "Faceauth"),
    ("tab.enroll", "Enroll", "Запись лица"),
    ("tab.test", "Test", "Проверка"),
    ("tab.models", "Models", "Модели"),
    ("tab.setup", "Setup", "Настройка"),
    ("field.user", "User:", "Пользователь:"),
    ("field.camera", "Camera:", "Камера:"),
    ("field.ir", "IR camera (ir_mode)", "ИК-камера (ir_mode)"),
    ("field.samples", "Samples:", "Снимков:"),
    ("field.label", "Label (optional):", "Метка (необязательно):"),
    ("field.target", "Save to:", "Сохранить в:"),
    (
        "field.new_variant",
        "New variant name:",
        "Имя нового варианта:",
    ),
    ("config.source", "Config: {0}", "Конфиг: {0}"),
    (
        "config.defaults",
        "Config: defaults (no faceauth.toml found)",
        "Конфиг: значения по умолчанию (faceauth.toml не найден)",
    ),
    ("camera.start", "Turn camera on", "Включить камеру"),
    ("camera.stop", "Turn camera off", "Выключить камеру"),
    ("camera.opening", "Opening camera…", "Открываю камеру…"),
    ("camera.off", "Camera is off", "Камера выключена"),
    ("camera.none", "No cameras found", "Камеры не найдены"),
    // Enroll tab
    (
        "enroll.mode_new",
        "New model (replace)",
        "Новая модель (заменить)",
    ),
    (
        "enroll.mode_add",
        "Add to the existing model",
        "Дополнить существующую",
    ),
    ("enroll.primary", "Main set", "Основной набор"),
    ("enroll.new_variant", "New variant…", "Новый вариант…"),
    ("enroll.start", "Start recording", "Начать запись"),
    ("enroll.progress", "Captured {0}/{1}", "Снято {0}/{1}"),
    (
        "enroll.saving",
        "Saving (administrator password may be requested)…",
        "Сохраняю (может потребоваться пароль)…",
    ),
    (
        "enroll.saved",
        "Saved {0} samples",
        "Сохранено снимков: {0}",
    ),
    (
        "enroll.none",
        "No usable face samples were captured",
        "Не удалось снять ни одного подходящего кадра",
    ),
    (
        "enroll.need_variant",
        "Enter a variant name (e.g. glasses)",
        "Введите имя варианта (например, glasses)",
    ),
    // Test tab
    ("test.start", "Start live test", "Начать проверку"),
    ("test.stop", "Stop", "Остановить"),
    (
        "test.score",
        "Score {0} (threshold {1}) — lower is better",
        "Оценка {0} (порог {1}) — чем меньше, тем лучше",
    ),
    (
        "test.streak",
        "Matching frames in a row: {0}/{1}",
        "Совпадений подряд: {0}/{1}",
    ),
    ("test.pass", "✔ Would unlock", "✔ Вход был бы выполнен"),
    ("test.waiting", "Waiting for a face…", "Жду лицо в кадре…"),
    // Models tab
    ("models.load", "Load model", "Загрузить модель"),
    (
        "models.none",
        "Nothing enrolled for this user",
        "Для пользователя ничего не записано",
    ),
    (
        "models.primary",
        "Main set: {0} samples",
        "Основной набор: {0} снимков",
    ),
    (
        "models.variant",
        "Variant “{0}”: {1} samples",
        "Вариант «{0}»: {1} снимков",
    ),
    ("models.updated", "Updated: {0}", "Обновлено: {0}"),
    ("models.delete", "Delete", "Удалить"),
    ("models.rename", "Rename", "Переименовать"),
    (
        "models.clear",
        "Delete the whole model",
        "Удалить всю модель",
    ),
    (
        "models.disable_hint",
        "Enable / disable face login for everyone: sudo faceauth disable | enable",
        "Включить / выключить вход по лицу для всех: sudo faceauth disable | enable",
    ),
    ("models.done", "Done", "Готово"),
    // Setup tab
    ("setup.run", "Run diagnostics", "Запустить диагностику"),
    (
        "setup.steps",
        "1. Diagnostics  2. Enroll  3. Test  4. Add the PAM line",
        "1. Диагностика  2. Запись лица  3. Проверка  4. Строка для PAM",
    ),
    (
        "setup.pam",
        "Add this line at the top of /etc/pam.d/sudo (and other services):",
        "Добавьте эту строку в начало /etc/pam.d/sudo (и других сервисов):",
    ),
    ("setup.copy", "Copy", "Скопировать"),
    (
        "setup.copied",
        "Copied to clipboard",
        "Скопировано в буфер обмена",
    ),
    // Generic
    ("status.error", "Error: {0}", "Ошибка: {0}"),
    ("status.busy", "Busy…", "Выполняется…"),
    (
        "status.need_user",
        "Enter a user name",
        "Введите имя пользователя",
    ),
];

/// Translated message for `key` (the key itself if unknown).
pub fn t(key: &'static str) -> &'static str {
    translate(key, lang())
}

fn translate(key: &'static str, lang: Lang) -> &'static str {
    MESSAGES
        .iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, en, ru)| if lang == Lang::Ru { *ru } else { *en })
        .unwrap_or(key)
}

/// [`t`] with `{0}`, `{1}` … replaced by `args`.
pub fn tf(key: &'static str, args: &[&dyn std::fmt::Display]) -> String {
    let mut s = t(key).to_string();
    for (i, a) in args.iter().enumerate() {
        s = s.replace(&format!("{{{i}}}"), &a.to_string());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_unique_and_translated() {
        let mut seen = std::collections::HashSet::new();
        for (k, en, ru) in MESSAGES {
            assert!(seen.insert(*k), "duplicate key {k}");
            assert!(!en.is_empty() && !ru.is_empty(), "{k}");
            let placeholders = |s: &str| (0..4).filter(|i| s.contains(&format!("{{{i}}}"))).count();
            assert_eq!(
                placeholders(en),
                placeholders(ru),
                "placeholders differ in {k}"
            );
        }
    }

    #[test]
    fn lookup() {
        assert_eq!(translate("tab.test", Lang::Ru), "Проверка");
        assert_eq!(translate("tab.test", Lang::En), "Test");
        assert_eq!(translate("missing.key", Lang::Ru), "missing.key");
        assert_eq!(lang_from("ru_RU.UTF-8"), Lang::Ru);
        assert_eq!(lang_from("en_US.UTF-8"), Lang::En);
        assert_eq!(lang_from(""), Lang::En);
    }

    #[test]
    fn formatting() {
        assert!(tf("enroll.progress", &[&3, &9]).contains("3/9"));
    }
}
