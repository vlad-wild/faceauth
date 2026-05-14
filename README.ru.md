# FaceAuth — система распознавания лиц на Rust

*⚠️ **Примечание о коде, сгенерированном с помощью искусственного интеллекта**: Эта проектная документация и код были сгенерированы с помощью искусственного интеллекта. Тщательно просмотрите их перед использованием в рабочей среде, поскольку код, сгенерированный с помощью искусственного интеллекта, может содержать ошибки, уязвимости в системе безопасности или неполные реализации. Тщательно протестируйте и рассмотрите возможность аудита безопасности.*

FaceAuth — это система аутентификации по лицу для Linux, написанная на Rust, вдохновленная проектом [Howdy](https://github.com/boltgolt/howdy). Она предоставляет PAM-модуль для входа в систему, разблокировки экрана и авторизации sudo с использованием камеры (включая ИК-камеры).

## Особенности

- **Захват видео** через OpenCV с поддержкой ручного управления экспозицией и поворота.
- **Детекция лиц** с использованием каскадов Хаара (OpenCV) или ONNX-моделей (например, UltraLightFaceDetector).
- **Извлечение эмбеддингов** с помощью нейросетевых моделей (MobileFaceNet, ArcFace и др.).
- **Сравнение с эталонными моделями** с пороговой проверкой.
- **Интеграция с PAM** через демон `faceauth-auth` (используется `pam_exec`).
- **CLI для управления** моделями пользователей, конфигурацией и тестирования.
- **Поддержка Quickshell** — виджет состояния и уведомления.
- **Поддержка IR-камер** — работа с инфракрасными камерами (например, Windows Hello) через настройку экспозиции и выбор устройства.

## Поддержка IR-камер и вход в темноте

Обычная RGB-веб-камера в полной темноте почти не видит лицо (нет видимого света). Чтобы вход **в темноте** был возможен, нужна **инфракрасная** камера с ИК-подсветкой (как у многих ноутбуков с Windows Hello): тогда кадр монохромный, но лицо подсвечено в ИК.

1. Найдите узел IR в V4L2:
   ```bash
   v4l2-ctl --list-devices
   ```

2. В **`/etc/faceauth/config.toml`** укажите это устройство и включите режим ИК:
   ```toml
   [video]
   device_path = "/dev/video3"
   ir_mode = true
   exposure = 100   # при необходимости подберите вручную под вашу IR-камеру
   ```

3. **Модель нужно снимать с той же IR-камеры**, с которой потом работает `faceauth-auth` (те же условия ИК и ракурс). Пример:
   ```bash
   sudo faceauth add -u "$USER" -s 10 -d 3 --ir
   ```
   Флаг `--ir` эквивалентен `ir_mode = true` в конфиге на время команды (удобно, если `faceauth.toml` в текущей папке без этого поля).

Что даёт **`ir_mode`** в коде: отключается отсев кадров по «темноте» (для ИК метрика яркости часто выглядит как «слишком тёмно»), а для Haar используется более мягкий `minNeighbors`, чтобы чаще находить лицо на монохроме.

Модель **MobileFaceNet** обучалась в основном на RGB; на ИК качество сравнения может быть чуть хуже. При частых отказах можно слегка поднять **`distance_threshold`** в `[recognition]` (осознанно, с учётом риска).

## Архитектура

```
┌─────────────────┐
│   PAM (pam_exec)│
└────────┬────────┘
         │ вызывает
┌────────▼────────┐
│ faceauth-auth   │
│ (бинарник Rust) │
└────────┬────────┘
         │ использует
┌────────▼─────────────────────────┐
│ Библиотеки FaceAuth              │
│  • camera — захват кадров        │
│  • detection — обнаружение лиц   │
│  • recognition — эмбеддинги      │
│  • database — хранение моделей   │
│  • config — конфигурация TOML    │
└──────────────────────────────────┘
```

## Установка (для Arch Linux)

FaceAuth можно установить двумя способами: сборка из исходников или через PKGBUILD (AUR).

### Зависимости

- **opencv** – библиотеки компьютерного зрения
- **opencv-data** – каскады Хаара для детекции лиц
- **v4l-utils** – утилиты для работы с видеокамерами
- **pam** – библиотека PAM (обычно уже установлена)
- **rust** и **cargo** – для сборки из исходников

### Способ 1: Сборка из исходников

1. Установите зависимости:
   ```bash
   sudo pacman -S opencv opencv-data v4l-utils rust
   ```

2. Клонируйте репозиторий и соберите проект:
   ```bash
   git clone https://github.com/yourusername/faceauth.git
   cd faceauth
   cargo build --release
   ```

3. Установите системные файлы:
   ```bash
   sudo cp target/release/faceauth /usr/local/bin/
   sudo cp target/release/faceauth-auth /usr/local/bin/
   sudo mkdir -p /etc/faceauth
   sudo cp faceauth.toml /etc/faceauth/config.toml
   ```

### Способ 2: Установка через PKGBUILD (AUR)

Если вы используете Arch Linux, можете собрать пакет с помощью предоставленного `PKGBUILD`.

1. Перейдите в директорию с PKGBUILD:
   ```bash
   cd faceauth
   ```

2. Соберите пакет:
   ```bash
   makepkg -si
   ```

   Или установите через AUR-хелпер (если пакет загружен в AUR):
   ```bash
   yay -S faceauth
   ```

   Пакет установит:
   - `/usr/bin/faceauth` – CLI утилита управления
   - `/usr/bin/faceauth-auth` – демон для PAM
   - `/usr/bin/faceauth-ui` – окно записи лица (опционально)
   - `/etc/faceauth/config.toml` – конфигурация по умолчанию
   - `/usr/share/doc/faceauth/README.md` – документация
   - `/usr/share/doc/faceauth/pam-example` – пример PAM-конфигурации
   - `/usr/lib/systemd/system/faceauth.service` – systemd-сервис (опционально)

3. После установки настройте PAM (см. ниже).

### Настройка PAM

Добавьте в `/etc/pam.d/system-auth` (или `/etc/pam.d/sudo`) строку:

```
auth [success=2 default=ignore] pam_exec.so quiet /usr/local/bin/faceauth-auth
auth [success=1 default=bad] pam_unix.so try_first_pass nullok
```

Имя учётной записи берётся из переменной окружения **`PAM_USER`**, которую выставляет `pam_exec` (в файлах pam.d **не** работает подстановка `$USER` из shell). При необходимости можно явно указать пользователя: `... faceauth-auth -u имя` (только для одного пользователя на машине).

### Добавление модели лица

```bash
sudo faceauth add --user $USER --samples 5
# IR / темнота: та же камера, что в /etc/faceauth/config.toml (device_path + ir_mode), например:
sudo faceauth add -u "$USER" -s 10 -d 3 --ir
```

### Несколько обликов и дополнение модели

Хранится **основной** набор эмбеддингов и именованные **варианты** (например `glasses`). При проверке используется лучшее совпадение среди всех наборов.

| Действие | Пример |
|----------|--------|
| Полная замена модели (сброс вариантов) | `faceauth add -u USER -s 10` |
| Дописать снимки в основной набор | `faceauth add -u USER -s 8 --append` |
| Вариант «в очках» (заменить набор с этим именем) | `faceauth add -u USER -s 10 --variant glasses` |
| Дописать в существующий вариант | `faceauth add -u USER -s 5 --variant glasses --append` |

В **`faceauth-ui`** те же режимы выбираются радиокнопками.

### Графический интерфейс (`faceauth-ui`)

Окно для записи лица без CLI: предпросмотр камеры, прогресс съёмки, те же настройки, что и у `faceauth add`.

```bash
cargo run --release --bin faceauth-ui
# или после установки:
faceauth-ui
```

Конфиг подхватывается в порядке: `./faceauth.toml` → `~/.config/faceauth/config.toml` → `/etc/faceauth/config.toml`. Относительные пути к ONNX в конфиге разрешаются относительно каталога файла конфигурации.

## Конфигурация

Пример конфигурационного файла `/etc/faceauth/config.toml`:

```toml
[video]
device_path = "/dev/video0"
timeout = 4
dark_threshold = 50.0
certainty = 3.5
max_height = 320.0
rotate = 0
exposure = -1
ir_mode = false

[detection]
model_path = "models/ultra_light_640.onnx"
use_cnn = false
confidence_threshold = 0.7

[recognition]
model_path = "models/mobilefacenet.onnx"
embedding_size = 128
distance_threshold = 0.6

[debug]
end_report = false
save_failed = false
save_successful = false
```

## Интеграция с Quickshell

FaceAuth может предоставлять статус аутентификации через D-Bus или Unix-сокет для отображения в Quickshell.

### Виджет состояния

Пример QML-виджета для Quickshell:

```qml
import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    property bool faceAuthReady: false

    Timer {
        interval: 5000
        running: true
        repeat: true
        onTriggered: {
            // Проверка статуса демона FaceAuth
            faceAuthReady = checkFaceAuthStatus()
        }
    }

    Image {
        source: faceAuthReady ? "face-ok.svg" : "face-off.svg"
        width: 24
        height: 24
    }
}
```

### Уведомления

При неудачной аутентификации FaceAuth может отправлять уведомления через `libnotify`.

## Безопасность

- Все данные (эмбеддинги) хранятся локально в зашифрованном домашнем каталоге.
- Аутентификация по лицу является дополнительным фактором, а не заменой пароля.
- Рекомендуется использовать ИК-камеру для защиты от фотографий.

## Разработка

### Структура проекта

```
src/
├── camera.rs      # Захват видео, обработка кадров
├── config.rs      # Загрузка/сохранение конфигурации
├── detection.rs   # Детекция лиц (Haar / ONNX)
├── enroll.rs      # Запись модели (CLI + UI)
├── recognition.rs # Извлечение эмбеддингов
├── database.rs    # Хранение и сравнение моделей
├── main.rs        # CLI утилита
└── bin/
    ├── auth.rs    # Демон для PAM
    └── faceauth-ui.rs  # Окно записи лица
```

### Добавление новой модели распознавания

1. Поместите файлы ONNX-модели в `models/`.
2. Обновите `config.toml` с путями к модели.
3. Реализуйте соответствующий препроцессинг в `recognition.rs`.

## Лицензия

MIT