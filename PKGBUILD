# Maintainer: Vlad Wild <ya.vlash1@yandex.ru>
pkgname=faceauth
pkgver=0.3.0
pkgrel=1
pkgdesc="Face authentication system for Linux using OpenVINO/ONNX"
arch=('x86_64' 'aarch64')
url="https://github.com/vlad-wild/faceauth"
license=('MIT')
depends=('opencv' 'v4l-utils' 'pam')
optdepends=('openvino: OpenVINO backend for NPU/GPU/CPU acceleration'
            'intel-npu-driver: Intel NPU support (>= 1.30 for Lunar Lake)'
            'linux-enable-ir-emitter: turn on IR emitters that stay dark by default'
            'polkit: enrollment and model management from faceauth-ui')
makedepends=('rust' 'cargo' 'clang' 'llvm' 'pkgconf' 'git')
backup=('etc/faceauth/config.toml')
install=faceauth.install
source=("git+$url.git#tag=v$pkgver")
sha256sums=('SKIP')

options=(!lto)

prepare() {
  cd "$pkgname"
  cargo fetch --locked --target "$CARCH-unknown-linux-gnu"
}

build() {
  cd "$pkgname"
  export RUSTUP_TOOLCHAIN=stable
  export CARGO_TARGET_DIR=target

  export CC=clang
  export CXX=clang++
  export CFLAGS="${CFLAGS}"
  export CXXFLAGS="${CXXFLAGS}"
  export OPENCV_CLANG_RUNTIME="$(clang --version | head -1)"

  cargo build --frozen --release --all-targets
}

check() {
  cd "$pkgname"
  cargo test --frozen
}

package() {
  cd "$pkgname"

  # Binaries
  install -Dm755 "target/release/faceauth" "$pkgdir/usr/bin/faceauth"
  install -Dm755 "target/release/faceauth-auth" "$pkgdir/usr/bin/faceauth-auth"
  install -Dm755 "target/release/faceauth-ui" "$pkgdir/usr/bin/faceauth-ui"
  install -Dm755 "target/release/faceauthd" "$pkgdir/usr/bin/faceauthd"

  # faceauthd: face unlock for unprivileged screen lockers (socket activated)
  install -Dm644 "packaging/faceauthd.socket" "$pkgdir/usr/lib/systemd/system/faceauthd.socket"
  install -Dm644 "packaging/faceauthd.service" "$pkgdir/usr/lib/systemd/system/faceauthd.service"

  # Models
  local model
  for model in MobileFaceNet.onnx ultra_light_640.onnx face_detection_yunet_2023mar.onnx; do
    install -Dm644 "models/$model" "$pkgdir/usr/share/faceauth/models/$model"
  done

  # Config (system-wide; enrollment and PAM both read it)
  install -Dm644 "packaging/config.toml" "$pkgdir/etc/faceauth/config.toml"
  install -Dm644 "packaging/config.toml" "$pkgdir/usr/share/doc/$pkgname/config.toml"

  # Root-only model store and OpenVINO cache
  install -Dm644 "packaging/faceauth.tmpfiles" "$pkgdir/usr/lib/tmpfiles.d/faceauth.conf"

  # polkit action for faceauth-ui (pkexec faceauth import/verify/…)
  install -Dm644 "packaging/org.faceauth.policy" "$pkgdir/usr/share/polkit-1/actions/org.faceauth.policy"

  # Documentation & license
  install -Dm644 "README.md" "$pkgdir/usr/share/doc/$pkgname/README.md"
  install -Dm644 "LICENSE" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"

  # PAM configuration example
  install -Dm644 "pam/faceauth" "$pkgdir/usr/share/doc/$pkgname/pam-example"
}

# vim:set ts=2 sw=2 et:
