# Maintainer: Your Name <you@example.com>
pkgname=faceauth
pkgver=0.1.0
pkgrel=1
pkgdesc="Face authentication system for Linux using OpenCV/ONNX"
arch=('x86_64' 'aarch64')
url="https://github.com/vlad-wild/faceauth"
license=('MIT')
depends=('opencv' 'opencv-data' 'v4l-utils' 'pam')
makedepends=('rust' 'cargo' 'clang' 'llvm')
source=("$pkgname-$pkgver.tar.gz")
sha256sums=('SKIP')

prepare() {
  cd "$pkgname-$pkgver"
  cargo fetch --locked --target "$CARCH-unknown-linux-gnu"
}

build() {
  cd "$pkgname-$pkgver"
  export RUSTUP_TOOLCHAIN=stable
  export CARGO_TARGET_DIR=target
  cargo build --frozen --release --all-targets
}

check() {
  cd "$pkgname-$pkgver"
  cargo test --frozen
}

package() {
  cd "$pkgname-$pkgver"

  # Binaries
  install -Dm755 "target/release/faceauth" "$pkgdir/usr/bin/faceauth"
  install -Dm755 "target/release/faceauth-auth" "$pkgdir/usr/bin/faceauth-auth"
  install -Dm755 "target/release/faceauth-ui" "$pkgdir/usr/bin/faceauth-ui"

  # Models
  install -Dm644 "models/MobileFaceNet.onnx" "/etc/$pkgdir/models/MobileFaceNet.onnx"
  install -Dm644 "models/ultra_light_640.onnx" "/etc/$pkgdir/models/ultra_light_640.onnx"

  install -Dm644 "README.md" "$pkgdir/usr/share/doc/$pkgname/README.md"
  install -Dm644 "faceauth.toml" "$pkgdir/etc/faceauth/config.toml"
  install -Dm644 "LICENSE" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
  
  # PAM configuration example
  install -Dm644 "pam/faceauth" "$pkgdir/usr/share/doc/$pkgname/pam-example"
  
  # Systemd service for daemon (optional)
  # install -Dm644 "systemd/faceauth.service" "$pkgdir/usr/lib/systemd/system/faceauth.service"
}

# vim:set ts=2 sw=2 et:
