# Maintainer: Vlad Wild <ya.vlash1@yandex.ru>
pkgname=faceauth
pkgver=0.2.0
pkgrel=1
pkgdesc="Face authentication system for Linux using OpenCV/ONNX"
arch=('x86_64' 'aarch64')
url="https://github.com/vlad-wild/faceauth"
license=('MIT')
depends=('opencv' 'v4l-utils' 'pam')
makedepends=('rust' 'cargo' 'clang' 'llvm' 'pkgconf')
source=("git+$url.git")
sha256sums=('SKIP')

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

  # Models
  install -Dm644 "models/MobileFaceNet.onnx" "$pkgdir/etc/faceauth/models/MobileFaceNet.onnx"
  install -Dm644 "models/ultra_light_640.onnx" "$pkgdir/etc/faceauth/models/ultra_light_640.onnx"

  # Config
  install -Dm644 "faceauth.toml" "$pkgdir/etc/faceauth/config.toml"

  # Documentation & license
  install -Dm644 "README.md" "$pkgdir/usr/share/doc/$pkgname/README.md"
  install -Dm644 "LICENSE" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"

  # PAM configuration example
  install -Dm644 "pam/faceauth" "$pkgdir/usr/share/doc/$pkgname/pam-example"
}

# vim:set ts=2 sw=2 et:
