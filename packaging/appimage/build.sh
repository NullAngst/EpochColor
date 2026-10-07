#!/usr/bin/env bash
# Build EpochColor-<version>-x86_64.AppImage.
#
# Works the same on a GitHub runner and on your own machine. Needs curl,
# python3 (with venv), tar and xz. Downloads a relocatable Python, installs EpochColor and its
# dependencies into it, adds a full FFmpeg build, and packs it all with
# appimagetool. PyTorch is not bundled; the app downloads the right build
# for the GPU on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-$ROOT/dist}"
PY_SERIES="${PY_SERIES:-3.12}"          # Python minor series, newest patch is picked
FFMPEG_BRANCH="${FFMPEG_BRANCH:-8.1}"   # FFmpeg release branch, newest build of it is picked
WORK="$(mktemp -d)"
APPDIR="$WORK/AppDir"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT" "$APPDIR/usr/bin" "$APPDIR/usr/share/doc/epochcolor"

echo "== Python $PY_SERIES"
# uv fetches python-build-standalone's relocatable builds, newest patch of
# the series, with no API calls to page through
python3 -m venv "$WORK/uv"
"$WORK/uv/bin/pip" -q install uv
"$WORK/uv/bin/uv" python install "$PY_SERIES" --install-dir "$WORK/pythons"
PYSRC="$(find "$WORK/pythons" -maxdepth 1 -type d -name "cpython-${PY_SERIES}.*-linux-x86_64-gnu" | head -1)"
[ -n "$PYSRC" ] || { echo "no Python $PY_SERIES build found"; exit 1; }
cp -a "$PYSRC" "$APPDIR/usr/python"
# this copy is ours to install into
rm -f "$APPDIR"/usr/python/lib/python*/EXTERNALLY-MANAGED
PY="$APPDIR/usr/python/bin/python3"

echo "== EpochColor"
"$PY" -m pip install --no-cache-dir --disable-pip-version-check "$ROOT[gui,raw,heic]"
VERSION="$("$PY" -c 'import epochcolor; print(epochcolor.__version__)')"
# leave out what never runs inside the AppImage
find "$APPDIR/usr/python" -depth -type d -name __pycache__ -exec rm -rf {} +
rm -rf "$APPDIR/usr/python/lib/python$PY_SERIES/test" "$APPDIR/usr/python/lib/python$PY_SERIES/idlelib"
"$PY" -m compileall -q "$APPDIR/usr/python/lib" || true

echo "== FFmpeg $FFMPEG_BRANCH"
FF="ffmpeg-n${FFMPEG_BRANCH}-latest-linux64-gpl-${FFMPEG_BRANCH}"
curl -fsSL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$FF.tar.xz" | tar -xJ -C "$WORK"
cp "$WORK/$FF/bin/ffmpeg" "$WORK/$FF/bin/ffprobe" "$APPDIR/usr/bin/"
cp "$WORK/$FF/LICENSE.txt" "$APPDIR/usr/share/doc/epochcolor/FFMPEG-LICENSE.txt" 2>/dev/null || true
"$APPDIR/usr/bin/ffmpeg" -hide_banner -version | head -1

echo "== AppDir"
cp "$ROOT/packaging/appimage/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"
cp "$ROOT/packaging/appimage/epochcolor.desktop" "$APPDIR/epochcolor.desktop"
cp "$ROOT/packaging/appimage/epochcolor.png" "$APPDIR/epochcolor.png"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp "$ROOT/packaging/appimage/epochcolor.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/"
cp "$ROOT/LICENSE" "$APPDIR/usr/share/doc/epochcolor/LICENSE"
ln -sf epochcolor.png "$APPDIR/.DirIcon"

echo "== appimagetool"
curl -fsSL -o "$WORK/appimagetool" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x "$WORK/appimagetool"
TARGET="$OUT/EpochColor-$VERSION-x86_64.AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$WORK/appimagetool" --no-appstream "$APPDIR" "$TARGET"
echo "built $TARGET ($(du -h "$TARGET" | cut -f1))"
