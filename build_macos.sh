#!/usr/bin/env bash
# Build dist/macos/PyABR3.app with PyInstaller, then check that it starts.
#
# The macOS build runs without acquisition hardware and uses config/abrs_test.cfg. It is
# for examining calibrations; the Calibrate button uses simulated hardware.
#
# Requires uv. Run from any directory: ./build_macos.sh
#
# --paths . is needed because this directory contains an __init__.py, which otherwise
# makes PyInstaller search for src/ from a higher level.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build_macos.sh must be run on macOS" >&2
    exit 1
fi

uv run --locked --group build pyinstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name PyABR3 \
    --icon icons/pyabr3.icns \
    --distpath dist/macos \
    --workpath build/macos \
    --paths . \
    --add-data config:config \
    --add-data protocols:protocols \
    --add-data calfiles:calfiles \
    main.py

QT_QPA_PLATFORM=offscreen dist/macos/PyABR3.app/Contents/MacOS/PyABR3 --smoke-test
echo "Built dist/macos/PyABR3.app"
