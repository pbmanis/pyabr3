#!/usr/bin/env bash
# Build dist/windows/PyABR3/PyABR3.exe with PyInstaller.
#
# Run in Git Bash on the Windows acquisition computer, which needs uv, NI-DAQmx and the
# TDT drivers: ./build_windows.sh
#
# config/ and protocols/ are copied next to the program, so they can be edited on the
# rig, and the program runs from that folder. abr_data/ is created there for the data.
# A console window is kept, because the program reports its status with print.
#
# --paths . is needed because this directory contains an __init__.py, which otherwise
# makes PyInstaller search for src/ from a higher level.
set -euo pipefail
cd "$(dirname "$0")"

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) ;;
    *)
        echo "build_windows.sh must be run in Git Bash on Windows" >&2
        exit 1
        ;;
esac

uv run --locked --group build pyinstaller \
    --noconfirm \
    --clean \
    --console \
    --name PyABR3 \
    --icon icons/pyabr3.ico \
    --distpath dist/windows \
    --workpath build/windows \
    --paths . \
    --collect-all nidaqmx \
    --collect-all tdt \
    --hidden-import pythoncom \
    --hidden-import pywintypes \
    --hidden-import win32com.client \
    --add-data nidaq:nidaq \
    --add-data tdt/mic_record.rcx:tdt \
    main.py

app_dir=dist/windows/PyABR3
cp -R config protocols "$app_dir"/
mkdir -p "$app_dir"/abr_data
echo "Built $app_dir/PyABR3.exe"
