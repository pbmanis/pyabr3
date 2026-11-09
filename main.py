"""
Entry point for pyabr3, used by "uv run python main.py" and by the PyInstaller builds.

    python main.py              uses config/abrs.cfg on Windows, otherwise config/abrs_test.cfg
    python main.py test         uses config/abrs_test.cfg
    python main.py --smoke-test opens the windows, then exits (used to check a build)
"""

import os
import platform
import sys
from pathlib import Path

SMOKE_TEST_MSEC = 2000


def set_working_directory():
    """pyabr3 reads config/ and protocols/ relative to the working directory.

    In a Windows build they are next to the program, so they can be edited and data
    can be written there. In a macOS build they are read-only resources in the app.
    """
    if not getattr(sys, "frozen", False):
        return
    if platform.system() == "Windows":
        os.chdir(Path(sys.executable).parent)
    else:
        os.chdir(sys._MEIPASS)


def main():
    set_working_directory()
    from pyqtgraph.Qt import QtCore, QtWidgets

    from src.pyabr3 import PyABR

    args = sys.argv[1:]
    smoke_test = "--smoke-test" in args
    default_config = "abr" if platform.system() == "Windows" else "test"
    configs = [a for a in args if a in ["abr", "test"]]
    prog = PyABR(configs[0] if configs else default_config)
    app = QtWidgets.QApplication.instance()
    if smoke_test:
        prog.open_calibration()
        prog.show_data_viewer()
        QtCore.QTimer.singleShot(SMOKE_TEST_MSEC, app.quit)
    status = app.exec()
    print("smoke test: windows opened" if smoke_test else "pyabr3 closed")
    sys.exit(status)


if __name__ == "__main__":
    main()
