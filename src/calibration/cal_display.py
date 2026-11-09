"""
Display of a speaker calibration, used in the Calibration dock of pyabr3.

Two views are shown, each with the noise floor:
    raw calibration   the levels measured at the fixed attenuator setting
    75 dB check       the levels measured when the attenuator is adjusted at each
                      frequency to give 75 dB SPL (the chk75db file)
"""

import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from src.calibration.cal_chooser import created_time
from src.calibration.engine import CHECK_SPL, HISTORY_DIR, load_mat_struct

COLOR_MEASURED = "#5AA9FF"
COLOR_BANDPASS = "#F5A623"
COLOR_NOISE = "#E86FB0"
COLOR_MAX = "#E0E0E0"
COLOR_TARGET = "#C8C8C8"
NOISE_FILL_ALPHA = 90
INVISIBLE_PEN = (0, 0, 0, 0)
LEGEND_OFFSET = (-10, 10)
LEGEND_BRUSH = (0, 0, 0, 190)
LEGEND_TEXT = "#E0E0E0"
CHECK_FIELDS = {"Freqs", "dBSPL_bp", "dBSPL_nf"}


def plot_noise_floor(panel, freqs: np.ndarray, noise: np.ndarray):
    """Plot the noise floor, clipped and shaded down to 0 dB SPL as in MATLAB."""
    clipped = np.clip(noise, 0.0, None)
    noise_curve = panel.plot(freqs, clipped, pen=pg.mkPen(COLOR_NOISE), name="Noise floor")
    zero_curve = panel.plot(freqs, np.zeros_like(freqs), pen=pg.mkPen(INVISIBLE_PEN))
    fill = pg.mkColor(COLOR_NOISE)
    fill.setAlpha(NOISE_FILL_ALPHA)
    panel.addItem(pg.FillBetweenItem(noise_curve, zero_curve, brush=pg.mkBrush(fill)))


def _read_check(path: Path) -> dict | None:
    """The contents of a check file, or None for files in the older formats."""
    try:
        check = load_mat_struct(path, "CHKCAL")
    except (KeyError, ValueError, OSError):
        return None
    return check if CHECK_FIELDS <= set(check) else None


def _check_candidates(cal_path: Path, speaker: str) -> list:
    folders = {cal_path.parent, cal_path.parent / HISTORY_DIR}
    if cal_path.parent.name == HISTORY_DIR:
        folders.add(cal_path.parent.parent)
    return [p for folder in folders for p in folder.glob(f"chk75db_{speaker:s}*.cal")]


def find_check(cal_path: Path, speaker: str):
    """Find the 75 dB check that was made for a calibration.

    This is the first check made after the calibration, on the same day. Times are
    from the file headers, because file system times change when files are moved.
    Returns (path, contents, time made), or None.
    """
    cal_path = Path(cal_path)
    cal_time = created_time(cal_path)
    if cal_time is None:
        return None
    found = []
    for path in _check_candidates(cal_path, speaker):
        made = created_time(path)
        if made is not None and made >= cal_time and made.date() == cal_time.date():
            found.append((made, path))
    for made, path in sorted(found):
        check = _read_check(path)
        if check is not None:
            return path, check, made
    return None


class CalibrationDisplay(pg.GraphicsLayoutWidget):
    """The raw calibration above the 75 dB check, on the same frequency axis."""

    def __init__(self):
        super().__init__()
        self.raw_panel = self.add_panel(0)
        self.check_panel = self.add_panel(1)
        self.check_panel.setXLink(self.raw_panel)

    def add_panel(self, row: int):
        panel = self.addPlot(row=row, col=0)
        panel.setLogMode(x=True, y=False)
        panel.setLabel("bottom", "Frequency", units="Hz")
        panel.setLabel("left", "dB SPL")
        panel.showGrid(x=True, y=True, alpha=0.15)
        panel.addLegend(offset=LEGEND_OFFSET, brush=LEGEND_BRUSH, labelTextColor=LEGEND_TEXT)
        return panel

    def show_calibration(self, cal_path: Path, caldata: dict):
        cal_path = Path(cal_path)
        self.plot_raw(cal_path, caldata)
        self.plot_check(find_check(cal_path, str(caldata["spkr"])))

    def plot_raw(self, cal_path: Path, caldata: dict):
        panel = self.raw_panel
        panel.clear()
        freqs = np.asarray(caldata["freqs"], dtype=float)
        order = np.argsort(freqs)
        freqs = freqs[order]
        plot_noise_floor(panel, freqs, np.asarray(caldata["db_nf"], dtype=float)[order])
        for key, color, name in [
            ("maxdb", COLOR_MAX, "Max SPL (0 dB attn)"),
            ("db_cs", COLOR_MEASURED, "Cosinor"),
            ("db_bp", COLOR_BANDPASS, "Bandpass"),
        ]:
            panel.plot(freqs, np.asarray(caldata[key], dtype=float)[order], pen=pg.mkPen(color), name=name)
        panel.setXRange(np.log10(freqs[0]), np.log10(freqs[-1]))
        panel.setTitle(
            f"Raw calibration: {caldata['spkr']!s}, attn {float(caldata['calattn']):.0f} dB, "
            f"{caldata['date']!s} ({cal_path.name:s})"
        )

    def plot_check(self, found):
        panel = self.check_panel
        panel.clear()
        if found is None:
            panel.setTitle("75 dB check: no check was made on the day of this calibration")
            return
        path, check, made = found
        freqs = np.atleast_1d(np.asarray(check["Freqs"], dtype=float))
        order = np.argsort(freqs)
        freqs = freqs[order]
        plot_noise_floor(panel, freqs, np.asarray(check["dBSPL_nf"], dtype=float)[order])
        panel.plot(
            [freqs[0], freqs[-1]],
            [CHECK_SPL, CHECK_SPL],
            pen=pg.mkPen(COLOR_TARGET, style=pg.QtCore.Qt.PenStyle.DashLine),
            name=f"{CHECK_SPL:.0f} dB target",
        )
        panel.plot(
            freqs,
            np.asarray(check["dBSPL_bp"], dtype=float)[order],
            pen=pg.mkPen(COLOR_BANDPASS),
            symbol="o",
            symbolSize=8,
            symbolBrush=COLOR_BANDPASS,
            symbolPen=None,
            name="Measured (bandpass)",
        )
        panel.setTitle(f"75 dB check: {made:%Y-%m-%d %H:%M:%S} ({path.name:s})")
