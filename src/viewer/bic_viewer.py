"""
Panel that shows the responses of an interleaved binaural interaction component (BIC)
recording: the average response to the left only, right only and binaural stimuli, in
three colors. It is a separate piece from the ABR data viewer, and can be docked in the
main window or used alone:

    python -m src.viewer.bic_viewer [path to a trial file (.p) of the recording]

The computation is in bic_files.py. The upper plot has the three conditions and the sum
of the left and right responses (L+R); the lower plot has the BIC, the binaural response
minus L+R. The BIC of every stimulus in the recording can be overlaid, to follow it as the
interaural time difference (ITD) is varied.
"""

import sys
from pathlib import Path
from typing import Callable

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from src.viewer import bic_files
from src.viewer.data_viewer import FILTER_DEFAULTS

CONDITION_COLORS = {"L": "#4C9BE8", "R": "#E8574C", "B": "#5CC46A"}  # left, right, binaural
COLOR_SUM = "#C8C8C8"  # L+R
COLOR_BIC = "#F5A623"
LINE_WIDTH = 2
LEGEND_OFFSET = (-10, 10)
ERROR_STYLE = "color: #E8574C"


class BICPanel(QtWidgets.QWidget):
    """Responses to the left, right and binaural conditions of one interleaved BIC run.

    Parameters
    ----------
    start_dir : callable, optional
        Returns the directory where the file dialog starts.
    """

    def __init__(self, start_dir: Callable[[], str] | None = None, parent=None):
        super().__init__(parent)
        self.start_dir = start_dir
        self.run = None  # abr_files.Run
        self.responses: dict = {}  # wave: bic_files.BICResponses
        self.build_gui()

    def build_gui(self):
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open BIC recording...")
        self.open_button.clicked.connect(self.open_dialog)
        self.file_label = QtWidgets.QLabel("No file")
        top.addWidget(self.open_button)
        top.addWidget(self.file_label, stretch=1)
        layout.addLayout(top)

        controls = QtWidgets.QHBoxLayout()
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.currentIndexChanged.connect(self.plot_wave)
        controls.addWidget(QtWidgets.QLabel("Stimulus"))
        controls.addWidget(self.wave_combo, stretch=1)
        hpf_default, lpf_default = FILTER_DEFAULTS["pyabr3"]
        self.hpf_box = self.filter_box(hpf_default)
        self.lpf_box = self.filter_box(lpf_default)
        controls.addWidget(QtWidgets.QLabel("High pass (Hz, 0 = off)"))
        controls.addWidget(self.hpf_box)
        controls.addWidget(QtWidgets.QLabel("Low pass (Hz, 0 = off)"))
        controls.addWidget(self.lpf_box)
        self.overlay_box = QtWidgets.QCheckBox("Overlay the BIC of all stimuli")
        self.overlay_box.toggled.connect(self.plot_wave)
        controls.addWidget(self.overlay_box)
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()  # the three conditions, and L+R
        self.plot.setLabel("bottom", "Time re slot start", units="ms")
        self.plot.setLabel("left", "Response", units="uV")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.bic_plot = pg.PlotWidget()  # binaural - (L+R)
        self.bic_plot.setLabel("bottom", "Time re slot start", units="ms")
        self.bic_plot.setLabel("left", "BIC", units="uV")
        self.bic_plot.showGrid(x=True, y=True, alpha=0.3)
        self.bic_plot.setXLink(self.plot)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.plot)
        splitter.addWidget(self.bic_plot)
        layout.addWidget(splitter, stretch=1)
        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

    def filter_box(self, value: float) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(0.0, 20000.0)
        box.setDecimals(0)
        box.setValue(value)
        box.editingFinished.connect(self.filters_changed)
        return box

    def filters(self) -> tuple:
        """High and low pass in Hz, with None for off."""
        hpf, lpf = self.hpf_box.value(), self.lpf_box.value()
        return (hpf if hpf > 0.0 else None, lpf if lpf > 0.0 else None)

    def show_error(self, message: str):
        self.status_label.setStyleSheet(ERROR_STYLE)
        self.status_label.setText(message)

    def show_status(self, message: str):
        self.status_label.setStyleSheet("")
        self.status_label.setText(message)

    def open_dialog(self):
        start = self.start_dir() if self.start_dir is not None else ""
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a trial file of a BIC recording", start, "pyabr3 trial files (*.p)"
        )
        if name:
            self.load(Path(name))

    def load(self, path: Path):
        """Read and average the recording that this trial file belongs to."""
        path = Path(path)
        self.file_label.setText(str(path.parent))
        try:
            self.run = bic_files.abr_files.find_run(path)
        except ValueError as e:
            self.show_error(str(e))
            return
        self.compute()

    def compute(self):
        """Average the recording with the present filter settings, and show it."""
        if self.run is None:
            return
        hpf, lpf = self.filters()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.responses = bic_files.compute_bic(self.run, hpf, lpf)
        except Exception as e:  # a file that is not BIC, or is damaged
            self.responses = {}
            self.wave_combo.blockSignals(True)
            self.wave_combo.clear()
            self.wave_combo.blockSignals(False)
            self.plot.clear()
            self.bic_plot.clear()
            self.show_error(str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        current = self.wave_combo.currentData()
        self.wave_combo.blockSignals(True)
        self.wave_combo.clear()
        for wave, responses in self.responses.items():
            self.wave_combo.addItem(bic_files.stimulus_label(responses.info), wave)
        index = self.wave_combo.findData(current)
        self.wave_combo.setCurrentIndex(max(index, 0))
        self.wave_combo.blockSignals(False)
        self.plot_wave()

    def filters_changed(self):
        if self.run is not None:
            self.compute()

    def plot_wave(self, *args):
        """Plot the three conditions and L+R of the selected stimulus, and the BIC."""
        self.plot.clear()
        self.bic_plot.clear()
        wave = self.wave_combo.currentData()
        if wave is None or wave not in self.responses:
            return
        responses = self.responses[wave]
        stimulus_line = pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine)
        self.plot.addLegend(offset=LEGEND_OFFSET)
        self.plot.addItem(pg.InfiniteLine(0.0, angle=90, pen=stimulus_line))
        for condition in bic_files.CONDITIONS:
            if condition not in responses.traces:
                continue
            self.plot.plot(
                responses.t_ms,
                responses.traces[condition],
                pen=pg.mkPen(CONDITION_COLORS[condition], width=LINE_WIDTH),
                name=bic_files.CONDITION_NAMES[condition],
            )
        if responses.sum_lr is not None:
            self.plot.plot(
                responses.t_ms,
                responses.sum_lr,
                pen=pg.mkPen(COLOR_SUM, width=LINE_WIDTH, style=QtCore.Qt.PenStyle.DashLine),
                name="L+R",
            )
        self.plot.setTitle(bic_files.stimulus_label(responses.info))
        self.plot_bic(wave)
        polarity = "alternating polarity averaged" if responses.info["alternating"] else "single polarity"
        counts = ", ".join(
            f"{bic_files.CONDITION_NAMES[c]} {n:d}" for c, n in responses.counts.items()
        )
        self.show_status(f"{responses.n_trials:d} trials per condition; {polarity}; responses averaged: {counts}")

    def plot_bic(self, wave):
        """Plot the BIC (binaural minus L+R) of one stimulus, or of all of them to compare."""
        self.bic_plot.addItem(pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine)))
        self.bic_plot.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen("#888888")))
        if self.overlay_box.isChecked():
            self.bic_plot.addLegend(offset=LEGEND_OFFSET)
            waves = list(self.responses)
            for k, w in enumerate(waves):
                bic = self.responses[w].bic
                if bic is None:
                    continue
                self.bic_plot.plot(
                    self.responses[w].t_ms,
                    bic,
                    pen=pg.mkPen(pg.intColor(k, hues=max(len(waves), 2)), width=LINE_WIDTH),
                    name=bic_files.stimulus_label(self.responses[w].info),
                )
            self.bic_plot.setTitle("BIC of all stimuli: binaural - (L+R)")
            return
        bic = self.responses[wave].bic
        if bic is None:
            self.bic_plot.setTitle("BIC: needs left, right and binaural responses")
            return
        self.bic_plot.plot(self.responses[wave].t_ms, bic, pen=pg.mkPen(COLOR_BIC, width=LINE_WIDTH))
        self.bic_plot.setTitle("BIC: binaural - (L+R)")


def main():
    app = pg.mkQApp("BIC viewer")
    app.setStyle("fusion")
    panel = BICPanel()
    panel.setWindowTitle("BIC viewer")
    panel.resize(1000, 700)
    panel.show()
    if len(sys.argv) > 1:
        panel.load(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
