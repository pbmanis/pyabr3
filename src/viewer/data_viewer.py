#!/usr/bin/env python
"""
Window for examining data saved by pyabr3.

    .p files    a recording: opening any trial loads all trials of that recording
    .pkl files  threshold summaries (all_tone_map_data.pkl) or other saved arrays

Run on its own from the top directory of the repository:
    python -m src.viewer.data_viewer [file]

It is also opened from pyabr3, with the "Load Data File" button.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph import dockarea as PGD
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtCore import pyqtSignal
from pylibrary.tools import cprint as CP

from src.viewer import abr_files

COLOR_RESPONSE = "#5AA9FF"
COLOR_TRIAL = "#C8C8C8"
COLOR_STIMULUS = "#F5A623"
SUBJECT_ALPHA = 70
TRACE_SPACING_FACTOR = 1.1
LEGEND_OFFSET = (-10, 10)
LEGEND_BRUSH = (0, 0, 0, 190)
LEGEND_TEXT = "#E0E0E0"
DEFAULT_HPF = 300.0  # Hz, as used for plotting in the analysis
CLICK_LABEL = "Click"


class RunLoader(QtCore.QThread):
    """Reads a recording away from the GUI thread."""

    signal_progress = pyqtSignal(int, int)
    signal_loaded = pyqtSignal(object)
    signal_failed = pyqtSignal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            data = abr_files.load_run(self.path, self.signal_progress.emit)
        except Exception:
            self.signal_failed.emit(traceback.format_exc())
        else:
            self.signal_loaded.emit(data)


class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.loader = None
        self.run_data = None
        self.responses = None
        self.frequency_labels: dict = {}
        self.updating = False  # True while the program itself sets parameters
        self.setWindowTitle("ABR Data Viewer")
        self.setGeometry(120, 60, 1250, 900)
        self.build_gui()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

    def build_parameters(self) -> list:
        return [
            {
                "name": "Actions",
                "type": "group",
                "children": [
                    {"name": "Open File", "type": "action"},
                    {"name": "Close", "type": "action"},
                ],
            },
            {
                "name": "File",
                "type": "group",
                "children": [
                    {"name": "Name", "type": "str", "value": "", "readonly": True},
                    {"name": "Status", "type": "str", "value": "No file", "readonly": True},
                ],
            },
            {
                "name": "Average",
                "type": "group",
                "children": [
                    {
                        "name": "High pass (Hz, 0 = off)",
                        "type": "float",
                        "value": DEFAULT_HPF,
                        "limits": [0.0, 20000.0],
                    },
                    {
                        "name": "Low pass (Hz, 0 = off)",
                        "type": "float",
                        "value": 3000.0,
                        "limits": [0.0, 40000.0],
                    },
                    {
                        "name": "Frequency",
                        "type": "list",
                        "limits": [CLICK_LABEL],
                        "value": CLICK_LABEL,
                    },
                    {
                        "name": "Interleaved set",
                        "type": "int",
                        "value": 0,
                        "limits": [0, 0],
                    },
                ],
            },
            {
                "name": "Single trial",
                "type": "group",
                "children": [
                    {"name": "Stimulus #", "type": "int", "value": 0, "limits": [0, 0]},
                    {"name": "Repetition", "type": "int", "value": 1, "limits": [1, 1]},
                ],
            },
        ]

    def build_gui(self):
        self.dockArea = PGD.DockArea()
        self.setCentralWidget(self.dockArea)
        self.Dock_Params = PGD.Dock("Parameters", size=(300, 900))
        self.Dock_Plots = PGD.Dock("Data", size=(950, 650))
        self.Dock_Info = PGD.Dock("Information", size=(950, 250))
        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_Plots, "right", self.Dock_Params)
        self.dockArea.addDock(self.Dock_Info, "bottom", self.Dock_Plots)

        self.ptreedata = Parameter.create(
            name="Viewer", type="group", children=self.build_parameters()
        )
        self.ptree = ParameterTree()
        self.ptree.setParameters(self.ptreedata, showTop=False)
        self.Dock_Params.addWidget(self.ptree)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.Dock_Plots.addWidget(self.plot_widget)
        self.info_tree = pg.DataTreeWidget()
        self.Dock_Info.addWidget(self.info_tree)
        self.panels: dict = {}

    def value(self, *path):
        return self.ptreedata.child(*path).value()

    def set_status(self, text: str):
        self.ptreedata.child("File", "Status").setValue(text)

    def show_error(self, message: str):
        CP.cprint("r", message)
        self.set_status("Error")
        QtWidgets.QMessageBox.critical(self, "Data Viewer", message)

    def command_dispatcher(self, param, changes):
        if self.updating:
            return
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            match path[0]:
                case "Actions":
                    match path[1]:
                        case "Open File":
                            self.open_file_dialog()
                        case "Close":
                            self.close()
                case "Average":
                    if change == "value":
                        self.update_average(path[1])
                case "Single trial":
                    if change == "value":
                        self.update_trial()

    def open_file_dialog(self):
        start = Path("abr_data") if Path("abr_data").is_dir() else Path.home()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", str(start), "ABR data (*.p *.pkl)"
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path: Path):
        if self.is_loading():
            return
        path = Path(path)
        self.ptreedata.child("File", "Name").setValue(path.name)
        match path.suffix:
            case ".p":
                self.start_loading_run(path)
            case ".pkl":
                self.show_summary(path)
            case _:
                self.show_error(f"Files of this type cannot be opened: {path.name:s}")

    def is_loading(self) -> bool:
        return self.loader is not None and self.loader.isRunning()

    def start_loading_run(self, path: Path):
        self.set_status("Reading")
        self.loader = RunLoader(path, parent=self)
        self.loader.signal_progress.connect(
            lambda count, total: self.set_status(f"Reading {count:d} / {total:d}")
        )
        self.loader.signal_failed.connect(self.show_error)
        self.loader.signal_loaded.connect(self.run_loaded)
        self.loader.start()

    def run_loaded(self, data: abr_files.RunData):
        self.run_data = data
        run = data.run
        self.updating = True
        n_sets = len(run.waves) if run.interleaved else 1
        self.ptreedata.child("Average", "Interleaved set").setLimits([0, n_sets - 1])
        self.ptreedata.child("Average", "Interleaved set").setValue(0)
        self.ptreedata.child("Single trial", "Stimulus #").setLimits([run.waves[0], run.waves[-1]])
        self.ptreedata.child("Single trial", "Stimulus #").setValue(run.waves[0])
        repetition = self.ptreedata.child("Single trial", "Repetition")
        repetition.setLimits([run.repetitions[0], run.repetitions[-1]])
        repetition.setValue(run.repetitions[0])
        self.updating = False

        self.build_run_panels()
        self.show_info(self.run_information(data))
        self.set_status(f"{len(run.files):d} trials, {run.stim:s}")
        self.update_average("Frequency labels")
        self.update_trial()

    @staticmethod
    def run_information(data: abr_files.RunData) -> dict:
        header = data.header
        calibration = header["calibration"]
        return {
            "subject": header["subject_data"],
            "protocol": header["protocol"],
            "recorded": str(header["timestamp"]),
            "sample rate (Hz)": data.sample_rate,
            "trials": len(data.run.files),
            "repetitions of each stimulus": data.n_reps,
            "calibration": {
                k: str(calibration[k]) for k in ["spkr", "mic", "date", "calattn", "filename"]
            },
        }

    def show_info(self, information):
        self.info_tree.setData(information)

    def build_run_panels(self):
        self.plot_widget.clear()
        self.panels = {}
        self.panels["average"] = self.plot_widget.addPlot(row=0, col=0, title="Average response")
        self.panels["average"].setLabel("bottom", "Time", units="ms")
        self.panels["average"].setLabel("left", "dB SPL")
        self.panels["trial"] = self.plot_widget.addPlot(row=1, col=0, title="Single trial")
        self.panels["trial"].setLabel("bottom", "Time", units="s")
        self.panels["trial"].setLabel("left", "Amplifier output", units="V")
        self.panels["stimulus"] = self.plot_widget.addPlot(row=2, col=0, title="Stimulus")
        self.panels["stimulus"].setLabel("bottom", "Time", units="s")
        self.panels["stimulus"].setLabel("left", "V")
        self.panels["stimulus"].setXLink(self.panels["trial"])
        for key in ["trial", "stimulus"]:
            self.panels[key].setDownsampling(auto=True, mode="peak")
            self.panels[key].setClipToView(True)
        self.plot_widget.ci.layout.setRowStretchFactor(0, 3)

    @staticmethod
    def frequency_label(frequency: float) -> str:
        return CLICK_LABEL if frequency == 0.0 else f"{frequency:.0f} Hz"

    def update_average(self, changed: str):
        if self.run_data is None:
            return
        if changed == "Frequency":  # no need to recalculate the averages
            self.plot_average()
            return
        hpf = self.value("Average", "High pass (Hz, 0 = off)") or None
        lpf = self.value("Average", "Low pass (Hz, 0 = off)") or None
        interleaved_set = self.value("Average", "Interleaved set")
        try:
            self.responses = abr_files.compute_responses(self.run_data, hpf, lpf, interleaved_set)
        except Exception:
            self.show_error(traceback.format_exc())
            return
        self.frequency_labels = {
            self.frequency_label(f): f for f in self.responses.frequencies
        }
        self.updating = True
        current = self.value("Average", "Frequency")
        choice = self.ptreedata.child("Average", "Frequency")
        choice.setLimits(list(self.frequency_labels))
        if current in self.frequency_labels:
            choice.setValue(current)
        self.updating = False
        self.plot_average()

    def plot_average(self):
        label = self.value("Average", "Frequency")
        frequency = self.frequency_labels[label]
        levels = self.responses.levels(frequency)
        traces = [self.responses.traces[(db, frequency)] for db in levels]
        spacing = TRACE_SPACING_FACTOR * max(np.ptp(trace) for trace in traces) or 1.0
        panel = self.panels["average"]
        panel.clear()
        for i, trace in enumerate(traces):
            panel.plot(self.responses.t_ms, trace + i * spacing, pen=pg.mkPen(COLOR_RESPONSE))
        ticks = [(i * spacing, f"{db:.0f}") for i, db in enumerate(levels)]
        panel.getAxis("left").setTicks([ticks])
        panel.setTitle(f"Average response: {label:s} (traces {spacing:.2f} µV apart)")

    def update_trial(self):
        if self.run_data is None:
            return
        wave = self.value("Single trial", "Stimulus #")
        repetition = self.value("Single trial", "Repetition")
        trial_panel, stimulus_panel = self.panels["trial"], self.panels["stimulus"]
        trial_panel.clear()
        stimulus_panel.clear()
        filename = self.run_data.run.files.get((wave, repetition))
        if filename is None:
            trial_panel.setTitle(f"Stimulus {wave:d}, repetition {repetition:d}: no file")
            return
        trial = abr_files.read_trial(filename)
        t = np.arange(len(trial["data"])) / trial["record_frequency"]
        trial_panel.plot(t, trial["data"], pen=pg.mkPen(COLOR_TRIAL))
        trial_panel.setTitle(f"Single trial: {filename.name:s}")
        self.plot_stimulus(trial, wave, stimulus_panel)

    @staticmethod
    def plot_stimulus(trial: dict, wave: int, panel):
        """Later files do not hold the waveform: "sound" is replaced by an empty dict."""
        keys = list(trial["stimuli"])
        entry = trial["stimuli"][keys[wave]] if wave < len(keys) else {}
        sound = entry.get("sound")
        if isinstance(sound, np.ndarray) and sound.size > 0:
            t = np.arange(len(sound)) / entry["rate"]
            panel.plot(t, sound, pen=pg.mkPen(COLOR_STIMULUS))
            panel.setTitle("Stimulus")
        else:
            panel.setTitle("Stimulus (the waveform is not stored in this file)")

    def show_summary(self, path: Path):
        try:
            data = abr_files.load_summary(path)
        except Exception:
            self.show_error(traceback.format_exc())
            return
        self.run_data = None
        self.show_info(data)
        self.plot_widget.clear()
        self.panels = {}
        if abr_files.is_threshold_summary(data):
            self.plot_thresholds(data)
            self.set_status("Threshold summary")
        elif isinstance(data, dict):
            arrays = abr_files.numeric_arrays(data)
            self.plot_arrays(arrays)
            self.set_status(f"{len(arrays):d} arrays plotted")
        else:
            self.set_status(f"Contents: {type(data).__name__:s}")

    def plot_thresholds(self, data: dict):
        """Mean threshold of each group, with a faint line for each subject."""
        panel = self.plot_widget.addPlot(row=0, col=0, title="Thresholds")
        self.panels["thresholds"] = panel
        panel.setLogMode(x=True, y=False)
        panel.setLabel("bottom", "Frequency", units="Hz")
        panel.setLabel("left", "Threshold (dB SPL)")
        panel.showGrid(x=True, y=True, alpha=0.15)
        panel.addLegend(offset=LEGEND_OFFSET, brush=LEGEND_BRUSH, labelTextColor=LEGEND_TEXT)
        frequencies = np.asarray(data["frequencies"], dtype=float)
        colors = data.get("plot_colors", {})
        for group, subjects in data["thresholds"].items():
            thresholds = np.asarray(subjects, dtype=float)
            if thresholds.ndim != 2 or not np.any(np.isfinite(thresholds)):
                continue
            color = pg.mkColor(colors.get(group, COLOR_RESPONSE))
            faint = pg.mkColor(color)
            faint.setAlpha(SUBJECT_ALPHA)
            for subject in thresholds:
                panel.plot(frequencies, subject, pen=pg.mkPen(faint), connect="finite")
            has_data = np.any(np.isfinite(thresholds), axis=0)
            mean = np.full(len(frequencies), np.nan)
            mean[has_data] = np.nanmean(thresholds[:, has_data], axis=0)
            panel.plot(
                frequencies,
                mean,
                pen=pg.mkPen(color, width=2),
                symbol="o",
                symbolSize=8,
                symbolBrush=color,
                symbolPen=None,
                connect="finite",
                name=f"{group!s} (n={len(thresholds):d})",
            )

    def plot_arrays(self, arrays: dict):
        for row, (name, array) in enumerate(arrays.items()):
            panel = self.plot_widget.addPlot(row=row, col=0, title=name)
            self.panels[name] = panel
            for trace in np.atleast_2d(array):
                panel.plot(trace, pen=pg.mkPen(COLOR_RESPONSE))

    def closeEvent(self, event):
        if self.is_loading():
            self.loader.wait(30000)
        super().closeEvent(event)


def main():
    app = pg.mkQApp("ABR Data Viewer")
    app.setStyle("fusion")
    window = DataViewer()
    window.show()
    if len(sys.argv) > 1:
        window.open_path(Path(sys.argv[1]))
    sys.exit(QtWidgets.QApplication.instance().exec())


if __name__ == "__main__":
    main()
