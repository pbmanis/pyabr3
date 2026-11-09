#!/usr/bin/env python
"""
Window for examining data saved by pyabr3.

    .p files    a recording: opening any trial loads all trials of that recording
    .pkl files  threshold summaries (all_tone_map_data.pkl) or other saved arrays
    .txt files  a run from the older MATLAB program: any of the files of a run

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
FILTER_DEFAULTS = {  # Hz: (high pass, low pass)
    "pyabr3": (300.0, 3000.0),  # as used for plotting in analyze_abr
    "matlab": (500.0, 2500.0),  # as set in ABRPlotting ABR_Reader.setup
}
CLICK_LABEL = "Click"
VIEW_AVERAGE = "Average only"
VIEW_SD = "Average with SD"
VIEW_TRACES = "Average with individual traces"
COLOR_SINGLE_TRACE = "#C8C8C8"
SINGLE_TRACE_ALPHA = 35
SD_FILL_ALPHA = 80
INVISIBLE_PEN = (0, 0, 0, 0)
MEAN_LINE_WIDTH = 2
MAX_SINGLE_TRACES = 100  # at each level; more than this makes redrawing slow
ROBUST_PERCENTILES = (1.0, 99.0)  # single trials: ignores occasional large artifacts


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


class TrialCalculator(QtCore.QThread):
    """Calculates the response in each repetition away from the GUI thread."""

    signal_progress = pyqtSignal(int, int)
    signal_done = pyqtSignal(object, object)  # request, responses (None if stopped)
    signal_failed = pyqtSignal(str)

    def __init__(self, data, request: tuple, hpf, lpf, interleaved_set: int, parent=None):
        super().__init__(parent)
        self.data = data
        self.request = request
        self.settings = (hpf, lpf, interleaved_set)

    def run(self):
        try:
            result = abr_files.compute_trial_responses(
                self.data,
                *self.settings,
                progress=self.signal_progress.emit,
                should_stop=self.isInterruptionRequested,
            )
        except Exception:
            self.signal_failed.emit(traceback.format_exc())
        else:
            self.signal_done.emit(self.request, result)


class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.loader = None
        self.run_data = None
        self.run_generation = 0  # identifies the recording that trial responses belong to
        self.calculator = None
        self.trial_request = None  # the settings used for trial_responses
        self.trial_responses = None
        self.file_status = ""
        self.matlab_run = None
        self.matlab_labels: dict = {}
        self.data_format = None
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
                        "value": FILTER_DEFAULTS["pyabr3"][0],
                        "limits": [0.0, 20000.0],
                    },
                    {
                        "name": "Low pass (Hz, 0 = off)",
                        "type": "float",
                        "value": FILTER_DEFAULTS["pyabr3"][1],
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
                    {"name": "Invert", "type": "bool", "value": False},
                    {
                        "name": "Display",
                        "type": "checklist",
                        "limits": [VIEW_AVERAGE, VIEW_SD, VIEW_TRACES],
                        "value": VIEW_AVERAGE,
                        "exclusive": True,
                    },
                ],
            },
            {
                "name": "MATLAB run",
                "type": "group",
                "children": [
                    {"name": "Run", "type": "list", "limits": [""], "value": ""},
                    {
                        "name": "Sample rate (Hz)",
                        "type": "float",
                        "value": abr_files.MATLAB_SAMPLE_RATE,
                        "limits": [1000.0, 1000000.0],
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

    def show_file_status(self, text: str):
        self.file_status = text
        self.set_status(text)

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
                    if change == "value" and len(path) == 2:  # not the radio buttons in Display
                        self.update_average(path[1])
                case "Single trial":
                    if change == "value":
                        self.update_trial()
                case "MATLAB run":
                    if change == "value":
                        self.matlab_setting_changed(path[1], data)

    def open_file_dialog(self):
        start = Path("abr_data") if Path("abr_data").is_dir() else Path.home()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", str(start), "ABR data (*.p *.pkl *.txt)"
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
            case ".txt":
                self.open_matlab_file(path)
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

    def use_filter_defaults(self, data_format: str):
        """Filter settings are kept between files of the same format."""
        if data_format == self.data_format:
            return
        self.data_format = data_format
        hpf, lpf = FILTER_DEFAULTS[data_format]
        self.updating = True
        self.ptreedata.child("Average", "High pass (Hz, 0 = off)").setValue(hpf)
        self.ptreedata.child("Average", "Low pass (Hz, 0 = off)").setValue(lpf)
        self.updating = False

    def run_loaded(self, data: abr_files.RunData):
        self.run_data = data
        self.run_generation += 1
        self.matlab_run = None
        self.use_filter_defaults("pyabr3")
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
        self.show_file_status(f"{len(run.files):d} trials, {run.stim:s}")
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

    def open_matlab_file(self, path: Path):
        """Show the run that this file belongs to, and list the other runs in its folder."""
        try:
            name = abr_files.matlab_run_name(path)
            runs = abr_files.list_matlab_runs(path.parent)
        except ValueError as e:
            self.show_error(str(e))
            return
        if name not in runs:
            self.show_error(f"No levels file ({name:s}-SPL.txt) was found for this run")
            return
        self.matlab_labels = {f"{run:s} ({kind:s})": run for run, kind in runs.items()}
        label = next(k for k, run in self.matlab_labels.items() if run == name)
        self.updating = True
        choice = self.ptreedata.child("MATLAB run", "Run")
        choice.setLimits(list(self.matlab_labels))
        choice.setValue(label)
        self.updating = False
        self.show_matlab_run(path.parent, name)

    def matlab_setting_changed(self, setting: str, value):
        if self.matlab_run is None:
            return
        if setting == "Run":
            self.show_matlab_run(self.matlab_run.directory, self.matlab_labels[value])
        else:
            self.update_average(setting)

    def show_matlab_run(self, directory: Path, name: str):
        try:
            run = abr_files.read_matlab_run(directory, name)
        except (ValueError, OSError) as e:
            self.show_error(str(e))
            return
        self.matlab_run = run
        self.run_data = None
        self.use_filter_defaults("matlab")
        self.plot_widget.clear()
        self.panels = {}
        self.panels["average"] = self.plot_widget.addPlot(row=0, col=0, title="Average response")
        self.panels["average"].setLabel("bottom", "Time", units="ms")
        self.panels["average"].setLabel("left", "dB SPL")
        self.ptreedata.child("File", "Name").setValue(name)
        self.show_file_status(f"MATLAB {run.stim:s} run, {len(run.waves):d} frequencies")
        self.show_info(
            {
                "folder": str(directory),
                "run": name,
                "type": run.stim,
                "levels (dB SPL)": run.levels,
                "frequencies (Hz)": [f for f in run.waves if f > 0.0],
            }
        )
        self.update_average("Run")

    def calculate_responses(self, hpf, lpf) -> abr_files.Responses:
        if self.matlab_run is not None:
            rate = self.value("MATLAB run", "Sample rate (Hz)")
            return abr_files.compute_matlab_responses(self.matlab_run, rate, hpf, lpf)
        interleaved_set = self.value("Average", "Interleaved set")
        return abr_files.compute_responses(self.run_data, hpf, lpf, interleaved_set)

    def update_average(self, changed: str):
        if self.run_data is None and self.matlab_run is None:
            return
        if changed in ["Frequency", "Invert", "Display"]:  # the averages have not changed
            self.plot_average()
            return
        hpf, lpf = self.filter_settings()
        try:
            self.responses = self.calculate_responses(hpf, lpf)
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
        polarity = -1.0 if self.value("Average", "Invert") else 1.0
        traces = [polarity * self.responses.traces[(db, frequency)] for db in levels]
        view, note = self.current_view()
        trials = None
        if view != VIEW_AVERAGE:
            trials = [polarity * self.trial_responses[(db, frequency)] for db in levels]
        spacing = TRACE_SPACING_FACTOR * self.trace_extent(view, traces, trials) or 1.0
        panel = self.panels["average"]
        panel.clear()
        t_ms = self.responses.t_ms
        for i, trace in enumerate(traces):
            offset = i * spacing
            if view == VIEW_SD:
                self.plot_sd(panel, t_ms, trace + offset, abr_files.trial_sd(trials[i]))
            elif view == VIEW_TRACES:
                self.plot_single_traces(panel, t_ms, self.shown_trials(trials[i]) + offset)
            width = MEAN_LINE_WIDTH if view != VIEW_AVERAGE else 1
            panel.plot(t_ms, trace + offset, pen=pg.mkPen(COLOR_RESPONSE, width=width))
        ticks = [(i * spacing, f"{db:.0f}") for i, db in enumerate(levels)]
        panel.getAxis("left").setTicks([ticks])
        panel.setTitle(f"Average: {label:s} (traces {spacing:.2f} µV apart){note:s}")

    def current_view(self) -> tuple:
        """The view that can be drawn now, and a note for the plot title.

        If the trial responses are not ready, the average is shown while they are
        calculated, and the plot is redrawn when they arrive.
        """
        view = self.value("Average", "Display")
        if view == VIEW_AVERAGE:
            return VIEW_AVERAGE, ""
        if self.matlab_run is not None:
            return VIEW_AVERAGE, "; single traces are not stored in MATLAB runs"
        request = self.current_trial_request()
        if request == self.trial_request:
            n_trials = next(iter(self.trial_responses.values())).shape[0]
            shown = min(n_trials, MAX_SINGLE_TRACES) if view == VIEW_TRACES else n_trials
            count = f"{shown:d} of {n_trials:d}" if shown < n_trials else f"{n_trials:d}"
            return view, f"; {count:s} trials"
        self.request_trial_responses(request)
        return VIEW_AVERAGE, "; calculating single traces"

    def filter_settings(self) -> tuple:
        hpf = self.value("Average", "High pass (Hz, 0 = off)") or None
        lpf = self.value("Average", "Low pass (Hz, 0 = off)") or None
        return hpf, lpf

    def current_trial_request(self) -> tuple:
        interleaved_set = self.value("Average", "Interleaved set")
        return (self.run_generation, *self.filter_settings(), interleaved_set)

    def is_calculating(self) -> bool:
        return self.calculator is not None and self.calculator.isRunning()

    def request_trial_responses(self, request: tuple):
        if self.is_calculating():
            if self.calculator.request != request:
                self.calculator.requestInterruption()  # restarted when it has stopped
            return
        *_, interleaved_set = request
        self.calculator = TrialCalculator(
            self.run_data, request, *self.filter_settings(), interleaved_set, parent=self
        )
        self.calculator.signal_progress.connect(
            lambda count, total: self.set_status(f"Calculating traces {count:d} / {total:d}")
        )
        self.calculator.signal_failed.connect(self.show_error)
        self.calculator.signal_done.connect(self.trial_responses_done)
        self.calculator.start()

    def trial_responses_done(self, request: tuple, responses):
        """Keep the result, unless the settings changed while it was being calculated."""
        self.set_status(self.file_status)
        self.calculator.wait()
        if responses is not None and request == self.current_trial_request():
            self.trial_request = request
            self.trial_responses = responses
        if self.run_data is not None:
            self.plot_average()

    @staticmethod
    def shown_trials(trials: np.ndarray) -> np.ndarray:
        if trials.shape[0] <= MAX_SINGLE_TRACES:
            return trials
        return trials[np.linspace(0, trials.shape[0] - 1, MAX_SINGLE_TRACES).astype(int)]

    @classmethod
    def trace_extent(cls, view: str, traces: list, trials) -> float:
        """The vertical extent of what is drawn at each level, for spacing the levels."""
        if view == VIEW_SD:
            sds = [abr_files.trial_sd(t) for t in trials]
            return max(np.max(m + s) - np.min(m - s) for m, s in zip(traces, sds))
        if view == VIEW_TRACES:
            low, high = ROBUST_PERCENTILES
            ranges = [np.percentile(cls.shown_trials(t), [low, high]) for t in trials]
            return max(top - bottom for bottom, top in ranges)
        return max(np.ptp(trace) for trace in traces)

    @staticmethod
    def plot_sd(panel, t_ms, mean: np.ndarray, sd: np.ndarray):
        upper = panel.plot(t_ms, mean + sd, pen=pg.mkPen(INVISIBLE_PEN))
        lower = panel.plot(t_ms, mean - sd, pen=pg.mkPen(INVISIBLE_PEN))
        fill = pg.mkColor(COLOR_RESPONSE)
        fill.setAlpha(SD_FILL_ALPHA)
        panel.addItem(pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(fill)))

    @staticmethod
    def plot_single_traces(panel, t_ms, trials: np.ndarray):
        """One item with gaps between the traces, which is much faster than one per trace."""
        n_trials, n_points = trials.shape
        x = np.tile(np.append(t_ms, np.nan), n_trials)
        y = np.column_stack((trials, np.full(n_trials, np.nan))).ravel()
        faint = pg.mkColor(COLOR_SINGLE_TRACE)
        faint.setAlpha(SINGLE_TRACE_ALPHA)
        panel.plot(x, y, pen=pg.mkPen(faint), connect="finite")

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
        self.matlab_run = None
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
        if self.is_calculating():
            self.calculator.requestInterruption()
            self.calculator.wait(30000)
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
