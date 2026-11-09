#!/usr/bin/env python
"""
Calibration window for the speaker and microphone calibrations.

Run on its own from the top directory of the repository:
    python -m src.calibration.calibration_gui
    python -m src.calibration.calibration_gui --mock   (simulated hardware)

It is also opened from pyabr3, with the "Calibrate" button in the Actions group.
"""

import argparse
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph import dockarea as PGD
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import pyqtSignal
from pylibrary.tools import cprint as CP

from src.calibration import calc, engine
from src.calibration.cal_display import (
    COLOR_BANDPASS,
    COLOR_MEASURED,
    COLOR_NOISE,
    plot_noise_floor,
)
from src.calibration.hardware import (
    CalibrationError,
    MockCalibrationHardware,
    PyStimCalibrationHardware,
)

# colors follow the measurement in every panel
COLOR_STIMULUS = "#C8C8C8"
OUTSIDE_WINDOW_ALPHA = 90
SPL_CURVE_STEP = 2000.0  # Hz, as in calibrations.m
LEGEND_OFFSET = (-10, 10)
LEGEND_BRUSH = (0, 0, 0, 190)
LEGEND_TEXT = "#E0E0E0"


class CalibrationWorker(QtCore.QThread):
    """Runs a calibration away from the GUI thread.

    run_fn is called with (stop_event, log, on_data), and returns the saved data.
    """

    signal_log = pyqtSignal(str)
    signal_data = pyqtSignal(object)
    signal_failed = pyqtSignal(str)
    signal_complete = pyqtSignal(object)

    def __init__(self, run_fn, parent=None):
        super().__init__(parent)
        self.run_fn = run_fn
        self.stop_event = engine.threading.Event()

    def run(self):
        try:
            result = self.run_fn(self.stop_event, self.signal_log.emit, self.signal_data.emit)
        except CalibrationError as e:
            self.signal_failed.emit(str(e))
        except Exception:
            self.signal_failed.emit(traceback.format_exc())
        else:
            self.signal_complete.emit(result)


def parse_frequencies(text: str, nyquist: float) -> np.ndarray:
    """Convert a list of frequencies in kHz (commas or spaces between them) to Hz."""
    tokens = [t for t in re.split(r"[,\s\[\]]+", text) if t]
    try:
        freqs = 1000.0 * np.array([float(t) for t in tokens])
    except ValueError as e:
        raise CalibrationError(f"Could not read the frequency list: {e!s}")
    if len(freqs) < 2:
        raise CalibrationError("At least two frequencies are needed")
    if np.any(freqs <= 0.0) or np.any(freqs >= nyquist):
        raise CalibrationError(f"Frequencies must be between 0 and {nyquist / 1000.0:.1f} kHz")
    return freqs


class CalibrationWindow(QtWidgets.QMainWindow):
    """Settings, plots and log for the calibrations.

    Parameters
    ----------
    hardware : CalibrationHardware
    config : dict
        The contents of the pyabr3 configuration file.
    embedded : bool
        True when opened from pyabr3.
    """

    def __init__(self, hardware, config: dict, embedded: bool = False):
        super().__init__()
        self.hw = hardware
        self.config = config
        self.embedded = embedded
        self.worker = None
        self.current_cmd = None
        self.panels: dict = {}
        self.setWindowTitle(f"ABR Calibration - {self.hw.description:s}")
        self.setGeometry(150, 80, 1300, 900)
        self.build_gui()
        self.apply_speaker_preset(self.ptreedata.child("Setup", "Speaker").value())
        self.build_speaker_panels()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

    def default_cal_dir(self) -> Path:
        """Where pyabr3 reads its calibration. Mock runs never write there."""
        if self.hw.is_mock:
            return Path(tempfile.gettempdir(), "pyabr3_mock_calibration")
        return Path(self.config["calfile"].replace("\\", "/")).parent

    def build_parameters(self) -> list:
        return [
            {
                "name": "Setup",
                "type": "group",
                "children": [
                    {
                        "name": "Microphone",
                        "type": "list",
                        "limits": list(engine.MICROPHONES),
                        "value": "7016#10252",
                    },
                    {
                        "name": "Speaker",
                        "type": "list",
                        "limits": list(engine.SPEAKERS),
                        "value": "MF1",
                    },
                    {"name": "Frequencies (kHz)", "type": "str", "value": ""},
                    {"name": "Ref SPL (maxtones)", "type": "float", "value": 110.0},
                    {"name": "Cal directory", "type": "str", "value": str(self.default_cal_dir())},
                    {"name": "Test calc (no save)", "type": "bool", "value": False},
                ],
            },
            {
                "name": "Manual settings (verify)",
                "type": "group",
                "children": [
                    {
                        "name": "Mic preamp gain (dB)",
                        "type": "float",
                        "value": 20.0,
                        "limits": [0.0, 60.0],
                    },
                    {"name": "Speaker driver attn (dB)", "type": "float", "value": -6.0},
                ],
            },
            {
                "name": "Programmable attenuator",
                "type": "group",
                "children": [
                    {"name": "Attn (dB)", "type": "float", "value": 20.0, "limits": [0.0, 120.0]},
                ],
            },
            {
                "name": "Actions",
                "type": "group",
                "children": [
                    {"name": "Microphone (94 dB)", "type": "action"},
                    {"name": "Microphone (104 dB)", "type": "action"},
                    {"name": "Calibrate Speaker", "type": "action"},
                    {"name": "Check Speaker (75 dB)", "type": "action"},
                    {"name": "Noise Floor", "type": "action"},
                    {"name": "Stop", "type": "action"},
                    {"name": "Close", "type": "action"},
                ],
            },
            {
                "name": "Status",
                "type": "group",
                "children": [
                    {
                        "name": "Hardware",
                        "type": "str",
                        "value": self.hw.description,
                        "readonly": True,
                    },
                    {"name": "State", "type": "str", "value": "Ready", "readonly": True},
                    {"name": "Progress", "type": "str", "value": "", "readonly": True},
                ],
            },
        ]

    def build_gui(self):
        ptreewidth = 300
        self.dockArea = PGD.DockArea()
        self.setCentralWidget(self.dockArea)
        self.Dock_Params = PGD.Dock("Parameters", size=(ptreewidth, 900))
        self.Dock_Plots = PGD.Dock("Measurements", size=(1000, 700))
        self.Dock_Log = PGD.Dock("Log", size=(1000, 200))
        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_Plots, "right", self.Dock_Params)
        self.dockArea.addDock(self.Dock_Log, "bottom", self.Dock_Plots)

        self.ptreedata = Parameter.create(
            name="Calibration", type="group", children=self.build_parameters()
        )
        self.ptree = ParameterTree()
        self.ptree.setParameters(self.ptreedata, showTop=False)
        self.Dock_Params.addWidget(self.ptree)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.Dock_Plots.addWidget(self.plot_widget)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.log_text.setFont(font)
        self.Dock_Log.addWidget(self.log_text)

    def add_panel(self, key: str, title: str, xlabel: str, ylabel: str, row: int, col: int):
        panel = self.plot_widget.addPlot(row=row, col=col, title=title)
        panel.setLabel("bottom", xlabel)
        panel.setLabel("left", ylabel)
        panel.showGrid(x=True, y=True, alpha=0.15)
        self.panels[key] = panel
        return panel

    @staticmethod
    def add_legend(panel):
        panel.addLegend(offset=LEGEND_OFFSET, brush=LEGEND_BRUSH, labelTextColor=LEGEND_TEXT)

    def fast_trace_panel(self, panel):
        panel.setDownsampling(auto=True, mode="peak")
        panel.setClipToView(True)

    def build_speaker_panels(self):
        self.plot_widget.clear()
        self.panels = {}
        self.fast_trace_panel(self.add_panel("stimulus", "Stimulus", "t (s)", "V", 0, 0))
        recording = self.add_panel("recording", "Microphone", "t (s)", "V", 0, 1)
        self.fast_trace_panel(recording)
        self.add_legend(recording)
        spectrum = self.add_panel("spectrum", "Spectrum near stimulus", "F (Hz)", "V²/Hz", 1, 0)
        self.add_legend(spectrum)
        spl = self.add_panel("spl", "Calibration and noise floor", "F (Hz)", "dB SPL", 1, 1)
        spl.setLogMode(x=True, y=False)
        self.add_legend(spl)

    def build_microphone_panels(self):
        self.plot_widget.clear()
        self.panels = {}
        self.fast_trace_panel(self.add_panel("stimulus", "Stimulus", "t (s)", "V", 0, 0))
        recording = self.add_panel("recording", "Microphone", "t (s)", "V", 1, 0)
        self.fast_trace_panel(recording)
        self.add_legend(recording)
        self.add_panel("spectrum", "Bandpassed spectrum", "F (Hz)", "V²/Hz", 2, 0).setXRange(
            900, 1100
        )
        self.add_panel("cosinor", "Cosinor amplitude", "F (Hz)", "V", 3, 0).setXRange(900, 1100)

    def build_noise_panels(self):
        self.plot_widget.clear()
        self.panels = {}
        self.fast_trace_panel(self.add_panel("recording", "Microphone", "t (s)", "V", 0, 0))
        spectrum = self.add_panel("spectrum", "Noise spectrum", "F (Hz)", "dB", 1, 0)
        spectrum.setLogMode(x=True, y=False)

    def command_dispatcher(self, param, changes):
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            match path[0]:
                case "Setup":
                    if path[1] == "Speaker" and change == "value":
                        self.apply_speaker_preset(data)
                case "Actions":
                    match path[1]:
                        case "Microphone (94 dB)":
                            self.start_run("microphone")
                        case "Microphone (104 dB)":
                            self.start_run("microphone104")
                        case "Calibrate Speaker":
                            self.start_run("calibrate")
                        case "Check Speaker (75 dB)":
                            self.start_run("checkcal")
                        case "Noise Floor":
                            self.start_run("noise_floor")
                        case "Stop":
                            self.stop()
                        case "Close":
                            self.close()

    def apply_speaker_preset(self, speaker: str):
        preset = engine.SPEAKERS[speaker]
        text = ", ".join(f"{f:g}" for f in preset["freqs_khz"])
        self.ptreedata.child("Setup", "Frequencies (kHz)").setValue(text)
        self.ptreedata.child("Setup", "Ref SPL (maxtones)").setValue(preset["maxtones"])
        self.ptreedata.child("Programmable attenuator", "Attn (dB)").setValue(preset["attn"])

    def value(self, *path):
        return self.ptreedata.child(*path).value()

    def confirm_settings(self, message: str) -> bool:
        """Manual settings cannot be read by the program, so the user must confirm them."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Hardware Settings",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def show_error(self, message: str):
        CP.cprint("r", message)
        self.append_log(f"ERROR: {message:s}")
        QtWidgets.QMessageBox.critical(self, "Calibration", message)

    def append_log(self, message: str):
        self.log_text.appendPlainText(message)

    def set_status(self, state: str, progress: str | None = None):
        self.ptreedata.child("Status", "State").setValue(state)
        if progress is not None:
            self.ptreedata.child("Status", "Progress").setValue(progress)

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def start_run(self, cmd: str):
        if self.is_running():
            return
        cal_dir = Path(self.value("Setup", "Cal directory"))
        try:
            self.hw.connect()
            if self.hw.is_mock:
                cal_dir.mkdir(parents=True, exist_ok=True)
            if not cal_dir.is_dir():
                raise CalibrationError(f"The calibration directory does not exist: {cal_dir!s}")
            run_fn = self.make_run_function(cmd, cal_dir)
        except CalibrationError as e:
            self.show_error(str(e))
            return

        speaker = self.value("Setup", "Speaker")
        speaker_run = cmd in ["calibrate", "checkcal"]
        attn = self.value("Programmable attenuator", "Attn (dB)") if speaker_run else None
        message = engine.verify_message(
            cmd,
            speaker,
            self.value("Setup", "Microphone"),
            self.value("Manual settings (verify)", "Mic preamp gain (dB)"),
            self.value("Manual settings (verify)", "Speaker driver attn (dB)"),
            attn,
        )
        if not self.confirm_settings(message):
            self.append_log("Calibration cancelled at the manual settings check.")
            return

        if self.hw.is_mock:
            self.hw.calibrator_db = {"microphone": 94.0, "microphone104": 104.0}.get(cmd)
        self.current_cmd = cmd
        if cmd == "noise_floor":
            self.build_noise_panels()
        elif cmd.startswith("microphone"):
            self.build_microphone_panels()
        else:
            self.build_speaker_panels()
        self.set_status(f"Running: {cmd:s}", "")
        self.worker = CalibrationWorker(run_fn, parent=self)
        self.worker.signal_log.connect(self.append_log)
        self.worker.signal_data.connect(self.update_plots)
        self.worker.signal_failed.connect(self.run_failed)
        self.worker.signal_complete.connect(self.run_complete)
        self.worker.start()

    def make_run_function(self, cmd: str, cal_dir: Path):
        gain = self.value("Manual settings (verify)", "Mic preamp gain (dB)")
        microphone = self.value("Setup", "Microphone")
        if cmd == "noise_floor":
            return lambda stop, log, data: engine.measure_noise_floor(
                self.hw, cal_dir, stop, log, data
            )
        if cmd.startswith("microphone"):
            settings = engine.MicSettings(
                microphone=microphone,
                ref_db=104.0 if cmd == "microphone104" else 94.0,
                gain=gain,
                cal_dir=cal_dir,
                test_mode=self.value("Setup", "Test calc (no save)"),
            )
            return lambda stop, log, data: engine.calibrate_microphone(
                self.hw, settings, stop, log, data
            )
        settings = engine.SpeakerSettings(
            speaker=self.value("Setup", "Speaker"),
            microphone=microphone,
            freqs=parse_frequencies(
                self.value("Setup", "Frequencies (kHz)"), self.hw.sample_freq / 2.0
            ),
            attn=self.value("Programmable attenuator", "Attn (dB)"),
            ref_spl=self.value("Setup", "Ref SPL (maxtones)"),
            gain=gain,
            cal_dir=cal_dir,
            check=cmd == "checkcal",
        )
        return lambda stop, log, data: engine.calibrate_speaker(self.hw, settings, stop, log, data)

    def stop(self):
        if self.is_running():
            self.worker.stop_event.set()
            self.set_status("Stopping")

    def run_failed(self, message: str):
        self.set_status("Error")
        self.show_error(message)

    def run_complete(self, result):
        if result is None:
            self.set_status("Stopped")
            return
        self.set_status("Done")
        if self.embedded and self.current_cmd == "calibrate":
            self.append_log("Use Read Cal File in pyabr3 to select this calibration.")

    def update_plots(self, data: dict):
        if self.current_cmd == "noise_floor":
            self.update_noise_plots(data)
        elif self.current_cmd is not None and self.current_cmd.startswith("microphone"):
            self.update_microphone_plots(data)
        else:
            self.update_speaker_plots(data)

    @staticmethod
    def window_plot(panel, x, y, ts1: int, ts2: int, color: str, name: str):
        """Plot the region used for measurement, with the rest of the trace dimmed."""
        dim = pg.mkColor(color)
        dim.setAlpha(OUTSIDE_WINDOW_ALPHA)
        panel.plot(x[:ts1], y[:ts1], pen=pg.mkPen(dim))
        panel.plot(x[ts1 - 1 : ts2], y[ts1 - 1 : ts2], pen=pg.mkPen(color), name=name)
        panel.plot(x[ts2 - 1 :], y[ts2 - 1 :], pen=pg.mkPen(dim))

    def update_speaker_plots(self, d: dict):
        i, freq = d["index"], d["freq"]
        self.set_status(self.ptreedata.child("Status", "State").value(), f"{i + 1:d} / {len(d['freqs']):d}")

        p = self.panels["stimulus"]
        p.clear()
        p.plot(d["tstim"], d["wave"], pen=pg.mkPen(COLOR_STIMULUS))

        p = self.panels["recording"]
        p.clear()
        ts1, ts2 = d["ts1"], d["ts2"]
        self.window_plot(p, d["trec"], d["ch2"], ts1, ts2, COLOR_MEASURED, "Raw")
        self.window_plot(p, d["trec"], d["ybp"], ts1, ts2, COLOR_BANDPASS, "Bandpass")
        self.window_plot(p, d["trec"], d["ynf"], ts1, ts2, COLOR_NOISE, "Noise")

        p = self.panels["spectrum"]
        p.clear()
        p.plot(d["bp_freqs"], d["bp_psd"], pen=pg.mkPen(COLOR_BANDPASS), name="Bandpass")
        p.plot(d["nf_freqs"], d["nf_psd"], pen=pg.mkPen(COLOR_NOISE), name="Noise")
        p.setXRange(freq - 1000.0, freq + 1000.0)

        self.plot_spl_summary(d)

    def plot_spl_summary(self, d: dict):
        n = d["index"] + 1
        order = np.argsort(d["freqs"][:n])
        freqs = d["freqs"][:n][order]
        spl = self.panels["spl"]
        spl.clear()
        plot_noise_floor(spl, freqs, d["dbspl_nf"][:n][order])
        dbspl_bp = d["dbspl_bp"][:n][order]
        spl.plot(
            freqs,
            dbspl_bp,
            pen=None,
            symbol="o",
            symbolSize=8,
            symbolBrush=COLOR_BANDPASS,
            symbolPen=None,
        )
        if n > 1:
            spl_freqs = np.arange(freqs[0], freqs[-1], SPL_CURVE_STEP)
            spl.plot(
                spl_freqs,
                calc.spl_at_f(freqs, dbspl_bp, spl_freqs),
                pen=pg.mkPen(COLOR_BANDPASS),
                name="Bandpass",
            )
            spl.setXRange(np.log10(freqs[0]), np.log10(freqs[-1]))

    def update_microphone_plots(self, d: dict):
        p = self.panels["stimulus"]
        p.clear()
        p.plot(d["tstim"], d["wave"], pen=pg.mkPen(COLOR_STIMULUS))
        p = self.panels["recording"]
        p.clear()
        p.plot(d["trec"], d["ch2"], pen=pg.mkPen(COLOR_MEASURED), name="Raw")
        p.plot(d["trec"], d["ybp"], pen=pg.mkPen(COLOR_BANDPASS), name="Bandpass")
        p = self.panels["spectrum"]
        p.clear()
        p.plot(d["bp_freqs"], d["bp_psd"], pen=pg.mkPen(COLOR_BANDPASS))
        p = self.panels["cosinor"]
        p.clear()
        p.plot(d["fr_c"], d["amp_c"], pen=pg.mkPen(COLOR_MEASURED))

    def update_noise_plots(self, d: dict):
        p = self.panels["recording"]
        p.clear()
        p.plot(d["trec"], d["ch2nf"], pen=pg.mkPen(COLOR_NOISE))
        p = self.panels["spectrum"]
        p.clear()
        p.plot(d["freqs"][1:], 10.0 * np.log10(d["psd"][1:]), pen=pg.mkPen(COLOR_NOISE))

    def closeEvent(self, event):
        if self.is_running():
            self.worker.stop_event.set()
            self.worker.wait(5000)
        self.hw.shutdown()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="ABR speaker and microphone calibration")
    parser.add_argument("--mock", action="store_true", help="use simulated hardware")
    parser.add_argument("--config", default="config/abrs.cfg", help="pyabr3 configuration file")
    args = parser.parse_args()
    app = pg.mkQApp("ABR Calibration")
    app.setStyle("fusion")
    config = configfile.readConfigFile(args.config)
    hardware = MockCalibrationHardware() if args.mock else PyStimCalibrationHardware(config)
    window = CalibrationWindow(hardware, config)
    window.show()
    sys.exit(QtWidgets.QApplication.instance().exec())


if __name__ == "__main__":
    main()
