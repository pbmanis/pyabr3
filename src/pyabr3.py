#!/usr/bin/env python
"""
pyabr3 : a python program for ABR recordings. 
Parameter setup is defined in config/.cfg files (text files)

This program uses pystim for hardware control,
and sound.py for stimulus generation. 

2024-2025 Paul B. Manis

"""

import atexit
import datetime
import copy
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph import dockarea as PGD
from pyqtgraph import reload as reload
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import QObject, pyqtSignal, pyqtSlot

from src import convert_nested_ordered_dict
from src import presenter_thread as presenter_thread
from src import protocol_reader as protocol_reader
from src import pystim3 as pystim3
from src import wave_generator as wave_generator
from src import sound as sound  # for waveform generation
from src.build_parametertree import build_parametertree
import src.read_calibration as read_calibration
import src.read_abr as read_abr
import src.message_box as MBox
from src.calibration import cal_chooser, cal_display, calibration_gui
from src.calibration.hardware import MockCalibrationHardware, PyStimCalibrationHardware
from src.viewer import abr_files, bic_viewer, data_viewer

MAX_CLICK = 105.1  # SPL for maximum click

THREAD_PERIOD = 20  # thread period, in msec

ONLINE_UV_PER_VOLT = abr_files.VOLTS_TO_UV / abr_files.AMPLIFIER_GAIN  # uV at the electrodes

# the user manual: in a build it is in manual/, when running from the source it is built by
# "make pdf" in doc_tex/
MANUAL_PATHS = [Path("manual", "pyabr3_manual.pdf"), Path("doc_tex", "build", "pyabr3_manual.pdf")]


class PyABR(QtCore.QObject):

    signal_pause = pyqtSignal()
    signal_resume = pyqtSignal()
    signal_stop = pyqtSignal()
    signal_quit = pyqtSignal()

    def __init__(self, configtype: str = "abr"):
        """Register shutdown only after success, so startup failures do not quit a partial object."""
        assert configtype in ["test", "abr"]
        if configtype == "test":
            configfilename = "config/abrs_test.cfg"
        else:
            configfilename = "config/abrs.cfg"
        self.QColor = QtGui.QColor
        # get the configuration file:
        self.config = configfile.readConfigFile(configfilename)
        self.PS = pystim3.PyStim(
            required_hardware=self.config["required_hardware"],
            ni_devicename=self.config["NI_device"],
        )
        # check hardware
        self.hardware, self.sfin, self.sfout = self.PS.getHardware()
        # get calibration file ("calfile_left", if present, is the same thing as "calfile")
        self.calfile = Path(self.config.get("calfile_left", self.config["calfile"]))
        if configtype == "test" and not self.calfile.is_file():
            # the configured path is on the rig: use the copy in the repository or app
            self.calfile = Path("calfiles", "frequency_MF1.cal")
        self.caldata = read_calibration.get_calibration_data(self.calfile)
        now = datetime.datetime.now()
        self.WG = wave_generator.WaveGenerator(caldata=self.caldata)
        # optional calibration of a separate right speaker (for binaural modes).
        # Without it, the right ear uses the same calibration as the left.
        self.calfile_right, self.caldata_right = self.load_right_calibration(configtype)
        self.WG.caldata_right = self.caldata_right
        self.Presenter = None  # the thread object that will be used for presenting stimuli
        self.cal_window = None  # the calibration window, when it has been opened
        self.data_viewer = None  # the data viewer window, when it has been opened
        self.wave_matrix: dict = {}
        self.ptreedata = None  # pyqtgraph parametertree  - set to None until the tree is built
        self.protocol = None
        self.debugFlag = True
        self.link_traces = True
        self.raw_clear_N = 10
        self.read_protocol_directory()
        self.acq_mode = "test"
        self.basefn = None  # base file path/name for saving data
        self.run_directory = None  # where the current recording is saved
        self.buildGUI()
        self.PR.update_protocol(self.ptreedata)
        self.update_devices(devices=self.hardware)
        self.counter = 0
        self.ch1 = None
        self.ch2 = None
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        super(PyABR, self).__init__()

        self.TrialTimer = pg.QtCore.QTimer()
        self.TrialTimer.timeout.connect(self.recurring_timer)
        self.TrialCounter = 1

        self.threadpool = QtCore.QThreadPool()
        self.Presenter = presenter_thread.Presenter(parameters=self)
        # self.TrialTimer.timeout.connect(self.Presenter.next_stimulus)
        # connect signals FROM the Presenter thread
        self.Presenter.signal_data_ready.connect(self.update_ABR)
        self.Presenter.signal_error.connect(self.thread_error)
        self.Presenter.signal_hardware_error.connect(self.hardware_error)
        self.Presenter.signal_finished.connect(self.finished)
        self.Presenter.signal_progress.connect(self.update_progress)
        self.Presenter.signal_repetitions_done.connect(self.clear_average)
        self.Presenter.signal_mode_status.connect(self.update_mode)
        self.signal_pause.connect(self.Presenter.pause)
        self.signal_resume.connect(self.Presenter.resume)
        self.signal_stop.connect(self.Presenter.stop)
        self.signal_stop.connect(self.PS.cleanup_NIDAQ_RP21)
        self.signal_quit.connect(self.Presenter.quit)

        # note after building the GUI, everything is handled througn callbacks
        # in the command_dispatcher function.
        atexit.register(self.quit)

    def read_protocol_directory(self):
        """read_protocol_directory Get the protocols from the protocols directory
        Chose the default (current) protocol, then read it.
        This must be called before the GUI is built, but can be called afterwards
        to update the list of available prototols.
        """
        self.known_protocols = [p.name for p in Path("protocols").glob("*.cfg")]
        self.current_protocol = self.known_protocols[0]
        self.PR = protocol_reader.ProtocolReader(ptreedata=self.ptreedata)
        self.PR.read_protocol(protocolname=self.current_protocol)
        self.stim = self.PR.get_current_protocol()

    def buildGUI(self):
        """Build GUI and window"""

        white = self.QColor(255, 255, 255)
        black = self.QColor(0, 0, 0)
        red = self.QColor(255, 0, 0)
        # Define the table style for various parts dark scheme
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, self.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Text, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Button, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, red)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Link, self.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, self.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, self.QColor(0, 255, 0))

        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        self.app.setPalette(dark_palette)
        self.app.setStyleSheet(
            "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
        )
        ptreewidth = 250
        right_docks_width = 1024 - ptreewidth - 20
        right_docs_height = 800
        self.win = pg.QtWidgets.QMainWindow()

        self.dockArea = PGD.DockArea()
        self.win.setCentralWidget(self.dockArea)
        self.Dock_Params = PGD.Dock("Parameters", size=(ptreewidth, 1024))
        self.Dock_Recording = PGD.Dock("Recording", size=(right_docks_width, 1024))
        self.Dock_Preview = PGD.Dock("Preview", size=(right_docks_width, 1024))
        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_Recording, "right", self.Dock_Params)
        self.dockArea.addDock(self.Dock_Preview, "below", self.Dock_Recording)
        self.Dock_Calibration = PGD.Dock("Calibration", size=(right_docks_width, 1024))
        self.dockArea.addDock(self.Dock_Calibration, "below", self.Dock_Preview)
        self.Dock_BIC = PGD.Dock("BIC", size=(right_docks_width, 1024))
        self.dockArea.addDock(self.Dock_BIC, "below", self.Dock_Calibration)
        self.bic_panel = bic_viewer.BICPanel(start_dir=self.bic_start_dir)
        self.Dock_BIC.addWidget(self.bic_panel)
        self.win.setWindowTitle("ABR Acquistion")
        self.win.setGeometry(100, 100, 1200, 1000)
        help_menu = self.win.menuBar().addMenu("Help")
        manual_action = help_menu.addAction("pyabr3 User Manual")
        manual_action.setShortcut(QtGui.QKeySequence.StandardKey.HelpContents)
        manual_action.triggered.connect(self.show_manual)
        # print(dir(self.dockArea))
        self.Dock_Params.raise_()
        self.ptree, self.ptreedata = build_parametertree(
            self.known_protocols,
            self.current_protocol,
            stimuli=self.stim["stimuli"],
            data_path_names=list(self.get_data_paths()),
        )
        self.PR.update_protocol(ptreedata=self.ptreedata)  # show the protocol's values
        self.update_calibration_labels()
        self.Dock_Params.addWidget(self.ptree)

        self.build_recording_graphs()
        self.calibration_display = cal_display.CalibrationDisplay()
        self.Dock_Calibration.addWidget(self.calibration_display)
        self.calibration_display.show_calibration(self.calfile, self.caldata)
        self.win.show()

    def build_recording_graphs(self):
        # add space for the graphs
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(50, 50, 50))
        view.setCentralItem(layout)

        self.plot_ABR_Raw = layout.addPlot()
        self.plot_ABR_Raw.getAxis("left").setLabel("uV", color="#ff0000")
        self.plot_ABR_Raw.setTitle("Raw ABR Signal", color="#ff0000")
        self.plot_ABR_Raw.getAxis("bottom").setLabel("t (msec)", color="#ff0000")
        # self.plot_ABR_Raw.setYRange(-10.0, 10.0)  # microvolts

        layout.nextRow()  # averaged abr trace
        self.plot_ABR_Average = layout.addPlot()  #
        if self.link_traces:
            self.plot_ABR_Average.setXLink(self.plot_ABR_Raw)
        self.plot_ABR_Average.setTitle("Average ABR Signal", color="#ff0000")
        self.plot_ABR_Average.getAxis("left").setLabel("uV", color="#0000ff")
        # self.plot_ABR_Average.setYRange(-10, 10.0)
        self.plot_ABR_Average.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        layout.nextRow()  # waveforms
        self.stimulus_waveform = layout.addPlot()  #
        if self.link_traces:
            self.stimulus_waveform.setXLink(self.plot_ABR_Raw)
        self.stimulus_waveform.setTitle("Waveform", color="#ff0000")
        self.stimulus_waveform.getAxis("left").setLabel("V", color="#0000ff")
        # self.stimulus_waveform.setYRange(-10, 10.0)
        self.stimulus_waveform.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        layout.nextRow()  #
        l2 = layout.addLayout(colspan=3, border=(50, 0, 0))  # embed a new layout
        l2.setContentsMargins(2, 2, 2, 2)
        self.plot_Amplitude_Level = l2.addPlot(Title="Amplitude-Intensity")
        self.plot_Amplitude_Level.getAxis("bottom").setLabel("dB (SPL)")
        self.plot_Amplitude_Level.getAxis("left").setLabel("uV")

        self.plt_map = l2.addPlot(Title="Map")
        self.plt_map.getAxis("bottom").setLabel("F (kHz)")
        self.plt_map.getAxis("left").setLabel("dB (SPL)")

        self.Dock_Recording.addWidget(view)

        
        view2 = pg.GraphicsView()
        layout2 = pg.GraphicsLayout(border=(50, 50, 50))
        view2.setCentralItem(layout2)
        self.cal_plot = layout2.addPlot()
        self.Dock_Preview.addWidget(view2)
        self.Dock_Recording.raise_()

    def command_dispatcher(self, param, changes):
        """command_dispatcher callback for parametertree
        This function is called when the parametertree is changed, including
        when the user clicks on a button or changes a value.

        Bug: this was connected in both __init__ and buildGUI, so each button press
        ran its action twice (including acquire). It is now connected only in __init__.

        Parameters
        ----------
        param : parametertree parameter
            parametertree parameter that was changed
        changes : This holds all the real changes
            _description_
        """

        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            match path[0]:
                case "Quit":
                    self.quit(atexit=False)

                case "Protocol":
                    self.current_protocol = data
                    self.PR.read_protocol(data, update=False)
                case "Actions":
                    match path[1]:
                        case "New Filename":
                            self.new_filename()
                        case "Test Acquisition":
                            self.acquire(mode="test")
                        case "Start Acquisition":
                            self.acquire(mode="collect")
                        case "Stop":
                            self.stop()
                        case "Pause":
                            self.pause()
                        case "Resume":
                            self.continue_trials()
                        case "Load Data File":
                            self.load_file()
                        case "Read Cal File":
                            self.load_cal_file()
                            self.update_calibration_labels()
                        case "Read Right Cal File":
                            self.load_right_cal_file()
                            self.update_calibration_labels()
                        case "Calibrate":
                            self.open_calibration()

        self.ptreedata.sigTreeStateChanged.disconnect(self.command_dispatcher)
        self.PR.update_protocol(ptreedata=self.ptreedata)
        self.protocol = self.PR.get_current_protocol()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

    def quit(self, atexit=True):
        self.TrialTimer.stop()
        self.signal_quit.emit()
        if self.Presenter is not None:
            self.Presenter.end_thread()
            self.threadpool.waitForDone(5 * THREAD_PERIOD)  # end thread'
        self.win.close()
        if not atexit:
            exit()

    def hardware_error(self, message: str):
        """The acquisition stopped because the circuit is not set up as needed (for example
        the sweep-done flag is not reset by the start trigger). Tell the user, and stop.
        Related: presenter_thread.Presenter.run, pystim3.CircuitError."""
        self.stop()
        MBox.MessageBox(message, title="Hardware")

    def thread_error(self, data):
        print("Thread Error: ", data)
        self.quit()

    def new_filename(self):
        """
        Create a new filename for the data file
        """

        self.protocol["filename"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ptreedata.child("Parameters").setValue(self.protocol)
        print("new file: ", self.protocol["filename"])

    def show_manual(self):
        """Open the user manual (PDF) in the system's PDF viewer."""
        manual = next((p for p in MANUAL_PATHS if p.is_file()), None)
        if manual is None:
            MBox.MessageBox(
                "The user manual was not found. Build it with 'make pdf' in doc_tex/.",
                title="User Manual",
            )
            return
        url = QtCore.QUrl.fromLocalFile(str(manual.resolve()))
        if not QtGui.QDesktopServices.openUrl(url):
            MBox.MessageBox(
                f"Could not open the user manual:\n{manual.resolve()}", title="User Manual"
            )

    def show_data_viewer(self):
        if self.data_viewer is None:
            self.data_viewer = data_viewer.DataViewer()
        self.data_viewer.show()
        self.data_viewer.raise_()
        return self.data_viewer

    def load_file(self):
        """
        Open a data file in the data viewer.

        Bug: this only read .pkl files, not the .p files that acquisition writes, and
        update_plots is empty, so nothing was shown. Related: src/viewer/
        """
        self.show_data_viewer().open_file_dialog()


    def bic_start_dir(self) -> str:
        """Where the BIC panel's file dialog starts: the current run if there is one, otherwise
        the first data location."""
        if self.run_directory is not None and Path(self.run_directory).is_dir():
            return str(self.run_directory)
        return str(next(iter(self.get_data_paths().values())))

    def load_right_calibration(self, configtype: str):
        """
        Read the calibration of the right speaker, if the configuration names one
        ("calfile_right"). Returns (path, caldata), or (None, None) when the right ear
        uses the calibration of the left (free-field operation, and the default).
        Related: wave_generator.WaveGenerator.caldata_for
        """
        name = self.config.get("calfile_right")
        if not name:
            return None, None
        path = Path(name)
        if configtype == "test" and not path.is_file():
            # the configured path is on the rig: with the test configuration, use the left
            print(f"Right calibration file {str(path):s} not found; using the left calibration")
            return None, None
        return path, read_calibration.get_calibration_data(path)

    def load_cal_file(self):
        """
        Choose a calibration file, and use it for stimuli and saved data.

        Bug: this reloaded the configured file and updated self.caldata only, so the
        wave generator kept the calibration read at startup. It now updates both.
        Related: src/calibration/cal_chooser.py
        """
        chosen = cal_chooser.choose_calibration_file(
            self.calfile.parent, current=self.calfile, parent=self.win
        )
        if chosen is None:
            return
        try:
            caldata = read_calibration.get_calibration_data(chosen)
            self.calibration_display.show_calibration(chosen, caldata)
        except Exception as e:  # older or damaged files fail in many different ways
            MBox.MessageBox(f"Could not read {chosen.name:s}: {e!s}", title="Calibration")
            return
        self.calfile = chosen
        self.caldata = caldata
        self.WG.caldata = self.caldata
        self.Dock_Calibration.raiseDock()

    def load_right_cal_file(self):
        """
        Choose the calibration of the right path (second speaker, NI ao1), used in the
        binaural modes. Until one is chosen (or "calfile_right" is in the configuration)
        the right path uses the left calibration.
        Related: load_cal_file, wave_generator.WaveGenerator.caldata_for
        """
        current = self.calfile_right if self.calfile_right is not None else self.calfile
        chosen = cal_chooser.choose_calibration_file(current.parent, current=current, parent=self.win)
        if chosen is None:
            return
        try:
            caldata = read_calibration.get_calibration_data(chosen)
            self.calibration_display.show_calibration(chosen, caldata)
        except Exception as e:  # older or damaged files fail in many different ways
            MBox.MessageBox(f"Could not read {chosen.name:s}: {e!s}", title="Calibration")
            return
        self.calfile_right = chosen
        self.caldata_right = caldata
        self.WG.caldata_right = self.caldata_right
        self.Dock_Calibration.raiseDock()

    def update_calibration_labels(self):
        """Show which calibration file each output path uses (Calibration group of the tree)."""
        self.ptreedata.child("Calibration", "Left (free-field) path").setValue(self.calfile.name)
        right = "same as left" if self.calfile_right is None else self.calfile_right.name
        self.ptreedata.child("Calibration", "Right path").setValue(right)

    def open_calibration(self):
        """
        Open the calibration window. Without the RP2.1 (for example, with the
        test configuration) the window uses simulated hardware, and says so in its title.
        """
        if self.cal_window is not None and self.cal_window.isVisible():
            self.cal_window.raise_()
            return
        if self.TrialTimer.isActive():
            MBox.MessageBox("Stop the acquisition before calibrating.", title="Calibration")
            return
        if "RP21" in self.hardware:
            hardware = PyStimCalibrationHardware(self.config, pystim=self.PS)
        else:
            hardware = MockCalibrationHardware()
        self.cal_window = calibration_gui.CalibrationWindow(hardware, self.config, embedded=True)
        self.cal_window.show()

    def plot_stimulus_wave(self, n: int = 0): # , n_sample_delay:int=65):
        first_sound = self.wave_matrix[list(self.wave_matrix.keys())[n]]
        # samp_delay = n_sample_delay/first_sound["rate"]
        # print("nsam del: ", n_sample_delay, "samp_delay: ", samp_delay)
        t = np.arange(0, len(first_sound["sound"]) / float(first_sound["rate"]), 1.0 / first_sound["rate"])
        if len(t) > len(first_sound["sound"]):
            t = t[:len(first_sound["sound"])]
        t = (t + 750e-6) * 1e3  # msec; measured delay added in circuit in pystim3.abr
        if (self.raw_clear_N > 0) and ((n % self.raw_clear_N) == 0):
            self.stimulus_waveform.clear()
            self.stimulus_waveform.setXRange(0, np.max(t))
        self.stimulus_waveform.plot(
            t,
            first_sound["sound"],
            pen=pg.mkPen("y"),
        )

        # self.stimulus_waveform.autoRange()

    def update_plots(self):
        pass

    def finished(self):
        """
        Called when the Presenter thread has finished with valid data
        Nominally, this might automatically save the data to a file
        """
        pass

    def update_progress(
        self,
        wavetype: str = "",
        wave_count: int = 0,
        max_wave_count: int = 0,
        rep_count: int = 0,
        max_rep_count: int = 0,
        db: float = 0.0,
        fr: float = 0.0,
    ):
        """Show the current stimulus in the Status group.

        Bug: Freq (kHz) showed Hz, and command_dispatcher copied each displayed level
        and frequency into the protocol's default_spl and default_frequency. The
        frequency is now shown in kHz, and the Status group no longer alters the protocol.
        """
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if wavetype in ["click", "tonepip"] and childs.name() == "dBSPL":
                        childs.setValue(f"{db:5.1f}")
                    if wavetype in ["tonepip"] and childs.name() == "Freq (kHz)":
                        childs.setValue(f"{fr / 1000.0:8.3f}")
                    if childs.name() == "Wave #":
                        childs.setValue(f"{wave_count+1:d} / {max_wave_count:d}")
                    if childs.name() == "Rep #":
                        childs.setValue(f"{rep_count:d} / {max_rep_count:d}")

    def update_mode(self, mode: str):
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if childs.name() == "Mode":
                        childs.setValue(f"{mode:s}")

    def update_devices(self, devices: str):
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if childs.name() == "Devices":
                        childs.setValue(f"{devices!s}")

    def recurring_timer(self):
        self.counter += 1  # nothing really to do here.
        time.sleep(0.01)

    def make_waveforms(self, wavetype, dbspls, frequencies):
        """
        thin wrapper for the wavegenerator waveofmr maker.
        """
        self.WG.setup(protocol=self.protocol, frequency=self.PS.Stimulus.NI_out_sampleFreq, config=self.config)  # output rate
        self.WG.make_waveforms(wavetype=wavetype, dbspls=dbspls, frequencies=frequencies)
        self.wave_matrix = self.WG.wave_matrix

    def make_and_plot(self, n: int, wavetype: str, dbspls: list, frequencies: list):
        self.make_waveforms(wavetype=wavetype, dbspls=dbspls, frequencies=frequencies)
        # for k in list(self.wave_matrix.keys()):
        #     print(k, np.max(self.wave_matrix[k]["sound"]))
        self.plot_stimulus_wave(n)

    def acquisition_in_progress(self) -> bool:
        """Return True while a protocol is running or paused. A presentation loop
        that has died on a hardware error does not count, so it cannot block the next run.
        """
        return (
            self.threadpool.activeThreadCount() > 0
            and self.Presenter._running
            and not self.Presenter._finished
        )

    def get_subject_id(self) -> str:
        """Return the Subject ID entered in the parameter tree (blank if none)."""
        return self.ptreedata.child("Subject Data", "Subject ID").value().strip()

    def get_data_paths(self) -> dict:
        """Return the data locations from the configuration, as {name: path}.
        A configuration with only the single "datapath" entry gives one location.
        """
        if "datapaths" in self.config:
            return dict(self.config["datapaths"])
        return {"default": self.config["datapath"]}

    def next_run_directory(self) -> Path:
        """Return the first unused directory for this subject and protocol, with a
        serial number as in acq4: <datapath>/2024-12-10-p572_000/interleaved_plateau.
        A repeat of the protocol goes in <datapath>/2024-12-10-p572_001/interleaved_plateau.
        """
        base = f"{self.basefn:s}-{self.get_subject_id():s}"
        protocol = Path(self.current_protocol).stem
        datapath = self.get_data_paths()[self.ptreedata.child("Data Path").value()]
        serial = 0
        while Path(datapath, f"{base:s}_{serial:03d}", protocol).exists():
            serial += 1
        return Path(datapath, f"{base:s}_{serial:03d}", protocol)

    def acquire(self, mode: str = "test"):
        """
        present and collect responses without saving - using
        abbreviated protocol
        Here, we just set things up, and display the stimulus
        that will be presented.

        Then we can start the player.
        In test mode, we do not save the data....

        Bug: update_ABR replaced the Subject ID with "testing", so no saved file
        identified its subject. Now a recording (mode "collect") will not start
        until a Subject ID has been entered, and that ID is what is saved.

        Bug: each acquisition started another Presenter.run loop, and the extra loop
        died on the "NIDAQ task" check in pystim3.play_sound, skipping a repetition.
        A loop is now started only if none is running, after the waveforms are set
        (see also Presenter.setWaveforms in presenter_thread.py).

        """
        if self.acquisition_in_progress():
            MBox.MessageBox("Press Stop before starting another acquisition.", title="Acquisition")
            return
        if mode == "collect" and not self.get_subject_id():
            MBox.MessageBox("Enter a Subject ID before starting.", title="Acquisition")
            return
        if mode == "collect" and any(c in self.get_subject_id() for c in '\\/:*?"<>|'):
            MBox.MessageBox('A Subject ID cannot contain \\ / : * ? " < > |', title="Acquisition")
            return
        if mode == "collect" and self.ptreedata.child("Data Path").value() not in self.get_data_paths():
            MBox.MessageBox("Choose a Data Path before starting.", title="Acquisition")
            return
        self.acq_mode = mode
        self.Dock_Recording.raiseDock()
        self.app.processEvents()
        self.protocol = self.PR.get_current_protocol()  # be sure we have current protocol data
        mode = protocol_reader.get_mode(self.protocol)
        required_hardware = protocol_reader.required_hardware_for_mode(
            self.config["required_hardware"], mode
        )
        try:
            self.PS = pystim3.PyStim(
                required_hardware=required_hardware,
                ni_devicename=self.config["NI_device"],
            )
        except pystim3.CircuitError as e:  # the circuit is not the one the acquisition needs
            MBox.MessageBox(str(e), title="Hardware")
            return
        except IOError as e:
            if mode == "freefield":
                raise
            MBox.MessageBox(str(e), title="Hardware")
            return
        self.PS.reset_hardware()
        self.TrialCounter = 1
        flist = self.protocol["stimuli"]["freqlist"]
        if self.protocol["protocol"]["stimulustype"] == "click":
            if len(flist) == 0:
                flist = [0] * len(self.protocol["stimuli"]["dblist"])
        try:
            self.make_and_plot(
                n=0,
                wavetype=self.protocol["protocol"]["stimulustype"],
                dbspls=self.protocol["stimuli"]["dblist"],
                frequencies=flist,
            )
        except wave_generator.UnreachableLevelError as e:
            MBox.MessageBox(
                f"{e!s}\n\nNothing was presented. Reduce the maximum level or the range, "
                "and save this as a new protocol.",
                title="Protocol",
            )
            return
        if mode == "collect":
            self.basefn = datetime.datetime.now().strftime("%Y-%m-%d")
            self.run_directory = self.next_run_directory()
            self.run_directory.mkdir(parents=True)
        if mode == "test":
            print("Starting acquisition in test mode")
        self.stimulus_waveform.enableAutoRange()
        self.Dock_Recording.raiseDock()
        self.Dock_Recording.update()
        self.TrialTimer.setInterval(int(self.protocol["stimuli"]["stimulus_period"]))
        self.TrialTimer.timeout.emit()
        self.TrialTimer.start()
        # finally, load up the data and trigger the thread to start.
        self.Presenter.setWaveforms(  # new instance of pystim3 each time
            wave=self.wave_matrix, protocol=self.protocol, sound_presentation_class=self.PS
        )
        if self.threadpool.activeThreadCount() == 0:  # the loop persists between acquisitions
            self.threadpool.start(self.Presenter.run)  # start the thread.

    def update_ABR(self, chdata, wave_counter, repetition_counter):
        """update_ABR Plot the most recent abr data, and keep
        track of the running average.
        *** this function is called from the Presenter thread ***

        Bug: every recording was written to abr_data/ under a name made only of the
        date, stimulus type and counters, so a repeat on the same day overwrote the
        first, and datapath was ignored. Each recording now has its own directory
        under datapath, from next_run_directory; the file names are unchanged.

        Bug: the plots are labeled msec and uV, but were drawn in s and V at the
        amplifier output. They are now drawn in msec and uV at the electrodes, as is
        the stimulus (plot_stimulus_wave). Saved data remain in V at the amplifier.

        Parameters
        ----------
        chdata : np.array
            The data from the ABR acquisition
        wave_counter : int
            index into which wave was played
        repetition_counter : _type_
            _description_

        Return
        ------
            Success: Bool
            True if data was acquired and displayed.
            False if data was not propery acquired (need to repeat stimulus)
        """
        self.update_success = False
        if chdata[0] is None:  # no data, so fake it
            rng = np.random.default_rng()
            chdata = [
                rng.standard_normal(self.PS.Ndata) * 1e-6,
                rng.standard_normal(self.PS.Ndata) * 1e-6,
            ]
        self.ch1_data = np.array(chdata[0]).squeeze()
        if len(self.ch1_data) == 0:  # didn't get the data... 
            print("Trial data is missing; retrying: ", self.TrialCounter, self.ch1_data.shape)
            return False
        # self.ch2_data = np.array(ch2[0]).T
        print("trial: ", self.TrialCounter, self.ch1_data.shape)
        if self.TrialCounter == 1:
            self.summed_buffer = self.ch1_data
            self.plot_ABR_Raw.clear()

        else:
            # print("buffer shapes: ", self.ch1_data.shape, self.summed_buffer.shape)
            if len(self.ch1_data) < len(self.summed_buffer):
                n_extend = len(self.summed_buffer) - len(self.ch1_data)
                # print("Lens: ", len(self.ch1_data), len(self.summed_buffer), n_extend)
                self.ch1_data = np.append(self.ch1_data, np.mean(self.ch1_data) * np.ones(n_extend))
                # print("Shape: ", self.ch1_data.shape, self.summed_buffer.shape)
            elif len(self.ch1_data) > len(self.summed_buffer):  # longer
                self.ch1_data = self.ch1_data[:len(self.summed_buffer)]
            
            self.summed_buffer = self.summed_buffer + self.ch1_data
        
        _, self.sfin, self.sfout = self.PS.getHardware()
        self.t_record_ch1 = np.linspace(0, 1e3 * len(self.ch1_data) / self.sfin, self.ch1_data.shape[0])  # msec
        if (self.raw_clear_N > 0) and (((self.TrialCounter-1) % self.raw_clear_N) == 0):
            self.plot_ABR_Raw.clear()
            self.plot_ABR_Raw.setXRange(0, np.max(self.t_record_ch1))

        self.plot_ABR_Raw.plot(self.t_record_ch1, self.ch1_data * ONLINE_UV_PER_VOLT, pen=pg.mkPen(pg.intColor(self.TrialCounter-1, hues=10, values=10)))
        # self.plot_ABR_Raw.setYRange(-15, 15)
 
        self.plot_ABR_Average.clear()
        self.t_record_sum = np.linspace(0, 1e3 * len(self.summed_buffer) / self.sfin, self.summed_buffer.shape[0])  # msec
        self.plot_ABR_Average.plot(
            self.t_record_sum,
            self.summed_buffer / float(self.TrialCounter) * ONLINE_UV_PER_VOLT,
            pen=pg.mkPen("g"),
        )
        self.update_success = True
        self.TrialCounter += 1    

        if self.acq_mode == "test":
            return True
        subject_data = {}
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Subject Data":
                for childs in child:
                    subject_data[childs.name()] = childs.value()
        subject_data["Subject ID"] = self.get_subject_id()  # same form as the directory name
        # now assemble data and save it
        write_time = datetime.datetime.now()
        wave_copy = copy.deepcopy(self.wave_matrix)
        for k in wave_copy.keys():
            wave_copy[k]["sound"] = {}
            if "sound_right" in wave_copy[k]:
                wave_copy[k]["sound_right"] = {}
        out_data = {
            "subject_data": subject_data,
            "calibration": self.caldata,
            "stimuli": wave_copy,
            "protocol": self.protocol,
            "wave_number": wave_counter,
            "repetition": repetition_counter,
            "data": self.ch1_data,
            "record_frequency": self.PS.trueFreq,
            "timestamp": write_time,
        }
        condition_tags = self.get_condition_tags(wave_counter)
        if condition_tags is not None:  # not in free-field files, which are unchanged
            out_data["ear_mode"] = condition_tags.pop("ear_mode")
            out_data["conditions"] = condition_tags
            # "calibration" is the left path; the right path may have its own
            out_data["calibration_right"] = (
                self.caldata_right if self.caldata_right is not None else self.caldata
            )
        fn = Path(
            self.run_directory,
            f"{self.basefn:s}_{self.protocol['protocol']['stimulustype']:s}_{wave_counter:03d}_{repetition_counter:03d}.p",
        )
        print("output file: ", fn)
        with open(fn, "wb") as fh:
            pickle.dump(out_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        del wave_copy    # delete the wave copy
        return True

    def get_condition_tags(self, wave_counter: int) -> dict | None:
        """
        Describe the stimulus conditions of a sweep, to be saved with its data.
        Returns None in free-field mode, so those files are unchanged.

        For the interleaved BIC the dictionary has "ear_mode" and, for each slot of
        the sweep: "slot_conditions" ("L", "R" or "B": left, right, binaural),
        "slot_polarities" (+1 or -1) and "slot_starts" (s, nominal, before the ITD).
        It also has "itd" (s; positive means the right ear lags), "frequency" (Hz;
        0 for clicks), "dbspl_left", "dbspl_right" and "attenuation" ([left, right], dB).
        Related: wave_generator.WaveGenerator._make_bic_interleaved
        """
        mode = protocol_reader.get_mode(self.protocol)
        if mode == "freefield":
            return None
        entry = list(self.wave_matrix.values())[wave_counter]
        tags = {"ear_mode": mode}
        for key in (
            "slot_conditions",
            "slot_polarities",
            "slot_starts",
            "itd",
            "frequency",
            "dbspl_left",
            "dbspl_right",
            "attenuation",
        ):
            tags[key] = entry[key]
        return tags

    def clear_average(self):
        """Start a new on-line average; called when a click or tone stimulus changes.

        Bug: only the plot was cleared, so the next sweep redrew a running average of
        all the levels so far. Resetting TrialCounter also restarts summed_buffer in
        update_ABR. Interleaved protocols have one stimulus, so they are not reset.
        """
        self.TrialCounter = 1
        self.plot_ABR_Average.clear()

    def stop(self):
        """stop:  Stop stimulus presentation dead. 
        *** This function sends a signal to the Presenter thread
        to cleanly stop the stimulus presentation. The thread
        continues to run in the background (doing nothing..).
        """
        self.signal_stop.emit()  # this calls the cleanup directly, so don't call here
        self.TrialTimer.stop()

    def pause(self):
        """pause:  Pause the stimulus presentation.
        *** This function sends a signal to the Presenter thread
        to pause the stimulus presentation. The thread
        continues to run in the background (doing nothing..).
        It can be resumed with the continue_trials function.
        """
        self.signal_pause.emit()
        self.TrialTimer.stop()

    def continue_trials(self):
        """continue_trials:  resume the stimulus presentation when it
        is paused. If the presentation is not paused, this function
        does nothing. The timing picks up where it left off.

        """
        if self.Presenter._paused:
            self.Presenter.resume()
            self.TrialTimer.start()


def _default_config_type():
    """Avoid the Windows lab calibration on macOS; mirror main.py's platform default."""
    return "abr" if sys.platform == "win32" else "test"


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1:]
    if len(cmd) > 0:
        print("cmd: ", cmd)
        prog = PyABR(cmd[0])
    else:
        prog = PyABR(_default_config_type())
    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()
