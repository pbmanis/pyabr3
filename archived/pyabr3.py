#!/usr/bin/env python
"""
pyabr : a python program for ABR recordings. 
Parameter setup is defined in .toml files (text files),
and displayed in a window.

"""

import sys
import datetime
import pickle
import platform
import time
import tomllib
from pathlib import Path

import numpy as np
import scipy.signal
import pyqtgraph as pg

# from PyQt4 import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

import sound  # for waveform generation
import PySounds  # for access to hardware

default_protocol = Path("protocols", "tonepip.toml")


class PyABR(object):
    def __init__(self):

        self.PS = PySounds.PySounds()
        # check hardware first
        self.hw, self.sfin, self.sfout = self.PS.getHardware()
        self.current_protocol = default_protocol
        self.ptreedata = None  # until the tree is built
        self.rawtoml = ""
        self.pars = None
        self.debugFlag = True

        print(f"Using Hardware: ", self.hw)
        self.protocols = list(Path("protocols").glob("*.toml"))
        print("Known protocols: ")
        for p in self.protocols:
            print(f"    {str(p):s}")

        self.current_protocol = self.protocols[0]
        self.buildGUI()
        # note after buildgui, all is done with callbacks

    def read_protocol(self):
        """
        Read the current protocol
        """
        assert self.ptreedata is not None  # do not call before building gui
        # self.pars = tomllib.load(self.current_protocol)
        with open(self.current_protocol, "r") as fh:
            self.rawtoml = fh.read()
        # paste the raw string into the text box for reference
        children = self.ptreedata.children()
        for c in children:  # find the parameter text box...
            if c.name() == "Parameters":
                c.setValue(self.rawtoml)
        print("set protocol?")
        print(self.rawtoml)
        print(self.pars)

    def buildGUI(self):
        """Build GUI and window"""

        app = pg.mkQApp()
        win = pg.QtWidgets.QWidget()
        layout = pg.QtWidgets.QGridLayout()
        win.setLayout(layout)
        win.setWindowTitle("ABR Acquistion")
        win.resize(1380, 1024)

        # Define parameters that control aquisition and buttons...
        params = [
            {
                "name": "Protocol",  # for selecting stimulus protocol
                "type": "list",
                "values": [str(p) for p in self.protocols],
                "value": str(self.protocols[0]),
            },
            {
                "name": "Parameters",  # for displaying stimulus parameters
                "type": "text",
                "value": self.rawtoml,  # in msec
                "default": self.pars,
                "readonly": True,
            },
            # },
            {"name": "Comment", "type": "text", "value": ""},  # open comment field
            {
                "name": "Actions",
                "type": "group",
                "children": [
                    {"name": "New Filename", "type": "action"},
                    {"name": "Test Acquisition", "type": "action"},
                    {"name": "Start Acquisition", "type": "action"},
                    {"name": "Stop/Pause", "type": "action"},
                    {"name": "Continue", "type": "action"},
                    {"name": "Save Visible", "type": "action"},
                    {"name": "Load File", "type": "action"},
                ],
            },
            {
                "name": "Status",
                "type": "group",
                "children": [
                    {"name": "dBSPL", "type": "float", "value": 0.0, "readonly": True},
                    {
                        "name": "Freq (kHz)",
                        "type": "float",
                        "value": 0.0,
                        "readonly": True,
                    },
                ],
            },
        ]

        ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="params", type="group", children=params)
        ptree.setParameters(self.ptreedata)
        ptree.setMaximumWidth(350)
        # build layout for plots and parameters
        layout.addWidget(ptree, 0, 0, 5, 1)  # Parameter Tree on left

        # add space for the graphs
        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(50, 50, 50))
        view.setCentralItem(l)

        layout.addWidget(view, 0, 1, 5, 4)  # data plots on right

        self.plt_abr = l.addPlot()
        self.plt_abr.getAxis("left").setLabel("uV", color="#ff0000")
        self.plt_abr.setTitle("Raw ABR Signal", color="#ff0000")
        self.plt_abr.getAxis("bottom").setLabel("t (msec)", color="#ff0000")
        self.plt_abr.setYRange(-10.0, 10.0)  # microvolts

        l.nextRow()  # averaged abr trace
        self.plt_avg = l.addPlot()  #
        self.plt_avg.setXLink(self.plt_abr)
        self.plt_avg.setTitle("Average ABR Signal", color="#ff0000")
        self.plt_avg.getAxis("left").setLabel("uV", color="#0000ff")
        self.plt_avg.setYRange(-10, 10.0)
        self.plt_avg.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        l.nextRow()  # waveforms
        self.waveform = l.addPlot()  #
        self.waveform.setXLink(self.plt_abr)
        self.waveform.setTitle("Waveform", color="#ff0000")
        self.waveform.getAxis("left").setLabel("V", color="#0000ff")
        self.waveform.setYRange(-10, 10.0)
        self.waveform.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        l.nextRow()  #
        l2 = l.addLayout(colspan=3, border=(50, 0, 0))  # embed a new layout
        l2.setContentsMargins(2, 2, 2, 2)
        self.plt_AI = l2.addPlot(Title="Amplitude-Intensity")
        self.plt_AI.getAxis("bottom").setLabel("dB (SPL)")
        self.plt_AI.getAxis("left").setLabel("uV")

        self.plt_map = l2.addPlot(Title="Map")
        self.plt_map.getAxis("bottom").setLabel("F (kHz)")
        self.plt_map.getAxis("left").setLabel("dB (SPL)")

        win.show()
        self.read_protocol()  # get a default protocol from disk
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

        if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
            pg.QtGui.QGuiApplication.instance().exec()

    def command_dispatcher(self, param, changes):

        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            if path is not None:
                childName = ".".join(path)
            else:
                childName = param.name
            if childName == "Protocol":
                self.current_protocol = data
                self.read_protocol()
            #
            # Actions
            #

            if len(path) <= 1:
                return
            if path[1] == "Start Acquisition":
                self.sequence_acquire()
            elif path[1] == "Test Acquisition":
                self.test_acquire()
            elif path[1] == "Stop/Pause":
                self.stop_pause()

    def sequence_acquire(self):
        """
        Present stimuli using sequences of intensities and/or
        frequencies
        """
        
        print("Starting sequenced acquisition")

        self.waveform.clear()
        if 'freqlist' not in list(self.pars["stimuli"].keys()):
            freqs = [1000.]
        else:
            freqs = self.pars["stimuli"]["freqlist"]
            for freq in freqs:
                for db in self.pars["stimuli"]["dblist"]:
                    for n in range(self.pars["stimuli"]["nreps"]):
                        self.make_waveforms(dbspl=db, frequency=freq)
                        self.waveform.plot(
                            self.wave_out.time,
                            self.wave_out.sound,
                            pen=pg.mkPen("y"),
                        )
                        self.waveform.enableAutoRange()

    def test_acquire(self):
        """
        present and collect responses without saving - using
        abbreviated protocol
        """
        
        print("Starting test acquisition")
        self.waveform.clear()
        self.make_waveforms()
        self.waveform.plot(
            self.wave_out.time, self.wave_out.sound, pen=pg.mkPen("y")
        )
        self.waveform.enableAutoRange()
        self.pars["stimuli"]["nreps"] = 10
        time.sleep(0.1)
        self.play_waveforms()
            

    def make_waveforms(self, dbspl=None, frequency=None):
        if self.pars["protocol"]["stimulustype"] == "click":
            if dbspl is None:
                dbspl = self.pars["stimuli"]["default_spl"]
            starts = np.cumsum(
                np.ones(self.pars["stimuli"]["nclicks"])*
                self.pars["stimuli"]["interval"]
            )
            starts += self.pars["stimuli"]["delay"] 
            self.wave_out = sound.ClickTrain(
                rate=self.sfout,
                duration=self.pars["stimuli"]["stim_dur"],
                dbspl=dbspl,
                click_duration=self.pars["stimuli"]["click_duration"],
                click_starts=starts,
                alternate=self.pars["stimuli"]["alternate"],
            )
            self.wave_out.generate()  # force generation

        if self.pars["protocol"]["stimulustype"] == "tonepip":
            if dbspl is None:
                dbspl = self.pars["stimuli"]["default_spl"]
            if frequency is None:
                frequency = self.pars["stimuli"]["default_frequency"]
            starts = np.cumsum(
                np.ones(self.pars["stimuli"]["npips"])*
                self.pars["stimuli"]["interval"]
            )
            starts += self.pars["stimuli"]["delay"] 
            self.wave_out = sound.TonePip(
                rate=self.sfout,
                duration=self.pars["stimuli"]["stim_dur"],
                f0=frequency,
                dbspl=dbspl,
                pip_duration=self.pars["stimuli"]["pip_duration"],
                ramp_duration=self.pars["stimuli"]["pip_risefall"],
                pip_start=starts,
                alternate=self.pars["stimuli"]["alternate"],
            )
            self.wave_out.generate()  # force generation

    def play_waveforms(self):
        self.TrialTimer = pg.QtCore.QTimer()  # get a Q timer
        self.TrialTimer.timeout.connect(self.next_trial)

        self.TrialCounter = 0
        for n in range(self.pars["stimuli"]["nreps"]):
            self.next_trial()

    def stop_pause(self):
        self.TrialTimer.stop()

    def continue_trials(self):
        self.next_trial()

    # callback routine to stop timer when thread times out.
    def next_trial(self):
        if self.debugFlag:
            print("NextTrial: entering")
        self.TrialTimer.stop()
        if self.TrialCounter <= self.pars["stimuli"]["nreps"]:
            self.TrialTimer.start(int(self.pars["stimuli"]["stimulus_period"]))
            self.PS.playSound(self.wave_out.sound, self.wave_out.sound, self.sfout)
            self.TrialCounter = self.TrialCounter + 1
        print("NextTrial: exiting")

        # (self.ch1, self.ch2) = Sounds.retrieveInputs()


if __name__ == "__main__":
    prog = PyABR()
