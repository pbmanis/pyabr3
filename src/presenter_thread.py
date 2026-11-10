
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph import reload as reload
from pyqtgraph.Qt.QtCore import QObject, pyqtSignal, pyqtSlot

from src import convert_nested_ordered_dict as convert_nested_ordered_dict
from src import protocol_reader as protocol_reader
from src import pystim3 as pystim3
from src import sound as sound  # for waveform generation


THREAD_PERIOD = 20  # thread period, in msec

MAX_CLICK = 105.1  # SPL for maximum click

class Presenter(QObject):  # (QtCore.QRunnable):
    """
    Presenter thread class
    Inherits from QObject to handle the presentation thread setup and signals.

    Sent signals are:

        finished
            No data
        error
            tuple (exctype, value, traceback.format_exc() )
        result
            object data returned from processing, anything
        mode_status
            status of the flags
        data_ready: a signal with the result data from the A/D converter
        progress:
            a signal with the current stimulus information.


    received signals/slots:
        setWaveforms
        Pause
        Resume
        Stop


    """

    # define the signals.

    signal_finished = pyqtSignal()  # protocol is ALL done
    signal_error = pyqtSignal(tuple)  # protocol has errored
    signal_data_ready = pyqtSignal(
        np.ndarray, int, int
    )  #  returns 2 channels in the array, then the wave number and trial counter
    signal_repetitions_done = pyqtSignal()
    signal_paused = pyqtSignal()  # status of running: paused or not
    signal_stop = pyqtSignal()  # stop protocol entirely

    signal_progress = pyqtSignal(
        str, int, int, int, int, float, float
    )  # update display of progress. Pass name, wave count, nwaves, rep count, nreps, db and fr
    signal_mode_status = pyqtSignal(str)  # notify changes in mode

    def __init__(self, parameters=None):  # fn, *args, **kwargs):

        super(Presenter, self).__init__()

        # Flags
        self._finished = False
        self._error = False
        self._paused = False
        self._running = False
        self._quit = False

        self._waveloaded = False
        self.wavedata = None
        self.wavetype = None
        self.protocol = None
        self.chdata = None
        self.parameters = parameters
        self.wave_counter = 0
        self.repetition_counter = 0

    @pyqtSlot()
    def run(self):
        """
        Initialize the runner function with passed args, kwargs.

        Within this function, we present successive stimuli.
        When the stimulus list is exhausted, we set the
        _finished signal and flag and return

        Calling the slot "setWaveforms" sets up and starts the acquisiton sequence.

        Bug: after the last stimulus the loop went on to the "Running" branch, so Mode
        alternated between Finished and Running. It now waits while finished.

        """
        self._running = True
        self.wave_counter = 0
        while True:  # while running, keep processing
            time.sleep(float(THREAD_PERIOD / 1000.0))  # Short delay to prevent excessive CPU usage
            # State management:
            if self._quit:  # this is the only way to exit the thread.
                return  # the final return (ends the thread)
            if not self._running:
                # always notify our state
                self.signal_mode_status.emit("Waiting to start")
                continue
            if self._finished:
                self.signal_mode_status.emit("Finished")
                continue
            if self._paused:
                self.signal_mode_status.emit("Paused")
                continue
            if not self._waveloaded:
                self.signal_mode_status.emit("No wave loaded")
            # running
            if self._running and not self._paused:
                self.signal_mode_status.emit("Running")
                self.next_stimulus()
                time.sleep(self.protocol["stimuli"]["stimulus_period"])

    def retrieve_data(self):
        res = self.sound_class.retrieveRP21_inputs()
        # print("presenter retrieve: ", res[0].shape)
        self.chdata = np.array(res)
        # self.ch2_data = np.array(res)
        self.signal_data_ready.emit(self.chdata, self.wave_counter, self.repetition_counter)

    def next_stimulus(self):
        """
        Play the next stimulus in the sequence and collect data.
        for nreps. When done, return the acquired data.

        This is done for each entry in the wave_matrix (wave_counter),
        and each entry is repeated n_repetition times.
        Each repetition is returned via a signal to the main program.

        """
        # print(
        #     f"next stim, wave count: {self.wave_counter:d}/{self.n_waves} "
        #     + f"repetition count: {self.repetition_counter:d}/{self.n_repetitions:d}"
        # )
        # print("    self._running: ", self._running)
        if not self._running or self.wavetype is None or not self._waveloaded:
            print("Presentation runner is missing something")
            return None, None

        if self.repetition_counter >= self.n_repetitions:
            self.wave_counter += 1
            self.repetition_counter = 0  # reset reps
            if self.wave_counter < self.n_waves:
                self.signal_repetitions_done.emit()
        if self.wave_counter >= self.n_waves:
            self.signal_finished.emit()
            self._finished = True
            return
        self.repetition_counter += 1
        if len(self.wavekeys[0]) > 2:
            sec_key = self.wavekeys[self.wave_counter][2]
        else:
            sec_key = 0.0
        self.signal_progress.emit(
            self.wavetype,
            self.wave_counter,
            self.n_waves,
            self.repetition_counter,
            self.n_repetitions,
            self.wavekeys[self.wave_counter][1],
            sec_key,
        )
       

        if self.wavetype in ["click", "tonepip"]:
            """
            present clicks and tone pips. Here, we use multiple waveforms in the array
            wave_matrix, and use the attenuation setting to adjust the attenuators.
            This is the "old way" of doing things - one intensity and frequency at
            a time.
            """
            if self.wave_counter >= self.n_waves:
                self.retrieve_data()
                self.signal_finished.emit()
                self._finished = True
                self._running = False
                return
            wave = self.wave_matrix[self.wavekeys[self.wave_counter]]["sound"]
            attn = self.wave_matrix[self.wavekeys[self.wave_counter]]["attenuation"][0]

            self.sound_class.play_sound(
                wave,
                wave,
                attns=[attn, attn],  # set both attenuators
            )
            self.retrieve_data()

        else:  # all other stimuli.
            """
            Present a single waveform that contains many stimulus frequencies
            and levels in one sweep, but many times. 
            All presentation is done with a single attenuation that is 
            calculated to provide the best dynamic range of the DAC output
            given the range of individual frequency levels that need to be
            accomodated. 
            Note that the db value that appears in the wave key is the
            same as the attenuation value.
            """

            attn = np.min([w[1] for w in self.wave_matrix.keys()])
            wave = self.wave_matrix[self.wavekeys[0]]["sound"]
            # print("wave keys: ", self.wavekeys)
            # attn = self.wavekeys[self.wave_counter][1]  # ]["attenuation"][0]
            self.sound_class.play_sound(
                wave, wave,attns=[attn, attn]
            )
            self.retrieve_data()

        return

    @pyqtSlot()
    def setWaveforms(self, wave: dict, protocol: dict, sound_presentation_class: object):
        """setWaveforms get the waveforms and protocol, and start the acquisition
        by setting self._running to True

        Parameters
        ----------
        wave : dict
            the wave matrix data
        protocol : dict
            the protocol dictionary that describes how the acquistion and
            stimulation should proceed
        sound_presentation_class : python function
            The class that will be called to present the sound through
            the hardware. Typically, this will be an object of type
            PyStim from pystim.py
        """
        # get the data we need and trigger the stimulus/acquisition.
        self._running = False
        self._finished = False  # the run loop is reused between acquisitions
        self.sound_class = sound_presentation_class
        self.wave_matrix = wave
        self.protocol = protocol
        self.wavekeys = list(self.wave_matrix.keys())
        self.wave_counter = 0  # reset counter of the waveforms
        self.repetition_counter = 0
        self.wavetype = self.wavekeys[0][0]  # get the first key of the first waveform.
        self.n_waves = len(self.wavekeys)
        self.n_repetitions = self.protocol["stimuli"]["nreps"]
        self._waveloaded = True
        self._running = True

    @pyqtSlot()
    def pause(self):
        # pause the thread
        if self._running:
            self._paused = True  # set True to cause the thread to skip
        # the stimulus presentations.

    @pyqtSlot()
    def resume(self):
        # resume stimulation
        if self._running and self._paused:
            self._paused = False

    @pyqtSlot()
    def stop(self):
        # stops stimulation, and
        self._running = False  # stops running
        self._paused = False  # also kills ability to resume.

    @pyqtSlot()
    def finished(self):
        self._finished = True
        self.signal_finished.emit()  # send a signal

    @pyqtSlot()
    def end_thread(self):
        """
        Stop the thread from running.
        This is needed at the end of the program to terminate cleanly.
        """
        self._paused = False  # resume if paused
        self._running = False  # set running False
        self.signal_finished.emit()  # send a signal

    @pyqtSlot()
    def quit(self):
        self._running = False
        self._quit = True
        self.signal_finished.emit()  # send a signal