#!/usr/bin/env python

# PySounds: a python Class for interacting with hardware to produce sounds and
# record signals.
#
# Output hardware is either an National Instruments DAC card or a system sound card
# If the NI DAC is available, TDT system 3 hardware is assumed as well for the
# attenuators (PA5) and an RP2.1 to input the startle response.
# Second channel of RP2.1 is collected as well. Use this for a microphone input
# to monitor sound in the chamber.
# If the system sound card is used, stimuli are generated and microphone input is
# collected, but they are not simultaneous. This is used only for testing.
#

# 12/17/2008 Paul B. Manis, Ph.D.
# UNC Chapel Hill
# Department of Otolaryngology/Head and Neck Surgery
# Supported by NIH Grants DC000425-22 and DC004551-07 to PBM.
# Copyright Paul Manis, 2008, 2009
#
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
"""
    Additional Terms:
    The author(s) would appreciate that any modifications to this program, or
    corrections of erros, be reported to the principal author, Paul Manis, at
    pmanis@med.unc.edu, with the subject line "PySounds Modifications". 
    
"""
import scipy.signal

# import matplotlib.pyplot as plt
import scipy
import pyaudio
import struct, ctypes
import numpy as np

REF_ES_dB = 86.0  # calibration info -  Assumes 10 dB padding with attenuator.
REF_ES_volt = 2.0  # output in volts to get refdb
REF_MAG_dB = 100.0  # right speaker is mag... different scaling.


class PySounds:
    def __init__(self):
        ################################################################################
        # the first thing we must do is find out what hardware is available and what
        # system we are on.
        ################################################################################
        self.debugFlag = False
        if self.debugFlag:
            print("PySounds: Checking Hardware and OS")
        try:
            import os

            if os.name !="nt":
                assert 0  # force use of pyaudio if not on windows xp/nt.
            if self.debugFlag:
                print("PySounds.init: OS is Windows (NT or XP)")
            # get the drivers and the activeX control (win32com)
            from nidaq import NIDAQ as nidaq_devs

            #            import nidaq
            import win32com.client

            if self.debugFlag:
                print("PySounds.init: Attempt to Assert num devs > 0:", end=" ")
            assert len(nidaq_devs.listDevices()) > 0
            self.dev0 = nidaq_devs.getDevice("Dev2")
            hwerr = 0
            if self.debugFlag:
                print("PySounds.init: found nidq devices.")
                print("devices: %s" % nidaq_devs.listDevices())
                print("getDevice:", end=" ")
                print("  ", self.dev0)

                print("\nAnalog Output Channels:", end=" ")
                # print "  AI: ", self.dev0.listAIChannels()
                print(" AO: ", self.dev0.listAOChannels())  # check output only

            # active x connection to attenuators
            # note - variables set at this scope level are global to source file
            self.PA5 = win32com.client.Dispatch("PA5.x")
            a = self.PA5.ConnectPA5("USB", 1)
            if a > 0 and self.debugFlag:
                print("PySounds.init: Connected to PA5 Attenuator 1")
            else:
                print("PySounds.init: Failed to connect to PA5 Attenuator 1")
                hwerr = 1
            self.PA5.SetAtten(120.0)
            a = self.PA5.ConnectPA5("USB", 2)
            if a > 0 and self.debugFlag:
                print("PySounds.init: Connected to PA5 Attenuator 2")
            else:
                print("PySounds.init: Failed to connect to PA5 Attenuator 2")
                hwerr = 1

            self.PA5.SetAtten(120.0)
            self.RP21 = win32com.client.Dispatch("RPco.x")  # connect to RP2.1
            a = self.RP21.ConnectRP2("USB", 1)
            if a > 0 and self.debugFlag:
                print("PySounds.init: RP2.1 Connect is good: %d" % (a))
            else:
                print("PySounds.init: Failed to connect to PA5 Attenuator 1")
                hwerr = 1
            self.RP21.ClearCOF()
            self.samp_cof_flag = 2  # 2 is for 24.4 kHz
            self.samp_flist = [
                6103.5256125,
                12210.703125,
                24414.0625,
                48828.125,
                97656.25,
                195312.5,
            ]
            if self.samp_cof_flag > 5:
                self.samp_cof_flag = 5
            a = self.RP21.LoadCOFsf("C:\\pyStartle\\startle2.rco", self.samp_cof_flag)
            if a > 0:
                print(
                    "PySounds.init: Connected to TDT RP2.1 and startle2.rco is loaded"
                )
            else:
                print("PySounds.init: Error loading startle2.rco?, error = %d" % (a))
                hwerr = 1
            self.hardware = "nidaq"
            self.out_sampleFreq = 100000
            self.in_sampleFreq = self.samp_flist[self.samp_cof_flag]
            if hwerr == 1:
                print("PySounds.init: ?? Error connecting to hardware")
                exit()
        except:
            if self.debugFlag:
                print("PySounds.init: OS or hardware only supports standard sound card")
            self.hardware = "pyaudio"
            self.out_sampleFreq = 44100.0
            self.in_sampleFreq = 44100.0

    def getHardware(self):
        return (self.hardware, self.out_sampleFreq, self.in_sampleFreq)

    # internal debug flag to control printing of intermediate messages
    def debugOn(self):
        self.debugFlag = True

    def debugOff(self):
        self.debugFlag = False

    ################################################################################
    # STIMULUS GENERATION ROUTINES
    #
    # transcribed from Matlab. P. Manis, Nov. 28-December 1 2008.
    ################################################################################

    def StimulusMaker(
        self,
        mode="tone",
        amp=1,
        freq=(1000, 3000, 4000),
        delay=0,
        duration=2000,
        rf=2.5,
        phase0=0,
        samplefreq=44100,
        ipi=20,
        nPresent=1,
        alternate=1,
        level=70,
        playSignal=False,
        plotSignal=False,
        channel=0,
    ):
        # generate a tsound (tone, bb noise, bpnoise)  pip with amplitude (V), frequency (Hz) (or frequencies, using a tuple)
        # delay (msec), duration (msec).
        # if no rf (risefall) time is given (units, msec), cosine^2 shaping with 5 msec ramp duration is applied.
        # if no phase is given, phase starts on 0, with positive slope.
        # level is in dB SPL as given by the reference calibration data above...
        #
        clock = (
            1000.0 / samplefreq
        )  # calculate the sample clock rate, and convert to points per msec (khz)
        #        uclock = 1000.*clock # microsecond clock
        #        phi = 2*np.pi*phase0/360.0 # convert phase from degrees to radians...
        Fs = 1000.0 / clock
        # phi = 0. # actually, always 0 phase for start
        w = []
        fil = self.rfShape(
            0, duration, samplefreq, rf
        )  # make the shape filter with 0 delay
        jd = int(np.floor(delay / clock))  # beginning of signal buildup (delay time)
        if jd < 0:
            jd = 0
        jpts = np.arange(0, len(fil))
        signal = np.zeros(len(jpts))
        siglen = len(signal)

        if mode == "tone":
            for i in range(0, len(freq)):
                signal = signal + fil * amp * np.sin(2 * np.pi * freq[i] * jpts / Fs)
                if self.debugFlag:
                    print("Generated Tone at %7.1fHz" % (freq[i]))

        elif mode == "bbnoise":
            signal = signal + fil * amp * np.normal(0, 1, siglen)
            if self.debugFlag:
                print("BroadBand Noise ")

        elif mode == "bpnoise":
            tsignal = fil * amp * np.random.normal(0, 1, siglen)
            # use freq[0] and freq[1] to set bandpass on the noise
            if self.debugFlag:
                print("freqs: HP: %6.1f    LP: %6.1f" % (freq[0], freq[1]))
            sf2 = samplefreq * 2.0  # nyquist limit
            if freq[0] > sf2 or freq[1] > sf2:
                print("freqs: ", freq)
                print("nyquist limit: ", sf2)
                print("sample frequ: ", samplefreq)
                print("coefficients not bounded [0, 1] for w... ")
                return np.array(signal)
            wp = [float(freq[0]) / sf2, float(freq[1]) / sf2]
            ws = [0.75 * float(freq[0]) / sf2, 1.25 * float(freq[1]) / sf2]
            (filter_b, filter_a) = scipy.signal.iirdesign(
                wp, ws, gpass=2.0, gstop=60.0, ftype="ellip"
            )
            if self.debugFlag:
                print("BandPass Noise %7.1f-%7.1f" % (freq[0], freq[1]))
            signal = scipy.signal.lfilter(filter_b, filter_a, tsignal)

        elif mode == "notchnoise":
            return np.array(signal)

        elif mode == "multitones":
            return np.array(signal)

        elif mode == "silence":
            return np.array(signal)

        # now build the waveform from the components
        w = np.zeros(int(np.ceil(ipi * (nPresent - 1) / clock) + jd + siglen))
        sign = np.ones(nPresent)
        if alternate == True:
            sign[list(range(1, nPresent, 2))] = -1
        id = int(np.floor(ipi / clock))
        for i in range(0, nPresent):  # for each pulse in the waveform
            j0 = jd + i * id  # compute start time
            w[list(range(j0, j0 + siglen))] = sign[i] * signal

        w = w * self.dbconvert(
            spl=level, chan=channel
        )  # aftera all the shaping ane scaling, we convert to generate a signal of w dB
        if playSignal == True:
            self.playSound(w, w, samplefreq)

        #        if plotSignal == True:
        #            self.plotSignal(w, w, clock)
        return np.array(w)

    #
    # Rise-fall shaping of a waveform. This routine generates an envelope with
    # 1 as the signal max, and 0 as the baseline (off), with cosine^2 shaping of
    # duration rf starting at delay (msec). The duration of the signal
    # includes the rise and fall, so the duration of the signal at full
    # amplitude is dur - 2*rf.
    # Note that since samplefreq is in Hz, delay, rf and duratio are
    # converted to seconds from the msec in the call.

    def rfShape(self, delay=0.0, duration=100.0, samplefreq=44100, rf=2.5):
        jd = int(
            np.floor((delay / 1000.0) * samplefreq)
        )  # beginning of signal buildup (delay time)
        if jd < 0:
            jd = 0
        je = int(
            np.floor(((delay + duration) / 1000.0) * samplefreq)
        )  # end of signal decay (duration + delay)
        #
        # build sin^2 filter from 0 to 90deg for shaping the waveform
        #
        nf = int(np.floor((rf / 1000.0) * samplefreq))  # number of points in the filter
        fo = 1.0 / (
            4.0 * (rf / 1000.0)
        )  # filter "frequency" in Hz - the 4 is because we use only 90deg for the rf component

        pts = np.arange(jd, jd + nf)
        fil = np.zeros(je)
        fil[jd:nf+jd] = np.sin(2 * np.pi * fo * pts * samplefreq) ** 2  # filter
        fil[list(range(jd + nf, je - nf))] = 1
        pts = list(range(je - nf, je))
        kpts = list(range(jd + nf, jd, -1))
        fil[pts] = fil[kpts]
        return fil

    #
    # insertGap takes a waveform and inserts a shaped gap into it.
    # currently, gap is all the way off, i.e., 0 intensity.
    # a future change is to include relative gap level (-dB from current waveform)
    #
    def insertGap(self, wave, delay=20, duration=20, rf=2.5, samplefreq=44100):
        fil = self.rfShape(
            delay, duration, samplefreq, rf
        )  # make the shape filter with 0 delay
        lenf = len(fil)
        lenw = len(wave)
        if lenw > lenf:
            fil = np.append(fil, np.zeros(lenw - lenf))
        if lenf > lenw:
            fil = np.append(fil, np.zeros(lenf - lenw))
        return wave * (1.0 - fil)

    #
    # compute voltage from reference dB level
    # db = 20 * log10 (Vsignal/Vref)
    #
    def dbconvert(self, spl=0, chan=0):
        ref = REF_ES_dB
        if chan == 1:
            ref = REF_MAG_dB

        zeroref = REF_ES_volt / (10.0 ** (ref / 20.0))
        sf = zeroref * 10.0 ** (spl / 20.0)
        # actually, the voltage needed to get spl out...
        if self.debugFlag:
            print("PySounds.dbconvert: scale = %f for %f dB" % (sf, spl))
        return sf  # return a scale factor to multiply by a waveform normalized to 1

    ################################################################################
    # hardware interactions:
    #
    # set the attenuators on the PA5.
    # If no args are given, set to max attenuation

    def setAttens(self, attenl=120, attenr=120):
        if self.hardware == "nidaq":
            self.PA5.ConnectPA5("USB", 1)
            self.PA5.SetAtten(attenl)
            self.PA5.ConnectPA5("USB", 2)
            self.PA5.SetAtten(attenr)

    #
    # playSound sends the sound out to an audio device. In the absence of NI card
    # and TDT system, it will use the system audio device (sound card, etc)
    # The waveform is played in stereo.
    # Postduration is given in seconds...
    def playSound(self, wavel, waver, samplefreq, postduration=0.35):
        if self.hardware in ["pyaudio"]:
            self.audio = pyaudio.PyAudio()
            print(dir(self.audio))
            chunk = 1024
            FORMAT = pyaudio.paFloat32
            CHANNELS = 2
            RATE = samplefreq
            if self.debugFlag:
                print("PySounds.playSound: samplefreq: %f" % (RATE))
            try:
                print('channels: ', CHANNELS, 'samplefreq: ', RATE)
                self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=int(RATE),
                output=True,
                input=False,
                frames_per_buffer=chunk,
            )
            except:
                raise ValueError('Error presenting audio through system')
            # play stream
            # print self.stream
            wave = np.zeros(2 * len(wavel))
            if len(wavel) != len(waver):
                print(
                    "PySounds.playSound: waves not matched in length: %d vs. %d (L,R)"
                    % (len(wavel), len(waver))
                )
                return
            (waver, clipr) = self.clip(waver, 20.0)
            (wavel, clipl) = self.clip(wavel, 20.0)
            wave[0::2] = waver
            wave[
                1::2
            ] = wavel  # order chosen so matches etymotic earphones on my macbookpro.
            postdur = int(float(postduration * self.in_sampleFreq))
            rwave = self.read_array(len(wavel) + postdur, CHANNELS)
            self.write_array(wave)
            self.HwOff()
            # self.stream.stop_stream()
            #             self.stream.close()
            #             self.audio.terminate()
            self.ch1 = rwave[0::2]
            self.ch2 = rwave[1::2]

        elif self.hardware in ["nidaq"]:
            self.task = self.dev0.createTask()  # creat a task for the NI 6731 board.
            self.task.CreateAOVoltageChan(
                "/Dev2/ao0", "ao0", -10.0, 10.0, self.nidaq.Val_Volts, None
            )
            self.task.CreateAOVoltageChan(
                "/Dev2/ao1", "ao1", -10.0, 10.0, self.nidaq.Val_Volts, None
            )  # use 2 channels
            wlen = 2 * len(wavel)
            self.task.CfgSampClkTiming(
                None,
                samplefreq,
                self.nidaq.Val_Rising,
                self.nidaq.Val_FiniteSamps,
                len(wavel),
            )
            # DAQmxCfgDigEdgeStartTrig (taskHandle, "PFI0", DAQmx_Val_Rising);
            self.task.SetStartTrigType(self.nidaq.Val_DigEdge)
            self.task.CfgDigEdgeStartTrig("PFI0", self.nidaq.Val_Rising)
            daqwave = np.zeros(wlen)
            (wavel, clipl) = self.clip(wavel, 10.0)
            (waver, clipr) = self.clip(waver, 10.0)

            daqwave[0 : len(wavel)] = wavel
            daqwave[
                len(wavel) :
            ] = waver  # concatenate channels (using "groupbychannel" in writeanalogf64)
            dur = wlen / float(samplefreq)
            self.task.write(daqwave)
            # now take in some acquisition...
            a = self.RP21.ClearCOF()
            if a <= 0:
                print("PySounds.playSound: Unable to clear RP2.1")
                return
            a = self.RP21.LoadCOFsf("C:\\pyStartle\\startle2.rco", self.samp_cof_flag)
            if a > 0 and self.debugFlag:
                print(
                    "PySounds.playSound: Connected to TDT RP2.1 and startle2.rco is loaded"
                )
            else:
                raise ValueError(
                    "PySounds.playSound: Error loading startle2.rco?, error = %d" % (a)
                )

            self.trueFreq = self.RP21.GetSFreq()
            Ndata = np.ceil(0.5 * (dur + postduration) * self.trueFreq)
            self.RP21.SetTagVal("REC_Size", Ndata)  # old version using serbuf  -- with
            # new version using SerialBuf, can't set data size - it is fixed.
            # however, old version could not read the data size tag value, so
            # could not determine when buffer was full/acquisition was done.
            self.setAttens(10.0, 10.0)  # set equal, but not at minimum...

            self.task.start()  # start the NI AO task
            a = self.RP21.Run()  # start the RP2.1 processor...
            a = self.RP21.SoftTrg(
                1
            )  # and trigger it. RP2.1 will in turn start the ni card
            while not self.task.isTaskDone():  # wait for AO to finish?
                if not self.PPGo:  # while waiting, check for stop.
                    self.RP21.Halt()
                    self.task.stop()
                    return
            self.task.stop()  # done, so stop the output.
            self.setAttens()  # attenuators down (there is noise otherwise)
            # read the data...
            curindex1 = self.RP21.GetTagVal("Index1")
            curindex2 = self.RP21.GetTagVal("Index2")
            while (
                curindex1 < Ndata or curindex2 < Ndata
            ):  # wait for input data to be sampled
                if not self.PPGo:  # while waiting, check for stop.
                    self.RP21.Halt()
                    return
                curindex1 = self.RP21.GetTagVal("Index1")
                curindex2 = self.RP21.GetTagVal("Index2")
            self.task.stop()
            self.ch2 = self.RP21.ReadTagV("Data_out2", 0, Ndata)
            # ch2 = ch2 - mean(ch2[1:int(Ndata/20)]) # baseline: first 5% of trace
            self.ch1 = self.RP21.ReadTagV("Data_out1", 0, Ndata)
            self.HwOff()
            # self.RP21.Halt()

        else:
            raise ValueError("No recognized hardware to use here")

    def retrieveInputs(self):
        return (self.ch1, self.ch2)

    def HwOff(self):  # turn the hardware off if you can.
        if self.hardware == "pyaudio":
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

        if self.hardware == "nidaq":
            self.task.stop()
            self.setAttens()
            self.RP21.Halt()

    # clip data to max value (+/-) to avoid problems with daqs
    def clip(self, data, maxval):
        if self.debugFlag:
            print(
                "PySounds.clip: max(data) = %f, %f and maxval = %f"
                % (max(data), min(data), maxval)
            )
        clip = 0
        u = np.where(data >= maxval)
        ul = list(np.transpose(u).flat)
        if len(ul) > 0:
            data[ul] = maxval
            clip = 1  # set a flag in case we want to know
            if self.debugFlag:
                print("PySounds.clip: clipping %d positive points" % (len(ul)))
        minval = -maxval
        v = np.where(data <= minval)
        vl = list(np.transpose(v).flat)
        if len(vl) > 0:
            data[vl] = minval
            clip = 1
            if self.debugFlag:
                print("PySounds.clip: clipping %d negative points" % (len(vl)))
        if self.debugFlag:
            print(
                "PySounds.clip: clipped max(data) = %f, %f and maxval = %f"
                % (max(data), min(data), maxval)
            )
        return (data, clip)

    ################################################################################
    # the following was taken from #http://hlzr.net/docs/pyaudio.html
    # it is used for reading and writing to the system audio devie
    #
    ################################################################################
    def write_array(self, data):
        """
        Outputs a numpy array to the audio port, using PyAudio.
        """
        # Make Buffer
        buffer_size = struct.calcsize("@f") * len(data)
        output_buffer = ctypes.create_string_buffer(buffer_size)

        # Fill Up Buffer
        # struct needs @fffff, one f for each float
        format = "@" + "f" * len(data)
        struct.pack_into(format, output_buffer, 0, *data)

        # Shove contents of buffer out audio port
        self.stream.write(output_buffer)

    def read_array(self, size, channels=1):
        input_str_buffer = np.zeros((size, 1))  # self.stream.read(size)
        input_float_buffer = struct.unpack(
            "@" + "f" * size * channels, input_str_buffer
        )
        return np.array(input_float_buffer)
