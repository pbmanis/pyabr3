"""
pystim3: a Python Class for interacting with various bits of hardware to produce sounds and
record signals.

The audio output hardware can be one of:
1. A National Instruments DAC card
2. system sound card
3. RP2.1 from Tucker Davis
4. RZ5D from Tucker Davis


If the NI DAC is available, TDT system 3 hardware is assumed as well for the
attenuators (PA5) and an RP2.1. or RZ5D. We don't use the RP2.1 for sound generation.
The RZ5D is limited by a 48 kHz maximum output rate, and thus to less than stimului 
with components at a maximum of 24 kHz. It is not suitable for the higher frequency sounds 
tht might be required for some small animal (mouse) ABRs.

Hardware on the Manis Lab Rig 5 (ABR) system includes:
RP2.1 (vintage)
RZ5D
NI6731 (high speed 4 channel 16-bit dac)
2 x PA5 attenuators

If the system sound card is used, stimuli are generated. This is used only for testing.

12/2024:
Switch to use tdtpy for TDT system control. This resulted in some significant
changes in approach.

12/17/2008-2024 Paul B. Manis, Ph.D.
UNC Chapel Hill
Department of Otolaryngology/Head and Neck Surgery
Supported by NIH Grants DC000425, DC004551 and DC015093 to PBM.
Tessa J. Ropp, Ph.D. also contributed to the development of this code.

Refactored and modified version, includes access to rz5d to help synchronize acquisition.
August, 2017 and later.

The Python requirements are listed in the requirements.txt file in the root directory of the repository.
Other requirements include:
nidaqmx for Python (https://nidaqmx-python.readthedocs.io/en/latest/)
pyaudio (https://people.csail.mit.edu/hubert/pyaudio/, or more recent versions; only for testing
when other hardware is not available).
tdt rco/rcx/rpx files for the TDT system. These are/were created with RPvdsEx, and should
reside in the tdt directory of the repository. The rco files are compiled versions of the
rcx files, and are used by the RP2.1 and RZ5D systems to control the hardware.

"""

"""
Old:
    (TDT manual system 3): 
    Sweep Control
    To use the sweep control circuit constructs the following names are required:
    zSwPeriod: The period of the sweep duration. This is set in OpenWorkbench
    and can not be modified during block acquisition.
    If it is necessary to change this value during the experiment, an
    *asynchronous next sweep control circuit* construct should be used
     See Asynchronous Next Sweep Control, page 324 for more information.
    317
    OpenEx User's Guide
    318
    zSwCount: The maximum number of sweeps before the signal is terminated.
    If this requires manual or external control, the value should be set to -1 through the OpenWorkbench protocol.

"""

import ctypes
from dataclasses import dataclass, field
from pyqtgraph.Qt.QtCore import QMutex
from pathlib import Path
import platform
import struct
import time
import numpy as np

# Check for the system we are running on, and what hardware/software is available.
opsys = platform.system()
nidaq_available = False
if opsys in ["nt", "Windows"]:
    try:
        print("Attempting import of nidaqmx")
        import nidaq.nidaq as nidaq
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, Edge, VoltageUnits

        nidaq_available = True
        print("    nidaq, nidaqmx and nidaqmx.constants were imported ok.")
        print("\nAttempting import of tdtpy")
        import tdt

        print("    tdtpy was imported ok.")

    except:
        raise ImportError(
            "Some required imports failed - check the system and the installation of the required packages"
        )

# If we are not on Windows, we can use the system sound card for testing.

if opsys in ["Darwin", "Linux"] or nidaq_available == False:
    import pyaudio


#  RZ5D State flags when using Synapse
RZ5D_Idle = 0
RZ5D_Preview = 2
RZ5D_Standby = 1
RZ5D_Run = 3

# default input and output rates for the sound card and NI card
def_soundcard_outputrate = 44100
def_soundcard_inrate = 44100
def_NI_outputrate = 500000


# define empty list function for dataclasses
def defemptylist():
    return []


@dataclass
class Stimulus_Status:
    """
    Create data structure for the status of the stimulus generator
    """

    controller: object = None
    running: bool = False
    stimulus_count: int = 0
    done: bool = False
    index: int = 0
    debugFlag: bool = False
    NI_devicename: str = ""
    NIDAQ_task: object = None
    RP21_proj: object = None
    required_hardware: list = field(default_factory=defemptylist)
    hardware: list = field(default_factory=defemptylist)
    max_repetitions: int = 10


class Stimulus_Parameters:
    """
    Data structure for the stimulus parameters,
    populated with some default values
    The defaults need to be checked, as the hardware
    setup should have instantiated them at their correct values.
    """

    NI_out_sampleFreq: float = def_NI_outputrate
    RP21_out_sampleFreq: float = 0.0
    RP21_in_sampleFreq: float = 0.0
    RP21_TriggerSource: str = "B"
    RZ5D_in_sampleFreq: float = 24410.0
    RZ5D_out_sampleFreq: float = 24410.0
    soundcard_out_sampleFreq: float = def_soundcard_outputrate
    soundcard_in_sampleFreq: float = def_soundcard_inrate
    atten_left: float = 120.0
    atten_right: float = 120.0


class PyStim:
    """PyStim class: a class to control the stimulus generation and data acquisition"""

    def __init__(
        self,
        required_hardware=["Soundcard"],
        ni_devicename: str = "Dev1",
        controller=None,
        acquisition_mode: str = "abr",
    ):
        """
        During initialization, we identify what hardware is available.

        Parameters
        ----------
        required_hardware : list : (Default: ['Soundcard'])
            A list of the names of devices we expect to be able to use
            For example: ['PA5', 'NIDAQ', 'RZ5D'] for an attenuator, an NI
            card (for DAC output) and the TDT RZ5D DSP unit. other combinations
            are possible (but not all have been tested or are useful)
        nidevicename : str (Default: 'dev1')
            The device name for the NI device we will use. Get this from
            the NIDAQmx system configuration utility that is provided by National Instruments.
        controller : object
            The parent class that provides the controls.
        acquisition_mode: str
            Either 'abr' or 'calibration'
        """

        self.State = Stimulus_Status()  # create instance of each data structure (class)
        self.State.required_hardware = required_hardware  # save the caller data
        self.State.NI_devicename = ni_devicename
        self.State.controller = controller
        self.Stimulus = Stimulus_Parameters()  # create an instance of the stimulus

        self.find_hardware(
            acquisition_mode=acquisition_mode
        )  # look for the required hardware and make sure we can talk to it.
        self.trueFreq = None
        self.ch1 = None  # These will be arrays to receive the a/d sampled data
        self.ch2 = None
        self.audio = None  # pyaudio object  - get later
        self.State.NIDAQ_task = None  # nidaq task object - get later
        self.stim_mutex = QMutex()

    def find_hardware(self, acquisition_mode="abr", verbose: bool = False):
        """
        Find the required hardware on the system.
        For non-windows systems, this just finds the system soundcard for testing
        Otherwise it looks for the requested hardware.
        Populates the available hardware in the list self.State.hardware.

        Parameters
        ----------
        None

        """
        if (
            opsys in ["Darwin", "Linux"] or nidaq_available is False
        ):  # If we are not on a Windows system, just set up soundcard
            print(f"Found operation system: {opsys}; We only support the sound card")
            self.setup_soundcard()
            self.State.hardware.append("Soundcard")
            #  TODO: check for other sound card sample rates, and use the maximum rate
            # or a specified rate from the configuration file.
        elif opsys == "Windows":
            if "NIDAQ" in self.State.required_hardware:  #  and self.setup_nidaq():
                self.State.hardware.append("NIDAQ")
                # self.setup_nidaq()



            if "RP21" in self.State.required_hardware:
                assert acquisition_mode in ["abr", "calibrate"]
                print("looking for RP21")

                if acquisition_mode == "abr":
                    try:
                        self.RP21_proj = tdt.DSPProject(interface="USB")
                    except:
                        raise ValueError("Unable to connect to device")
                    if self.setup_RP21(
                        # "c:\\TDT\\OpenEx\\MyProjects\\Tetrode\\RCOCircuits\\tone_search.rcx"
                        "c:\\users\\experimenters\\desktop\\pyabr\\tdt\\abrs_v2.rcx",
                        acquisition_mode="abr",
                    ):
                        self.State.hardware.append("RP21")
                    else:
                        print("RP21 expected, but was not found")
                        raise NotImplementedError("RP21 expected, but was not found")
                # elif acquisition_mode == "calibrate":
                #     if self.setup_RP21(
                #         "c:\\Users\\experimenters\\Desktop\\pyabr\\tdt\\mic_record.rcx",
                #         acquisition_mode="calibrate"):
                #         self.State.hardware.append("RP21")
                #     else:
                #         print("RP21 expected, but was not found")
                #         raise NotImplementedError("RP21 expected, but was not found")

                else:
                    raise ValueError(
                        f"RP21 acquisition mode must be 'abr' or 'calibrate'; got: '{acquisition_mode:s}'"
                    )
            if "PA5" in self.State.required_hardware and self.setup_PA5():
                self.State.hardware.append("PA5")

            if "RZ5D" in self.State.required_hardware and self.setup_RZ5D():
                self.State.hardware.append("RZ5D")
        else:
            raise NotImplementedError("Unknown operating system: {opsys}")

        print("Hardware found: ", self.State.hardware)

    def getHardware(self):
        """getHardware: get some information about the hardware setup

        Returns
        -------
        tuple
            The hardware state data
            the current output and input sample rates

        Raises
        ------
        ValueError
            if not valid hardware is in the State list of hardware.
        """
        # print("Hardware: self.state.hardware: ", self.State.hardware)
        if "NIDAQ" in self.State.hardware:
            sfout = self.Stimulus.NI_out_sampleFreq
        elif "RP21" in self.State.hardware:
            sfout = self.Stimulus.RP21_out_sampleFreq
        elif "Soundcard" in self.State.hardware:
            sfout = self.Stimulus.soundcard_out_sampleFreq
        else:
            raise ValueError("pystim3.getHardware: No Valid OUTPUT hardware found")
        if "RP21" in self.State.hardware:
            sfin = self.Stimulus.RP21_in_sampleFreq
        elif "Soundcard" in self.State.hardware:
            sfin = self.Stimulus.soundcard_in_sampleFreq
        else:
            raise ValueError("pystim3.getHardware: No Valid INTPUT hardware found")
        return (self.State.hardware, sfin, sfout)

    def reset_hardware(self):
        """
        Reset the hardware to initial state
        """
        if "RZ5D" in self.State.hardware:
            if self.RZ5D is not None:
                self.RZ5D.setModeStr("Idle")
        if "PA5" in self.State.hardware:
            if self.PA5 is not None:
                self.PA5.SetAtten(120.0)
        if "RP21" in self.State.hardware:
            if self.RP21_circuit is not None:
                self.RP21_circuit.stop()
        # if "NIDAQ" in self.State.hardware:  # not needed - using inside context manager
        #     if self.State.NIDAQ_task is not None:
        #         self.State.NIDAQ_task.close()
        # if "pyaudio" in self.State.hardware:  # not needed - using context manager
        #     if self.audio is not None:
        #         self.audio.terminate()

    def setup_soundcard(self):
        if self.State.debugFlag:
            print(
                "pystim:setup_soundcard: Your OS or available hardware only supports a standard sound card"
            )
        self.State.hardware.append("pyaudio")

    def setup_nidaq(self):
        # get the drivers and the activeX control (win32com)
        self.NIDevice = nidaqmx.system.System.local()
        self.NIDevicename = self.NIDevice.devices.device_names
        self.Stimulus.NI_out_sampleFreq = def_NI_outputrate  # output frequency, in Hz
        return True

    def show_nidaq(self):
        """
        Report some information regardign the nidaq setup
        """
        print("pystim:show_nidaq found the follwing nidaq devices:")
        print(f"    {self.NIDevice.devices.device_names:s}")
        # print ("devices: %s" % nidaq.NIDAQ.listDevices())
        print("    ", self.NIDevice)
        print(
            f"\nAnalog Output Channels: {self.NIDevice.devices[self.NIDevicename].ao_physical_chans.channel_names}"
        )

    def setup_PA5(self, devnum=1):
        """
        Set up the ActiveX connection to one TDT PA5 attenuator

        Parameters
        ----------
        devnum : int (default = 1)
            The device number to connect to for the attenuator
        """

        self.PA5 = tdt.util.connect_pa5(interface="USB", device_id=1)
        self.PA5.SetAtten(120.0)  # set all attenuators to maximum attenuation
        return True

    def setup_RP21(self, rcofile: str = "", acquisition_mode="abr"):
        """
        Make an ActiveX connection to theTDT RP2.1 Real-Time Processor through tdtpy
        and load the RCX file.

        Parameters
        ----------
        rcofile : str (default : '')
            The RCX file to connect to. Must be the full path.
        acquisition_mode : str (default : 'abr')
            The acquisition mode to use. Either 'abr' or 'calibrate'
        """
        if self.State.debugFlag:
            print("Setting up RP21")
        self.RP21_rcxfile = rcofile
        if not Path(self.RP21_rcxfile).is_file():
            raise FileNotFoundError(
                f"The required RP2.1 RCX file was not found \n    (Looking for {self.RP21_rcxfile})"
            )

        self.RP21_circuit = self.RP21_proj.load_circuit(self.RP21_rcxfile, "RP2")
        print(f"Using: {self.RP21_rcxfile}")
        # acquisition_mode = 'calibrate'
        if acquisition_mode == "abr":
            self.samp_cof_flag = 2  # set for 24 kHz
        elif acquisition_mode == "calibrate":
            self.samp_cof_flag = 5
        else:
            raise ValueError(
                f"Acquistion mode must be either 'abr' or 'calibrate', got '{acquisition_mode:s}'"
            )
        self.samp_flist = [
            6103.5256125,
            12210.703125,
            24414.0625,
            48828.125,
            97656.25,
            195312.5,
        ]
        if self.samp_cof_flag > 5 or self.samp_cof_flag < 0:
            raise ValueError("RP2.1 sample rate flag is out of bounds: [0, 5]", self.samp_cof_flag)
        self.RP21_circuit.convert(self.samp_flist[self.samp_cof_flag], "fs", "nPer")

        # set the input and output sample frequencies to the same value
        self.Stimulus.RP21_out_sampleFreq = self.samp_flist[self.samp_cof_flag]
        self.Stimulus.RP21_in_sampleFreq = self.samp_flist[self.samp_cof_flag]
        print("RP2.1 set up, in samp freq = ", self.Stimulus.RP21_in_sampleFreq)
        return True

    def show_RP21(self):
        """
        TODO: maybe report RP2.1 info: cof rate, loaded circuit, sample freqs
        """
        pass

    def setup_RZ5D(self):
        """setup_RZ5D attach to the RZ5D through the SynapseAPI

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        IOError
            _description_
        """
        try:
            self.RZ5D = tdt.SynapseAPI()
            if self.RZ5D.getModeStr() != "Idle":
                self.RZ5D.setModeStr("Idle")
            return True
        except:
            raise IOError("pystim.setup_RZ5D: RZ5D requested, but not found")

    def get_RZ5D_Params(self):
        self.RZ5DParams = {}  # keep a local copy of the parameters
        self.RZ5DParams["device_name"] = self.RZ5D.getGizmoNames()
        self.RZ5DParams["device status"] = self.RZ5D.getModeStr()

    def show_RZ5D(self):
        print("Device Status: {0:d}".format(self.RZ5DParams["device_status"]))

    def get_RZ5D_Mode(self):
        return self.RZ5D.getModeStr()

    def RZ5D_close(self):
        if self.RZ5D.getModeStr() != "Idle":
            self.RZ5D.setModeStr("Idle")

    def cleanup_NIDAQ_RP21(self):
        self.stim_mutex.lock()
        try:
            if "PA5" in self.State.hardware:
                self.setAttens()  # attenuators down (there is noise otherwise)
        except:
            pass

        if "RP21" in self.State.hardware:
            self.RP21_circuit.stop()  # self.RP21.Halt()

        self.stim_mutex.unlock()

    # internal debug flag to control printing of intermediate messages
    def debugOn(self):
        self.State.debugFlag = True

    def debugOff(self):
        self.State.debugFlag = False

    def setAttens(self, atten_left=120.0, atten_right=120.0):
        if "PA5" in self.State.hardware:
            # self.PA5.ConnectPA5("USB", 1)
            # self.PA5.SetAtten(atten_left)
            # if atten_right is not None:
            #     self.PA5.ConnectPA5("USB", 2)
            #     self.PA5.SetAtten(atten_right)
            self.PA5.SetAtten(atten_left)
            # if atten_right is not None:
            #     self.PA5_2.SetAtten(atten_right)

    def _compute_out_sampling_info(
        self, nstim_wave: int, postduration: float, out_sampleFreq: float
    ):
        self.stim_dur = postduration + nstim_wave / float(out_sampleFreq)
        self.t_stim = np.linspace(0, nstim_wave / out_sampleFreq, nstim_wave)
        self.stimulus_points = int(self.stim_dur / float(out_sampleFreq))


    def _update_output_points(self, out_sampleFreq:float):
        self.stim_dur = self.rec_dur
        self.stimulus_points = int(self.stim_dur/float(out_sampleFreq))
        self.t_stim = np.linspace(0, self.stimulus_points / out_sampleFreq, self.stimulus_points)


    def _compute_in_sampling_info(self, in_sampleFreq: float, out_sampleFreq: float, block_size: int=1024):
        # dur = postduration + n_wave / float(self.Stimulus.RP21_out_sampleFreq)
        
        if self.stim_dur is None:
            raise ValueError("Stimulus duration must be set before computing acquisition points")
        self.acquisition_points = int(np.ceil(self.stim_dur * in_sampleFreq))
        # make acquisition a multiple of 1024 points
        n = int(self.acquisition_points / block_size)
        if self.acquisition_points % block_size > 0:
            n = n + 1
        self.acquisition_points = n * block_size
        self.rec_dur = self.acquisition_points/in_sampleFreq
        self.t_record = np.linspace(
            0,
            self.rec_dur,
            self.acquisition_points,
        )
        self._update_output_points(out_sampleFreq = out_sampleFreq)

    def play_sound(
        self,
        wavel,
        waver=None,
        postduration=0.000,
        attns=[20.0, 20.0],
    ):
        """
        play_sound sends the sound out to an audio device.
        In the absence of NI card, and TDT system, we (try to) use the system audio device (sound card, etc)
        The waveform is played in both channels on sound cards, possibly on both channels
        for other devices if there are 2 channels.

        Parameters
        ----------
        wavel : numpy array of floats
            Left channel waveform
        waver : numpy of floats
            Right channel waveform
        postduration : float (default: 0.35)
            Time after end of stimulus, in seconds
        attns : 2x1 list (default: [20., 20.])
            Attenuator settings to use for this stimulus
        storedata : bool (default: True)
            flag to force storage of data at end of run

        """

        # if we are just using pyaudio (linux, MacOS), set it up now
        if "pyaudio" in self.State.hardware:
            self.play_audio(wavel, postduration)

        # set up waveforms for output on NIDAQ or RP21
        self.stim_dur = None
        self.rec_dur = None
        if "NIDAQ" in self.State.hardware:
            # print("***** NIDAQ ******")
            if self.State.NIDAQ_task is not None:
                raise ValueError("NIDAQ task has not been released")
            self._compute_out_sampling_info(
                nstim_wave=len(wavel),
                postduration=postduration,
                out_sampleFreq=self.Stimulus.NI_out_sampleFreq,
            )

            # print("dur, postdur: ", dur, postduration, len(self.waveout), ndata, self.Stimulus.NI_out_sampleFreq)
        else:
            self.t_stim = np.linspace(
                0, len(wavel) / self.Stimulus.RP21_out_sampleFreq, len(wavel)
            )

        if "RP21" in self.State.hardware:
            # print("***** RP21 ****** ")
            # get input sampling, and adjust the output array as well
            self._compute_in_sampling_info(in_sampleFreq=self.Stimulus.RP21_in_sampleFreq,
                                           out_sampleFreq=self.Stimulus.NI_out_sampleFreq)

        # if "RZ5D" in self.State.hardware:
        #     nstim_wave = len(wavel)
        #     rec_dur = postduration + nstim_wave / float(self.Stimulus.RZ5D_out_sampleFreq)
        #     self.waveout = wavel
        #     self.stimulus_points = int(rec_dur / float(self.Stimulus.RZ5D_out_sampleFreq))
        #     self.acquisition_points = int(np.ceil((rec_dur) * self.Stimulus.RZ5D_in_sampleFreq))
        #     self.t_record = np.linspace(
        #         0,
        #         float(self.acquisition_points) / self.Stimulus.RZ5D_in_sampleFreq,
        #         self.acquisition_points,
        #     )

        if self.stim_dur is None:
            raise ValueError("IO output devices not recognized? ", self.State.hardware)
        if self.rec_dur is None:
            return ValueError("IO input devices not recognized", self.State.hardware)
        # print("stim_dur, rec_dur: ", self.stim_dur, self.rec_dur)

        if "PA5" in self.State.hardware:
            self.setAttens(atten_left=attns[0], atten_right=attns[1])

        # the following call not only sets up the NIDAQ, but also starts the RP2.1
        # to trigger stimulation AND acquistion. When this returns, the
        # sampled data is in self.ch1 and self.ch2, along with a timebase.

        self.acquire_with_devices(wavel, None, timeout=self.rec_dur * 1.5, re_armable=False)

        if "PA5" in self.State.hardware:
            self.setAttens()  # attenuators down (there is noise otherwise)
        self.ch1 = self.ch1.squeeze()

    def stop_nidaq(self, task_handle=None):
        """
        Only stop the DAC, not the RZ5D
        This is used when reloading a new stimulus.
        """
        self.stim_mutex.lock()
        if task_handle is None:
            task_handle = self.State.NIDAQ_task
        if task_handle is not None:
            task_handle.stop()
            task_handle.close()  # release resources
            self.State.NIDAQ_task = None  # need to destroy value
            self.State.running = False
        self.stim_mutex.unlock()

    def acquire_with_devices(
        self,
        wavel,
        waver=None,
        repetitions: int = 1,
        timeout: float = 1200.0,
        re_armable: bool = False,
    ):
        """
        Set up and initialize the NIDAQ card for output,
        then let it run and keep up with each task completion
        so it can be retriggered on the next trigger pulse.
        Configured so that if we are currently running, the run is immediately stopped
        so we can setup right away.
        """

        self.waveout = wavel
        self.repetitions = repetitions
        self.State.stimulus_count = 0
        (self.waveout, clipl) = self.clip(self.waveout, 10.0)  # clip the wave if it's >10V
        self.start_time = time.time()
        self.timeout = timeout
        self.load_and_arm(re_arm=re_armable)

    def load_and_arm(self, re_arm: bool = False):
        """
        if NIDAQ: Initial setup of NI card for AO.
        Creates a task for the card, sets parameters, clock rate,
        and does setup if needed.
        A callback is registered so that when the task is done, the
        board is either released or re-armed for the next trigger.
        The callback is used so that the task does not block the GUI.
        """

        self.re_arm = re_arm
        this_starttime = time.time()
        self.ch1 = None
        if "NIDAQ" not in self.State.hardware:
            raise ValueError("Only use NIDAQ for output at this time")

        """ Should try:

        with (nidaqmx.task.Task(NIDACOUT) as self.State.NIDAQ_task,
            tdt.DSPProject(interface="USB") as self.RP21_proj,
            tdt.PA5(interface="USB") as self.PA5): 
        etc with setup calls.
            setup calls for EACH device should be in a subroutine.
            
            
        """
        with nidaqmx.task.Task("NI_DAC_out") as self.State.NIDAQ_task:
            channel_name = f"/{self.State.NI_devicename:s}/ao0"
            self.State.NIDAQ_task.ao_channels.add_ao_voltage_chan(  # can only do this once...
                channel_name, min_val=-10.0, max_val=10.0, units=VoltageUnits.VOLTS
            )
            self.State.NIDAQ_task.timing.cfg_samp_clk_timing(
                rate=self.Stimulus.NI_out_sampleFreq,
                source="",
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=len(self.waveout),
            )
            self.State.NIDAQ_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source="/Dev1/PFI0",
                trigger_edge=Edge.RISING,
            )
            self.State.NIDAQ_task.write(self.waveout)
            self.State.NIDAQ_task.start()
            # now we can configure the RP21
            if "RP21" in self.State.hardware:
                self.RP21_circuit.cset_tag("record_del_n", 2, "ms", "n")
                # print("new # acquisition points: ", self.acquisition_points)
                self.RP21_circuit.set_tag("sampled_wave_n", self.acquisition_points)
                recdur = self.acquisition_points / self.Stimulus.RP21_in_sampleFreq + (
                    0.002 / self.Stimulus.RP21_in_sampleFreq
                )
                self.RP21_circuit.cset_tag("record_dur_n", recdur, "s", "n")
                self.RP21_circuit.cset_tag("play_dur_n", 1, "ms", "n")
                self.databuffer = self.RP21_circuit.get_buffer("sampled_wave", "r")

                self.RP21_circuit.start(0.25)  # start, but wait a bit...
                self.ch1 = self.databuffer.acquire(
                    self.Stimulus.RP21_TriggerSource, "running", False
                )  # should not return until done

            # now wait for the stimulus to be completed
            while not self.State.NIDAQ_task.is_task_done():
                now_time = time.time()
                if now_time - this_starttime > 2.0:
                    failed = True
                    print("arming nidaq/task execution FAILED")
                    break
                else:
                    time.sleep(0.05)

        if "RP21" in self.State.hardware:
            self.ch1 = self.databuffer.read(None)
        # print(self.ch1.shape)
            self.RP21_circuit.stop()

        self.State.NIDAQ_task = None

        return True

    def retrieveRP21_inputs(self):
        if self.ch1 is None or self.ch1.shape[0] == 0:
            self.ch1 = self.databuffer.read(None)

        self.ch2 = np.zeros_like(self.ch1)
        return (self.ch1, self.ch2)

    def HwOff(self):  # turn the hardware off.

        if "Soundcard" in self.State.hardware:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
            except:
                pass  # possible we never created teh stream...

        if "NIDAQ" in self.State.hardware:
            if self.State.NIDAQ_task is not None:
                self.Stat.NIDAQ_task.register_done_event(None)
                self.stop_nidaq(task_handle=self.State.NIDAQ_task)

        if "RP21" in self.State.hardware:
            self.RP21_circuit.stop()
            # self.RP21.Halt()

        if "RZ5D" in self.State.hardware:
            self.RZ5D_close()

    # clip data to max value (+/-) to avoid problems with daqs
    def clip(self, data, maxval):
        if self.State.debugFlag:
            print(
                "pysounds.clip: max(data) = %f, %f and maxval = %f" % (max(data), min(data), maxval)
            )
        clip = 0
        u = np.where(data >= maxval)
        ul = list(np.transpose(u).flat)
        if len(ul) > 0:
            data[ul] = maxval
            clip = 1  # set a flag in case we want to know
            if self.State.debugFlag:
                print("pysounds.clip: clipping %d positive points" % (len(ul)))
        minval = -maxval
        v = np.where(data <= minval)
        vl = list(np.transpose(v).flat)
        if len(vl) > 0:
            data[vl] = minval
            clip = 1
            if self.State.debugFlag:
                print("pysounds.clip: clipping %d negative points" % (len(vl)))
        if self.State.debugFlag:
            print(
                "pysounds.clip: clipped max(data) = %f, %f and maxval = %f"
                % (np.max(data), np.min(data), maxval)
            )
        return (data, clip)

    def play_audio(self, postduration: float, wavel: np.ndarray):
        dur = len(wavel) / float(self.Stimulus.soundcard_out_sampleFreq)
        self.acquisition_points = int(
            np.ceil((dur + postduration) * self.Stimulus.soundcard_out_sampleFreq)
        )
        with pyaudio.PyAudio() as self.audio:
            chunk = 1024
            FORMAT = pyaudio.paFloat32
            # CHANNELS = 2
            CHANNELS = 1
            if self.State.debugFlag:
                print(
                    f"pystim.play_sound: samplefreq: {self.Stimulus.soundcard_out_sampleFreq:.1f} Hz"
                )
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=int(self.Stimulus.soundcard_out_sampleFreq),
                output=True,
                input=True,
                frames_per_buffer=chunk,
            )
            wave = np.zeros(2 * len(wavel))
            if len(wavel) != len(waver):
                print(
                    f"pystim.play_sound: L,R output waves are not the same length: L = {len(wavel):d}, R = {len(waver):d}"
                )
                return
            (waver, clipr) = self.clip(waver, 20.0)
            (wavel, clipl) = self.clip(wavel, 20.0)
            wave[0::2] = waver
            wave[1::2] = wavel  # order chosen so matches etymotic earphones on my macbookpro.
            postdur = int(float(postduration * self.Stimulus.soundcard_in_sampleFreq))

            write_array(self.stream, wave)
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        return


"""
the following was taken from #http://hlzr.net/docs/pyaudio.html
it is used for reading and writing to the system audio device

"""


def write_array(stream, data):
    """
    Outputs a numpy array to the audio port, using PyAudio.
    """
    # Make Buffer
    buffer_size = struct.calcsize("@f") * len(data)
    output_buffer = ctypes.create_string_buffer(buffer_size)

    # Fill Up Buffer
    # struct needs @fffff, one f for each float
    dataformat = "@" + "f" * len(data)
    struct.pack_into(dataformat, output_buffer, 0, *data)

    # Shove contents of buffer out audio port
    stream.write(output_buffer)


def read_array(stream, size, channels=1):
    input_str_buffer = np.zeros((size, 1))  # stream.read(size)
    input_float_buffer = struct.unpack("@" + "f" * size * channels, input_str_buffer)
    return np.array(input_float_buffer)


if __name__ == "__main__":

    p = PyStim(required_hardware=["PA5", "NIDAQ", "RP21"], ni_devicename="dev1")

    ni_sample_frequency = 500000
    t = np.arange(0, 2.0, 1.0 / ni_sample_frequency)
    w = 10 * np.cos(2 * np.pi * 1000.0 * t)
    # import matplotlib.pyplot as mpl
    # mpl.plot(t, w)
    # mpl.show()

    p.play_sound(w)

    # p.stop_nidaq()
