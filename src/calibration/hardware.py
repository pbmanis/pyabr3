"""
Hardware access for the calibrations.

CalibrationHardware defines the small interface that the calibration engine uses.
PyStimCalibrationHardware implements it on the NI, RP2.1 and PA5 connections made
by pystim3. MockCalibrationHardware simulates a speaker, microphone and calibrator
with known properties, so the calibration logic can be tested without hardware.

Calibration cannot use PyStim.play_sound, because load_and_arm is tied to the tags
in the ABR circuit. Recording uses mic_record.rcx, whose tags are used here in the
same way as in calstim.m and rp_setup.m (no-globals branch).
"""

import threading
import time
from pathlib import Path

import numpy as np
import scipy.signal
from pylibrary.tools import cprint as CP

MIC_RECORD_RCX = Path(__file__).parents[2] / "tdt" / "mic_record.rcx"
RP21_COF_FLAG = 5  # 195312.5 Hz (hardware_initialization.m, rp_setup.m)
RP21_CAL_SAMPLE_FREQ = 195312.5
NI_CAL_SAMPLE_FREQ = 500000.0
MAX_ATTN = 120.0


class CalibrationError(Exception):
    """A problem that stops a calibration, with a message for the user."""


class CalibrationStopped(Exception):
    """The user pressed Stop."""


class CalibrationHardware:
    """Interface between the calibration engine and the hardware."""

    is_mock = False
    description = ""
    sample_freq = RP21_CAL_SAMPLE_FREQ  # A/D rate, Hz
    ni_freq = NI_CAL_SAMPLE_FREQ  # D/A rate, Hz

    def connect(self):
        pass

    def set_attn(self, attn: float):
        """Set the attenuator; out of range values give maximum attenuation (set_attn.m)."""
        if not 0.0 <= attn <= MAX_ATTN:  # also catches NaN
            attn = MAX_ATTN
        self._set_attn(attn)

    def _set_attn(self, attn: float):
        raise NotImplementedError

    def calstim(self, wave: np.ndarray, n_points: int, stop: threading.Event) -> np.ndarray:
        """Play wave and return n_points of the microphone channel (calstim.m)."""
        raise NotImplementedError

    def shutdown(self):
        pass


class PyStimCalibrationHardware(CalibrationHardware):
    """Calibration hardware using the NI, RP2.1 and PA5 connections made by pystim3.

    Parameters
    ----------
    config : dict
        The contents of the pyabr3 configuration file.
    pystim : PyStim, optional
        An existing instance, when called from pyabr3. pyabr3 creates a new
        instance for each acquisition, so replacing its circuit here is safe.
    """

    description = "NI / RP2.1 / PA5"

    def __init__(self, config: dict, pystim=None):
        self.config = config
        self.PS = pystim
        self.rp = None

    def connect(self):
        if self.rp is not None:
            return
        from src import pystim3

        self.pystim3 = pystim3
        if pystim3.opsys != "Windows" or not pystim3.nidaq_available:
            raise CalibrationError(
                "Calibration requires the NI and TDT hardware on Windows. "
                "Use mock hardware (--mock) on this computer."
            )
        if self.PS is None:
            # PyStim cannot create an RP2.1 connection in calibrate mode, so
            # the RP2.1 is connected below.
            hardware = [h for h in self.config["required_hardware"] if h != "RP21"]
            self.PS = pystim3.PyStim(
                required_hardware=hardware, ni_devicename=self.config["NI_device"]
            )
            self.PS.RP21_proj = pystim3.tdt.DSPProject(interface="USB")
        if "RP21" not in self.PS.State.hardware:
            self.PS.State.hardware.append("RP21")
        self.PS.setup_RP21(str(MIC_RECORD_RCX), acquisition_mode="calibrate")
        self.rp = self.PS.RP21_circuit._iface  # calls below are those made in MATLAB
        self._load_cof()
        self.ni_freq = float(self.PS.Stimulus.NI_out_sampleFreq)

    def _set_attn(self, attn: float):
        self.PS.setAttens(atten_left=attn, atten_right=attn)

    def _load_cof(self):
        if self.rp.ClearCOF() == 0:
            raise CalibrationError("Failed to clear the RP2.1 circuit")
        if self.rp.LoadCOFsf(str(MIC_RECORD_RCX), RP21_COF_FLAG) == 0:
            raise CalibrationError(f"Failed to load {MIC_RECORD_RCX!s}")
        self.sample_freq = float(self.rp.GetSFreq())

    def _load_circuit(self, n_points: int):
        """Load and run mic_record, then check the status of the RP2.1 (rp_setup.m)."""
        self._load_cof()
        self.rp.Run()
        status = int(self.rp.GetStatus())
        for bit, name in enumerate(["connecting to", "loading circuit on", "running circuit on"]):
            if not status & (1 << bit):
                raise CalibrationError(f"Error {name} the RP2.1")
        if self.rp.SetTagVal("REC_Size", n_points) == 0:
            raise CalibrationError("Failed to set REC_Size on the RP2.1")

    def _wait_for_recording(self, n_points: int, stop: threading.Event):
        end_time = time.time() + n_points / self.sample_freq
        while time.time() < end_time:
            if stop.is_set():
                raise CalibrationStopped()
            time.sleep(0.01)

    def calstim(self, wave: np.ndarray, n_points: int, stop: threading.Event) -> np.ndarray:
        P = self.pystim3
        device = self.PS.State.NI_devicename
        wave, _ = self.PS.clip(wave, 10.0)
        with P.nidaqmx.task.Task("NI_DAC_out") as task:
            task.ao_channels.add_ao_voltage_chan(
                f"/{device:s}/ao0", min_val=-10.0, max_val=10.0, units=P.VoltageUnits.VOLTS
            )
            task.timing.cfg_samp_clk_timing(
                rate=self.ni_freq,
                source="",
                sample_mode=P.AcquisitionType.FINITE,
                samps_per_chan=len(wave),
            )
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=f"/{device:s}/PFI0", trigger_edge=P.Edge.RISING
            )
            task.write(wave)
            self._load_circuit(n_points)
            task.start()
            self.rp.SoftTrg(1)
            try:
                self._wait_for_recording(n_points, stop)
            except CalibrationStopped:  # the 'Stop' branch of rp_setup.m
                self.rp.SoftTrg(0)
                self.rp.Halt()
                self.set_attn(MAX_ATTN)
                task.stop()
                raise
            self.rp.SoftTrg(2)
            ch2 = np.array(self.rp.ReadTagV("data_out2", 0, n_points), dtype=float)
            task.stop()
        self.set_attn(MAX_ATTN)
        self.rp.SoftTrg(2)
        self.rp.Halt()
        return ch2

    def shutdown(self):
        if self.rp is None:
            return
        try:
            self.rp.Halt()
            self.set_attn(MAX_ATTN)
        except Exception as e:
            CP.cprint("r", f"Calibration hardware shutdown: {e!s}")


class MockCalibrationHardware(CalibrationHardware):
    """Simulated speaker, microphone and sound level calibrator.

    Parameters
    ----------
    spl_at_0_attn : callable
        SPL at 0 dB attenuation for a full scale (10 V) tone at frequency f.
    mv_per_pa : float
        Microphone sensitivity.
    preamp_gain : float
        Microphone preamplifier gain, in dB.
    calibrator_db : float or None
        Level of the 1 kHz calibrator, or None when it is off.
    noise_vrms : float
        Broadband noise at the A/D input.
    hum_40k_v : float
        Amplitude of a 40 kHz power supply component, removed by the notch filter.
    hum_20k_v : float
        Amplitude of a 20 kHz power supply component, removed by the notch filter.
    """

    is_mock = True
    description = "MOCK (simulated hardware)"

    def __init__(
        self,
        spl_at_0_attn=None,
        mv_per_pa: float = 3.85,
        preamp_gain: float = 20.0,
        calibrator_db: float | None = None,
        noise_vrms: float = 2e-5,
        hum_40k_v: float = 2e-4,
        hum_20k_v: float = 2e-4,
        seed: int = 1,
    ):
        self.spl_at_0_attn = spl_at_0_attn or self.default_speaker
        self.mv_per_pa = mv_per_pa
        self.preamp_gain = preamp_gain
        self.calibrator_db = calibrator_db
        self.noise_vrms = noise_vrms
        self.hum_40k_v = hum_40k_v
        self.hum_20k_v = hum_20k_v
        self.rng = np.random.default_rng(seed)
        self.attn = MAX_ATTN
        self.attn_history: list = []
        self.calstim_count = 0
        self.stop_after_calls: int | None = None
        self.shutdown_called = False

    @staticmethod
    def default_speaker(f):
        """A speaker with a smooth roll off above 20 kHz."""
        return 105.0 - 12.0 * (np.log2(np.maximum(f, 20000.0) / 20000.0)) ** 2

    def _set_attn(self, attn: float):
        self.attn = attn
        self.attn_history.append(attn)

    def _pa_to_volts(self, pa):
        return pa * self.mv_per_pa * 1e-3 * 10.0 ** (self.preamp_gain / 20.0)

    def calstim(self, wave: np.ndarray, n_points: int, stop: threading.Event) -> np.ndarray:
        self.calstim_count += 1
        if self.stop_after_calls is not None and self.calstim_count >= self.stop_after_calls:
            stop.set()
        if stop.is_set():
            raise CalibrationStopped()
        t = np.arange(n_points) / self.sample_freq
        # NI (500 kHz) to RP2.1 (195312.5 Hz) is exactly 25/64
        played = scipy.signal.resample_poly(wave, 25, 64)[:n_points]
        pressure_wave = np.zeros(n_points)
        pressure_wave[: len(played)] = played / 10.0

        spectrum = np.abs(np.fft.rfft(wave))
        tone_freq = np.argmax(spectrum) * self.ni_freq / len(wave)
        peak_pa = np.sqrt(2.0) * 20e-6 * 10.0 ** (self.spl_at_0_attn(tone_freq) / 20.0)
        volts = self._pa_to_volts(pressure_wave * peak_pa * 10.0 ** (-self.attn / 20.0))
        if self.calibrator_db is not None:
            pa = np.sqrt(2.0) * 20e-6 * 10.0 ** (self.calibrator_db / 20.0)
            volts = volts + self._pa_to_volts(pa) * np.sin(2.0 * np.pi * 1000.0 * t)
        volts = volts + self.hum_40k_v * np.sin(2.0 * np.pi * 40000.0 * t)
        volts = volts + self.hum_20k_v * np.sin(2.0 * np.pi * 20000.0 * t)
        return volts + self.noise_vrms * self.rng.standard_normal(n_points)

    def shutdown(self):
        self.shutdown_called = True
