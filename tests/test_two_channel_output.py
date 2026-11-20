"""
Tests for the second attenuator and two-channel NI output (ABR_BIC_PLAN.md, step 3),
using fake hardware. The free-field (one channel, one PA5) behavior must not change.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_two_channel_output.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from src import pystim3


class FakePA5:
    def __init__(self):
        self.atten = []

    def SetAtten(self, value):
        self.atten.append(value)


class FakeTask:
    """Records the calls load_and_arm makes on an NI task."""

    last = None

    def __init__(self, name):
        FakeTask.last = self
        self.channels = []
        self.timing_kwargs = None
        self.written = None
        self.ao_channels = SimpleNamespace(
            add_ao_voltage_chan=lambda name, **kw: self.channels.append(name)
        )
        self.timing = SimpleNamespace(cfg_samp_clk_timing=self._timing)
        self.triggers = SimpleNamespace(
            start_trigger=SimpleNamespace(cfg_dig_edge_start_trig=lambda **kw: None)
        )

    def _timing(self, **kw):
        self.timing_kwargs = kw

    def write(self, data):
        self.written = np.array(data)

    def start(self):
        pass

    def stop(self):
        pass

    def is_task_done(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def ps(monkeypatch):
    """A PyStim without hardware: one PA5 and the NI card, as in free-field operation."""
    p = pystim3.PyStim.__new__(pystim3.PyStim)
    p.State = pystim3.Stimulus_Status()
    p.State.hardware = ["NIDAQ", "PA5"]
    p.State.NI_devicename = "Dev1"
    p.Stimulus = pystim3.Stimulus_Parameters()
    p.PA5 = FakePA5()
    p.PA5_2 = None
    p.acquisition_points = 1024
    fake_nidaqmx = SimpleNamespace(task=SimpleNamespace(Task=FakeTask))
    monkeypatch.setattr(pystim3, "nidaqmx", fake_nidaqmx, raising=False)
    monkeypatch.setattr(pystim3, "AcquisitionType", SimpleNamespace(FINITE=1), raising=False)
    monkeypatch.setattr(pystim3, "Edge", SimpleNamespace(RISING=1), raising=False)
    monkeypatch.setattr(pystim3, "VoltageUnits", SimpleNamespace(VOLTS=1), raising=False)
    return p


def test_one_pa5_ignores_right_attenuation(ps):
    ps.setAttens(atten_left=30.0, atten_right=50.0)
    assert ps.PA5.atten == [30.0]


def test_second_pa5_gets_the_right_attenuation(ps):
    ps.PA5_2 = FakePA5()
    ps.State.hardware.append("PA5_2")
    ps.setAttens(atten_left=30.0, atten_right=50.0)
    assert ps.PA5.atten == [30.0]
    assert ps.PA5_2.atten == [50.0]
    ps.setAttens()  # attenuators down: both at maximum
    assert ps.PA5.atten[-1] == 120.0
    assert ps.PA5_2.atten[-1] == 120.0


def test_reset_hardware_includes_second_pa5(ps):
    ps.PA5_2 = FakePA5()
    ps.State.hardware.append("PA5_2")
    ps.reset_hardware()
    assert ps.PA5.atten == [120.0]
    assert ps.PA5_2.atten == [120.0]


def test_setup_pa5_connects_the_requested_device(ps, monkeypatch):
    connected = []

    def connect_pa5(interface, device_id):
        connected.append(device_id)
        return FakePA5()

    monkeypatch.setattr(
        pystim3, "tdt", SimpleNamespace(util=SimpleNamespace(connect_pa5=connect_pa5)), raising=False
    )
    ps.PA5 = None
    assert ps.setup_PA5() is True
    assert ps.setup_PA5(devnum=2) is True
    assert connected == [1, 2]
    assert ps.PA5.atten == [120.0] and ps.PA5_2.atten == [120.0]
    with pytest.raises(ValueError):
        ps.setup_PA5(devnum=3)


def test_missing_second_pa5_gives_a_clear_error(ps, monkeypatch):
    def connect_pa5(interface, device_id):
        raise RuntimeError("no device")

    monkeypatch.setattr(
        pystim3, "tdt", SimpleNamespace(util=SimpleNamespace(connect_pa5=connect_pa5)), raising=False
    )
    with pytest.raises(IOError, match="second PA5"):
        ps.setup_PA5(devnum=2)


def test_freefield_output_is_one_channel_on_ao0(ps):
    wave = np.linspace(-12.0, 12.0, 1000)  # beyond the 10 V limit
    ps.acquire_with_devices(wave, None)
    task = FakeTask.last
    assert task.channels == ["/Dev1/ao0"]
    assert task.written.ndim == 1
    assert task.timing_kwargs["samps_per_chan"] == 1000
    assert task.written.max() == 10.0 and task.written.min() == -10.0


def test_free_field_callers_passing_both_waves_stay_single_channel(ps, monkeypatch):
    """Presenter passes the same wave as waver in free-field; that must not select ao1."""
    seen = {}
    monkeypatch.setattr(
        ps, "acquire_with_devices", lambda wavel, waver, **kw: seen.update(waver=waver)
    )
    monkeypatch.setattr(ps, "_compute_out_sampling_info", lambda **kw: setattr(ps, "stim_dur", 0.1))
    monkeypatch.setattr(ps, "_compute_in_sampling_info", lambda **kw: setattr(ps, "rec_dur", 0.1))
    ps.State.hardware.append("RP21")
    ps.ch1 = np.zeros(4)
    wave = np.zeros(100)
    ps.play_sound(wave, wave, attns=[10.0, 10.0])
    assert seen["waver"] is None


def test_two_channel_output_uses_ao0_and_ao1(ps):
    left = np.linspace(-1.0, 1.0, 500)
    right = np.linspace(-12.0, 12.0, 500)
    ps.acquire_with_devices(left, right)
    task = FakeTask.last
    assert task.channels == ["/Dev1/ao0:1"]
    assert task.written.shape == (2, 500)
    assert task.timing_kwargs["samps_per_chan"] == 500
    np.testing.assert_allclose(task.written[0], left)
    assert task.written[1].max() == 10.0 and task.written[1].min() == -10.0  # clipped


def test_two_channel_play_sound_needs_matching_waves(ps):
    with pytest.raises(ValueError, match="same length"):
        ps.play_sound(np.zeros(100), np.zeros(90), two_channel=True)
    with pytest.raises(ValueError, match="same length"):
        ps.play_sound(np.zeros(100), None, two_channel=True)


def test_two_channel_play_sound_needs_second_pa5(ps):
    with pytest.raises(ValueError, match="PA5_2"):
        ps.play_sound(np.zeros(100), np.zeros(100), two_channel=True)


# --- calibration hardware: output path --------------------------------------------------

from src.calibration.hardware import CalibrationError, MAX_ATTN, PyStimCalibrationHardware


@pytest.fixture
def cal_hw(ps):
    hw = PyStimCalibrationHardware(config={}, pystim=ps)
    hw.pystim3 = pystim3
    hw.rp = SimpleNamespace(
        SoftTrg=lambda n: None, ReadTagV=lambda *a: np.zeros(4), Halt=lambda: None
    )
    hw._load_circuit = lambda n_points: None
    hw._wait_for_recording = lambda n_points, stop: None
    hw.ni_freq = 500000.0
    return hw


def test_calibration_defaults_to_the_left_path(cal_hw, ps):
    assert cal_hw.output_path == "left"
    assert cal_hw.ao_channel == "ao0"
    cal_hw._set_attn(40.0)
    assert ps.PA5.atten == [40.0]


def test_calibration_output_path_must_be_left_or_right(cal_hw):
    with pytest.raises(ValueError):
        cal_hw.output_path = "both"


def test_calibration_left_plays_on_ao0(cal_hw):
    cal_hw.calstim(np.zeros(100), 4, stop=None)
    assert FakeTask.last.channels == ["/Dev1/ao0"]


def test_calibration_right_plays_on_ao1_with_second_attenuator(cal_hw, ps):
    ps.PA5_2 = FakePA5()
    ps.State.hardware.append("PA5_2")
    cal_hw.output_path = "right"
    assert cal_hw.ao_channel == "ao1"
    cal_hw._set_attn(40.0)
    assert ps.PA5.atten == [MAX_ATTN]  # the other path stays off
    assert ps.PA5_2.atten == [40.0]
    cal_hw.calstim(np.zeros(100), 4, stop=None)
    assert FakeTask.last.channels == ["/Dev1/ao1"]


def test_calibration_right_connects_second_pa5_when_missing(cal_hw, ps, monkeypatch):
    monkeypatch.setattr(
        pystim3, "tdt", SimpleNamespace(util=SimpleNamespace(connect_pa5=lambda **kw: FakePA5())), raising=False
    )
    cal_hw.output_path = "right"
    cal_hw.connect()  # rp is set, so this only makes sure the second PA5 is there
    assert "PA5_2" in ps.State.hardware
    assert ps.PA5_2 is not None


def test_calibration_right_without_second_pa5_is_a_calibration_error(cal_hw, monkeypatch):
    def connect_pa5(**kw):
        raise RuntimeError("no device")

    monkeypatch.setattr(
        pystim3, "tdt", SimpleNamespace(util=SimpleNamespace(connect_pa5=connect_pa5)), raising=False
    )
    cal_hw.output_path = "right"
    with pytest.raises(CalibrationError, match="second PA5"):
        cal_hw.connect()
