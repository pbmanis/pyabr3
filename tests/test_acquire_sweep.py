"""
Tests for PyStim._acquire_sweep and the circuit check, with a fake RP2.1 circuit and
buffer on a simulated clock: the latched data_ready flag (iCompare of the buffer index
against compare_count, into a JK flip-flop that the start trigger resets).

There is no fallback: a circuit without the tags, or a flag that the start trigger does
not clear, is a CircuitError.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_acquire_sweep.py
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pyqtgraph as pg
import pytest

from src import pystim3

N = 1024
RATE = 1e5  # A/D rate, so a sweep takes about 10 ms


class FakeCircuit:
    """data_ready is high once a sweep has been recorded. A trigger named in reset_by
    clears it (the circuit's reset); stuck means that nothing clears it."""

    def __init__(
        self, reset_by=("B",), stuck=False, never=False, clear_delay=0.0, tags=("data_ready", "compare_count")
    ):
        self.tags = {t: 0 for t in tags}
        self.reset_by = reset_by
        self.stuck = stuck
        self.never = never
        self.clear_delay = clear_delay  # s from the reset trigger to the flag clearing
        self.clear_at = None
        self.t_sweep = None  # start of the sweep being recorded
        self.flag = False  # the latched flag, as left by the previous sweep
        self.triggers = []
        self.set_calls = []

    def elapsed(self):
        return None if self.t_sweep is None else time.time() - self.t_sweep

    def get_tag(self, name):
        if name not in self.tags:
            raise KeyError(name)
        if name == "data_ready":
            if self.clear_at is not None and time.time() >= self.clear_at:
                self.flag = False
                self.clear_at = None
            elapsed = self.elapsed()
            if elapsed is not None and elapsed >= N / RATE and not self.never:
                self.flag = True
            return 1 if self.flag else 0
        return self.tags[name]

    def set_tag(self, name, value):
        if name not in self.tags:
            raise KeyError(name)
        self.tags[name] = value
        self.set_calls.append((name, value))

    def trigger(self, name):
        self.triggers.append(name)
        if name in self.reset_by and not self.stuck:
            if self.clear_delay > 0.0:
                self.clear_at = time.time() + self.clear_delay
            else:
                self.flag = False
        if name == "B":  # the start trigger: the sweep begins
            self.t_sweep = time.time()


class FakeBuffer:
    def __init__(self, circuit, channels=1, compression=1):
        self.circuit = circuit
        self.channels = channels
        self.compression = compression
        self.calls = []
        self.data = 1.0 + np.arange(N)

    def reset_read(self, index):
        self.calls.append(("reset_read", index))

    @property
    def write_index(self):
        elapsed = self.circuit.elapsed()
        return 0 if elapsed is None else min(N, int(elapsed * RATE))

    def _read(self, offset, length):
        self.calls.append(("_read", offset, length))
        return self.data[np.newaxis, offset : offset + length]


def make_ps(circuit, channels=1, compression=1, reset_trigger=None):
    ps = pystim3.PyStim.__new__(pystim3.PyStim)
    ps.Stimulus = pystim3.Stimulus_Parameters()
    ps.Stimulus.RP21_in_sampleFreq = RATE
    ps.Stimulus.RP21_ResetTrigger = reset_trigger
    ps.acquisition_points = N
    ps.RP21_circuit = circuit
    ps.databuffer = FakeBuffer(circuit, channels, compression)
    return ps


def test_data_ready_ends_the_sweep_and_the_data_are_read_from_the_start():
    circuit = FakeCircuit()
    ps = make_ps(circuit)
    data = ps._acquire_sweep()
    np.testing.assert_array_equal(data, 1.0 + np.arange(N))
    assert circuit.set_calls == [("compare_count", N - 1)]  # the last index of the buffer
    assert circuit.triggers == [ps.Stimulus.RP21_TriggerSource]
    assert ps.databuffer.calls == [("reset_read", 0), ("_read", 0, N)]


def test_a_following_sweep_works_after_the_flag_was_set_by_the_last_one():
    circuit = FakeCircuit()
    ps = make_ps(circuit)
    assert ps._acquire_sweep() is not None  # leaves data_ready set
    assert circuit.flag
    assert ps._acquire_sweep() is not None  # the start trigger clears it, so it can be waited for


def test_compare_count_is_in_raw_index_units():
    circuit = FakeCircuit()
    ps = make_ps(circuit, channels=2, compression=1)  # two channels interleaved: two raw per sample
    ps._acquire_sweep()
    assert circuit.set_calls == [("compare_count", 2 * N - 1)]


def test_compare_count_is_set_before_the_trigger():
    order = []
    circuit = FakeCircuit()
    set_tag, trigger = circuit.set_tag, circuit.trigger
    circuit.set_tag = lambda name, value: (order.append("set"), set_tag(name, value))[1]
    circuit.trigger = lambda name: (order.append("trigger"), trigger(name))[1]
    make_ps(circuit)._acquire_sweep()
    assert order == ["set", "trigger"]


def test_a_flag_that_the_start_trigger_does_not_clear_is_an_error():
    circuit = FakeCircuit(stuck=True)
    circuit.flag = True  # left set by the last sweep, as on the rig
    ps = make_ps(circuit)
    with pytest.raises(pystim3.CircuitError, match="stayed set after the start trigger") as e:
        ps._acquire_sweep()
    assert "before the trigger: True" in str(e.value)
    assert ps.databuffer.calls == [("reset_read", 0)]  # nothing was read


def test_an_error_names_what_was_seen_when_the_flag_was_clear_before_the_trigger():
    circuit = FakeCircuit()
    circuit.trigger = lambda name: circuit.triggers.append(name) or setattr(circuit, "flag", True)  # the trigger sets it
    ps = make_ps(circuit)
    with pytest.raises(pystim3.CircuitError, match="before the trigger: False"):
        ps._acquire_sweep()


def test_a_reset_trigger_is_fired_first_and_checked():
    circuit = FakeCircuit(reset_by=("A",))  # the start trigger does not reset; trigger A does
    circuit.flag = True
    ps = make_ps(circuit, reset_trigger="A")
    data = ps._acquire_sweep()
    assert data is not None and len(data) == N
    assert circuit.triggers == ["A", "B"]


def test_a_reset_trigger_that_does_not_clear_the_flag_is_an_error(monkeypatch):
    monkeypatch.setattr(pystim3, "DATA_READY_RESET_S", 0.01)
    circuit = FakeCircuit(stuck=True)
    circuit.flag = True
    ps = make_ps(circuit, reset_trigger="A")
    with pytest.raises(pystim3.CircuitError, match="after the reset trigger 'A'"):
        ps._acquire_sweep()
    assert circuit.triggers == ["A"]  # the sweep was not started


def test_no_reset_trigger_by_default():
    assert pystim3.Stimulus_Parameters().RP21_ResetTrigger is None
    assert pystim3.Stimulus_Parameters().RP21_TriggerSource == "B"


def test_a_sweep_that_never_completes_returns_none(capsys):
    circuit = FakeCircuit(never=True)
    ps = make_ps(circuit)
    assert ps._acquire_sweep() is None  # a failed sweep: the trial is repeated
    assert "data_ready did not come" in capsys.readouterr().out


def test_circuit_error_is_an_ioerror():
    assert issubclass(pystim3.CircuitError, IOError)


# --- the circuit is checked when it is loaded ---------------------------------------------


class FakeProject:
    def __init__(self, tags):
        self.circuit = SimpleNamespace(tags={t: (1, "F") for t in tags}, convert=lambda *a: None)

    def load_circuit(self, path, name):
        return self.circuit


def setup_ps(tags, tmp_path):
    ps = pystim3.PyStim.__new__(pystim3.PyStim)
    ps.State = pystim3.Stimulus_Status()
    ps.Stimulus = pystim3.Stimulus_Parameters()
    ps.RP21_proj = FakeProject(tags)
    rcx = tmp_path / "abrs_v3.rcx"
    rcx.write_bytes(b"")
    return ps, rcx


def test_circuit_with_all_tags_is_accepted(tmp_path):
    ps, rcx = setup_ps(pystim3.CIRCUIT_TAGS, tmp_path)
    assert ps.setup_RP21(str(rcx), acquisition_mode="abr") is True


def test_circuit_without_the_tags_is_an_error_that_names_them(tmp_path):
    ps, rcx = setup_ps(("sampled_wave", "sampled_wave_n"), tmp_path)
    with pytest.raises(pystim3.CircuitError, match="data_ready, compare_count"):
        ps.setup_RP21(str(rcx), acquisition_mode="abr")


def test_calibration_circuit_is_not_checked_for_the_abr_tags(tmp_path):
    ps, rcx = setup_ps((), tmp_path)
    assert ps.setup_RP21(str(rcx), acquisition_mode="calibrate") is True


# --- what happens to the acquisition when there is a circuit error ------------------------


def test_a_circuit_error_stops_the_presenter_and_reports_it():
    import threading

    from src import presenter_thread

    class Failing:
        def play_sound(self, *args, **kwargs):
            raise pystim3.CircuitError("the circuit is wrong")

    wave = {("click", 80.0, 0): {"sound": np.zeros(10), "attenuation": [10.0]}}
    protocol = {"stimuli": {"nreps": 3, "stimulus_period": 0.0}}
    presenter = presenter_thread.Presenter()
    messages = []
    presenter.signal_hardware_error.connect(messages.append)
    presenter.setWaveforms(wave=wave, protocol=protocol, sound_presentation_class=Failing())
    app = pg.mkQApp("test")  # the signal is delivered to this thread, through its event loop
    thread = threading.Thread(target=presenter.run)
    thread.start()
    deadline = time.time() + 5.0
    while not messages and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    presenter._quit = True  # end the loop
    thread.join(timeout=5.0)
    assert messages == ["the circuit is wrong"]
    assert presenter._running is False
    assert not thread.is_alive()  # the loop did not die from the exception, and has now been told to quit


def test_the_window_stops_and_shows_the_circuit_error(monkeypatch):
    from src import pyabr3

    events = []
    monkeypatch.setattr(pyabr3.MBox, "MessageBox", lambda message, title="": events.append(("box", message, title)))
    app = SimpleNamespace(stop=lambda: events.append(("stop",)))
    pyabr3.PyABR.hardware_error(app, "the circuit is wrong")
    assert events == [("stop",), ("box", "the circuit is wrong", "Hardware")]


# --- a slow reset, and compare_count before the circuit starts ----------------------------


def test_a_flag_that_clears_a_little_after_the_trigger_is_waited_for(capsys):
    circuit = FakeCircuit(clear_delay=0.003)  # the sweep takes 10 ms: half of it is allowed
    circuit.flag = True
    ps = make_ps(circuit)
    data = ps._acquire_sweep()
    assert data is not None and len(data) == N
    assert "data_ready cleared" in capsys.readouterr().out


def test_the_wait_for_the_flag_to_clear_is_limited_to_half_the_sweep():
    circuit = FakeCircuit(clear_delay=0.5)  # far too slow
    circuit.flag = True
    ps = make_ps(circuit)
    t0 = time.time()
    with pytest.raises(pystim3.CircuitError, match="stayed set"):
        ps._acquire_sweep()
    assert time.time() - t0 < 0.1  # 0.5 * 10 ms, not the whole 0.1 s allowed for long sweeps


def test_compare_count_in_raw_units_for_the_buffer():
    ps = make_ps(FakeCircuit(), channels=1, compression=1)
    assert ps._compare_count() == N - 1
    ps = make_ps(FakeCircuit(), channels=2, compression=1)
    assert ps._compare_count() == 2 * N - 1
    ps = make_ps(FakeCircuit(), channels=1, compression=2)
    assert ps._compare_count() == N // 2 - 1


class FakeTask:
    def __init__(self, name):
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=lambda *a, **k: None)
        self.timing = SimpleNamespace(cfg_samp_clk_timing=lambda **k: None)
        self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(cfg_dig_edge_start_trig=lambda **k: None))

    def write(self, data):
        pass

    def start(self):
        pass

    def is_task_done(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RigCircuit(FakeCircuit):
    """A circuit that also records the order of what load_and_arm does to it."""

    def __init__(self):
        super().__init__(tags=("data_ready", "compare_count", "sampled_wave_n"))
        self.events = []
        self.buffer = None

    def set_tag(self, name, value):
        self.events.append(("set", name, value))
        super().set_tag(name, value)

    def cset_tag(self, *args):
        self.events.append(("cset", args[0]))

    def get_buffer(self, name, mode):
        return self.buffer

    def start(self, *args):
        self.events.append(("start",))

    def stop(self):
        self.events.append(("stop",))


def test_compare_count_is_set_before_the_circuit_starts(monkeypatch):
    circuit = RigCircuit()
    ps = make_ps(circuit)
    circuit.buffer = ps.databuffer
    ps.State = pystim3.Stimulus_Status()
    ps.State.hardware = ["NIDAQ", "RP21"]
    ps.State.NI_devicename = "Dev1"
    ps.waveout = np.zeros(100)
    monkeypatch.setattr(pystim3, "nidaqmx", SimpleNamespace(task=SimpleNamespace(Task=FakeTask)), raising=False)
    monkeypatch.setattr(pystim3, "AcquisitionType", SimpleNamespace(FINITE=1), raising=False)
    monkeypatch.setattr(pystim3, "Edge", SimpleNamespace(RISING=1), raising=False)
    monkeypatch.setattr(pystim3, "VoltageUnits", SimpleNamespace(VOLTS=1), raising=False)
    ps.load_and_arm()
    names = [e for e in circuit.events if e[0] in ("start", "stop") or e[:2] == ("set", "compare_count")]
    assert names[0] == ("set", "compare_count", N - 1)  # before the circuit starts
    assert names[1] == ("start",)
    assert ps.ch1 is not None and len(ps.ch1) == N
