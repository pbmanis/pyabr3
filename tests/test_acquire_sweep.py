"""
Tests for PyStim._acquire_sweep, with a fake RP2.1 circuit and buffer on a simulated
clock: the latched data_ready flag (iCompare of the buffer index against compare_count,
reset by the start trigger), and the buffer-index fallback for circuits without it.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_acquire_sweep.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from src import pystim3

N = 1024
RATE = 1e5  # A/D rate, so a sweep takes about 10 ms


class FakeCircuit:
    """data_ready goes high when the sweep has been recorded, and is cleared by the trigger."""

    def __init__(self, tags=("running", "data_ready", "compare_count"), stale_ready=False, never=False):
        self.tags = {t: 0 for t in tags}
        self.stale_ready = stale_ready
        self.never = never
        self.t_trigger = None
        self.triggers = []
        self.set_calls = []

    def elapsed(self):
        return None if self.t_trigger is None else time.time() - self.t_trigger

    def get_tag(self, name):
        if name not in self.tags:
            raise KeyError(name)
        done = self.elapsed() is not None and self.elapsed() >= N / RATE and not self.never
        if name == "data_ready":
            return 1 if (done or self.stale_ready) else 0
        if name == "running":
            return 1 if (self.elapsed() is not None and not done) else 0
        return self.tags[name]

    def set_tag(self, name, value):
        if name not in self.tags:
            raise KeyError(name)
        self.tags[name] = value
        self.set_calls.append((name, value))

    def trigger(self, name):
        self.t_trigger = time.time()
        self.triggers.append(name)


class FakeBuffer:
    def __init__(self, circuit, channels=1, compression=1):
        self.circuit = circuit
        self.channels = channels
        self.compression = compression
        self.calls = []
        self.pending_calls = 0
        self.data = 1.0 + np.arange(N)

    def reset_read(self, index):
        self.calls.append(("reset_read", index))

    def _stored(self):
        elapsed = self.circuit.elapsed()
        return 0 if elapsed is None or self.circuit.never else min(N, int(elapsed * RATE))

    def pending(self):
        self.pending_calls += 1
        return self._stored()

    @property
    def write_index(self):
        return self._stored()

    def read(self, n):
        self.calls.append(("read", n))
        if n > self._stored():
            raise IOError("not enough samples")
        return self.data[np.newaxis, :n]

    def _read(self, offset, length):
        self.calls.append(("_read", offset, length))
        return self.data[np.newaxis, offset : offset + length]


def make_ps(circuit, channels=1, compression=1):
    ps = pystim3.PyStim.__new__(pystim3.PyStim)
    ps.Stimulus = pystim3.Stimulus_Parameters()
    ps.Stimulus.RP21_in_sampleFreq = RATE
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
    assert ps.databuffer.pending_calls == 0  # the wrapping index was not used


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


def test_a_flag_that_the_trigger_did_not_clear_is_not_trusted(capsys):
    circuit = FakeCircuit(stale_ready=True)
    ps = make_ps(circuit)
    data = ps._acquire_sweep()  # falls back to the buffer index, which has the whole sweep
    assert data is not None and len(data) == N
    assert ("read", N) in ps.databuffer.calls
    assert ps.databuffer.pending_calls > 0
    assert "still set after the trigger" in capsys.readouterr().out


def test_circuit_without_the_tags_is_read_by_the_buffer_index(capsys):
    circuit = FakeCircuit(tags=("running",))
    ps = make_ps(circuit)
    data = ps._acquire_sweep()
    assert len(data) == N
    assert circuit.set_calls == []  # nothing is set that the circuit does not have
    assert ps.databuffer.calls == [("reset_read", 0), ("read", N)]
    assert "waiting on the buffer index" in capsys.readouterr().out
    assert ps._data_ready_ok is False


def test_a_sweep_that_never_completes_returns_none(capsys):
    circuit = FakeCircuit(never=True)
    ps = make_ps(circuit)
    assert ps._acquire_sweep() is None
    assert "not captured" in capsys.readouterr().out


def test_tag_probe_is_made_once():
    circuit = FakeCircuit()
    ps = make_ps(circuit)
    ps._acquire_sweep()
    assert ps._data_ready_ok is True
    circuit.tags.pop("data_ready")  # would fail a second probe
    assert ps._has_data_ready() is True
