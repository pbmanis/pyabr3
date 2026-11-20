"""
Tests for the check that a record fits in the buffer of the RP2.1 circuit, and for the
examples in the manual (doc_tex/chapters/03_acquisition.tex, "The record must fit in the
buffer of the circuit"): the numbers in the tables are computed here.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_recording_length.py
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

from src import pyabr3, pystim3

FS = 24414.0625


@pytest.mark.parametrize(
    "duration,samples,record",
    [  # the first table of the manual
        (0.645, 15748, 16384),
        (1.05, 25635, 26624),
        (1.2, 29297, 29696),
        (2.0, 48829, 49152),
        (5.0, 122071, 122880),
        (10.0, 244141, 244736),
    ],
)
def test_the_examples_of_the_manual_record_lengths(duration, samples, record):
    assert int(np.ceil(duration * FS)) == samples
    assert pystim3.recording_points(duration, FS) == record
    assert record % 1024 == 0 and record - samples < 1024


@pytest.mark.parametrize(
    "capacity,rate,longest",
    [  # the second table of the manual
        (26624, 24414.0625, 1.0905),
        (26624, 48828.125, 0.5453),
        (26624, 97656.25, 0.2726),
        (256000, 24414.0625, 10.4858),
        (256000, 48828.125, 5.2429),
        (256000, 97656.25, 2.6214),
    ],
)
def test_the_examples_of_the_manual_longest_waveforms(capacity, rate, longest):
    assert pystim3.longest_recording(capacity, rate) == pytest.approx(longest, abs=5e-5)


def test_the_longest_waveform_just_fits_and_a_little_more_does_not():
    for capacity in (26624, 256000, 30000):
        longest = pystim3.longest_recording(capacity, FS)
        assert pystim3.recording_points(longest, FS) <= capacity
        assert pystim3.recording_points(longest + 2.0 / FS, FS) > capacity


def test_the_record_length_is_what_the_program_has_always_used():
    """recording_points replaced the rounding in _compute_in_sampling_info: same numbers."""

    def old(duration, rate, block=1024):
        points = int(np.ceil(duration * rate))
        n = int(points / block)
        if points % block > 0:
            n = n + 1
        return n * block

    for duration in np.linspace(0.001, 3.0, 400):
        assert pystim3.recording_points(duration, FS) == old(duration, FS)
    ps = pystim3.PyStim.__new__(pystim3.PyStim)
    ps.stim_dur = 1.2
    ps._update_output_points = lambda out_sampleFreq: None
    ps._compute_in_sampling_info(in_sampleFreq=FS, out_sampleFreq=500000.0)
    assert ps.acquisition_points == 29696


def test_the_message_gives_the_sizes_the_calculation_and_the_cures():
    message = pystim3.buffer_too_small_message(29696, 26624, FS, duration=1.2)
    assert "a waveform of 1.200 s at 24414.0625 Hz is 29297 samples" in message
    assert "rounded up to a multiple of 1024" in message
    assert "needs 29696 samples (1.2163 s)" in message
    assert "holds only 26624 samples (1.0905 s)" in message
    assert "set when the circuit is designed" in message
    assert "wave_duration 1.090 s or less" in message  # rounded down: 1.091 s would not fit
    assert "at least 29696 samples (words)" in message
    assert "manual" in message


def test_the_message_without_a_duration_has_no_calculation():
    message = pystim3.buffer_too_small_message(29696, 26624, FS)
    assert "is 29297 samples" not in message
    assert "needs 29696 samples" in message


def ps_with_circuit(size_max, hardware=("NIDAQ", "RP21")):
    ps = pystim3.PyStim.__new__(pystim3.PyStim)
    ps.State = pystim3.Stimulus_Status()
    ps.State.hardware = list(hardware)
    ps.Stimulus = pystim3.Stimulus_Parameters()
    ps.Stimulus.RP21_in_sampleFreq = FS
    ps.RP21_circuit = SimpleNamespace(
        get_buffer=lambda name, mode: SimpleNamespace(size_max=size_max) if name == "sampled_wave" else None
    )
    return ps


def test_the_capacity_is_read_from_the_circuit():
    assert ps_with_circuit(256000).buffer_capacity() == 256000


def test_a_waveform_that_is_too_long_is_refused_before_anything_is_presented():
    ps = ps_with_circuit(26624)
    with pytest.raises(pystim3.CircuitError, match="does not fit in the buffer") as e:
        ps.check_recording_fits(1.2)
    assert "holds only 26624 samples" in str(e.value)


def test_a_waveform_that_fits_exactly_is_accepted():
    ps_with_circuit(26624).check_recording_fits(1.05)  # the record is exactly 26624 samples
    with pytest.raises(pystim3.CircuitError):
        ps_with_circuit(26624).check_recording_fits(1.0906)  # one block more


def test_the_larger_buffer_holds_the_protocols_that_are_used():
    ps = ps_with_circuit(256000)
    for duration in (0.645, 1.2, 2.0, 10.0):
        ps.check_recording_fits(duration)
    with pytest.raises(pystim3.CircuitError):
        ps.check_recording_fits(10.5)  # the record is 256000 + 1024 samples at most


def test_nothing_is_checked_without_the_rp21():
    ps_with_circuit(1024, hardware=("Soundcard",)).check_recording_fits(100.0)


def test_the_longest_waveform_of_a_protocol_is_the_one_that_sets_the_record():
    app = SimpleNamespace(
        wave_matrix={
            "a": {"sound": np.zeros(500000), "rate": 500000.0},  # 1 s
            "b": {"sound": np.zeros(600000), "rate": 500000.0},  # 1.2 s
            "c": {"sound": np.zeros(100000), "rate": 500000.0},
        }
    )
    assert pyabr3.PyABR.longest_waveform_duration(app) == pytest.approx(1.2)
