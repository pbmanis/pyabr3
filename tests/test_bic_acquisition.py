"""
Tests for presenting and tagging the interleaved BIC (ABR_BIC_PLAN.md, step 5), with
fake hardware. Free-field presentation and files must not change.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_bic_acquisition.py
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

from src import presenter_thread, protocol_reader, pyabr3, read_calibration, wave_generator

CALFILE = Path(REPO, "calfiles", "frequency_MF1.cal")


class FakeSound:
    """Stands in for pystim3.PyStim; records play_sound calls."""

    def __init__(self):
        self.calls = []

    def play_sound(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def retrieveRP21_inputs(self):
        return (np.zeros(4), np.zeros(4))


def bic_protocol(**stimuli):
    p = {
        "protocol": {"protocolname": "bic", "stimulustype": "click"},
        "stimuli": {
            "mode": "bic_interleaved",
            "wave_duration": 0.4,  # used by the free-field click, not by the BIC
            "stimulus_duration": 5e-4,
            "stimulus_risefall": 1e-4,
            "nstim": 2,
            "delay": 0.005,
            "interval": 40e-3,
            "nreps": 2,
            "alternate": True,
            "itd": 100e-6,
            "dblist": [80.0],
            "dblist_right": [70.0],
            "freqlist": [],
        },
    }
    p["stimuli"].update(stimuli)
    return p


def wave_matrix_for(protocol):
    caldata = read_calibration.get_calibration_data(CALFILE)
    wg = wave_generator.WaveGenerator(caldata=caldata)
    wg.setup(protocol=protocol, frequency=500000.0, config={"click_reference_dbspl": 105.1})
    wg.make_waveforms(
        wavetype=protocol["protocol"]["stimulustype"],
        dbspls=protocol["stimuli"]["dblist"],
        frequencies=None,
    )
    return wg.wave_matrix


# --- hardware requested for the mode -----------------------------------------------------


def test_freefield_requests_the_configured_hardware_unchanged():
    required = ["NIDAQ", "RP21", "PA5"]
    assert protocol_reader.required_hardware_for_mode(required, "freefield") == required


@pytest.mark.parametrize("mode", ["monaural_left", "monaural_right", "binaural", "bic_interleaved"])
def test_binaural_modes_add_the_second_attenuator(mode):
    required = ["NIDAQ", "RP21", "PA5"]
    assert protocol_reader.required_hardware_for_mode(required, mode) == required + ["PA5_2"]
    assert required == ["NIDAQ", "RP21", "PA5"]  # the configuration is not modified


def test_no_second_attenuator_without_a_first_one_or_twice():
    assert protocol_reader.required_hardware_for_mode(["NIDAQ", "RP21"], "binaural") == ["NIDAQ", "RP21"]
    both = ["NIDAQ", "PA5", "PA5_2"]
    assert protocol_reader.required_hardware_for_mode(both, "binaural") == both


# --- presenting -------------------------------------------------------------------------


def make_presenter(protocol, wave_matrix, sound):
    presenter = presenter_thread.Presenter()
    presenter.setWaveforms(wave=wave_matrix, protocol=protocol, sound_presentation_class=sound)
    return presenter


def test_bic_entry_plays_two_channels_with_each_attenuator():
    protocol = bic_protocol()
    matrix = wave_matrix_for(protocol)
    (entry,) = matrix.values()
    sound = FakeSound()
    presenter = make_presenter(protocol, matrix, sound)
    assert presenter._present_once() is True
    ((args, kwargs),) = sound.calls
    assert args[0] is entry["sound"]
    assert args[1] is entry["sound_right"]
    assert kwargs["attns"] == pytest.approx([105.1 - 80.0, 105.1 - 70.0])
    assert kwargs["two_channel"] is True


def test_freefield_entry_is_played_as_before():
    protocol = bic_protocol()
    del protocol["stimuli"]["mode"]
    matrix = wave_matrix_for(protocol)
    (entry,) = matrix.values()
    sound = FakeSound()
    presenter = make_presenter(protocol, matrix, sound)
    assert presenter._present_once() is True
    ((args, kwargs),) = sound.calls
    assert args[0] is entry["sound"] and args[1] is entry["sound"]
    assert kwargs == {"attns": [entry["attenuation"][0]] * 2}  # no two_channel argument


# --- tagging ----------------------------------------------------------------------------


def test_free_field_files_have_no_condition_tags():
    protocol = bic_protocol()
    del protocol["stimuli"]["mode"]
    app = SimpleNamespace(protocol=protocol, wave_matrix=wave_matrix_for(protocol))
    assert pyabr3.PyABR.get_condition_tags(app, 0) is None


def test_bic_files_are_tagged_with_the_conditions():
    protocol = bic_protocol()
    matrix = wave_matrix_for(protocol)
    app = SimpleNamespace(protocol=protocol, wave_matrix=matrix)
    tags = pyabr3.PyABR.get_condition_tags(app, 0)
    assert tags["ear_mode"] == "bic_interleaved"
    assert tags["slot_conditions"] == ["L", "L", "R", "R", "B", "B"] * 2
    assert tags["slot_polarities"] == [1, -1] * 6
    assert len(tags["slot_starts"]) == 12
    assert tags["itd"] == pytest.approx(100e-6)
    assert tags["dbspl_left"] == 80.0 and tags["dbspl_right"] == 70.0
    assert tags["attenuation"] == pytest.approx([25.1, 35.1])
    assert tags["frequency"] == 0.0


def test_tags_follow_the_wave_number():
    protocol = bic_protocol(dblist=[80.0, 60.0], dblist_right=[75.0, 55.0])
    app = SimpleNamespace(protocol=protocol, wave_matrix=wave_matrix_for(protocol))
    assert pyabr3.PyABR.get_condition_tags(app, 0)["dbspl_left"] == 80.0
    assert pyabr3.PyABR.get_condition_tags(app, 1)["dbspl_left"] == 60.0
    assert pyabr3.PyABR.get_condition_tags(app, 1)["dbspl_right"] == 55.0


# --- the conditions presented one at a time ---------------------------------------------


def test_sequential_modes_are_presented_in_order_on_two_channels():
    protocol = bic_protocol(mode="bic_sequential")
    matrix = wave_matrix_for(protocol)
    sound = FakeSound()
    presenter = make_presenter(protocol, matrix, sound)
    entries = list(matrix.values())
    assert len(entries) == 3  # L, R, B
    for wave_counter, entry in enumerate(entries):
        presenter.wave_counter = wave_counter
        assert presenter._present_once() is True
    assert len(sound.calls) == 3
    for ((args, kwargs), entry) in zip(sound.calls, entries):
        assert args[0] is entry["sound"] and args[1] is entry["sound_right"]
        assert kwargs["attns"] == pytest.approx(entry["attenuation"])
        assert kwargs["two_channel"] is True
    assert [e["condition"] for e in entries] == ["L", "R", "B"]
    # the ear that does not play is at maximum attenuation
    assert entries[0]["attenuation"][1] == 120.0 and entries[1]["attenuation"][0] == 120.0


@pytest.mark.parametrize("mode", ["monaural_left", "monaural_right", "binaural"])
def test_single_condition_modes_play_two_channels(mode):
    protocol = bic_protocol(mode=mode)
    matrix = wave_matrix_for(protocol)
    sound = FakeSound()
    presenter = make_presenter(protocol, matrix, sound)
    assert presenter._present_once() is True
    ((_, kwargs),) = sound.calls
    assert kwargs["two_channel"] is True


def test_sequential_sweeps_are_tagged_with_their_condition():
    protocol = bic_protocol(mode="bic_sequential")
    app = SimpleNamespace(protocol=protocol, wave_matrix=wave_matrix_for(protocol))
    for wave_counter, condition in enumerate("LRB"):
        tags = pyabr3.PyABR.get_condition_tags(app, wave_counter)
        assert tags["ear_mode"] == "bic_sequential"
        assert tags["slot_conditions"] == [condition] * 2
        assert tags["slot_polarities"] == [1, -1]
        assert tags["dbspl_left"] == 80.0 and tags["dbspl_right"] == 70.0


@pytest.mark.parametrize("mode", ["monaural_left", "monaural_right", "binaural", "bic_sequential"])
def test_these_modes_request_the_second_attenuator(mode):
    assert protocol_reader.required_hardware_for_mode(["NIDAQ", "PA5"], mode) == ["NIDAQ", "PA5", "PA5_2"]
