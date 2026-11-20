"""
Tests for the interleaved BIC waveforms (ABR_BIC_PLAN.md, step 4): wave_generator
WaveGenerator.make_waveforms with "mode: bic_interleaved". Free-field stimuli are
checked in test_freefield_baseline.py.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_bic_waveforms.py
"""

import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from src import read_calibration, wave_generator

CALFILE = Path(REPO, "calfiles", "frequency_MF1.cal")
SFOUT = 500000.0
CONFIG = {"click_reference_dbspl": 105.1}


def protocol(stimulustype="click", **stimuli):
    p = {
        "protocol": {"protocolname": "bic", "stimulustype": stimulustype},
        "stimuli": {
            "mode": "bic_interleaved",
            "wave_duration": 1.2,
            "stimulus_duration": 5e-4,
            "stimulus_risefall": 1e-4,
            "nstim": 3,
            "delay": 0.005,
            "interval": 40e-3,
            "nreps": 10,
            "alternate": True,
            "itd": 0.0,
            "default_frequency": 4000.0,
            "default_spl": 80.0,
            "dblist": [80.0],
            "freqlist": [],
        },
    }
    p["stimuli"].update(stimuli)
    return p


@pytest.fixture
def caldata():
    return read_calibration.get_calibration_data(CALFILE)


def make(caldata, stimulustype="click", dbspls=None, frequencies=None, caldata_right=None, **stimuli):
    wg = wave_generator.WaveGenerator(caldata=caldata)
    wg.caldata_right = caldata_right
    prot = protocol(stimulustype, **stimuli)
    wg.setup(protocol=prot, frequency=SFOUT, config=copy.deepcopy(CONFIG))
    wg.make_waveforms(
        wavetype=stimulustype,
        dbspls=prot["stimuli"]["dblist"] if dbspls is None else dbspls,
        frequencies=frequencies,
    )
    return wg


def slot_window(entry, k, itd_samples=0):
    start = int(round(entry["slot_starts"][k] * SFOUT)) + itd_samples
    return slice(start, start + int(round(40e-3 * SFOUT)))


def test_slot_layout_with_alternation(caldata):
    wg = make(caldata)
    (key,) = wg.wave_matrix.keys()
    entry = wg.wave_matrix[key]
    assert key == ("click", 80.0, 0.0, 80.0, 0.0)
    assert entry["slot_conditions"] == ["L", "L", "R", "R", "B", "B"] * 3
    assert entry["slot_polarities"] == [1, -1] * 9
    starts = np.array(entry["slot_starts"])
    np.testing.assert_allclose(np.diff(starts), 40e-3)
    assert starts[0] == pytest.approx(0.005)
    assert wg.nwaves == 1


def test_slot_layout_without_alternation(caldata):
    entry = next(iter(make(caldata, alternate=False).wave_matrix.values()))
    assert entry["slot_conditions"] == ["L", "R", "B"] * 3
    assert entry["slot_polarities"] == [1] * 9


def test_waveform_length_and_channels(caldata):
    entry = next(iter(make(caldata).wave_matrix.values()))
    nslots = 18
    expected = int(round((0.005 + nslots * 40e-3 + 40e-3) * SFOUT))
    assert len(entry["sound"]) == len(entry["sound_right"]) == expected
    assert entry["rate"] == SFOUT
    assert np.max(np.abs(entry["sound"])) == pytest.approx(read_calibration.SPLCAL_Click_Voltage)


def test_monaural_slots_are_silent_in_the_other_channel(caldata):
    entry = next(iter(make(caldata).wave_matrix.values()))
    for k, condition in enumerate(entry["slot_conditions"]):
        window = slot_window(entry, k)
        left = np.max(np.abs(entry["sound"][window]))
        right = np.max(np.abs(entry["sound_right"][window]))
        assert (left > 0) == (condition in ("L", "B")), (k, condition)
        assert (right > 0) == (condition in ("R", "B")), (k, condition)


def test_polarity_alternates_within_each_pair(caldata):
    for stimulustype, freqs in (("click", None), ("tonepip", [4000.0])):
        entry = next(iter(make(caldata, stimulustype, frequencies=freqs).wave_matrix.values()))
        left = entry["sound"]
        for k in range(0, 18, 2):  # slots k and k+1 are a + and - pair
            first, second = left[slot_window(entry, k)], left[slot_window(entry, k + 1)]
            if np.max(np.abs(first)) > 0:
                np.testing.assert_allclose(second, -first, atol=1e-12)


def test_itd_delays_the_lagging_ear_in_binaural_slots(caldata):
    itd = 200e-6  # 100 samples at 500 kHz
    entry = next(iter(make(caldata, itd=itd).wave_matrix.values()))
    assert entry["itd"] == itd
    k = entry["slot_conditions"].index("B")
    left, right = entry["sound"], entry["sound_right"]
    i_left = np.flatnonzero(left[slot_window(entry, k)])[0]
    i_right = np.flatnonzero(right[slot_window(entry, k)])[0]
    assert i_right - i_left == 100  # positive itd: the right ear lags
    # the monaural slots are not delayed
    kl = entry["slot_conditions"].index("L")
    kr = entry["slot_conditions"].index("R")
    assert np.flatnonzero(left[slot_window(entry, kl)])[0] == np.flatnonzero(right[slot_window(entry, kr)])[0]


def test_negative_itd_delays_the_left_ear(caldata):
    entry = next(iter(make(caldata, itd=-200e-6).wave_matrix.values()))
    k = entry["slot_conditions"].index("B")
    i_left = np.flatnonzero(entry["sound"][slot_window(entry, k)])[0]
    i_right = np.flatnonzero(entry["sound_right"][slot_window(entry, k)])[0]
    assert i_left - i_right == 100


def test_click_attenuation_per_ear(caldata):
    wg = make(caldata, dblist=[80.0], dblist_right=[70.0])
    (key,) = wg.wave_matrix.keys()
    assert key == ("click", 80.0, 0.0, 70.0, 0.0)
    assert wg.wave_matrix[key]["attenuation"] == pytest.approx([105.1 - 80.0, 105.1 - 70.0])


def test_right_level_defaults_to_left(caldata):
    entry = next(iter(make(caldata, dblist=[60.0]).wave_matrix.values()))
    assert entry["dbspl_right"] == 60.0
    assert entry["attenuation"][0] == pytest.approx(entry["attenuation"][1])


def test_levels_are_paired_by_position(caldata):
    wg = make(caldata, dblist=[80.0, 60.0], dblist_right=[75.0, 55.0])
    assert list(wg.wave_matrix.keys()) == [("click", 80.0, 0.0, 75.0, 0.0), ("click", 60.0, 0.0, 55.0, 0.0)]
    assert wg.nwaves == 2


def test_dblist_strings_are_evaluated(caldata):
    wg = make(caldata, dblist="[80.0, 60.0]")
    assert wg.nwaves == 2


def test_tone_attenuation_uses_each_ears_calibration(caldata):
    quieter = copy.deepcopy(caldata)
    quieter["maxdb"] = np.asarray(quieter["maxdb"], dtype=float) - 6.0
    wg = make(caldata, "tonepip", frequencies=[8000.0, 4000.0], caldata_right=quieter, dblist=[50.0])
    assert [k[2] for k in wg.wave_matrix] == [8000.0, 4000.0]
    for entry in wg.wave_matrix.values():
        assert entry["attenuation"][1] == pytest.approx(entry["attenuation"][0] - 6.0)
        assert np.max(np.abs(entry["sound_right"])) == pytest.approx(read_calibration.SPLCAL_Tone_Voltage, rel=1e-3)


def test_unreachable_levels_are_reported_for_each_ear(caldata):
    with pytest.raises(wave_generator.UnreachableLevelError, match="right ear"):
        make(caldata, dblist=[80.0], dblist_right=[110.0])
    with pytest.raises(wave_generator.UnreachableLevelError, match="left ear"):
        make(caldata, dblist=[110.0], dblist_right=[80.0])
    quieter = copy.deepcopy(caldata)
    quieter["maxdb"] = np.asarray(quieter["maxdb"], dtype=float) - 10.0
    wg = wave_generator.WaveGenerator(caldata=caldata)
    level = wg.compute_attn_db_for_dbspl(8000.0, 0.0)[1][0] - 5.0  # left can reach it, right cannot
    make(caldata, "tonepip", frequencies=[8000.0], dblist=[level])
    with pytest.raises(wave_generator.UnreachableLevelError):
        make(caldata, "tonepip", frequencies=[8000.0], caldata_right=quieter, dblist=[level])


def test_bad_protocols_are_rejected(caldata):
    with pytest.raises(ValueError, match="same length"):
        make(caldata, dblist=[80.0, 70.0], dblist_right=[80.0])
    with pytest.raises(ValueError, match="itd"):
        make(caldata, itd=0.04)
    with pytest.raises(ValueError, match="nstim"):
        make(caldata, nstim=0)


# --- the conditions presented one at a time (approach a) --------------------------------


def starts_in(entry, channel):
    """The sample indices where each stimulus of a sweep starts in a channel."""
    wave = entry[channel]
    onsets = np.flatnonzero((wave[1:] != 0) & (wave[:-1] == 0)) + 1
    return onsets if wave[0] == 0 else np.r_[0, onsets]


def test_single_modes_have_one_entry_with_the_right_ears_playing(caldata):
    for mode, condition, left_plays, right_plays in (
        ("monaural_left", "L", True, False),
        ("monaural_right", "R", False, True),
        ("binaural", "B", True, True),
    ):
        wg = make(caldata, mode=mode)
        (key,) = wg.wave_matrix.keys()
        entry = wg.wave_matrix[key]
        assert key == ("click", 80.0, 0.0, 80.0, 0.0, condition)
        assert entry["condition"] == condition and entry["mode"] == mode
        assert (np.max(np.abs(entry["sound"])) > 0) == left_plays
        assert (np.max(np.abs(entry["sound_right"])) > 0) == right_plays
        assert entry["slot_conditions"] == [condition] * 3
        assert wg.nwaves == 1


def test_sweep_is_nstim_stimuli_with_alternating_polarity(caldata):
    entry = next(iter(make(caldata, mode="monaural_left", nstim=4).wave_matrix.values()))
    assert len(entry["sound"]) == int(round(1.2 * SFOUT))
    assert entry["slot_polarities"] == [1, -1, 1, -1]
    onsets = starts_in(entry, "sound")
    np.testing.assert_array_equal(onsets, np.round((0.005 + 0.04 * np.arange(4)) * SFOUT).astype(int))
    peaks = entry["sound"][onsets]
    assert peaks[0] > 0 > peaks[1] and peaks[2] > 0 > peaks[3]


def test_no_alternation_gives_one_polarity(caldata):
    entry = next(iter(make(caldata, mode="binaural", alternate=False).wave_matrix.values()))
    assert entry["slot_polarities"] == [1, 1, 1]
    assert np.min(entry["sound"]) == 0.0 and np.min(entry["sound_right"]) == 0.0


def test_binaural_itd_delays_the_lagging_ear(caldata):
    entry = next(iter(make(caldata, mode="binaural", itd=200e-6).wave_matrix.values()))
    np.testing.assert_array_equal(starts_in(entry, "sound_right") - starts_in(entry, "sound"), 100)
    entry = next(iter(make(caldata, mode="binaural", itd=-200e-6).wave_matrix.values()))
    np.testing.assert_array_equal(starts_in(entry, "sound") - starts_in(entry, "sound_right"), 100)


def test_monaural_modes_ignore_itd(caldata):
    entry = next(iter(make(caldata, mode="monaural_right", itd=200e-6).wave_matrix.values()))
    np.testing.assert_array_equal(
        starts_in(entry, "sound_right"), np.round((0.005 + 0.04 * np.arange(3)) * SFOUT).astype(int)
    )


def test_sequential_runs_each_condition_in_turn_for_each_stimulus(caldata):
    wg = make(caldata, mode="bic_sequential", dblist=[80.0, 60.0], dblist_right=[75.0, 55.0])
    assert list(wg.wave_matrix.keys()) == [
        ("click", 80.0, 0.0, 75.0, 0.0, "L"),
        ("click", 80.0, 0.0, 75.0, 0.0, "R"),
        ("click", 80.0, 0.0, 75.0, 0.0, "B"),
        ("click", 60.0, 0.0, 55.0, 0.0, "L"),
        ("click", 60.0, 0.0, 55.0, 0.0, "R"),
        ("click", 60.0, 0.0, 55.0, 0.0, "B"),
    ]
    assert wg.nwaves == 6


def test_sequential_tone_pips_loop_over_frequencies(caldata):
    wg = make(caldata, "tonepip", frequencies=[8000.0, 4000.0], mode="bic_sequential", dblist=[50.0])
    assert [(k[2], k[5]) for k in wg.wave_matrix] == [
        (8000.0, "L"), (8000.0, "R"), (8000.0, "B"), (4000.0, "L"), (4000.0, "R"), (4000.0, "B")
    ]


def test_the_ear_that_does_not_play_has_maximum_attenuation(caldata):
    wg = make(caldata, mode="bic_sequential", dblist=[80.0], dblist_right=[70.0])
    attenuation = {k[5]: e["attenuation"] for k, e in wg.wave_matrix.items()}
    assert attenuation["L"] == pytest.approx([25.1, 120.0])
    assert attenuation["R"] == pytest.approx([120.0, 35.1])
    assert attenuation["B"] == pytest.approx([25.1, 35.1])


def test_only_the_ears_that_play_need_to_reach_the_level(caldata):
    # the right ear cannot reach 110 dB SPL, but a left-only stimulus does not use it
    make(caldata, mode="monaural_left", dblist=[80.0], dblist_right=[110.0])
    with pytest.raises(wave_generator.UnreachableLevelError, match="right ear"):
        make(caldata, mode="monaural_right", dblist=[80.0], dblist_right=[110.0])
    with pytest.raises(wave_generator.UnreachableLevelError, match="right ear"):
        make(caldata, mode="binaural", dblist=[80.0], dblist_right=[110.0])


def test_sequential_tags_match_the_interleaved_ones(caldata):
    entry = next(iter(make(caldata, mode="binaural", nstim=3, itd=1e-4).wave_matrix.values()))
    for key in ("slot_conditions", "slot_polarities", "slot_starts", "itd", "frequency",
                "dbspl_left", "dbspl_right", "attenuation"):
        assert key in entry


def test_sequential_sweep_must_fit_the_waveform(caldata):
    with pytest.raises(ValueError, match="wave_duration"):
        make(caldata, mode="monaural_left", nstim=40, wave_duration=1.0)  # needs 1.6 s


def test_sequential_free_field_protocols_are_unchanged(caldata):
    wg = make(caldata, mode="freefield")  # the free-field click path, not the BIC code
    assert all(len(k) == 3 for k in wg.wave_matrix)
