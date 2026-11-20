"""
Tests for the interleaved BIC viewer (ABR_BIC_PLAN.md, step 7), using generated data
files in the format that pyabr3 writes for the interleaved BIC.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_bic_viewer.py
"""

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import pyqtgraph as pg

from src.viewer import abr_files, bic_files, bic_viewer

RATE = 24414.0625  # RP2.1 sample rate
DELAY = 0.005
INTERVAL = 0.04
LATENCY = 3e-3  # the response peaks this long after the start of its slot
SIGMA = 5e-4
AMPLITUDE = {"L": 1.0, "R": 2.0, "B": 3.0}  # V at the amplifier; 100, 200, 300 uV at the electrodes
ARTIFACT = 5.0  # V, reverses with the polarity of the stimulus
ARTIFACT_LATENCY = 2e-4


def gaussian(t, center):
    return np.exp(-0.5 * ((t - center) / SIGMA) ** 2)


def make_trial(
    alternate=True, nstim=2, dbspl_left=80.0, dbspl_right=70.0, itd=2e-4, amplitude=1.0, amplitudes=None,
    only=None, ear_mode="bic_interleaved",
):
    """only: a condition ("L", "R" or "B"): a sweep of that condition alone, as in the
    modes that present the conditions one at a time (give an ear_mode for it)."""
    amplitudes = AMPLITUDE if amplitudes is None else amplitudes
    group = [("L", 1), ("L", -1), ("R", 1), ("R", -1), ("B", 1), ("B", -1)] if alternate else [
        ("L", 1), ("R", 1), ("B", 1)
    ]
    if only is not None:
        group = [(only, 1), (only, -1)] if alternate else [(only, 1)]
    slots = group * nstim
    starts = [DELAY + k * INTERVAL for k in range(len(slots))]
    duration = DELAY + len(slots) * INTERVAL + INTERVAL
    t = np.arange(int(duration * RATE)) / RATE
    data = np.zeros_like(t)
    for (condition, polarity), start in zip(slots, starts):
        data += amplitude * amplitudes[condition] * gaussian(t, start + LATENCY)
        data += polarity * ARTIFACT * gaussian(t, start + ARTIFACT_LATENCY)
    return {
        "calibration": {"spkr": "MF1"},
        "protocol": {
            "protocol": {"stimulustype": "click"},
            "stimuli": {"delay": DELAY, "interval": INTERVAL, "nstim": nstim, "mode": "bic_interleaved"},
        },
        "data": data,
        "record_frequency": RATE,
        "ear_mode": ear_mode,
        "conditions": {
            "slot_conditions": [c for c, _ in slots],
            "slot_polarities": [p for _, p in slots],
            "slot_starts": starts,
            "itd": itd,
            "frequency": 0.0,
            "dbspl_left": dbspl_left,
            "dbspl_right": dbspl_right,
            "attenuation": [25.1, 35.1],
        },
    }


def write_run(directory: Path, trials: dict) -> Path:
    """trials: {(wave, repetition): trial dictionary}"""
    directory.mkdir(parents=True, exist_ok=True)
    for (wave, rep), trial in trials.items():
        with open(directory / f"2025-01-01_click_{wave:03d}_{rep:03d}.p", "wb") as fh:
            pickle.dump(trial, fh)
    return directory / "2025-01-01_click_000_001.p"


def peak(responses, condition):
    trace = responses.traces[condition]
    i = int(np.argmax(trace))
    return responses.t_ms[i], trace[i]


def test_conditions_are_separated_and_the_artifact_cancels(tmp_path):
    first = write_run(tmp_path, {(0, 1): make_trial()})
    (responses,) = bic_files.load_bic_run(first).values()
    assert set(responses.traces) == {"L", "R", "B"}
    for condition, amplitude in AMPLITUDE.items():
        latency_ms, value = peak(responses, condition)
        assert latency_ms == pytest.approx(LATENCY * 1000.0, abs=0.1)
        assert value == pytest.approx(amplitude * 100.0, rel=0.05)  # uV
        t = responses.t_ms
        before = responses.traces[condition][(t > ARTIFACT_LATENCY * 1000 - 0.1) & (t < ARTIFACT_LATENCY * 1000 + 0.1)]
        assert np.max(np.abs(before)) < 0.05 * amplitude * 100.0  # artifact (500 uV) is gone


def test_without_alternation_the_artifact_is_not_removed(tmp_path):
    first = write_run(tmp_path, {(0, 1): make_trial(alternate=False)})
    (responses,) = bic_files.load_bic_run(first).values()
    t = responses.t_ms
    window = (t > ARTIFACT_LATENCY * 1000 - 0.1) & (t < ARTIFACT_LATENCY * 1000 + 0.1)
    assert np.max(responses.traces["L"][window]) > 100.0  # the 500 uV artifact is there
    assert responses.info["alternating"] is False


def test_time_zero_is_the_start_of_the_slot(tmp_path):
    first = write_run(tmp_path, {(0, 1): make_trial()})
    (responses,) = bic_files.load_bic_run(first).values()
    assert responses.t_ms[0] == pytest.approx(-1.0)
    assert responses.t_ms[1] - responses.t_ms[0] == pytest.approx(1000.0 / abr_files.RESAMPLED_RATE)
    assert len(responses.t_ms) == int(round(INTERVAL * abr_files.RESAMPLED_RATE))


def test_trials_and_slots_are_counted(tmp_path):
    trials = {(0, rep): make_trial(nstim=2) for rep in (1, 2, 3)}
    (responses,) = bic_files.load_bic_run(write_run(tmp_path, trials)).values()
    assert responses.n_trials == 3
    assert responses.counts == {"L": 12, "R": 12, "B": 12}  # 4 slots per condition, 3 trials


def test_repetitions_are_averaged_with_equal_weight(tmp_path):
    trials = {(0, 1): make_trial(amplitude=1.0), (0, 2): make_trial(amplitude=3.0)}
    (responses,) = bic_files.load_bic_run(write_run(tmp_path, trials)).values()
    assert peak(responses, "L")[1] == pytest.approx(2.0 * 100.0, rel=0.05)


def test_each_wave_has_its_own_stimulus(tmp_path):
    trials = {
        (0, 1): make_trial(dbspl_left=80.0, dbspl_right=80.0),
        (1, 1): make_trial(dbspl_left=60.0, dbspl_right=50.0, itd=0.0),
    }
    first = write_run(tmp_path, trials)
    responses = bic_files.load_bic_run(first)
    assert list(responses) == [0, 1]
    assert responses[1].info["dbspl_left"] == 60.0 and responses[1].info["dbspl_right"] == 50.0
    infos = bic_files.wave_info(abr_files.find_run(first))
    assert infos[0]["dbspl_right"] == 80.0
    assert bic_files.stimulus_label(infos[0]) == "80 dB SPL, click, ITD 0.20 ms"
    assert bic_files.stimulus_label(responses[1].info) == "60/50 dB SPL (L/R), click, ITD 0.00 ms"


def test_tone_label_has_the_frequency():
    info = {"dbspl_left": 70.0, "dbspl_right": 70.0, "frequency": 8000.0, "itd": 0.0}
    assert bic_files.stimulus_label(info) == "70 dB SPL, 8 kHz, ITD 0.00 ms"


def test_filtering_keeps_the_conditions_apart(tmp_path):
    first = write_run(tmp_path, {(0, 1): make_trial()})
    (responses,) = bic_files.load_bic_run(first, hpf=300.0, lpf=3000.0).values()
    assert peak(responses, "B")[1] > peak(responses, "R")[1] > peak(responses, "L")[1] > 0


def test_a_recording_that_is_not_bic_is_rejected(tmp_path):
    trial = make_trial()
    del trial["conditions"], trial["ear_mode"]  # a free-field file has neither
    first = write_run(tmp_path, {(0, 1): trial})
    with pytest.raises(ValueError, match="not a binaural"):
        bic_files.load_bic_run(first)


def test_slot_outside_the_sweep_is_an_error():
    with pytest.raises(ValueError, match="not inside"):
        bic_files.slot_responses(np.zeros(1000), 1000.0, [0.5, 2.0], 0.1, 0.0)


def test_slot_responses_cut_the_expected_samples():
    sweep = np.arange(1000.0)  # one unit per sample at 1000 Hz
    slots = bic_files.slot_responses(sweep, 1000.0, [0.1, 0.3], 0.05, 0.0)
    assert slots.shape == (2, 50)
    assert slots[0, 0] == 99.0  # starts PRESTIMULUS_S (1 sample) before 0.1 s
    assert slots[1, 0] == 299.0


def test_average_by_condition():
    slots = np.array([[1.0], [3.0], [10.0], [0.0]])
    means, counts = bic_files.average_by_condition(slots, ["L", "L", "R", "R"])
    assert means["L"][0] == 2.0 and means["R"][0] == 5.0
    assert counts == {"L": 2, "R": 2}
    assert "B" not in means


# --- the panel --------------------------------------------------------------------------


@pytest.fixture
def panel(qtbot=None):
    pg.mkQApp("test")
    return bic_viewer.BICPanel()


def test_panel_shows_three_labeled_conditions(panel, tmp_path):
    first = write_run(tmp_path, {(0, 1): make_trial(), (1, 1): make_trial(dbspl_left=60.0, dbspl_right=60.0)})
    panel.load(first)
    assert panel.wave_combo.count() == 2
    names = [item.opts["name"] for item in panel.plot.listDataItems()]
    assert names == ["Left", "Right", "Binaural", "L+R"]
    pens = {item.opts["name"]: item.opts["pen"].color().name() for item in panel.plot.listDataItems()}
    assert len(set(pens.values())) == 4  # all different colors
    panel.wave_combo.setCurrentIndex(1)
    assert "60 dB SPL" in panel.plot.getPlotItem().titleLabel.text
    assert len(panel.plot.listDataItems()) == 4


def test_bic_is_binaural_minus_the_sum_of_left_and_right(tmp_path):
    amplitudes = {"L": 1.0, "R": 2.0, "B": 2.0}  # 100, 200, 200 uV: the binaural is smaller than L+R
    first = write_run(tmp_path, {(0, 1): make_trial(amplitudes=amplitudes)})
    (responses,) = bic_files.load_bic_run(first).values()
    assert responses.has_bic
    np.testing.assert_allclose(responses.sum_lr, responses.traces["L"] + responses.traces["R"])
    np.testing.assert_allclose(responses.bic, responses.traces["B"] - responses.sum_lr)
    i = int(np.argmin(responses.bic))
    assert responses.t_ms[i] == pytest.approx(LATENCY * 1000.0, abs=0.1)
    assert responses.bic[i] == pytest.approx(-100.0, rel=0.05)  # 200 - (100 + 200) uV
    assert np.max(responses.sum_lr) == pytest.approx(300.0, rel=0.05)


def sequential_trials(reps=2, **kwargs):
    """Trials of a bic_sequential run: wave 0 is left only, 1 right only, 2 binaural."""
    trials = {}
    for wave, condition in enumerate("LRB"):
        for rep in range(1, reps + 1):
            trials[(wave, rep)] = make_trial(only=condition, ear_mode="bic_sequential", **kwargs)
    return trials


def test_conditions_presented_one_at_a_time_are_combined(tmp_path):
    amplitudes = {"L": 1.0, "R": 2.0, "B": 2.0}
    first = write_run(tmp_path, sequential_trials(reps=2, amplitudes=amplitudes))
    responses = bic_files.load_bic_run(first)
    assert list(responses) == [0]  # one stimulus, under its first wave
    (result,) = responses.values()
    assert result.has_bic
    assert result.n_trials == 2  # trials of each condition
    assert result.counts == {"L": 8, "R": 8, "B": 8}  # 4 slots in each of 2 trials per condition
    assert peak(result, "L")[1] == pytest.approx(100.0, rel=0.05)
    assert peak(result, "R")[1] == pytest.approx(200.0, rel=0.05)
    assert result.bic[int(np.argmin(result.bic))] == pytest.approx(-100.0, rel=0.05)
    assert result.info["ear_mode"] == "bic_sequential"


def test_each_stimulus_of_a_sequential_run_is_combined_separately(tmp_path):
    trials = {}
    for k, (level, itd) in enumerate([(80.0, 0.0), (80.0, 4e-4)]):
        for wave, condition in enumerate("LRB"):
            trials[(3 * k + wave, 1)] = make_trial(
                only=condition, ear_mode="bic_sequential", dbspl_left=level, dbspl_right=level, itd=itd
            )
    responses = bic_files.load_bic_run(write_run(tmp_path, trials))
    assert list(responses) == [0, 3]
    assert [r.info["itd"] for r in responses.values()] == [0.0, 4e-4]
    assert all(r.has_bic for r in responses.values())


@pytest.mark.parametrize("mode,condition", [("monaural_left", "L"), ("monaural_right", "R"), ("binaural", "B")])
def test_a_single_condition_run_shows_that_condition(tmp_path, mode, condition):
    first = write_run(tmp_path, {(0, 1): make_trial(only=condition, ear_mode=mode)})
    (responses,) = bic_files.load_bic_run(first).values()
    assert list(responses.traces) == [condition]
    assert not responses.has_bic and responses.bic is None


def test_no_bic_without_all_three_conditions():
    responses = bic_files.BICResponses(np.zeros(3), {"L": np.ones(3), "R": np.ones(3)}, {}, 1)
    assert not responses.has_bic
    assert responses.bic is None
    assert responses.sum_lr is not None
    assert bic_files.BICResponses(np.zeros(3), {"L": np.ones(3)}, {}, 1).sum_lr is None


def test_panel_plots_the_bic_and_can_overlay_all_stimuli(panel, tmp_path):
    trials = {
        (0, 1): make_trial(itd=0.0, amplitudes={"L": 1.0, "R": 1.0, "B": 1.5}),
        (1, 1): make_trial(itd=4e-4, amplitudes={"L": 1.0, "R": 1.0, "B": 1.2}),
        (2, 1): make_trial(itd=8e-4, amplitudes={"L": 1.0, "R": 1.0, "B": 1.0}),
    }
    panel.hpf_box.setValue(0.0)  # the default filter would reduce these broad pulses
    panel.lpf_box.setValue(0.0)
    panel.load(write_run(tmp_path, trials))
    assert len(panel.bic_plot.listDataItems()) == 1
    assert "BIC" in panel.bic_plot.getPlotItem().titleLabel.text
    panel.overlay_box.setChecked(True)
    items = panel.bic_plot.listDataItems()
    assert len(items) == 3
    assert [i.opts["name"] for i in items] == [
        "80/70 dB SPL (L/R), click, ITD 0.00 ms",
        "80/70 dB SPL (L/R), click, ITD 0.40 ms",
        "80/70 dB SPL (L/R), click, ITD 0.80 ms",
    ]
    troughs = [float(np.min(i.yData)) for i in items]  # the BIC is negative: B is below L+R
    assert troughs == pytest.approx([-50.0, -80.0, -100.0], rel=0.1)
    panel.overlay_box.setChecked(False)
    assert len(panel.bic_plot.listDataItems()) == 1


def test_panel_reports_an_unreadable_recording(panel, tmp_path):
    trial = make_trial()
    del trial["conditions"], trial["ear_mode"]
    panel.load(write_run(tmp_path, {(0, 1): trial}))
    assert "not a binaural" in panel.status_label.text()
    assert len(panel.plot.listDataItems()) == 0
    assert len(panel.bic_plot.listDataItems()) == 0
