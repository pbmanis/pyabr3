"""Tests for src/viewer, using generated data files in the format written by pyabr3."""

import datetime
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import pyqtgraph as pg

from src.viewer import abr_files, data_viewer

RATE = 97656.25
DATE = "2024-11-13"
NSTIM = 8
INTERVAL = 0.02
PERIOD = 0.01
N_POINTS = int(RATE * 0.25)
LPF = 3000.0
HPF = 300.0


def calibration():
    return {"spkr": "MF1", "mic": "7016#10252", "date": "09-Dec-2024", "calattn": 20,
            "filename": Path("frequency_MF1.cal")}  # fmt: skip


def protocol(**stimuli):
    return {
        "protocol": {"stimulustype": "click"},
        "stimuli": {
            "delay": 0.003,
            "interval": INTERVAL,
            "stimulus_period": PERIOD,
            "nstim": NSTIM,
            "nreps": 3,
        }
        | stimuli,
        "recording": {"LPF": LPF},
    }


def write_trial(directory, stim, wave, rep, data, table, proto):
    trial = {
        "subject_data": {"Subject ID": "x", "Age": "P30", "Sex": "F", "Weight": 20.0,
                         "Strain": "CBA", "Genotype": "WT", "Treat Group": ""},
        "calibration": calibration(),
        "stimuli": table,
        "protocol": proto,
        "wave_number": wave,
        "repetition": rep,
        "data": data,
        "record_frequency": RATE,
        "timestamp": datetime.datetime(2024, 11, 13, 12, 0, 0),
    }  # fmt: skip
    with open(directory / f"{DATE:s}_{stim:s}_{wave:03d}_{rep:03d}.p", "wb") as fh:
        pickle.dump(trial, fh)


def signal(seed, amplitude=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(N_POINTS) / RATE
    return amplitude * 1e-3 * np.sin(2 * np.pi * 800.0 * t) + 1e-4 * rng.standard_normal(N_POINTS)


@pytest.fixture
def click_dir(tmp_path):
    directory = tmp_path / "clicks"
    directory.mkdir()
    table = {("click", 90.0, 0): {"sound": np.ones(100), "rate": RATE},
             ("click", 50.0, 0): {"sound": {}, "rate": RATE}}  # fmt: skip
    proto = protocol(dblist=[90.0, 50.0], freqlist=[])
    for wave in range(2):
        for rep in range(1, 4):
            write_trial(directory, "click", wave, rep, signal(10 * wave + rep), table, proto)
    return directory


@pytest.fixture
def tone_dir(tmp_path):
    directory = tmp_path / "tones"
    directory.mkdir()
    table = {("tonepip", 30.0, 2000.0): {}, ("tonepip", 30.0, 4000.0): {},
             ("tonepip", 60.0, 2000.0): {}, ("tonepip", 60.0, 4000.0): {}}  # fmt: skip
    proto = protocol(dblist="[30.0, 60.0]", freqlist="[2000.0, 4000.0]")
    for wave in range(4):
        for rep in range(1, 4):
            write_trial(directory, "tonepip", wave, rep, signal(wave + rep), table, proto)
    return directory


@pytest.fixture
def interleaved_dir(tmp_path):
    directory = tmp_path / "interleaved_plateau"
    directory.mkdir()
    entry = {
        "dbspls": np.array([10.0, 10.0, 20.0, 20.0]),
        "frequencies": np.array([1000.0, 2000.0, 1000.0, 2000.0]),
        "sound": {},
    }
    table = {("interleaved_plateau", 0.0): entry, ("interleaved_plateau", 10.0): entry}
    proto = protocol(dblist="np.linspace(10.0, 20.0, 2)", freqlist=[1000.0, 2000.0], nreps=3)
    for wave in range(2):
        for rep in range(1, 4):
            write_trial(
                directory, "interleaved_plateau", wave, rep, signal(rep, 1.0 + wave), table, proto
            )
    return directory


def first_file(directory):
    return sorted(directory.glob("*.p"))[0]


def test_find_run_collects_only_the_same_recording(click_dir):
    (click_dir / "2024-11-14_click_000_001.p").write_bytes(b"")
    (click_dir / "notes.p").write_bytes(b"")
    run = abr_files.find_run(first_file(click_dir))
    assert run.date == DATE and run.stim == "click"
    assert run.waves == [0, 1] and run.repetitions == [1, 2, 3]
    assert len(run.files) == 6


def test_find_run_handles_stimulus_names_with_underscores(interleaved_dir):
    run = abr_files.find_run(first_file(interleaved_dir))
    assert run.stim == "interleaved_plateau" and run.interleaved


def test_name_that_is_not_a_trial_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="trial file"):
        abr_files.find_run(tmp_path / "notes.p")


@pytest.mark.parametrize("fixture_name,index", [("click_dir", 0), ("tone_dir", 1)])
def test_click_and_tone_averages_match_the_analysis(fixture_name, index, request):
    directory = request.getfixturevalue(fixture_name)
    data = abr_files.load_run(first_file(directory))
    responses = abr_files.compute_responses(data, HPF, LPF)
    wave = data.run.waves[index]
    expected, t, _ = abr_files.ANALYZER.average_within_traces(
        directory, wave, data.header["protocol"], DATE, high_pass_filter=HPF, low_pass_filter=LPF
    )
    _, db, fr = list(data.header["stimuli"])[wave]
    np.testing.assert_allclose(responses.traces[(db, fr)], expected * 100.0, atol=1e-9)
    np.testing.assert_allclose(responses.t_ms, t * 1000.0, atol=1e-9)


def test_interleaved_averages_match_the_analysis(interleaved_dir):
    data = abr_files.load_run(first_file(interleaved_dir))
    responses = abr_files.compute_responses(data, HPF, LPF)
    proto = data.header["protocol"] | {
        "stimuli": data.header["protocol"]["stimuli"] | {"dblist": [10.0, 20.0]}
    }
    expected, _, _, _ = abr_files.ANALYZER.average_across_traces(
        interleaved_dir, 0, proto, DATE, high_pass_filter=HPF, low_pass_filter=LPF
    )
    pips = [(10.0, 1000.0), (10.0, 2000.0), (20.0, 1000.0), (20.0, 2000.0)]
    for i, key in enumerate(pips):
        np.testing.assert_allclose(responses.traces[key], expected[i] * 100.0, atol=1e-9)


def test_interleaved_set_selects_the_stimulus_set(interleaved_dir):
    data = abr_files.load_run(first_file(interleaved_dir))
    first = abr_files.compute_responses(data, None, None, interleaved_set=0)
    second = abr_files.compute_responses(data, None, None, interleaved_set=1)
    key = (10.0, 1000.0)
    assert np.ptp(second.traces[key]) > 1.5 * np.ptp(first.traces[key])


def test_missing_repetition_is_left_out_of_the_average(click_dir):
    missing = click_dir / f"{DATE:s}_click_001_002.p"
    missing.rename(click_dir / "moved.txt")
    data = abr_files.load_run(first_file(click_dir))
    assert data.n_reps == {0: 3, 1: 2}


def test_responses_list_levels_and_frequencies(tone_dir):
    responses = abr_files.compute_responses(abr_files.load_run(first_file(tone_dir)))
    assert responses.frequencies == [2000.0, 4000.0]
    assert responses.levels(2000.0) == [30.0, 60.0]


def test_numeric_arrays_skips_what_cannot_be_plotted():
    arrays = abr_files.numeric_arrays(
        {"abr": [1.0, 2.0, 3.0], "map": np.ones((3, 2)), "name": "x", "one": 4.0, "obj": object()}
    )
    assert set(arrays) == {"abr", "map"}


# ---- window


@pytest.fixture
def qapp():
    return pg.mkQApp()


@pytest.fixture
def viewer(qapp):
    window = data_viewer.DataViewer()
    window.errors = []
    window.show_error = window.errors.append
    yield window
    window.close()
    qapp.processEvents()


def open_and_wait(viewer, qapp, path, timeout=30.0):
    viewer.open_path(path)
    deadline = time.time() + timeout
    while viewer.is_loading() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not viewer.is_loading()
    qapp.processEvents()
    assert viewer.errors == []


def test_window_shows_click_recording(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    assert viewer.value("File", "Status") == "6 trials, click"
    assert viewer.value("Average", "Frequency") == "Click"
    assert len(viewer.panels["average"].listDataItems()) == 2
    assert len(viewer.panels["trial"].listDataItems()) == 1
    assert len(viewer.panels["stimulus"].listDataItems()) == 1
    assert viewer.value("Single trial", "Stimulus #") == 0


def test_levels_are_ordered_lowest_at_the_bottom(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    ticks = viewer.panels["average"].getAxis("left")._tickLevels[0]
    assert [label for _, label in ticks] == ["50", "90"]
    assert ticks[0][0] < ticks[1][0]


def test_frequency_choice_changes_the_plot_title(viewer, qapp, tone_dir):
    open_and_wait(viewer, qapp, first_file(tone_dir))
    assert viewer.value("Average", "Frequency") == "2000 Hz"
    viewer.ptreedata.child("Average", "Frequency").setValue("4000 Hz")
    assert "4000 Hz" in viewer.panels["average"].titleLabel.text


def test_changing_filter_recalculates(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    before = viewer.responses.traces[(90.0, 0.0)].copy()
    viewer.ptreedata.child("Average", "Low pass (Hz, 0 = off)").setValue(0.0)
    assert not np.allclose(before, viewer.responses.traces[(90.0, 0.0)])


def test_stored_waveform_is_absent_for_later_files(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    viewer.ptreedata.child("Single trial", "Stimulus #").setValue(1)
    assert len(viewer.panels["stimulus"].listDataItems()) == 0
    assert "not stored" in viewer.panels["stimulus"].titleLabel.text


def test_missing_trial_is_reported_in_the_plot_title(viewer, qapp, click_dir):
    (click_dir / f"{DATE:s}_click_001_002.p").rename(click_dir / "moved.txt")
    open_and_wait(viewer, qapp, first_file(click_dir))
    viewer.ptreedata.child("Single trial", "Stimulus #").setValue(1)
    viewer.ptreedata.child("Single trial", "Repetition").setValue(2)
    assert "no file" in viewer.panels["trial"].titleLabel.text


def test_window_shows_interleaved_recording(viewer, qapp, interleaved_dir):
    open_and_wait(viewer, qapp, first_file(interleaved_dir))
    assert viewer.value("Average", "Frequency") == "1000 Hz"
    viewer.ptreedata.child("Average", "Interleaved set").setValue(1)
    assert viewer.errors == []


def test_window_shows_threshold_summary_and_skips_empty_groups(viewer, qapp, tmp_path):
    path = tmp_path / "all_tone_map_data.pkl"
    nan = float("nan")
    summary = {
        "frequencies": [2000.0, 4000.0, 8000.0],
        "thresholds": {
            "Young": [[40.0, nan, 50.0], [60.0, nan, 70.0]],
            "Empty": [[nan, nan, nan]],
            "None": [],
        },
        "plot_colors": {"Young": "#6495EDFF"},
    }
    with open(path, "wb") as fh:
        pickle.dump(summary, fh)
    open_and_wait(viewer, qapp, path)
    assert viewer.value("File", "Status") == "Threshold summary"
    named = {i.name(): i for i in viewer.panels["thresholds"].listDataItems() if i.name()}
    assert set(named) == {"Young (n=2)"}
    y = named["Young (n=2)"].yData
    assert y[0] == 50.0 and np.isnan(y[1]) and y[2] == 60.0


def test_window_plots_other_pkl_arrays(viewer, qapp, tmp_path):
    path = tmp_path / "visible.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"abr": np.arange(10.0), "AI": np.ones((2, 5)), "label": "x"}, fh)
    open_and_wait(viewer, qapp, path)
    assert viewer.value("File", "Status") == "2 arrays plotted"
    assert len(viewer.panels["AI"].listDataItems()) == 2


def test_corrupt_file_is_reported(viewer, qapp, tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not a pickle")
    viewer.open_path(path)
    assert len(viewer.errors) == 1


def test_unsupported_file_type_is_reported(viewer, tmp_path):
    viewer.open_path(tmp_path / "data.txt")
    assert "cannot be opened" in viewer.errors[0]


# ---- pyabr3 button


def test_pyabr3_opens_the_viewer_and_shows_the_file_dialog(qapp, monkeypatch):
    from src.pyabr3 import PyABR

    calls = []
    monkeypatch.setattr(data_viewer.DataViewer, "open_file_dialog", lambda self: calls.append(1))
    stub = SimpleNamespace(data_viewer=None)
    stub.show_data_viewer = lambda: PyABR.show_data_viewer(stub)
    PyABR.load_file(stub)
    first = stub.data_viewer
    assert isinstance(first, data_viewer.DataViewer) and calls == [1]
    PyABR.load_file(stub)
    assert stub.data_viewer is first
    first.close()


def test_version_comes_from_the_package():
    import src

    assert src.__version__ == "0.4.1"
