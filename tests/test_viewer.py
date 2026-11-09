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
    viewer.open_path(tmp_path / "data.xyz")
    assert "cannot be opened" in viewer.errors[0]


# ---- average with SD or single traces


def set_display(viewer, view):
    viewer.ptreedata.child("Average", "Display").setValue(view)


def wait_for_traces(viewer, qapp, timeout=30.0):
    deadline = time.time() + timeout
    while viewer.is_calculating() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not viewer.is_calculating()
    qapp.processEvents()
    assert viewer.errors == []


def test_mean_of_single_trial_responses_equals_the_average(click_dir):
    data = abr_files.load_run(first_file(click_dir))
    average = abr_files.compute_responses(data, HPF, LPF)
    trials = abr_files.compute_trial_responses(data, HPF, LPF)
    assert trials[(90.0, 0.0)].shape == (3, len(average.t_ms))
    np.testing.assert_allclose(
        np.mean(trials[(90.0, 0.0)], axis=0), average.traces[(90.0, 0.0)], atol=1e-3
    )


def test_trial_responses_use_only_the_selected_interleaved_set(interleaved_dir):
    data = abr_files.load_run(first_file(interleaved_dir))
    trials = abr_files.compute_trial_responses(data, None, None, interleaved_set=1)
    average = abr_files.compute_responses(data, None, None, interleaved_set=1)
    assert trials[(10.0, 1000.0)].shape[0] == 3
    assert len(trials) == 4
    np.testing.assert_allclose(
        np.mean(trials[(10.0, 1000.0)], axis=0), average.traces[(10.0, 1000.0)], atol=1e-3
    )


def test_trial_calculation_can_be_stopped(click_dir):
    data = abr_files.load_run(first_file(click_dir))
    assert abr_files.compute_trial_responses(data, should_stop=lambda: True) is None


def test_trial_sd_uses_n_minus_one_and_handles_one_trial():
    trials = np.array([[1.0, 2.0], [3.0, 6.0]])
    np.testing.assert_allclose(abr_files.trial_sd(trials), [np.sqrt(2.0), 2.0 * np.sqrt(2.0)])
    assert abr_files.trial_sd(trials[:1]).tolist() == [0.0, 0.0]


def test_display_is_a_set_of_radio_buttons_defaulting_to_average(viewer):
    display = viewer.ptreedata.child("Average", "Display")
    assert display.opts["exclusive"] and display.value() == data_viewer.VIEW_AVERAGE
    assert [c.name() for c in display.children()] == [
        data_viewer.VIEW_AVERAGE, data_viewer.VIEW_SD, data_viewer.VIEW_TRACES
    ]  # fmt: skip


def test_sd_view_shades_each_level(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    set_display(viewer, data_viewer.VIEW_SD)
    assert "calculating" in viewer.panels["average"].titleLabel.text
    wait_for_traces(viewer, qapp)
    panel = viewer.panels["average"]
    fills = [i for i in panel.items if isinstance(i, pg.FillBetweenItem)]
    assert len(fills) == 2
    assert len(panel.listDataItems()) == 6  # upper, lower and mean at two levels
    assert "3 trials" in panel.titleLabel.text


def test_single_trace_view_draws_faint_traces_behind_the_mean(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    set_display(viewer, data_viewer.VIEW_TRACES)
    wait_for_traces(viewer, qapp)
    items = viewer.panels["average"].listDataItems()
    assert len(items) == 4  # traces then mean, for two levels
    assert items[0].opts["pen"].color().alpha() == data_viewer.SINGLE_TRACE_ALPHA
    assert items[1].opts["pen"].width() == data_viewer.MEAN_LINE_WIDTH


def test_returning_to_average_shows_only_the_mean(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    set_display(viewer, data_viewer.VIEW_SD)
    wait_for_traces(viewer, qapp)
    set_display(viewer, data_viewer.VIEW_AVERAGE)
    assert len(viewer.panels["average"].listDataItems()) == 2


def test_changing_the_filter_recalculates_the_single_traces(viewer, qapp, click_dir):
    open_and_wait(viewer, qapp, first_file(click_dir))
    set_display(viewer, data_viewer.VIEW_SD)
    wait_for_traces(viewer, qapp)
    before = viewer.trial_responses[(90.0, 0.0)].copy()
    viewer.ptreedata.child("Average", "Low pass (Hz, 0 = off)").setValue(1000.0)
    wait_for_traces(viewer, qapp)
    assert viewer.trial_request[2] == 1000.0
    assert not np.allclose(before, viewer.trial_responses[(90.0, 0.0)])


def test_many_trials_are_limited_when_drawn_and_said_so(viewer):
    trials = np.zeros((512, 10))
    assert viewer.shown_trials(trials).shape == (data_viewer.MAX_SINGLE_TRACES, 10)
    assert viewer.shown_trials(np.zeros((20, 10))).shape == (20, 10)


def test_matlab_run_says_single_traces_are_not_available(viewer, matlab_dir):
    set_display(viewer, data_viewer.VIEW_SD)
    viewer.open_path(matlab_dir / f"{RUN:s}-p.txt")
    panel = viewer.panels["average"]
    assert "not stored" in panel.titleLabel.text
    assert len(panel.listDataItems()) == 3
    assert viewer.calculator is None


# ---- MATLAB text files

RUN = "20170419-0849"
TONE_RUN = "20170419-0901"
MATLAB_LEVELS = [20.0, 40.0, 60.0]
FS_MATLAB = 100000.0
N_MATLAB = 2500


def matlab_trace(amplitude):
    t = np.arange(N_MATLAB) / FS_MATLAB
    return amplitude * 1e-6 * np.sin(2 * np.pi * 1000.0 * t)


def write_columns(path, columns, ending="\r\n"):
    rows = np.column_stack(columns)
    lines = ["\t".join(f"  {v:.7e}" for v in row) + "\t" for row in rows]
    path.write_bytes((ending.join(lines) + ending).encode("ascii"))


def write_numbers(path, numbers):
    path.write_bytes("".join(f"  {v:.7e}\t\r\n" for v in numbers).encode("ascii"))


@pytest.fixture
def matlab_dir(tmp_path):
    """A click run, whose columns run from high to low level, and a tone run."""
    write_numbers(tmp_path / f"{RUN:s}-SPL.txt", MATLAB_LEVELS)
    high_to_low = [matlab_trace(a) for a in [3.0, 2.0, 1.0]]
    write_columns(tmp_path / f"{RUN:s}-p.txt", high_to_low)
    write_columns(tmp_path / f"{RUN:s}-n.txt", [2.0 * c for c in high_to_low], ending="\r")

    write_numbers(tmp_path / f"{TONE_RUN:s}-SPL.txt", MATLAB_LEVELS)
    write_numbers(tmp_path / f"{TONE_RUN:s}-kHz.txt", [8000.0, 48000.0])
    low_to_high = [matlab_trace(a) for a in [1.0, 2.0, 3.0]]
    write_columns(tmp_path / f"{TONE_RUN:s}-p-8000.000.txt", low_to_high)
    write_columns(tmp_path / f"{TONE_RUN:s}-n-8000.000.txt", low_to_high)
    not_reached = low_to_high[:2] + [np.full(N_MATLAB, np.nan)]
    write_columns(tmp_path / f"{TONE_RUN:s}-p-48000.000.txt", not_reached)
    write_columns(tmp_path / f"{TONE_RUN:s}-n-48000.000.txt", not_reached)
    return tmp_path


def test_matlab_run_name_comes_from_any_file_of_the_run():
    assert abr_files.matlab_run_name(Path(f"{RUN:s}-p.txt")) == RUN
    assert abr_files.matlab_run_name(Path(f"{TONE_RUN:s}-n-5600.000.txt")) == TONE_RUN
    with pytest.raises(ValueError):
        abr_files.matlab_run_name(Path("notes.txt"))


def test_matlab_runs_in_a_folder_are_listed_by_type(matlab_dir):
    assert abr_files.list_matlab_runs(matlab_dir) == {RUN: "click", TONE_RUN: "tone"}


def test_click_columns_are_reversed_and_polarities_averaged(matlab_dir):
    run = abr_files.read_matlab_run(matlab_dir, RUN)
    assert run.stim == "click" and run.levels == MATLAB_LEVELS
    responses = abr_files.compute_matlab_responses(run)
    # (p + n) / 2 = 1.5 * the amplitude in the p file; 20 dB is the last column
    np.testing.assert_allclose(responses.traces[(20.0, 0.0)], 1.5 * matlab_trace(1.0) * 1e6, atol=1e-5)
    np.testing.assert_allclose(responses.traces[(60.0, 0.0)], 1.5 * matlab_trace(3.0) * 1e6, atol=1e-5)


def test_tone_columns_keep_the_order_of_the_levels(matlab_dir):
    responses = abr_files.compute_matlab_responses(abr_files.read_matlab_run(matlab_dir, TONE_RUN))
    np.testing.assert_allclose(responses.traces[(60.0, 8000.0)], matlab_trace(3.0) * 1e6, atol=1e-5)
    assert responses.frequencies == [8000.0, 48000.0]


def test_level_that_was_not_recorded_is_left_out(matlab_dir):
    responses = abr_files.compute_matlab_responses(abr_files.read_matlab_run(matlab_dir, TONE_RUN))
    assert responses.levels(48000.0) == [20.0, 40.0]
    assert responses.levels(8000.0) == MATLAB_LEVELS


def test_matlab_time_base_at_100_khz_is_25_ms(matlab_dir):
    responses = abr_files.compute_matlab_responses(abr_files.read_matlab_run(matlab_dir, RUN))
    assert len(responses.t_ms) == N_MATLAB
    assert responses.t_ms[1] == pytest.approx(0.01)


def test_matlab_time_base_is_limited_to_25_ms_at_a_lower_rate(matlab_dir):
    responses = abr_files.compute_matlab_responses(abr_files.read_matlab_run(matlab_dir, RUN), 50000.0)
    assert len(responses.t_ms) == 1250
    assert responses.t_ms[-1] < 25.0


def test_matlab_filter_matches_abrfuncs_filter(matlab_dir):
    run = abr_files.read_matlab_run(matlab_dir, RUN)
    responses = abr_files.compute_matlab_responses(run, hpf=500.0, lpf=2500.0)
    trace = run.waves[0.0][:, 0]
    nyq = FS_MATLAB / 2.0
    b, a = abr_files.scipy.signal.iirfilter(4, (500.0 / nyq, 2500.0 / nyq), btype="band", ftype="butter")
    expected = abr_files.scipy.signal.filtfilt(b, a, trace, padlen=int(len(trace) / 10))
    np.testing.assert_allclose(responses.traces[(20.0, 0.0)], expected * 1e6, atol=1e-9)


def test_missing_point_at_the_end_of_a_trace_is_filled(tmp_path):
    write_numbers(tmp_path / f"{RUN:s}-SPL.txt", [20.0])
    trace = matlab_trace(1.0)
    trace[-1] = np.nan
    write_columns(tmp_path / f"{RUN:s}-p.txt", [trace])
    write_columns(tmp_path / f"{RUN:s}-n.txt", [trace])
    responses = abr_files.compute_matlab_responses(
        abr_files.read_matlab_run(tmp_path, RUN), hpf=500.0, lpf=2500.0
    )
    assert np.isfinite(responses.traces[(20.0, 0.0)]).all()


def test_run_with_a_missing_polarity_is_reported(matlab_dir):
    (matlab_dir / f"{RUN:s}-n.txt").rename(matlab_dir / "moved.dat")
    with pytest.raises(ValueError, match="No complete pair"):
        abr_files.read_matlab_run(matlab_dir, RUN)


def test_window_opens_matlab_click_run_from_any_of_its_files(viewer, matlab_dir):
    viewer.open_path(matlab_dir / f"{RUN:s}-n.txt")
    assert viewer.errors == []
    assert viewer.value("File", "Status") == "MATLAB click run, 1 frequencies"
    assert viewer.value("Average", "Frequency") == "Click"
    assert viewer.value("Average", "High pass (Hz, 0 = off)") == 500.0
    assert viewer.value("Average", "Low pass (Hz, 0 = off)") == 2500.0
    assert len(viewer.panels["average"].listDataItems()) == 3


def test_window_switches_between_runs_in_the_folder(viewer, matlab_dir):
    viewer.open_path(matlab_dir / f"{RUN:s}-SPL.txt")
    viewer.ptreedata.child("MATLAB run", "Run").setValue(f"{TONE_RUN:s} (tone)")
    assert viewer.errors == []
    assert viewer.value("File", "Name") == TONE_RUN
    assert viewer.value("Average", "Frequency") == "8000 Hz"
    viewer.ptreedata.child("Average", "Frequency").setValue("48000 Hz")
    assert len(viewer.panels["average"].listDataItems()) == 2


def test_sample_rate_setting_changes_the_time_axis(viewer, matlab_dir):
    viewer.open_path(matlab_dir / f"{RUN:s}-p.txt")
    assert viewer.responses.t_ms[1] == pytest.approx(0.01)
    viewer.ptreedata.child("MATLAB run", "Sample rate (Hz)").setValue(50000.0)
    assert viewer.responses.t_ms[1] == pytest.approx(0.02)


def test_invert_flips_the_plotted_traces(viewer, matlab_dir):
    viewer.open_path(matlab_dir / f"{RUN:s}-p.txt")
    before = viewer.panels["average"].listDataItems()[0].yData.copy()
    viewer.ptreedata.child("Average", "Invert").setValue(True)
    after = viewer.panels["average"].listDataItems()[0].yData
    np.testing.assert_allclose(after, -before, atol=1e-9)


def test_filter_settings_change_only_when_the_format_changes(viewer, qapp, matlab_dir, click_dir):
    viewer.open_path(matlab_dir / f"{RUN:s}-p.txt")
    viewer.ptreedata.child("Average", "High pass (Hz, 0 = off)").setValue(700.0)
    viewer.open_path(matlab_dir / f"{TONE_RUN:s}-SPL.txt")
    assert viewer.value("Average", "High pass (Hz, 0 = off)") == 700.0
    open_and_wait(viewer, qapp, first_file(click_dir))
    assert viewer.value("Average", "High pass (Hz, 0 = off)") == 300.0
    assert viewer.matlab_run is None


def test_text_file_that_is_not_from_matlab_is_reported(viewer, tmp_path):
    viewer.open_path(tmp_path / "notes.txt")
    assert "Not the name" in viewer.errors[0]


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
