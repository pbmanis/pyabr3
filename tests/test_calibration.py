"""
Tests for src/calibration, using mock hardware.

calc.bandpass is an equivalent of the MATLAB bandpass function, so these tests check
its behavior, not agreement with MATLAB. That requires data from the running system.

Existing .cal files in calfiles/ are read, and never written.

Run from the top directory of the repository. --confcutdir is needed because the
__init__.py in the directory above this repository cannot be imported:
    python -m pytest --confcutdir=tests tests/test_calibration.py
"""

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import pyqtgraph as pg
import scipy.io

from src import read_calibration
from src.calibration import cal_chooser, cal_display, calc, calibration_gui, engine
from src.calibration.hardware import (
    CalibrationError,
    CalibrationStopped,
    MockCalibrationHardware,
    PyStimCalibrationHardware,
)

FS = 195312.5
REAL_FREQUENCY_CAL = REPO / "calfiles" / "frequency_MF1.cal"
REAL_MICROPHONE_CAL = REPO / "calfiles" / "microphone_7016#10252.cal"
REAL_CHECK_CAL = REPO / "calfiles" / "chk75db_MF1.cal"
MIC = "7016#10252"
FREQS = np.array([64000.0, 32000.0, 16000.0, 8000.0, 4000.0])


def field_names(path, name):
    return list(scipy.io.loadmat(path, appendmat=False)[name].dtype.names)


def run_mic(hw, tmp_path, ref_db=94.0, **kw):
    hw.calibrator_db = ref_db
    settings = engine.MicSettings(microphone=MIC, ref_db=ref_db, cal_dir=tmp_path, **kw)
    result = engine.calibrate_microphone(hw, settings, threading.Event(), log=lambda m: None)
    hw.calibrator_db = None
    return result


def speaker_settings(tmp_path, **kw):
    args = dict(speaker="TestSpk", microphone=MIC, freqs=FREQS, attn=30.0, ref_spl=110.0,
                cal_dir=tmp_path)  # fmt: skip
    return engine.SpeakerSettings(**(args | kw))


def run_speaker(hw, tmp_path, stop=None, **kw):
    return engine.calibrate_speaker(
        hw, speaker_settings(tmp_path, **kw), stop or threading.Event(), log=lambda m: None
    )


@pytest.fixture
def hw():
    return MockCalibrationHardware()


@pytest.fixture
def calibrated_mic(hw, tmp_path):
    run_mic(hw, tmp_path)
    return hw


# ---- calculations


def test_tonepip_shape_matches_tonepip_m():
    w, uclock = calc.tonepip(10.0, 1000.0, 0.0, 200.0, 5.0, 0, 500000.0, 10, 1, 0)
    assert len(w) == 100000
    assert uclock == pytest.approx(2.0)
    assert w[0] == 0.0
    assert np.max(np.abs(w)) == pytest.approx(10.0, abs=1e-3)
    # sign is -1 in tonepip.m: first half cycle after the start is negative
    assert w[100] < 0.0


def test_tonepip_without_ramp():
    w, _ = calc.tonepip(10.0, 5000.0, 0.0, 10, 0.0, 0, 500000.0, 10, 1, 0)
    assert len(w) == 5000


def test_octave_calc():
    low, high = calc.octave_calc(8000.0, 8, FS)
    assert low == pytest.approx(8000.0 / 2 ** (1 / 8))
    assert high == pytest.approx(8000.0 * 2 ** (1 / 8))
    assert calc.octave_calc(87000.0, 5, FS)[1] == np.floor(FS / 2.2)


def test_spl_at_f_accepts_descending_frequencies():
    assert calc.spl_at_f([8000.0, 2000.0], [40.0, 20.0], 4000.0) == pytest.approx(30.0)
    assert np.isnan(calc.spl_at_f([8000.0, 2000.0], [40.0, 20.0], 16000.0))


def test_cosinor_recovers_amplitude_and_mesor():
    t = np.arange(5000) / FS
    y = 0.5 + 2.0 * np.cos(2 * np.pi * 3000.0 * t + 0.7)
    amp, mesor = calc.cosinor(t, y, 2 * np.pi * 3000.0)
    assert amp == pytest.approx(2.0, abs=1e-9)
    assert mesor == pytest.approx(0.5, abs=1e-9)


def test_bandpass_passes_band_and_compensates_delay():
    t = np.arange(39062) / FS
    x = np.sin(2 * np.pi * 8000.0 * t)
    y = calc.bandpass(x, calc.octave_calc(8000.0, 8, FS), FS, 60.0, 0.85)
    mid = slice(10000, 30000)
    assert calc.rms(y[mid]) == pytest.approx(calc.rms(x[mid]), rel=1e-3)
    assert np.max(np.abs(y[mid] - x[mid])) < 2e-3


def test_bandpass_rejects_stopband():
    t = np.arange(39062) / FS
    x = np.sin(2 * np.pi * 2000.0 * t)
    y = calc.bandpass(x, calc.octave_calc(8000.0, 8, FS), FS, 60.0, 0.85)
    assert calc.rms(y[10000:30000]) < 2e-3 * calc.rms(x)


@pytest.mark.parametrize("low,high,f0", [(39000.0, 41000.0, 40000.0), (19500.0, 20500.0, 20000.0)])
def test_notches_remove_power_supply_noise_and_pass_32k(low, high, f0):
    t = np.arange(39062) / FS
    sos = calc.design_notch(FS, low, high)
    mid = slice(10000, 30000)
    assert calc.rms(calc.apply_notch(sos, np.sin(2 * np.pi * f0 * t))[mid]) < 1e-3
    assert calc.rms(calc.apply_notch(sos, np.sin(2 * np.pi * 32000.0 * t))[mid]) == pytest.approx(
        np.sqrt(0.5), rel=1e-3
    )


def test_compute_spl_reproduces_values_in_matlab_files():
    """Same-day MATLAB calibration: the formula must return the stored SPL exactly."""
    cal = engine.load_mat_struct(REAL_FREQUENCY_CAL, "CAL")
    mic = engine.load_mat_struct(REAL_MICROPHONE_CAL, "MIC")
    assert cal["Date"] == mic["Date"]
    np.testing.assert_allclose(calc.compute_spl(cal["Vmeas"], mic), cal["dBSPL"], atol=1e-9)
    np.testing.assert_allclose(calc.compute_spl(cal["Vmeas_bp"], mic), cal["dBSPL_bp"], atol=1e-9)


def test_mic_sensitivity_reproduces_values_in_matlab_file():
    mic = engine.load_mat_struct(REAL_MICROPHONE_CAL, "MIC")
    assert calc.compute_dBPerVPa(mic["Vref_bp"], mic) == pytest.approx(mic["dBPerVPa"], abs=1e-9)
    assert calc.compute_mVPerPa(mic) == pytest.approx(mic["mVPerPa"], abs=1e-9)


def test_stored_maxdb_is_bandpass_plus_attenuation():
    cal = engine.load_mat_struct(REAL_FREQUENCY_CAL, "CAL")
    np.testing.assert_allclose(cal["maxdB"], cal["dBSPL_bp"] + cal["CalAttn"], atol=1e-9)


# ---- microphone calibration


def test_microphone_recovers_sensitivity(hw, tmp_path):
    mic = run_mic(hw, tmp_path)
    # the 94 dB calibrator is treated as 1 Pa, but is actually 1.0024 Pa
    assert mic["mVPerPa"] == pytest.approx(3.85 * 1.00237, rel=2e-3)
    assert mic["Vref_c"] == pytest.approx(mic["Vref_bp"], rel=5e-3)


def test_microphone_file_format_matches_matlab(hw, tmp_path):
    run_mic(hw, tmp_path)
    short = tmp_path / f"microphone_{MIC:s}.cal"
    assert field_names(short, "MIC") == field_names(REAL_MICROPHONE_CAL, "MIC")
    data = read_calibration.get_microphone_calibration(short)
    assert data["microphone"] == MIC
    assert data["cal_gain"] == 20.0
    dated = list(tmp_path.glob(f"calibration_history/microphone_{MIC:s}_*.cal"))
    assert len(dated) == 1


def test_microphone_test_mode_uses_synthetic_signal_and_does_not_save(hw, tmp_path):
    mic = run_mic(hw, tmp_path, test_mode=True)
    assert mic["Vrms"] == pytest.approx(0.05 / np.sqrt(2), rel=1e-3)
    assert list(tmp_path.iterdir()) == []


def test_speaker_is_at_maximum_attenuation_during_microphone_calibration(hw, tmp_path):
    run_mic(hw, tmp_path)
    assert set(hw.attn_history) == {120.0}


def test_104_db_calibrator_offset_is_preserved_from_matlab(hw, tmp_path):
    """Records a problem in soundfuncs.compute_spl, deliberately not changed in the port.

    RefSig is used as the reference, but mVPerPa is already referred to 94 dB, so
    SPL is reported 10 dB high after a calibration with the 104 dB setting.
    """
    mic = run_mic(hw, tmp_path, ref_db=104.0)
    assert mic["mVPerPa"] == pytest.approx(3.85 * 1.00237, rel=2e-3)
    result = run_speaker(hw, tmp_path, freqs=np.array([16000.0, 8000.0]))
    assert result["dBSPL"].ravel() == pytest.approx([85.0, 85.0], abs=0.2)  # true level is 75


# ---- speaker calibration


def test_speaker_recovers_known_levels(calibrated_mic, tmp_path):
    result = run_speaker(calibrated_mic, tmp_path)
    expected = calibrated_mic.default_speaker(FREQS) - 30.0
    np.testing.assert_allclose(result["dBSPL"].ravel(), expected, atol=0.15)
    np.testing.assert_allclose(result["dBSPL_bp"].ravel(), expected, atol=0.15)
    assert np.all(result["dBSPL_nf"].ravel() < expected - 15.0)


def test_maxdb_uses_bandpass_plus_attenuation(calibrated_mic, tmp_path):
    result = run_speaker(calibrated_mic, tmp_path)
    np.testing.assert_allclose(result["maxdB"], result["dBSPL_bp"] + 30.0, atol=1e-12)


def test_speaker_alternates_stimulus_and_noise_floor_attenuation(hw, tmp_path):
    run_mic(hw, tmp_path)
    hw.attn_history.clear()
    run_speaker(hw, tmp_path, freqs=np.array([8000.0, 4000.0]))
    # calstim leaves the attenuator alone in the mock, so this is the engine's sequence
    assert hw.attn_history == [30.0, 120.0, 30.0, 120.0, 120.0]


def test_speaker_file_format_readable_by_pyabr3(calibrated_mic, tmp_path):
    run_speaker(calibrated_mic, tmp_path)
    short = tmp_path / "frequency_TestSpk.cal"
    assert field_names(short, "CAL") == field_names(REAL_FREQUENCY_CAL, "CAL")
    assert len(list(tmp_path.glob("calibration_history/frequency_TestSpk_*.cal"))) == 1
    data = read_calibration.get_calibration_data(short)
    assert data["spkr"] == "TestSpk"
    assert data["mic"] == MIC
    assert data["calattn"] == 30.0
    assert data["refspl"] == 110.0
    assert data["freqs"].tolist() == FREQS.tolist()
    assert data["maxdb"].shape == FREQS.shape


def test_check_mode_produces_75_db_and_matlab_fields(calibrated_mic, tmp_path):
    run_speaker(calibrated_mic, tmp_path)
    result = run_speaker(calibrated_mic, tmp_path, check=True)
    # levels above what the speaker can produce stay at the 0 dB attenuation level
    expected = np.minimum(75.0, calibrated_mic.default_speaker(FREQS))
    np.testing.assert_allclose(result["dBSPL_bp"].ravel(), expected, atol=0.3)
    assert result["dBSPL"] == 75.0  # the target, as in MATLAB
    path = tmp_path / "chk75db_TestSpk.cal"
    assert len(list(tmp_path.glob("calibration_history/chk75db_TestSpk_*.cal"))) == 1
    assert field_names(path, "CHKCAL") == field_names(REAL_CHECK_CAL, "CHKCAL")


def test_check_mode_clips_attenuation_at_zero(calibrated_mic, tmp_path):
    run_speaker(calibrated_mic, tmp_path)
    result = run_speaker(calibrated_mic, tmp_path, check=True)
    # the mock produces 71.2 dB at 64 kHz with 0 dB attenuation, below the 75 dB target
    assert result["attndB"].ravel()[0] == 0.0
    assert calibrated_mic.attn == 120.0


def test_stop_saves_nothing_and_leaves_maximum_attenuation(calibrated_mic, tmp_path):
    calibrated_mic.calstim_count = 0
    calibrated_mic.stop_after_calls = 3
    assert run_speaker(calibrated_mic, tmp_path) is None
    assert calibrated_mic.attn == 120.0
    assert list(tmp_path.glob("frequency_*")) == []


def test_missing_microphone_file_is_reported(hw, tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        run_speaker(hw, tmp_path)


def test_preamp_gain_must_match_microphone_calibration(calibrated_mic, tmp_path):
    with pytest.raises(CalibrationError, match="preamp gain"):
        run_speaker(calibrated_mic, tmp_path, gain=40.0)


def test_out_of_range_attenuation_goes_to_maximum(hw):
    for value in [-1.0, 121.0, float("nan")]:
        hw.set_attn(value)
        assert hw.attn == 120.0


def test_verify_message_names_manual_settings():
    text = engine.verify_message("microphone104", "MF1", MIC, 20.0, -6.0, None)
    assert "104 dBSPL" in text and "Preamp Gain is 20 dB" in text and "attenuation is -6 dB" in text
    text = engine.verify_message("calibrate", "MF1", MIC, 20.0, -6.0, 20.0)
    assert "Programmable attenuator will be set to 20.0 dB" in text


# ---- pyabr3 hardware adapter, with fake devices


class FakeRP:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            return {"GetSFreq": FS, "GetStatus": 7, "ClearCOF": 1, "LoadCOFsf": 1,
                    "SetTagVal": 1}.get(name, 0)  # fmt: skip

        return call

    def ReadTagV(self, tag, start, n):
        self.calls.append(("ReadTagV", tag, start, n))
        return [0.0] * n


class FakeTask:
    log: list = []

    def __init__(self, *a):
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=lambda *a, **k: None)
        self.timing = SimpleNamespace(cfg_samp_clk_timing=lambda **k: FakeTask.log.append(k))
        self.triggers = SimpleNamespace(
            start_trigger=SimpleNamespace(
                cfg_dig_edge_start_trig=lambda **k: FakeTask.log.append(k)
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, w):
        FakeTask.log.append(("write", len(w)))

    def start(self):
        FakeTask.rp.calls.append(("task_start",))

    def stop(self):
        FakeTask.log.append("stop")


def fake_adapter():
    FakeTask.log = []
    hw = PyStimCalibrationHardware(config={})
    hw.rp = FakeRP()
    FakeTask.rp = hw.rp
    attns = []
    hw.PS = SimpleNamespace(
        State=SimpleNamespace(NI_devicename="Dev1"),
        clip=lambda w, v: (w, 0),
        setAttens=lambda atten_left, atten_right: attns.append(atten_left),
    )
    hw.pystim3 = SimpleNamespace(
        nidaqmx=SimpleNamespace(task=SimpleNamespace(Task=FakeTask)),
        VoltageUnits=SimpleNamespace(VOLTS=1),
        AcquisitionType=SimpleNamespace(FINITE=1),
        Edge=SimpleNamespace(RISING=1),
    )
    return hw, attns


def test_adapter_records_in_the_order_used_by_calstim():
    hw, attns = fake_adapter()
    n_points = 2000
    data = hw.calstim(np.zeros(1000), n_points, threading.Event())
    names = [c[0] for c in hw.rp.calls]
    assert len(data) == n_points
    assert names.index("ClearCOF") < names.index("LoadCOFsf") < names.index("Run")
    assert ("SoftTrg", 1) in hw.rp.calls
    assert ("ReadTagV", "data_out2", 0, n_points) in hw.rp.calls
    assert ("SetTagVal", "REC_Size", n_points) in hw.rp.calls
    assert names.index("ReadTagV") < names.index("Halt")  # as in calstim.m: read, then halt
    assert hw.rp.calls.index(("SoftTrg", 2)) < names.index("ReadTagV")
    assert attns == [120.0]
    # the NI task must be armed before the RP2.1 is triggered
    assert hw.rp.calls.index(("task_start",)) < hw.rp.calls.index(("SoftTrg", 1))
    assert {"trigger_source": "/Dev1/PFI0", "trigger_edge": 1} in FakeTask.log


def test_adapter_stop_halts_and_attenuates():
    hw, attns = fake_adapter()
    stop = threading.Event()
    stop.set()
    with pytest.raises(CalibrationStopped):
        hw.calstim(np.zeros(1000), 2000, stop)
    assert ("SoftTrg", 0) in hw.rp.calls
    assert ("Halt",) in hw.rp.calls
    assert attns == [120.0]


def test_adapter_refuses_to_connect_without_lab_hardware():
    if sys.platform == "win32":
        pytest.skip("only without the lab hardware")
    with pytest.raises(CalibrationError, match="mock"):
        PyStimCalibrationHardware(config={}).connect()


# ---- GUI


@pytest.fixture
def qapp():
    return pg.mkQApp()


@pytest.fixture
def window(qapp, tmp_path):
    win = calibration_gui.CalibrationWindow(MockCalibrationHardware(), {"calfile": "x/y.cal"})
    win.ptreedata.child("Setup", "Cal directory").setValue(str(tmp_path))
    win.ptreedata.child("Setup", "Frequencies (kHz)").setValue("16, 8, 4")
    win.prompts = []
    win.errors = []
    win.confirm_settings = lambda message: win.prompts.append(message) or True
    win.show_error = lambda message: win.errors.append(message)
    yield win
    win.close()
    qapp.processEvents()


def press(window, name):
    window.ptreedata.child("Actions", name).activate()


def wait_until_idle(window, qapp, timeout=60.0):
    deadline = time.time() + timeout
    while window.is_running() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not window.is_running()
    qapp.processEvents()


def test_window_runs_microphone_then_speaker_and_plots(window, qapp, tmp_path):
    press(window, "Microphone (94 dB)")
    wait_until_idle(window, qapp)
    assert window.errors == []
    assert (tmp_path / f"microphone_{MIC:s}.cal").is_file()
    assert "Calibrator Standard is set to 94" in window.prompts[0]
    assert len(window.panels["cosinor"].listDataItems()) == 1

    press(window, "Calibrate Speaker")
    wait_until_idle(window, qapp)
    assert window.errors == []
    assert (tmp_path / "frequency_MF1.cal").is_file()
    assert "Speaker driver attenuation" in window.prompts[1]
    assert window.value("Status", "State") == "Done"
    assert window.value("Status", "Progress") == "3 / 3"
    names = [i.name() for i in window.panels["spl"].listDataItems()]
    assert {"Noise floor", "Bandpass"} <= set(names)
    assert window.hw.attn == 120.0


def test_declining_the_prompt_does_not_run(window, qapp, tmp_path):
    window.confirm_settings = lambda message: False
    press(window, "Microphone (94 dB)")
    qapp.processEvents()
    assert window.worker is None
    assert list(tmp_path.iterdir()) == []


def test_speaker_run_without_microphone_file_reports_error(window, qapp):
    press(window, "Calibrate Speaker")
    wait_until_idle(window, qapp)
    qapp.processEvents()
    assert len(window.errors) == 1
    assert "not found" in window.errors[0]
    assert window.value("Status", "State") == "Error"


def test_stop_button_ends_run_without_saving(window, qapp, tmp_path):
    press(window, "Microphone (94 dB)")
    wait_until_idle(window, qapp)
    window.hw.calstim_count = 0
    window.hw.stop_after_calls = 2
    press(window, "Calibrate Speaker")
    wait_until_idle(window, qapp)
    assert window.value("Status", "State") == "Stopped"
    assert list(tmp_path.glob("frequency_*")) == []


def test_changing_speaker_applies_its_preset(window):
    window.ptreedata.child("Setup", "Speaker").setValue("ES1")
    assert window.value("Programmable attenuator", "Attn (dB)") == 0.0
    assert window.value("Setup", "Ref SPL (maxtones)") == 83.9
    assert window.value("Setup", "Frequencies (kHz)").startswith("1, 2, 3")


def test_all_speakers_kept_and_none_measure_at_the_notch_frequencies():
    assert {"MF1", "EC1", "ES1", "EC1#4101 closed field", "ES1#3539 free field",
            "MF1#1013 free field", "MF1#1956 closed field",
            "FF1#1013 free field (dome)"} == set(engine.SPEAKERS)  # fmt: skip
    for preset in engine.SPEAKERS.values():
        khz = np.array(preset["freqs_khz"])
        assert not np.any((khz >= 19.5) & (khz <= 20.5))
        assert not np.any((khz >= 39.0) & (khz <= 41.0))
    assert engine.SPEAKERS["MF1#1013 free field"]["attn"] == 20.0


def test_frequency_inside_a_notch_gives_a_warning(calibrated_mic, tmp_path):
    messages = []
    engine.calibrate_speaker(
        calibrated_mic,
        speaker_settings(tmp_path, freqs=np.array([20000.0, 8000.0])),
        threading.Event(),
        log=messages.append,
    )
    assert any("20000 Hz is inside" in m for m in messages)


def test_second_calibration_keeps_the_earlier_history_file(calibrated_mic, tmp_path, monkeypatch):
    run_speaker(calibrated_mic, tmp_path)
    first = list(tmp_path.glob("calibration_history/frequency_TestSpk_*.cal"))
    real_datetime = engine.datetime.datetime

    class Later(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz) + engine.datetime.timedelta(hours=1)

    monkeypatch.setattr(engine.datetime, "datetime", Later)
    run_speaker(calibrated_mic, tmp_path)
    second = list(tmp_path.glob("calibration_history/frequency_TestSpk_*.cal"))
    assert len(first) == 1 and len(second) == 2


def test_noise_floor_writes_csv_as_matlab(hw, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "NOISE_RECORD_DURATION", 0.2)
    results = []
    path = engine.measure_noise_floor(hw, tmp_path, threading.Event(), lambda m: None, results.append)
    assert path.name.startswith("Noise_Measure_") and path.suffix == ".csv"
    lines = path.read_text().splitlines()
    n_points = int(np.floor(FS * 0.2 * 1.5))
    assert lines[0] == "Time,Voltage(V)"
    assert len(lines) == n_points + 1
    assert hw.attn_history[0] == 120.0
    assert len(results[0]["ch2nf"]) == n_points


def test_window_measures_noise_floor(window, qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "NOISE_RECORD_DURATION", 0.2)
    press(window, "Noise Floor")
    wait_until_idle(window, qapp)
    assert window.errors == []
    assert len(list(tmp_path.glob("Noise_Measure_*.csv"))) == 1
    assert len(window.panels["spectrum"].listDataItems()) == 1


def test_bad_frequency_lists_are_rejected():
    with pytest.raises(CalibrationError):
        calibration_gui.parse_frequencies("4, abc", 97656.0)
    with pytest.raises(CalibrationError):
        calibration_gui.parse_frequencies("4, 120", 97656.0)
    assert calibration_gui.parse_frequencies("[1, 2.5 4]", 97656.0).tolist() == [1000, 2500, 4000]


def test_mock_window_never_defaults_to_the_lab_calibration_directory(qapp):
    win = calibration_gui.CalibrationWindow(
        MockCalibrationHardware(), {"calfile": "e:\\lab\\frequency_MF1.cal"}
    )
    assert "pyabr3_mock_calibration" in win.value("Setup", "Cal directory")
    win.close()


# ---- calibration file chooser


def write_header(path, created):
    header = f"MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: {created:s}".ljust(116)
    path.write_bytes(header.encode("ascii"))


def make_cal_files(tmp_path):
    """File system times run opposite to the header times, as after files are moved."""
    history = tmp_path / "calibration_history"
    history.mkdir()
    times = {
        history / "frequency_MF1_01-Jan-2026_09-00-00.cal": "Thu Jan  1 09:00:00 2026",
        history / "frequency_MF1_02-Jan-2026_09-00-00.cal": "Fri Jan  2 09:00:00 2026",
        tmp_path / "frequency_MF1.cal": "Sat Jan  3 10:30:15 2026",
    }
    for i, (path, created) in enumerate(times.items()):
        write_header(path, created)
        when = 3000 - 1000 * i
        os.utime(path, (when, when))
    (tmp_path / "microphone_x.cal").write_bytes(b"")
    return list(times)


def test_chooser_lists_current_and_history_files_newest_by_internal_time(tmp_path):
    files = make_cal_files(tmp_path)
    found = cal_chooser.list_calibration_files(tmp_path)
    assert [f[0] for f in found] == [files[2], files[1], files[0]]


def test_created_time_reads_matlab_file_header():
    assert cal_chooser.created_time(REAL_FREQUENCY_CAL) == cal_chooser.datetime.datetime(
        2024, 12, 9, 11, 0, 20
    )


def test_created_time_reads_files_written_by_this_program(tmp_path):
    path = tmp_path / "frequency_x.cal"
    engine.save_mat_struct(path, "CAL", {"Date": "x"})
    assert cal_chooser.created_time(path).date() == cal_chooser.datetime.date.today()


def test_file_without_a_readable_time_is_listed_last_as_unknown(qapp, tmp_path):
    files = make_cal_files(tmp_path)
    bad = tmp_path / "frequency_bad.cal"
    bad.write_bytes(b"not a mat file")
    dialog = cal_chooser.CalibrationFileChooser(tmp_path)
    last = dialog.table.rowCount() - 1
    assert dialog.table.item(last, 0).text() == "frequency_bad.cal"
    assert dialog.table.item(last, 1).text() == cal_chooser.UNKNOWN_TIME
    assert len(files) + 1 == dialog.table.rowCount()


def test_chooser_shows_names_and_times_and_selects_current(qapp, tmp_path):
    files = make_cal_files(tmp_path)
    dialog = cal_chooser.CalibrationFileChooser(tmp_path, current=files[0])
    assert dialog.table.rowCount() == 3
    assert dialog.table.item(0, 0).text() == "frequency_MF1.cal"
    assert dialog.table.item(2, 0).text().startswith("calibration_history/")
    assert dialog.table.item(0, 1).text() == "2026-01-03 10:30:15"
    assert dialog.selected_path() == files[0]


def test_chooser_defaults_to_newest_when_current_is_not_listed(qapp, tmp_path):
    files = make_cal_files(tmp_path)
    dialog = cal_chooser.CalibrationFileChooser(tmp_path, current=tmp_path / "other.cal")
    assert dialog.selected_path() == files[2]


def test_chooser_can_switch_to_another_folder(qapp, tmp_path, monkeypatch):
    first = tmp_path / "a"
    other = tmp_path / "b"
    first.mkdir()
    other.mkdir()
    write_header(first / "frequency_A.cal", "Thu Jan  1 09:00:00 2026")
    write_header(other / "frequency_B.cal", "Fri Jan  2 09:00:00 2026")
    write_header(other / "frequency_C.cal", "Sat Jan  3 09:00:00 2026")
    dialog = cal_chooser.CalibrationFileChooser(first)
    monkeypatch.setattr(
        cal_chooser.QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(other)
    )
    dialog.choose_folder()
    assert dialog.table.rowCount() == 2
    assert dialog.selected_path() == other / "frequency_C.cal"


# ---- calibration display


def set_header_time(path, created):
    data = bytearray(path.read_bytes())
    header = f"MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: {created:s}".ljust(116)
    data[:116] = header.encode("ascii")
    path.write_bytes(bytes(data))


def write_check(folder, name, created, level=75.0):
    path = folder / name
    engine.save_mat_struct(
        path,
        "CHKCAL",
        {
            "Freqs": np.array([[8000.0, 2000.0, 4000.0]]),
            "dBSPL_bp": np.array([[level], [level + 1.0], [level - 1.0]]),
            "dBSPL_nf": np.array([[30.0], [-5.0], [20.0]]),
        },
    )
    set_header_time(path, created)
    return path


def write_cal(folder, name, created):
    path = folder / name
    engine.save_mat_struct(path, "CAL", {"Date": "x"})
    set_header_time(path, created)
    return path


CAL_TIME = "Mon Oct 28 10:44:17 2024"


def test_find_check_uses_the_first_check_after_the_calibration(tmp_path):
    cal = write_cal(tmp_path, "frequency_MF1.cal", CAL_TIME)
    write_check(tmp_path, "chk75db_MF1_a.cal", "Mon Oct 28 10:00:00 2024")
    expected = write_check(tmp_path, "chk75db_MF1_b.cal", "Mon Oct 28 10:47:50 2024")
    write_check(tmp_path, "chk75db_MF1_c.cal", "Mon Oct 28 15:00:00 2024")
    assert cal_display.find_check(cal, "MF1")[0] == expected


def test_find_check_ignores_checks_on_another_day_or_for_another_speaker(tmp_path):
    cal = write_cal(tmp_path, "frequency_MF1.cal", CAL_TIME)
    write_check(tmp_path, "chk75db_MF1_a.cal", "Tue Oct 29 09:00:00 2024")
    write_check(tmp_path, "chk75db_ES1.cal", "Mon Oct 28 11:00:00 2024")
    assert cal_display.find_check(cal, "MF1") is None


def test_find_check_looks_in_the_history_folder(tmp_path):
    history = tmp_path / engine.HISTORY_DIR
    history.mkdir()
    cal = write_cal(tmp_path, "frequency_MF1.cal", CAL_TIME)
    expected = write_check(history, "chk75db_MF1_x.cal", "Mon Oct 28 10:47:50 2024")
    assert cal_display.find_check(cal, "MF1")[0] == expected


def test_find_check_for_a_history_calibration_looks_in_the_parent_folder(tmp_path):
    history = tmp_path / engine.HISTORY_DIR
    history.mkdir()
    cal = write_cal(history, "frequency_MF1_x.cal", CAL_TIME)
    expected = write_check(tmp_path, "chk75db_MF1.cal", "Mon Oct 28 10:47:50 2024")
    assert cal_display.find_check(cal, "MF1")[0] == expected


def test_find_check_skips_older_formats(tmp_path):
    cal = write_cal(tmp_path, "frequency_MF1.cal", CAL_TIME)
    old = tmp_path / "chk75db_MF1_old.cal"
    engine.save_mat_struct(old, "CHKCAL", {"Date": "x"})
    set_header_time(old, "Mon Oct 28 10:50:00 2024")
    assert cal_display.find_check(cal, "MF1") is None


def data_names(panel):
    return {i.name() for i in panel.listDataItems() if i.name()}


def test_display_shows_raw_calibration_and_check_with_noise_floor(qapp, tmp_path):
    cal = write_cal(tmp_path, "frequency_MF1.cal", CAL_TIME)
    write_check(tmp_path, "chk75db_MF1.cal", "Mon Oct 28 10:47:50 2024")
    caldata = read_calibration.get_calibration_data(REAL_FREQUENCY_CAL)
    display = cal_display.CalibrationDisplay()
    display.show_calibration(cal, caldata)
    assert data_names(display.raw_panel) == {
        "Noise floor", "Max SPL (0 dB attn)", "Cosinor", "Bandpass"
    }  # fmt: skip
    assert data_names(display.check_panel) == {"Noise floor", "75 dB target", "Measured (bandpass)"}
    measured = next(
        i for i in display.check_panel.listDataItems() if i.name() == "Measured (bandpass)"
    )
    assert measured.xData.tolist() == [2000.0, 4000.0, 8000.0]  # sorted by frequency
    assert measured.yData.tolist() == [76.0, 74.0, 75.0]
    noise = next(i for i in display.check_panel.listDataItems() if i.name() == "Noise floor")
    assert noise.yData.tolist() == [0.0, 20.0, 30.0]  # clipped at 0 dB SPL
    assert "chk75db_MF1.cal" in display.check_panel.titleLabel.text


def test_display_says_when_there_is_no_check_and_clears_the_previous_one(qapp, tmp_path):
    cal = write_cal(tmp_path, "frequency_MF1.cal", CAL_TIME)
    write_check(tmp_path, "chk75db_MF1.cal", "Mon Oct 28 10:47:50 2024")
    other = write_cal(tmp_path, "frequency_other.cal", "Tue Oct 29 10:44:17 2024")
    caldata = read_calibration.get_calibration_data(REAL_FREQUENCY_CAL)
    display = cal_display.CalibrationDisplay()
    display.show_calibration(cal, caldata)
    display.show_calibration(other, caldata)
    assert display.check_panel.listDataItems() == []
    assert "no check" in display.check_panel.titleLabel.text
    assert data_names(display.raw_panel) >= {"Noise floor", "Bandpass"}


# ---- pyabr3 button


def test_pyabr3_has_calibrate_button():
    from src.build_parametertree import build_parametertree

    _, tree = build_parametertree(["p.cfg"], "p.cfg", stimuli={"wave_duration": 1.0})
    assert tree.child("Actions", "Calibrate").type() == "action"
    assert "Save Visible" not in [c.name() for c in tree.child("Actions").children()]


def pyabr_stub(hardware, running=False):
    return SimpleNamespace(
        hardware=hardware,
        PS=object(),
        config={"calfile": "x/y.cal"},
        TrialTimer=SimpleNamespace(isActive=lambda: running),
        cal_window=None,
    )


def test_pyabr3_opens_one_window_and_uses_mock_without_rp21(qapp):
    from src.pyabr3 import PyABR

    stub = pyabr_stub(["Soundcard"])
    PyABR.open_calibration(stub)
    first = stub.cal_window
    assert first.hw.is_mock and first.embedded
    PyABR.open_calibration(stub)  # a second press raises the open window
    assert stub.cal_window is first
    first.close()


def test_pyabr3_uses_its_own_hardware_connection_on_the_rig(qapp):
    from src.pyabr3 import PyABR

    stub = pyabr_stub(["NIDAQ", "RP21", "PA5"])
    PyABR.open_calibration(stub)
    assert isinstance(stub.cal_window.hw, PyStimCalibrationHardware)
    assert stub.cal_window.hw.PS is stub.PS
    stub.cal_window.hw.rp = None
    stub.cal_window.close()


def test_pyabr3_refuses_calibration_during_acquisition(qapp, monkeypatch):
    from src import pyabr3

    messages = []
    monkeypatch.setattr(pyabr3.MBox, "MessageBox", lambda *a, **k: messages.append(a))
    stub = pyabr_stub(["NIDAQ", "RP21", "PA5"], running=True)
    pyabr3.PyABR.open_calibration(stub)
    assert stub.cal_window is None
    assert len(messages) == 1


def cal_stub(tmp_path):
    shown, raised = [], []
    return SimpleNamespace(
        shown=shown,
        raised=raised,
        calibration_display=SimpleNamespace(show_calibration=lambda p, d: shown.append((p, d))),
        Dock_Calibration=SimpleNamespace(raiseDock=lambda: raised.append(1)),
        calfile=tmp_path / "frequency_MF1.cal",
        caldata="old",
        WG=SimpleNamespace(caldata="old"),
        win=None,
    )


def test_pyabr3_uses_the_chosen_calibration_for_stimuli(tmp_path, monkeypatch):
    from src import pyabr3

    monkeypatch.setattr(
        pyabr3.cal_chooser, "choose_calibration_file", lambda *a, **k: REAL_FREQUENCY_CAL
    )
    stub = cal_stub(tmp_path)
    pyabr3.PyABR.load_cal_file(stub)
    assert stub.calfile == REAL_FREQUENCY_CAL
    assert stub.shown == [(REAL_FREQUENCY_CAL, stub.caldata)]
    assert stub.raised == [1]
    assert stub.caldata["spkr"] == "MF1"
    assert stub.WG.caldata is stub.caldata


def test_pyabr3_keeps_its_calibration_when_chooser_is_cancelled(tmp_path, monkeypatch):
    from src import pyabr3

    monkeypatch.setattr(pyabr3.cal_chooser, "choose_calibration_file", lambda *a, **k: None)
    stub = cal_stub(tmp_path)
    pyabr3.PyABR.load_cal_file(stub)
    assert stub.caldata == "old" and stub.WG.caldata == "old"


def test_pyabr3_button_press_runs_its_action_once(qapp, monkeypatch):
    """Builds the real pyabr3 window with the test configuration and a stored calibration."""
    import atexit

    from src import pyabr3

    real_read = pyabr3.read_calibration.get_calibration_data
    monkeypatch.setattr(
        pyabr3.read_calibration, "get_calibration_data", lambda fn: real_read(REAL_FREQUENCY_CAL)
    )
    calls = []
    monkeypatch.setattr(pyabr3.PyABR, "new_filename", lambda self: calls.append(1))
    monkeypatch.chdir(REPO)
    prog = pyabr3.PyABR("test")
    atexit.unregister(prog.quit)
    prog.ptreedata.child("Actions", "New Filename").activate()
    qapp.processEvents()
    assert calls == [1]
    prog.win.close()


def test_pyabr3_test_config_starts_without_the_rig_calibration_file(qapp, monkeypatch):
    import atexit

    from src import pyabr3

    monkeypatch.chdir(REPO)
    prog = pyabr3.PyABR("test")
    atexit.unregister(prog.quit)
    assert prog.calfile == Path("calfiles", "frequency_MF1.cal")
    assert prog.caldata["spkr"] == "MF1"
    assert {"Recording", "Preview", "Calibration"} <= set(prog.dockArea.docks)
    assert "frequency_MF1.cal" in prog.calibration_display.raw_panel.titleLabel.text
    prog.win.close()
