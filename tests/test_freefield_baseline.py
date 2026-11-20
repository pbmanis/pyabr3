"""
Regression test: free-field waveforms and attenuations must not change.

The binaural interaction component (BIC) work (see ABR_BIC_PLAN.md) must not change
how the existing click, tone pip and interleaved plateau stimuli are generated. The
expected values in freefield_baseline.json were computed with WaveGenerator before any
BIC change. They are summaries (length, sum, extremes, attenuations), compared with a
tolerance, not bit-for-bit.

To regenerate the baseline after a deliberate change to the stimuli (human use only):
    python tests/test_freefield_baseline.py --write

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_freefield_baseline.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
from pyqtgraph import configfile

from src import convert_nested_ordered_dict, read_calibration, wave_generator

BASELINE = Path(__file__).with_name("freefield_baseline.json")
CALFILE = Path(REPO, "calfiles", "frequency_MF1.cal")
SFOUT = 500000.0  # NI output rate


def load_protocol(name: str) -> dict:
    protocol = configfile.readConfigFile(Path(REPO, "protocols", name))
    return convert_nested_ordered_dict.convert_nested_ordered_dict(protocol)


def make_cases() -> dict:
    """The protocols to check: existing protocol files, and a tone pip protocol made
    from the click protocol (the tonepip*.cfg files do not currently parse)."""
    click = load_protocol("clickIO.cfg")
    tonepip = load_protocol("clickIO.cfg")
    tonepip["protocol"]["stimulustype"] = "tonepip"
    tonepip["stimuli"]["stimulus_duration"] = 5e-3  # a click is too short for a pip
    tonepip["stimuli"]["freqlist"] = [4000.0, 8000.0]
    tonepip["stimuli"]["dblist"] = [80.0, 50.0]
    plateau = load_protocol("interleaved_plateau.cfg")
    return {"click": click, "tonepip": tonepip, "interleaved_plateau": plateau}


def summarize(protocol: dict) -> dict:
    config = configfile.readConfigFile(str(Path(REPO, "config", "abrs.cfg")))
    caldata = read_calibration.get_calibration_data(CALFILE)
    wg = wave_generator.WaveGenerator(caldata=caldata)
    wg.setup(protocol=protocol, frequency=SFOUT, config=config)
    stimulustype = protocol["protocol"]["stimulustype"]
    stimuli = protocol["stimuli"]
    dbspls = stimuli["dblist"]
    if isinstance(dbspls, str):
        dbspls = eval(dbspls)  # the plateau protocol holds it as a string
    wg.make_waveforms(wavetype=stimulustype, dbspls=dbspls, frequencies=None)
    summary = {}
    for key, entry in wg.wave_matrix.items():
        sound = np.asarray(entry["sound"])
        summary[repr(tuple(float(k) if isinstance(k, (float, np.floating)) else k for k in key))] = {
            "n": int(sound.shape[0]),
            "ndim": int(sound.ndim),
            "sum": float(np.sum(sound)),
            "abs_sum": float(np.sum(np.abs(sound))),
            "max": float(np.max(sound)),
            "min": float(np.min(sound)),
            "attenuation": [float(a) for a in np.atleast_1d(entry["attenuation"])]
            if "attenuation" in entry
            else None,
        }
    return summary


def compute_all() -> dict:
    return {name: summarize(protocol) for name, protocol in make_cases().items()}


@pytest.fixture(scope="module")
def current():
    return compute_all()


@pytest.fixture(scope="module")
def expected():
    return json.loads(BASELINE.read_text())


@pytest.mark.parametrize("name", ["click", "tonepip", "interleaved_plateau"])
def test_freefield_waves_unchanged(name, current, expected):
    assert list(current[name].keys()) == list(expected[name].keys())
    for key, exp in expected[name].items():
        got = current[name][key]
        assert got["n"] == exp["n"]
        assert got["ndim"] == exp["ndim"]
        for field in ("sum", "abs_sum", "max", "min"):
            assert got[field] == pytest.approx(exp[field], rel=1e-9, abs=1e-9), (key, field)
        assert got["attenuation"] == pytest.approx(exp["attenuation"], rel=1e-9, abs=1e-9)


def test_mode_defaults_to_freefield_without_changing_protocol():
    from src import protocol_reader

    protocol = load_protocol("clickIO.cfg")
    before = json.dumps(protocol, sort_keys=True, default=str)
    assert protocol_reader.get_mode(protocol) == "freefield"
    assert json.dumps(protocol, sort_keys=True, default=str) == before
    assert "mode" not in protocol["stimuli"]


def test_existing_protocol_reads_unchanged(monkeypatch):
    from src import protocol_reader

    monkeypatch.chdir(REPO)  # read_protocol reads protocols/ relative to the working directory
    reader = protocol_reader.ProtocolReader(None)
    reader.read_protocol("clickIO.cfg")
    assert "mode" not in reader.get_current_protocol()["stimuli"]


@pytest.mark.parametrize("mode", ["freefield", "monaural_left", "monaural_right", "binaural", "bic_interleaved"])
def test_known_modes_accepted(mode):
    from src import protocol_reader

    assert protocol_reader.get_mode({"stimuli": {"mode": mode}}) == mode


def test_unknown_mode_rejected():
    from src import protocol_reader

    with pytest.raises(ValueError, match="Unknown stimulus mode"):
        protocol_reader.get_mode({"stimuli": {"mode": "stereo"}})


if __name__ == "__main__":
    if "--write" in sys.argv:
        BASELINE.write_text(json.dumps(compute_all(), indent=1))
        print("wrote", BASELINE)
