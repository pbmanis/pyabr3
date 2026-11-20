"""
Tests for per-ear calibration (ABR_BIC_PLAN.md, step 2).

Without a separate right-speaker calibration, both ears use the one calibration, so
free-field operation is unchanged. Reads calfiles/frequency_MF1.cal; never writes it.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_per_ear_calibration.py
"""

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from src import pyabr3, read_calibration, wave_generator

CALFILE = Path(REPO, "calfiles", "frequency_MF1.cal")


@pytest.fixture
def caldata():
    return read_calibration.get_calibration_data(CALFILE)


@pytest.fixture
def wg(caldata):
    return wave_generator.WaveGenerator(caldata=caldata)


def quieter(caldata, db=6.0):
    """A copy of the calibration for a speaker that is db quieter at every frequency."""
    other = copy.deepcopy(caldata)
    other["maxdb"] = np.asarray(other["maxdb"], dtype=float) - db
    return other


def test_right_ear_uses_left_calibration_by_default(wg, caldata):
    assert wg.caldata_right is None
    assert wg.caldata_for("left") is caldata
    assert wg.caldata_for("right") is caldata
    attn, maxdb = wg.compute_attn_db_for_dbspl(8000.0, 60.0)
    for ear in ("left", "right"):
        attn_ear, maxdb_ear = wg.compute_attn_db_for_dbspl(8000.0, 60.0, ear=ear)
        assert attn_ear[0] == pytest.approx(attn[0])
        assert maxdb_ear[0] == pytest.approx(maxdb[0])


def test_separate_right_calibration_changes_only_the_right(wg, caldata):
    wg.caldata_right = quieter(caldata, 6.0)
    left = wg.compute_attn_db_for_dbspl(8000.0, 60.0, ear="left")[0][0]
    right = wg.compute_attn_db_for_dbspl(8000.0, 60.0, ear="right")[0][0]
    assert left == pytest.approx(wg.compute_attn_db_for_dbspl(8000.0, 60.0)[0][0])
    assert right == pytest.approx(left - 6.0)  # a quieter speaker needs less attenuation


def test_unknown_ear_rejected(wg):
    with pytest.raises(ValueError, match="ear must be"):
        wg.caldata_for("both")


def test_tone_level_check_uses_the_requested_ear(wg, caldata):
    freq = 8000.0
    maxdb = wg.compute_attn_db_for_dbspl(freq, 0.0)[1][0]
    wg.caldata_right = quieter(caldata, 10.0)
    level = maxdb - 5.0  # reachable by the left speaker, not by the quieter right one
    wg.check_tone_levels([freq], [level])
    wg.check_tone_levels([freq], [level], ear="left")
    with pytest.raises(wave_generator.UnreachableLevelError):
        wg.check_tone_levels([freq], [level], ear="right")


def test_click_reference_per_ear(wg):
    wg.config = {"click_reference_dbspl": 105.1}
    assert wg.click_reference_dbspl() == 105.1
    assert wg.click_reference_dbspl("right") == 105.1  # no right value: same as left
    wg.config["click_reference_dbspl_right"] = 99.0
    assert wg.click_reference_dbspl("left") == 105.1
    assert wg.click_reference_dbspl("right") == 99.0


def test_right_calibration_file_is_optional():
    app = SimpleNamespace(config={"calfile": str(CALFILE)})
    assert pyabr3.PyABR.load_right_calibration(app, "abr") == (None, None)
    app.config["calfile_right"] = ""
    assert pyabr3.PyABR.load_right_calibration(app, "abr") == (None, None)


def test_right_calibration_file_is_read():
    app = SimpleNamespace(config={"calfile": str(CALFILE), "calfile_right": str(CALFILE)})
    path, caldata = pyabr3.PyABR.load_right_calibration(app, "abr")
    assert path == CALFILE
    assert "maxdb" in caldata


def test_missing_right_file_falls_back_only_in_test_configuration():
    app = SimpleNamespace(config={"calfile_right": "no/such/file.cal"})
    assert pyabr3.PyABR.load_right_calibration(app, "test") == (None, None)
    with pytest.raises(ValueError):
        pyabr3.PyABR.load_right_calibration(app, "abr")
