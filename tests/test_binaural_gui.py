"""
Tests for the binaural controls of the main window (ABR_BIC_PLAN.md, step 6): the
protocol fields shown in the parameter tree, and the calibration of each output path.
The Parameters group is read-only: protocols are used as saved.

Run from the top directory of the repository:
    python -m pytest --confcutdir=tests tests/test_binaural_gui.py
"""

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

import pytest
import pyqtgraph as pg

from src import protocol_reader, pyabr3
from src.build_parametertree import build_parametertree

CALFILE = Path(REPO, "calfiles", "frequency_MF1.cal")


@pytest.fixture
def tree(monkeypatch):
    pg.mkQApp("test")
    monkeypatch.chdir(REPO)  # read_protocol reads protocols/ relative to the working directory
    reader = protocol_reader.ProtocolReader(None)
    reader.read_protocol("clickIO.cfg")
    _, ptreedata = build_parametertree(["clickIO.cfg"], "clickIO.cfg", stimuli={"wave_duration": 1.2})
    return reader, ptreedata


def field(ptreedata, name):
    return ptreedata.child("Parameters", name)


def test_freefield_protocol_hides_the_binaural_fields(tree):
    reader, ptreedata = tree
    reader.update_protocol(ptreedata)
    assert field(ptreedata, "mode").value() == "freefield"
    assert not field(ptreedata, "itd").opts["visible"]
    assert not field(ptreedata, "dblist_right").opts["visible"]


def test_bic_protocol_shows_its_fields(tree):
    reader, ptreedata = tree
    reader._protocol = copy.deepcopy(reader._protocol)
    reader._protocol["stimuli"].update(mode="bic_interleaved", itd=2e-4, dblist_right=[70.0, 60.0])
    reader.update_protocol(ptreedata)
    assert field(ptreedata, "mode").value() == "bic_interleaved"
    assert field(ptreedata, "itd").value() == pytest.approx(2e-4)
    assert field(ptreedata, "dblist_right").value() == "[70.0, 60.0]"
    assert field(ptreedata, "itd").opts["visible"]
    assert field(ptreedata, "dblist_right").opts["visible"]


def test_fields_are_hidden_again_for_a_following_freefield_protocol(tree):
    reader, ptreedata = tree
    free = copy.deepcopy(reader._protocol)
    reader._protocol = copy.deepcopy(free)
    reader._protocol["stimuli"].update(mode="binaural", itd=1e-4)
    reader.update_protocol(ptreedata)
    assert field(ptreedata, "itd").opts["visible"]
    reader._protocol = free
    reader.update_protocol(ptreedata)
    assert not field(ptreedata, "itd").opts["visible"]
    assert field(ptreedata, "itd").value() == 0.0
    assert field(ptreedata, "mode").value() == "freefield"


def test_parameters_group_stays_read_only(tree):
    _, ptreedata = tree
    for name in ("mode", "itd", "dblist_right", "dblist"):
        assert field(ptreedata, name).opts["readonly"]


def test_the_existing_fields_are_unchanged(tree):
    reader, ptreedata = tree
    reader.update_protocol(ptreedata)
    stimuli = reader.get_current_protocol()["stimuli"]
    assert field(ptreedata, "nstim").value() == stimuli["nstim"]
    assert field(ptreedata, "interval").value() == pytest.approx(stimuli["interval"])
    assert "mode" not in stimuli  # showing the mode does not add it to the protocol


def test_calibration_labels_name_the_files_of_the_paths(tree):
    _, ptreedata = tree
    app = SimpleNamespace(ptreedata=ptreedata, calfile=Path("cal/frequency_MF1.cal"), calfile_right=None)
    pyabr3.PyABR.update_calibration_labels(app)
    assert ptreedata.child("Calibration", "Left (free-field) path").value() == "frequency_MF1.cal"
    assert ptreedata.child("Calibration", "Right path").value() == "same as left"
    app.calfile_right = Path("cal/frequency_ES1.cal")
    pyabr3.PyABR.update_calibration_labels(app)
    assert ptreedata.child("Calibration", "Right path").value() == "frequency_ES1.cal"


class FakeDock:
    raised = 0

    def raiseDock(self):
        FakeDock.raised += 1


def right_cal_app(chosen):
    shown = []
    app = SimpleNamespace(
        calfile=Path("somewhere/frequency_MF1.cal"),
        calfile_right=None,
        caldata_right=None,
        WG=SimpleNamespace(caldata_right=None),
        calibration_display=SimpleNamespace(show_calibration=lambda path, data: shown.append(path)),
        Dock_Calibration=FakeDock(),
        win=None,
        shown=shown,
    )
    return app


def test_right_calibration_is_chosen_and_given_to_the_wave_generator(monkeypatch):
    monkeypatch.setattr(pyabr3.cal_chooser, "choose_calibration_file", lambda *a, **k: CALFILE)
    app = right_cal_app(CALFILE)
    pyabr3.PyABR.load_right_cal_file(app)
    assert app.calfile_right == CALFILE
    assert app.caldata_right is not None
    assert app.WG.caldata_right is app.caldata_right
    assert app.shown == [CALFILE]
    assert app.calfile == Path("somewhere/frequency_MF1.cal")  # the left path is not changed


def test_cancelling_the_right_calibration_dialog_changes_nothing(monkeypatch):
    monkeypatch.setattr(pyabr3.cal_chooser, "choose_calibration_file", lambda *a, **k: None)
    app = right_cal_app(None)
    pyabr3.PyABR.load_right_cal_file(app)
    assert app.calfile_right is None and app.WG.caldata_right is None


def test_unreadable_right_calibration_is_reported_and_not_used(monkeypatch, tmp_path):
    bad = tmp_path / "frequency_bad.cal"
    bad.write_text("not a calibration")
    messages = []
    monkeypatch.setattr(pyabr3.cal_chooser, "choose_calibration_file", lambda *a, **k: bad)
    monkeypatch.setattr(pyabr3.MBox, "MessageBox", lambda message, title="": messages.append(message))
    app = right_cal_app(bad)
    pyabr3.PyABR.load_right_cal_file(app)
    assert app.calfile_right is None and app.WG.caldata_right is None
    assert "frequency_bad.cal" in messages[0]
