"""
Reading and averaging of pyabr3 data files for the data viewer.

A recording is a set of files named <date>_<stimulus>_<wave>_<repetition>.p, one for
each trial. The averages use the same arithmetic as average_within_traces and
average_across_traces in analyze_abr.py. Those functions cannot be called directly,
because they get the stimulus type from the name of the folder, and the folder that
pyabr3 writes to is not named for the stimulus.

Levels and frequencies are taken from the stimulus table stored in each file, since
the protocol's dblist can be a string of code, such as "np.linspace(10.0, 90.0, 9)".
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.analyze_abr import AnalyzeABR

TRIAL_NAME = re.compile(r"^(?P<date>[^_]+)_(?P<stim>.+)_(?P<wave>\d+)_(?P<rep>\d+)\.p$")
RESAMPLED_RATE = 1e5  # Hz; removes the drift of the RP2.1 clock against the stimulus
AMPLIFIER_GAIN = 1e4
VOLTS_TO_UV = 1e6
ANALYZER = AnalyzeABR()


@dataclass
class Run:
    """The trial files of one recording."""

    directory: Path
    date: str
    stim: str
    files: dict = field(default_factory=dict)  # (wave, repetition): path

    @property
    def waves(self) -> list:
        return sorted({wave for wave, _ in self.files})

    @property
    def repetitions(self) -> list:
        return sorted({rep for _, rep in self.files})

    @property
    def interleaved(self) -> bool:
        return self.stim.lower().startswith("interleaved")


@dataclass
class RunData:
    run: Run
    header: dict  # the contents of the first trial
    rep_averages: dict  # wave: mean across repetitions, in V at the amplifier output
    n_reps: dict  # wave: number of repetitions averaged
    sample_rate: float


@dataclass
class Responses:
    t_ms: np.ndarray
    traces: dict  # (dB SPL, frequency): response in uV at the electrodes

    @property
    def frequencies(self) -> list:
        return sorted({fr for _, fr in self.traces})

    def levels(self, frequency: float) -> list:
        return sorted(db for db, fr in self.traces if fr == frequency)


def find_run(path: Path) -> Run:
    """Find all of the trial files that belong to the same recording as this file."""
    path = Path(path)
    match = TRIAL_NAME.match(path.name)
    if match is None:
        raise ValueError(f"Not the name of a pyabr3 trial file: {path.name:s}")
    run = Run(directory=path.parent, date=match["date"], stim=match["stim"])
    for candidate in path.parent.glob("*.p"):
        other = TRIAL_NAME.match(candidate.name)
        if other is not None and (other["date"], other["stim"]) == (run.date, run.stim):
            run.files[(int(other["wave"]), int(other["rep"]))] = candidate
    return run


def read_trial(path: Path) -> dict:
    """One trial, with the delay before the stimulus removed, as for the analysis."""
    return ANALYZER.read_abr_file(path)


def load_run(path: Path, progress: Callable | None = None) -> RunData:
    """Read a recording, and average the repetitions of each stimulus."""
    run = find_run(path)
    totals: dict = {}
    n_reps: dict = {}
    header = None
    for count, ((wave, _), filename) in enumerate(sorted(run.files.items())):
        trial = read_trial(filename)
        if header is None:
            header = trial
        totals[wave] = totals.get(wave, 0.0) + trial["data"]
        n_reps[wave] = n_reps.get(wave, 0) + 1
        if progress is not None:
            progress(count + 1, len(run.files))
    rep_averages = {wave: totals[wave] / n_reps[wave] for wave in totals}
    return RunData(run, header, rep_averages, n_reps, header["record_frequency"])


def _filter_and_resample(data: np.ndarray, sample_rate: float, hpf, lpf) -> np.ndarray:
    filtered = data
    if hpf is not None:
        filtered = ANALYZER.FILT.SignalFilter_HPFButter(
            filtered, hpf, sample_rate, NPole=4, bidir=True
        )
    if lpf is not None:
        filtered = ANALYZER.FILT.SignalFilter_LPFButter(
            filtered, lpf, sample_rate, NPole=4, bidir=True
        )
    duration = len(filtered) / sample_rate
    t_original = np.linspace(0, duration, len(filtered))
    return np.interp(np.arange(0, duration, 1.0 / RESAMPLED_RATE), t_original, filtered)


def _to_microvolts(trace: np.ndarray) -> np.ndarray:
    return trace / AMPLIFIER_GAIN * VOLTS_TO_UV


def compute_responses(data: RunData, hpf=None, lpf=None, interleaved_set: int = 0) -> Responses:
    """Filtered average response for each level and frequency.

    Click and tone trials repeat one stimulus, so the responses in a trial are averaged
    together. Interleaved trials hold one response per level and frequency.
    interleaved_set selects which interleaved stimulus set is shown (the analysis uses
    only the first).
    """
    stimuli = data.header["protocol"]["stimuli"]
    table = data.header["stimuli"]
    keys = list(table)
    traces = {}
    if data.run.interleaved:
        wave = data.run.waves[interleaved_set]
        entry = table[keys[wave]]
        abr = _filter_and_resample(data.rep_averages[wave], data.sample_rate, hpf, lpf)
        one_response = int(stimuli["stimulus_period"] * RESAMPLED_RATE)
        n_responses = len(entry["dbspls"])
        pieces = np.split(abr[: one_response * n_responses], n_responses)
        for db, fr, piece in zip(entry["dbspls"], entry["frequencies"], pieces):
            traces[(float(db), float(fr))] = _to_microvolts(piece)
    else:
        one_response = int(stimuli["interval"] * RESAMPLED_RATE)
        n_responses = int(stimuli["nstim"])
        for wave in data.run.waves:
            if wave >= len(keys):
                continue
            abr = _filter_and_resample(data.rep_averages[wave], data.sample_rate, hpf, lpf)
            pieces = np.split(abr[: one_response * n_responses], n_responses)
            _, db, fr = keys[wave]
            traces[(float(db), float(fr))] = _to_microvolts(np.mean(pieces, axis=0))
    return Responses(np.arange(one_response) / RESAMPLED_RATE * 1000.0, traces)


def load_summary(path: Path):
    """Read a .pkl file."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def is_threshold_summary(data) -> bool:
    """True for the format of all_tone_map_data.pkl, written by read_abr.py."""
    return isinstance(data, dict) and {"frequencies", "thresholds"} <= set(data)


def numeric_arrays(data: dict) -> dict:
    """The top level values of a dict that can be plotted as 1-D or 2-D arrays."""
    arrays = {}
    for key, value in data.items():
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.ndim in [1, 2] and array.size > 1:
            arrays[str(key)] = array
    return arrays
