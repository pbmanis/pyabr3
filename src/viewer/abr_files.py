"""
Reading and averaging of pyabr3 data files for the data viewer.

A recording is a set of files named <date>_<stimulus>_<wave>_<repetition>.p, one for
each trial. The averages use the same arithmetic as average_within_traces and
average_across_traces in analyze_abr.py. Those functions cannot be called directly,
because they get the stimulus type from the name of the folder, and the folder that
pyabr3 writes to is not named for the stimulus.

Levels and frequencies are taken from the stimulus table stored in each file, since
the protocol's dblist can be a string of code, such as "np.linspace(10.0, 90.0, 9)".

Older recordings from the MATLAB program (ABR4) are text files, read here with the
same arithmetic as read_dataset, get_combineddata, getClickData and getToneData in
ABRPlotting/src/abr_reader.py. For a run named yyyymmdd-HHMM:
    <run>-SPL.txt                levels
    <run>-kHz.txt                frequencies in Hz (tone runs only)
    <run>-p.txt, <run>-n.txt     click responses, one column for each level
    <run>-p-<freq>.txt, -n-...   tone responses for each frequency
ABRPlotting needs its catalog and coding sheets, so it cannot be called from here.
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.signal

from src.analyze_abr import AnalyzeABR

TRIAL_NAME = re.compile(r"^(?P<date>[^_]+)_(?P<stim>.+)_(?P<wave>\d+)_(?P<rep>\d+)\.p$")
RESAMPLED_RATE = 1e5  # Hz; removes the drift of the RP2.1 clock against the stimulus
AMPLIFIER_GAIN = 1e4
VOLTS_TO_UV = 1e6
ANALYZER = AnalyzeABR()
MATLAB_RUN_NAME = re.compile(r"^(?P<run>\d{8}-\d{4})(-.*)?\.txt$")
MATLAB_SAMPLE_RATE = 1e5  # acquire4.m interpolates the responses to InterpFreq = 100 kHz
MATLAB_MAX_MS = 25.0  # read_dataset keeps the first 25 ms
MATLAB_FILTER_ORDER = 4


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


def _selected_waves(data: RunData, interleaved_set: int) -> list:
    """The stimuli to show: one interleaved set, or all of the click or tone stimuli."""
    n_stimuli = len(data.header["stimuli"])
    waves = [data.run.waves[interleaved_set]] if data.run.interleaved else data.run.waves
    return [wave for wave in waves if wave < n_stimuli]


def _responses_in_trace(data: RunData, wave: int, trace: np.ndarray, hpf, lpf) -> dict:
    """The response to each level and frequency in one trace, as (dB, frequency): uV."""
    stimuli = data.header["protocol"]["stimuli"]
    key = list(data.header["stimuli"])[wave]
    entry = data.header["stimuli"][key]
    abr = _filter_and_resample(trace, data.sample_rate, hpf, lpf)
    if data.run.interleaved:
        one_response = int(stimuli["stimulus_period"] * RESAMPLED_RATE)
        n_responses = len(entry["dbspls"])
        pieces = np.split(abr[: one_response * n_responses], n_responses)
        return {
            (float(db), float(fr)): _to_microvolts(piece)
            for db, fr, piece in zip(entry["dbspls"], entry["frequencies"], pieces)
        }
    one_response = int(stimuli["interval"] * RESAMPLED_RATE)
    n_responses = int(stimuli["nstim"])
    pieces = np.split(abr[: one_response * n_responses], n_responses)
    _, db, fr = key
    return {(float(db), float(fr)): _to_microvolts(np.mean(pieces, axis=0))}


def _response_duration(data: RunData) -> int:
    stimuli = data.header["protocol"]["stimuli"]
    period = stimuli["stimulus_period"] if data.run.interleaved else stimuli["interval"]
    return int(period * RESAMPLED_RATE)


def compute_responses(data: RunData, hpf=None, lpf=None, interleaved_set: int = 0) -> Responses:
    """Filtered average response for each level and frequency.

    Click and tone trials repeat one stimulus, so the responses in a trial are averaged
    together. Interleaved trials hold one response per level and frequency.
    interleaved_set selects which interleaved stimulus set is shown (the analysis uses
    only the first).
    """
    traces = {}
    for wave in _selected_waves(data, interleaved_set):
        traces |= _responses_in_trace(data, wave, data.rep_averages[wave], hpf, lpf)
    return Responses(np.arange(_response_duration(data)) / RESAMPLED_RATE * 1000.0, traces)


def compute_trial_responses(
    data: RunData,
    hpf=None,
    lpf=None,
    interleaved_set: int = 0,
    progress: Callable | None = None,
    should_stop: Callable | None = None,
) -> dict | None:
    """The response in each repetition, as (dB, frequency): array (n_trials, n_points).

    The trials are read again and filtered one at a time, so that only the responses
    are held in memory. Single precision is enough for display. Returns None if stopped.
    """
    waves = _selected_waves(data, interleaved_set)
    files = [path for (wave, _), path in sorted(data.run.files.items()) if wave in waves]
    wave_of = {path: wave for (wave, _), path in data.run.files.items()}
    collected: dict = {}
    for count, path in enumerate(files):
        if should_stop is not None and should_stop():
            return None
        trial = read_trial(path)
        responses = _responses_in_trace(data, wave_of[path], trial["data"], hpf, lpf)
        for key, response in responses.items():
            collected.setdefault(key, []).append(response.astype(np.float32))
        if progress is not None:
            progress(count + 1, len(files))
    return {key: np.stack(responses) for key, responses in collected.items()}


def trial_sd(trials: np.ndarray) -> np.ndarray:
    """Standard deviation across trials (n-1 weighting; zero for a single trial)."""
    if trials.shape[0] < 2:
        return np.zeros(trials.shape[1])
    return np.std(trials, axis=0, ddof=1)


@dataclass
class MatlabRun:
    """A run from the MATLAB program: mean of the two polarities at each frequency."""

    directory: Path
    name: str
    levels: list  # dB SPL, ascending, in the order of the columns of each array
    waves: dict  # frequency (0.0 for clicks): array of shape (n_points, n_levels), in V

    @property
    def stim(self) -> str:
        return "click" if list(self.waves) == [0.0] else "tone"


def matlab_run_name(path: Path) -> str:
    match = MATLAB_RUN_NAME.match(Path(path).name)
    if match is None:
        raise ValueError(f"Not the name of a MATLAB ABR file: {Path(path).name:s}")
    return match["run"]


def list_matlab_runs(directory: Path) -> dict:
    """All of the runs in a folder, as run name: "click" or "tone"."""
    runs = {}
    for spl_file in sorted(Path(directory).glob("*-SPL.txt")):
        name = spl_file.name[: -len("-SPL.txt")]
        runs[name] = "tone" if Path(directory, f"{name:s}-kHz.txt").is_file() else "click"
    return runs


def _read_text_columns(path: Path) -> np.ndarray:
    """Files have \\r, \\n or \\r\\n line endings, depending on how they were copied."""
    rows = [row.split() for row in re.split(r"[\r\n]+", path.read_text()) if row.strip()]
    return np.array(rows, dtype=float)


def _read_numbers(path: Path) -> list:
    return [float(value) for value in path.read_text().split()]


def read_matlab_run(directory: Path, name: str) -> MatlabRun:
    """Read a run, combining the responses to the two stimulus polarities."""
    directory = Path(directory)
    levels = _read_numbers(directory / f"{name:s}-SPL.txt")
    frequency_file = directory / f"{name:s}-kHz.txt"
    is_tone = frequency_file.is_file()
    suffixes = (
        {f: f"-{f:.3f}" for f in _read_numbers(frequency_file)} if is_tone else {0.0: ""}
    )
    waves = {}
    for frequency, suffix in suffixes.items():
        positive = directory / f"{name:s}-p{suffix:s}.txt"
        negative = directory / f"{name:s}-n{suffix:s}.txt"
        if not positive.is_file() or not negative.is_file():
            continue  # as in read_dataset, a run with a missing polarity is skipped
        combined = (_read_text_columns(positive) + _read_text_columns(negative)) / 2.0
        if combined.shape[1] != len(levels):
            raise ValueError(
                f"{positive.name:s} has {combined.shape[1]:d} columns for {len(levels):d} levels"
            )
        # The columns of click files are in the opposite order to the levels (getClickData)
        waves[frequency] = combined if is_tone else combined[:, ::-1]
    if not waves:
        raise ValueError(f"No complete pair of response files for run {name:s}")
    return MatlabRun(directory, name, levels, waves)


def _filter_matlab(trace: np.ndarray, sample_rate: float, hpf, lpf) -> np.ndarray:
    """Zero phase Butterworth filter, as ABRFuncs.filter in ABRPlotting."""
    nyquist = sample_rate / 2.0
    if hpf is not None and lpf is not None:
        band, cutoff = "band", (hpf / nyquist, lpf / nyquist)
    elif hpf is not None:
        band, cutoff = "high", hpf / nyquist
    elif lpf is not None:
        band, cutoff = "low", lpf / nyquist
    else:
        return trace
    b, a = scipy.signal.iirfilter(N=MATLAB_FILTER_ORDER, Wn=cutoff, btype=band, ftype="butter")
    return scipy.signal.filtfilt(b, a, trace, padlen=int(len(trace) / 10))


def _fill_nonfinite(trace: np.ndarray) -> np.ndarray:
    """Replace isolated missing points, such as a NaN at the end of a trace."""
    finite = np.isfinite(trace)
    if finite.all():
        return trace
    indices = np.arange(len(trace))
    return np.interp(indices, indices[finite], trace[finite])


def compute_matlab_responses(
    run: MatlabRun, sample_rate: float = MATLAB_SAMPLE_RATE, hpf=None, lpf=None
) -> Responses:
    """Filtered responses of a MATLAB run, in uV.

    Levels that were not recorded (a column of NaN) are left out. These files already
    have the amplifier gain removed.
    """
    n_points = next(iter(run.waves.values())).shape[0]
    t_ms = np.arange(n_points) / sample_rate * 1000.0
    keep = t_ms < MATLAB_MAX_MS
    traces = {}
    for frequency, waves in run.waves.items():
        for i, level in enumerate(run.levels):
            column = waves[:, i]
            if not np.any(np.isfinite(column)):
                continue
            filtered = _filter_matlab(
                _fill_nonfinite(column[keep]), sample_rate, hpf, lpf
            )
            traces[(level, frequency)] = filtered * VOLTS_TO_UV
    return Responses(t_ms[keep], traces)


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
