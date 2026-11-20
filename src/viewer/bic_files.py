"""
Reading and averaging of interleaved binaural interaction component (BIC) recordings.

A sweep of an interleaved BIC recording holds slots, one every "interval" seconds, in
the order L, R, B (left only, right only, both ears), repeated nstim times. With
alternating polarity each slot is presented twice, in opposite polarity: L+ L- R+ R-
B+ B-. The files record which slot is which (the "conditions" entry that
PyABR.get_condition_tags saves), so this module only needs the files.

For each trial the sweep is filtered and resampled as in abr_files, and cut into one
response for each slot. The responses to the same condition are averaged, which includes
both polarities, so the stimulus artifact (which reverses with polarity) cancels. The
average of the trials is then in uV at the electrodes. The BIC is the binaural response
minus the sum of the left and right responses (BICResponses.bic). Time is relative to the nominal
start of the slot, so in the binaural slots the ITD delays the lagging ear's stimulus.

Related: wave_generator.WaveGenerator._make_bic_interleaved, abr_files.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.viewer import abr_files

CONDITIONS = ("L", "R", "B")
CONDITION_NAMES = {"L": "Left", "R": "Right", "B": "Binaural"}
BIC_MODES = ("bic_interleaved", "bic_sequential", "monaural_left", "monaural_right", "binaural")
PRESTIMULUS_S = 1e-3  # each response starts this long before the slot, as for the ABR viewer


@dataclass
class BICResponses:
    t_ms: np.ndarray  # 0 is the nominal start of the slot
    traces: dict  # condition ("L", "R", "B"): mean response in uV at the electrodes
    counts: dict  # condition: number of slot responses averaged, over all trials
    n_trials: int
    info: dict = field(default_factory=dict)  # the stimulus: itd, levels, frequency...

    @property
    def has_bic(self) -> bool:
        return all(c in self.traces for c in CONDITIONS)

    @property
    def sum_lr(self) -> np.ndarray | None:
        """The sum of the left and right responses (abr_sum), or None if one is missing."""
        if "L" not in self.traces or "R" not in self.traces:
            return None
        return self.traces["L"] + self.traces["R"]

    @property
    def bic(self) -> np.ndarray | None:
        """The binaural interaction component: the binaural response minus the sum of the
        left and right responses, abr_binaural - abr_sum (see Fig 1 C-F of J Neurosci
        38(40):8563). None if a condition is missing."""
        if not self.has_bic:
            return None
        return self.traces["B"] - self.sum_lr


def slot_responses(
    trace: np.ndarray, sample_rate: float, slot_starts, window: float, trace_start: float
) -> np.ndarray:
    """Cut a sweep into one response for each slot.

    Parameters
    ----------
    trace : the sweep, sampled at sample_rate
    slot_starts : the nominal start of each slot, in seconds of sweep time
    window : the length of each response, in seconds (the slot interval)
    trace_start : the sweep time of the first sample of trace, in seconds

    Returns an array (n_slots, n_points). Each response starts PRESTIMULUS_S before
    its slot. Raises ValueError if a slot is not inside the trace.
    """
    n_points = int(round(window * sample_rate))
    slots = np.empty((len(slot_starts), n_points))
    for k, start in enumerate(slot_starts):
        first = int(round((start - PRESTIMULUS_S - trace_start) * sample_rate))
        if first < 0 or first + n_points > len(trace):
            raise ValueError(
                f"Slot {k:d} (start {start:g} s) is not inside the recorded sweep "
                f"({len(trace) / sample_rate:g} s)"
            )
        slots[k] = trace[first : first + n_points]
    return slots


def average_by_condition(slots: np.ndarray, conditions) -> tuple[dict, dict]:
    """Mean response and number of slots for each condition present."""
    conditions = np.asarray(conditions)
    means = {}
    counts = {}
    for condition in CONDITIONS:
        selected = conditions == condition
        if np.any(selected):
            means[condition] = np.mean(slots[selected], axis=0)
            counts[condition] = int(np.sum(selected))
    return means, counts


def check_bic_trial(trial: dict):
    if trial.get("ear_mode") not in BIC_MODES or "conditions" not in trial:
        raise ValueError("This is not a binaural (BIC) recording")


def trial_condition_means(trial: dict, hpf=None, lpf=None) -> tuple[dict, dict]:
    """The mean response to each condition within one trial, in V at the amplifier output.

    trial is as returned by abr_files.read_trial, which has removed the delay before
    the stimulus (all but 1 ms of it), so the sweep time of its first sample is that
    delay, as read_abr_file computes it.
    """
    check_bic_trial(trial)
    tags = trial["conditions"]
    sample_rate = trial["record_frequency"]
    stimuli = trial["protocol"]["stimuli"]
    trace_start = int((float(stimuli["delay"]) - 1e-3) * sample_rate) / sample_rate
    sweep = abr_files._filter_and_resample(trial["data"], sample_rate, hpf, lpf)
    slots = slot_responses(
        sweep, abr_files.RESAMPLED_RATE, tags["slot_starts"], float(stimuli["interval"]), trace_start
    )
    return average_by_condition(slots, tags["slot_conditions"])


def stimulus_info(trial: dict) -> dict:
    """What was presented, from the tags of a trial."""
    check_bic_trial(trial)
    tags = trial["conditions"]
    return {
        "ear_mode": trial["ear_mode"],
        "itd": float(tags["itd"]),
        "dbspl_left": float(tags["dbspl_left"]),
        "dbspl_right": float(tags["dbspl_right"]),
        "frequency": float(tags["frequency"]),
        "attenuation": [float(a) for a in tags["attenuation"]],
        "alternating": bool(np.any(np.asarray(tags["slot_polarities"]) < 0)),
        "interval": float(trial["protocol"]["stimuli"]["interval"]),
    }


def stimulus_label(info: dict) -> str:
    """A short description, such as '80/70 dB SPL, 8 kHz, ITD 0.20 ms'."""
    left, right = info["dbspl_left"], info["dbspl_right"]
    level = f"{left:g} dB SPL" if left == right else f"{left:g}/{right:g} dB SPL (L/R)"
    stimulus = f"{info['frequency'] / 1000.0:g} kHz" if info["frequency"] > 0 else "click"
    return f"{level}, {stimulus}, ITD {info['itd'] * 1000.0:.2f} ms"


def stimulus_key(info: dict) -> tuple:
    """What identifies a stimulus: trials with the same key are of the same stimulus."""
    return (info["frequency"], info["dbspl_left"], info["dbspl_right"], info["itd"])


def wave_info(run: abr_files.Run) -> dict:
    """The stimulus of each wave (stimulus set) in a run, as wave: info. Reads one file each."""
    infos = {}
    for wave in run.waves:
        path = next(path for (w, _), path in sorted(run.files.items()) if w == wave)
        infos[wave] = stimulus_info(abr_files.read_trial(path))
    return infos


def compute_bic(
    run: abr_files.Run, hpf=None, lpf=None, progress: Callable | None = None
) -> dict:
    """Average the responses of each condition, for each stimulus of a run.

    Returns {wave: BICResponses}, with one entry for each stimulus (frequency, levels and
    ITD), under the number of the first wave that has it. An interleaved run has all of
    the conditions in each wave. When the conditions were presented one at a time
    (monaural_left, monaural_right, binaural, bic_sequential) they are in different waves
    of the same stimulus, and are combined here. The trials of each condition are
    averaged with equal weight; n_trials is the largest number of trials of a condition.
    """
    sums: dict = {}  # group: condition: sum of the trial means
    n_trials: dict = {}  # group: condition: number of trials
    counts: dict = {}  # group: condition: number of slot responses
    infos: dict = {}
    group_of: dict = {}  # stimulus key: first wave with it
    for count, ((wave, _), path) in enumerate(sorted(run.files.items())):
        trial = abr_files.read_trial(path)
        means, n_slots = trial_condition_means(trial, hpf, lpf)
        info = stimulus_info(trial)
        group = group_of.setdefault(stimulus_key(info), wave)
        if group not in sums:
            sums[group], n_trials[group], counts[group], infos[group] = {}, {}, {}, info
        for condition, mean in means.items():
            sums[group][condition] = sums[group].get(condition, 0.0) + mean
            n_trials[group][condition] = n_trials[group].get(condition, 0) + 1
            counts[group][condition] = counts[group].get(condition, 0) + n_slots[condition]
        if progress is not None:
            progress(count + 1, len(run.files))
    responses = {}
    for group in sums:
        n_points = len(next(iter(sums[group].values())))
        t_ms = (np.arange(n_points) / abr_files.RESAMPLED_RATE - PRESTIMULUS_S) * 1000.0
        traces = {
            c: abr_files._to_microvolts(total / n_trials[group][c])
            for c, total in sums[group].items()
        }
        responses[group] = BICResponses(
            t_ms, traces, counts[group], max(n_trials[group].values()), infos[group]
        )
    return responses


def load_bic_run(path: Path, hpf=None, lpf=None, progress: Callable | None = None) -> dict:
    """Find the run that a trial file belongs to, and average it. See compute_bic."""
    return compute_bic(abr_files.find_run(Path(path)), hpf, lpf, progress)
