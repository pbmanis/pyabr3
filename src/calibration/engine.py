"""
Speaker, microphone and noise floor procedures, ported from calibrations.m in the
no-globals branch of ABR4.

No GUI is used here: results are passed to callbacks, so that the procedures can
run in a thread and be tested with mock hardware.

The .cal files are MATLAB files with the fields in the same order as those written
by MATLAB, because read_calibration.py reads the fields by position.

The current calibration is written to <cal_dir>/<name>.cal, as in MATLAB. A copy goes
in <cal_dir>/calibration_history. MATLAB named that copy with the date only, so a second
calibration on one day replaced the first. The name here also includes the time.
"""

import datetime
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.io
from pylibrary.tools import cprint as CP

from src.calibration import calc
from src.calibration.hardware import CalibrationError, CalibrationStopped

TONE_VOLTAGE = 10.0  # CALIBRATION.SPLCAL.ToneVoltage (hardware_initialization.m)
CHECK_SPL = 75.0  # level to test for correct calibration at all frequencies
SPEAKER_RECORD_DURATION = 0.5  # seconds
SPEAKER_RECORD_FACTOR = 1.5  # the recording is this much longer than the tone
SPEAKER_RISE_FALL = 5.0  # msec
SPEAKER_OCTAVE_FRACTION = 5  # measurement band is +/- 1/5 octave
SPEAKER_STEEPNESS = 0.9
MIC_RECORD_DURATION = 1.0  # seconds
MIC_TEST_FREQ = 5000.0  # for the daq channel, away from the calibrator frequency
MIC_NOMINAL_FREQ = 1000.0
MIC_STEEPNESS = 0.8
NOISE_RECORD_DURATION = 10.0  # seconds
NOISE_RECORD_FACTOR = 1.5
NOISE_TONE_FREQ = 1000.0  # the tone has zero amplitude: this only sets up the output
STOPBAND_ATTENUATION = 60.0
HISTORY_DIR = "calibration_history"
NOTCH_BANDS = [(39000.0, 41000.0), (19500.0, 20500.0)]  # power supply noise, in this order

_MF_KHZ = [87, 84, 82, 78, 76, 72, 68, 64, 60, 56, 52, 48, 44, 42, 38, 36, 34, 32, 30, 28, 26,
           24, 22, 18, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3.2, 2.4, 2, 1.6, 1.2, 1]  # fmt: skip
_ES_KHZ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 22, 24, 26, 28, 30, 32, 36, 38, 42, 44, 48,
           52, 56, 60, 64]  # fmt: skip
_MF_SETTINGS = {"freqs_khz": _MF_KHZ, "attn": 20.0, "maxtones": 110.0}
_ES_SETTINGS = {"freqs_khz": _ES_KHZ, "attn": 0.0, "maxtones": 83.9}

# From abr4.m and calibrations.m. EC1, ES1 and MF1 are the names used in no-globals. The
# longer names, from the main branch, use the settings of the same speaker family.
SPEAKERS = {
    "MF1": _MF_SETTINGS,
    "EC1": _ES_SETTINGS,
    "ES1": _ES_SETTINGS,
    "EC1#4101 closed field": _ES_SETTINGS,
    "ES1#3539 free field": _ES_SETTINGS,
    "MF1#1013 free field": _MF_SETTINGS,
    "MF1#1956 closed field": _MF_SETTINGS,
    "FF1#1013 free field (dome)": _MF_SETTINGS,
}

# manufacturer's calibration information, reported after a microphone calibration
MICROPHONES = {
    "7012#39279": "Manufacturer, June 5, 2008: 1/2\" 7012 mic: 16.22 mV/Pa, -35.8 dB re 1V/Pa",
    "7016#9945": "Manufacturer, June 3, 2010: 1/4\" 7016 mic: 3.55 mV/Pa, -49.0 dB re 1V/Pa",
    "7016#10252": "Manufacturer, 2018: 1/4\" 7016 mic: 3.85 mV/Pa, -48.3 dB re 1V/Pa",
}


@dataclass
class SpeakerSettings:
    speaker: str
    microphone: str
    freqs: np.ndarray  # Hz
    attn: float  # programmable attenuator setting, dB
    ref_spl: float  # CALIBRATION.SPLCAL.maxtones
    gain: float = 20.0  # microphone preamplifier, dB
    cal_dir: Path = field(default_factory=Path)
    check: bool = False  # True: check an existing calibration at CHECK_SPL


@dataclass
class MicSettings:
    microphone: str
    ref_db: float = 94.0  # calibrator level
    gain: float = 20.0  # microphone preamplifier, dB
    cal_dir: Path = field(default_factory=Path)
    test_mode: bool = False  # use a synthetic signal, and do not save


def default_log(message: str):
    CP.cprint("g", message)


def verify_message(cmd: str, speaker: str, mic: str, gain: float, driver_attn: float, attn) -> str:
    """Text of the manual settings check (verifyConfiguration.m), with the attenuators added."""
    lines = [
        "Please check the following:\n",
        "Microphone:",
        f"   Microphone type = {mic:s}",
        f"   Microphone Preamp Gain is {gain:.0f} dB",
    ]
    if cmd == "microphone":
        lines.append("   Calibrator Standard is set to 94 dBSPL and ON")
    elif cmd == "microphone104":
        lines.append("   Calibrator Standard is set to 104 dBSPL and ON")
    lines += [
        "\nSpeaker:",
        f"   Speaker is {speaker:s}",
        f"   Speaker driver attenuation is {driver_attn:.0f} dB",
    ]
    if attn is not None:
        lines.append(f"   Programmable attenuator will be set to {attn:.1f} dB")
    lines.append("\nAre your settings correct?")
    return "\n".join(lines)


def load_mat_struct(path: Path, name: str) -> dict:
    """Read a structure from a .cal file into a dict, keyed by the MATLAB field names."""
    if not Path(path).is_file():
        raise CalibrationError(f"Calibration file not found: {path!s}")
    data = scipy.io.loadmat(path, appendmat=False, squeeze_me=True)[name]
    return {k: data[k].item() for k in data.dtype.names}


def save_mat_struct(path: Path, name: str, data: dict):
    scipy.io.savemat(path, {name: data}, appendmat=False, format="5", long_field_names=True)


def save_with_history(cal_dir: Path, stem: str, name: str, data: dict, log: Callable):
    """Write the current calibration file, and a copy with the date and time in its name."""
    history = cal_dir / HISTORY_DIR
    history.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    dated = history / f"{stem:s}_{stamp:s}.cal"
    current = cal_dir / f"{stem:s}.cal"
    save_mat_struct(dated, name, data)
    save_mat_struct(current, name, data)
    log(f"Saved in {current!s} and {dated!s}")


def _column(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, 1)


def _date_strings() -> tuple:
    now = datetime.datetime.now()
    return now.strftime("%d-%b-%Y"), now.strftime("%d-%b-%Y %H:%M:%S")


def _check_stop(stop: threading.Event):
    if stop.is_set():
        raise CalibrationStopped()


def calibrate_speaker(
    hw,
    settings: SpeakerSettings,
    stop: threading.Event,
    log: Callable = default_log,
    on_step: Callable | None = None,
):
    """Measure the speaker at each frequency, or check an existing calibration.

    Returns the dictionary saved in the file, or None if the user stopped the run.
    The attenuator is left at maximum attenuation.
    """
    try:
        return _calibrate_speaker(hw, settings, stop, log, on_step)
    except CalibrationStopped:
        log("Calibration stopped: nothing was saved.")
        return None
    finally:
        hw.set_attn(120.0)


def _calibrate_speaker(hw, settings, stop, log, on_step):
    speaker, cal_dir = settings.speaker, Path(settings.cal_dir)
    spkr_freq = np.asarray(settings.freqs, dtype=float)
    nfreq = len(spkr_freq)
    mic = load_mat_struct(cal_dir / f"microphone_{settings.microphone:s}.cal", "MIC")
    if float(mic["Gain"]) != float(settings.gain):
        raise CalibrationError(
            f"The microphone calibration used a preamp gain of {float(mic['Gain']):.1f} dB, "
            f"but the gain is set to {settings.gain:.1f} dB. Use the same gain, "
            "or calibrate the microphone at this gain."
        )
    if float(mic["RefSig"]) != 94.0:
        log(
            f"WARNING: microphone calibration RefSig is {float(mic['RefSig']):.0f} dB. "
            "compute_spl (as in MATLAB) will report levels offset by "
            f"{float(mic['RefSig']) - 94.0:.0f} dB."
        )
    for low, high in NOTCH_BANDS:
        for f in spkr_freq[(spkr_freq >= low) & (spkr_freq <= high)]:
            log(f"WARNING: {f:.0f} Hz is inside the {low:.0f}-{high:.0f} Hz notch filter.")
    cal = None
    if settings.check:
        cal = load_mat_struct(cal_dir / f"frequency_{speaker:s}.cal", "CAL")

    action = "Checking" if settings.check else "Running"
    log(f"{action} Speaker Calibration on {_date_strings()[0]:s}")
    log(
        f"Speaker: {speaker:s} Microphone: {settings.microphone:s} "
        f"Attenuation: {settings.attn:7.1f} dB Reference level: {CHECK_SPL:7.2f} dBSPL"
    )
    log(f"Mic cal date: {mic['Date']!s}  Mic RefSig: {float(mic['RefSig']):7.2f} "
        f"Mic mVPerPa: {float(mic['mVPerPa']):7.2f}")

    results = {k: np.zeros(nfreq) for k in
               ["dbspl_cs", "dbspl_bp", "dbspl_nf", "vrms_cs", "vrms_bp", "vrms_nf", "maxdb"]}  # fmt: skip
    attndB = np.zeros(nfreq) + settings.attn
    duration = SPEAKER_RECORD_DURATION
    fs = hw.sample_freq
    sample_dt = 1.0 / fs
    n_points = int(np.floor(fs * duration * SPEAKER_RECORD_FACTOR))
    trec = np.arange(n_points) * sample_dt
    ts1 = int(np.floor(duration * 0.25 / sample_dt))
    ts2 = int(np.floor(duration * 0.75 / sample_dt))
    window = slice(ts1 - 1, ts2)  # MATLAB ts1:ts2 is 1-based and inclusive
    notches = [calc.design_notch(fs, low, high) for low, high in NOTCH_BANDS]
    log(f"{'F(Hz)':>8s} {'maxfreq':>8s} {'Mic(mV)':>8s} {'BP(mV)':>8s} {'NF(mV)':>8s} "
        f"{'dBWB':>7s} {'dbBP':>7s} {'Attn(dB)':>8s} {'Max dB':>7s} {'dbNF':>7s}")

    def notch_filter(x):
        for sos in notches:
            x = calc.apply_notch(sos, x)
        return x

    for i, freq in enumerate(spkr_freq):
        _check_stop(stop)
        wave, _ = calc.tonepip(
            TONE_VOLTAGE, freq, 0.0, duration * 1000.0, SPEAKER_RISE_FALL, 0, hw.ni_freq, 10, 1, 0
        )
        if settings.check:
            spl_f = calc.spl_at_f(cal["Freqs"], cal["dBSPL_bp"], freq)
            attn = spl_f - CHECK_SPL + settings.attn
            if attn < 0:
                attn = 0.0
        else:
            attn = settings.attn
        attndB[i] = attn

        hw.set_attn(attn)
        ch2 = notch_filter(hw.calstim(wave, n_points, stop))
        hw.set_attn(-1)  # maximum attenuation, for the noise floor measurement
        ch2nf = notch_filter(hw.calstim(wave, n_points, stop))

        bpfreqs = calc.octave_calc(freq, SPEAKER_OCTAVE_FRACTION, fs)
        ybp = calc.bandpass(ch2, bpfreqs, fs, STOPBAND_ATTENUATION, SPEAKER_STEEPNESS)
        ynf = calc.bandpass(ch2nf, bpfreqs, fs, STOPBAND_ATTENUATION, SPEAKER_STEEPNESS)
        results["vrms_bp"][i] = calc.rms(ybp[window])
        results["vrms_nf"][i] = calc.rms(ynf[window])

        amp_cosinor, fr_cosinor = calc.compute_cosinors(freq, trec[window], ch2[window])
        k = int(np.argmax(amp_cosinor))
        mfreq = fr_cosinor[k]
        results["vrms_cs"][i] = amp_cosinor[k] / np.sqrt(2.0)  # convert cosinor to RMS

        bp_psd, bp_freqs = calc.periodogram(ybp, fs)
        nf_psd, nf_freqs = calc.periodogram(ynf, fs)

        results["dbspl_cs"][i] = calc.compute_spl(results["vrms_cs"][i], mic)
        results["dbspl_bp"][i] = calc.compute_spl(results["vrms_bp"][i], mic)
        results["dbspl_nf"][i] = calc.compute_spl(results["vrms_nf"][i], mic)
        results["maxdb"][i] = results["dbspl_bp"][i] + attndB[i]

        log(f"{freq:8.1f} {mfreq:8.1f} {1000 * results['vrms_cs'][i]:8.3f} "
            f"{1000 * results['vrms_bp'][i]:8.3f} {1000 * results['vrms_nf'][i]:8.3f} "
            f"{results['dbspl_cs'][i]:7.1f} {results['dbspl_bp'][i]:7.1f} {attndB[i]:8.1f} "
            f"{results['maxdb'][i]:7.1f} {results['dbspl_nf'][i]:7.1f}")
        if on_step is not None:
            on_step({
                "index": i, "freq": freq, "freqs": spkr_freq, "attn": attndB.copy(),
                "tstim": np.arange(len(wave)) / hw.ni_freq, "wave": wave,
                "trec": trec, "ch2": ch2, "ybp": ybp, "ynf": ynf, "ts1": ts1, "ts2": ts2,
                "bp_freqs": bp_freqs, "bp_psd": bp_psd, "nf_freqs": nf_freqs, "nf_psd": nf_psd,
                "dbspl_bp": results["dbspl_bp"].copy(), "dbspl_nf": results["dbspl_nf"].copy(),
            })  # fmt: skip

    date, date_time = _date_strings()
    common_end = {
        "Gain": float(settings.gain),
        "CalAttn": float(settings.attn),
        "Speaker": speaker,
        "Microphone": settings.microphone,
        "Date": date,
        "DateTime": date_time,  # a string: scipy cannot write a MATLAB datetime object
    }
    start = {
        "RefSPL": float(settings.ref_spl),
        "Freqs": spkr_freq.reshape(1, -1),
        "maxdB": _column(results["maxdb"]),
    }
    if settings.check:
        out = start | {
            "dBSPL": CHECK_SPL,  # the target level, as in MATLAB: not a measurement
            "dBSPL_nf": _column(results["dbspl_nf"]),
            "dBSPL_bp": _column(results["dbspl_bp"]),
            "Vmeas": _column(results["vrms_cs"]),
            "Vmeas_bp": _column(results["vrms_bp"]),
            "attndB": _column(attndB),
        } | common_end
        save_with_history(cal_dir, f"chk75db_{speaker:s}", "CHKCAL", out, log)
        return out

    out = start | {
        "dBSPL": _column(results["dbspl_cs"]),
        "dBSPL_bp": _column(results["dbspl_bp"]),
        "dBSPL_nf": _column(results["dbspl_nf"]),
        "Vmeas": _column(results["vrms_cs"]),
        "Vmeas_bp": _column(results["vrms_bp"]),
    } | common_end
    save_with_history(cal_dir, f"frequency_{speaker:s}", "CAL", out, log)
    return out


def calibrate_microphone(
    hw,
    settings: MicSettings,
    stop: threading.Event,
    log: Callable = default_log,
    on_result: Callable | None = None,
):
    """Measure the microphone sensitivity using the sound level calibrator.

    Returns the MIC dictionary, or None if the user stopped the run.
    """
    try:
        return _calibrate_microphone(hw, settings, stop, log, on_result)
    except CalibrationStopped:
        log("Calibration stopped: nothing was saved.")
        return None
    finally:
        hw.set_attn(120.0)


def _calibrate_microphone(hw, settings, stop, log, on_result):
    mic_name, cal_dir = settings.microphone, Path(settings.cal_dir)
    hw.set_attn(120.0)  # no extraneous sound
    wave, _ = calc.tonepip(TONE_VOLTAGE, MIC_TEST_FREQ, 0.0, 10, 0.0, 0, hw.ni_freq, 10, 1, 0)
    fs = hw.sample_freq
    n_points = int(np.floor(MIC_RECORD_DURATION * fs))
    trec = np.arange(n_points) / fs
    ch2 = hw.calstim(wave, n_points, stop)
    log("Calibration sound recorded")
    if settings.test_mode:
        ch2 = 0.05 * np.sin(trec * 1000 * 2.0 * np.pi)
        ch2 = ch2 - np.mean(ch2)

    fr = np.arange(0.95 * MIC_NOMINAL_FREQ, 1.05 * MIC_NOMINAL_FREQ + 0.5)
    amp_c, fr_c = calc.compute_cosinors(fr, trec, ch2)
    imax_c = int(np.argmax(amp_c))
    ybp = calc.bandpass(ch2, [940.0, 1020.0], fs, STOPBAND_ATTENUATION, MIC_STEEPNESS)
    bp_psd, bp_freqs = calc.periodogram(ybp, fs)
    imax_bp = int(np.argmax(bp_psd))

    date, _ = _date_strings()
    mic = {
        "Gain": float(settings.gain),
        "RefSig": float(settings.ref_db),
        "Vrms": calc.rms(ch2),
        "Vref_c": float(amp_c[imax_c] / np.sqrt(2.0)),  # convert to RMS
        "Vref_bp": calc.rms(ybp),
        "Microphone": mic_name,
        "Date": date,
    }
    mic["dBPerVPa"] = float(calc.compute_dBPerVPa(mic["Vref_bp"], mic))
    mic["mVPerPa"] = float(calc.compute_mVPerPa(mic))

    log(f"MIC Vref (RMS): {mic['Vrms']:12.6f} Vrms(V) (raw trace)")
    log(f"MIC Vref (BP) : {mic['Vref_bp']:12.6f} Vrms(V) at {bp_freqs[imax_bp]:12.3f} Hz (bandpassed)")
    log(f"MIC Vref (cos): {mic['Vref_c']:12.6f} Vrms(V) at {fr_c[imax_c]:12.3f} Hz")
    log(MICROPHONES.get(mic_name, "No manufacturer information for this microphone"))
    log(f"Measured Transfer factor: {mic['mVPerPa']:9.5f} mV/Pa, {mic['dBPerVPa']:5.1f} dB re 1V/Pa")
    log(f"Measured with Mic Amp Gain = {settings.gain:.1f} dB, "
        f"Standard is {settings.ref_db:.1f} dB, bandpassed")

    if on_result is not None:
        on_result({
            "tstim": np.arange(len(wave)) / hw.ni_freq, "wave": wave, "trec": trec, "ch2": ch2,
            "ybp": ybp, "bp_freqs": bp_freqs, "bp_psd": bp_psd, "fr_c": fr_c, "amp_c": amp_c,
        })  # fmt: skip
    if settings.test_mode:
        log("Test mode: microphone calibration was not saved.")
        return mic
    save_with_history(cal_dir, f"microphone_{mic_name:s}", "MIC", mic, log)
    return mic


def measure_noise_floor(
    hw,
    cal_dir: Path,
    stop: threading.Event,
    log: Callable = default_log,
    on_result: Callable | None = None,
):
    """Record the microphone with no stimulus, and write the trace to a .csv file.

    Returns the path of the file, or None if the user stopped the run.
    """
    try:
        return _measure_noise_floor(hw, Path(cal_dir), stop, log, on_result)
    except CalibrationStopped:
        log("Noise floor measurement stopped: nothing was saved.")
        return None
    finally:
        hw.set_attn(120.0)


def _measure_noise_floor(hw, cal_dir, stop, log, on_result):
    fs = hw.sample_freq
    n_points = int(np.floor(fs * NOISE_RECORD_DURATION * NOISE_RECORD_FACTOR))
    trec = np.arange(n_points) / fs
    wave, _ = calc.tonepip(
        0.0, NOISE_TONE_FREQ, 0.0, NOISE_RECORD_DURATION * 1000.0, SPEAKER_RISE_FALL, 0,
        hw.ni_freq, 10, 1, 0,
    )  # fmt: skip
    hw.set_attn(-1)
    ch2nf = hw.calstim(wave, n_points, stop)
    log("Recording finished")
    psd, freqs = calc.noise_spectrum(ch2nf, fs)
    if on_result is not None:
        on_result({"trec": trec, "ch2nf": ch2nf, "psd": psd, "freqs": freqs})
    stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    filename = cal_dir / f"Noise_Measure_{stamp:s}.csv"
    np.savetxt(
        filename,
        np.column_stack((trec, ch2nf)),
        delimiter=",",
        header="Time,Voltage(V)",
        comments="",
        fmt="%.15g",
    )
    log(f"Wrote to csv: {filename!s}")
    return filename
