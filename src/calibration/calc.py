"""
Calculations for the speaker and microphone calibrations.

Each function is a direct port of the MATLAB function of the same name in ABR4
(soundfuncs.m, tonepip.m, cosinor.m, and the subfunctions in calibrations.m),
from the no-globals branch.
The arithmetic is intentionally unchanged, so that results can be validated
against the MATLAB program.
"""

import numpy as np
import scipy.signal


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def compute_spl(mic_vrms_V, mic: dict):
    """SPL from a microphone voltage (rms, V), given a MIC structure (soundfuncs.m)."""
    return mic["RefSig"] - mic["Gain"] + 20.0 * np.log10(mic_vrms_V * 1e3 / mic["mVPerPa"])


def compute_dBPerVPa(mic_vrms_V: float, mic: dict) -> float:
    """Microphone sensitivity, stated relative to 1 Pa (soundfuncs.m)."""
    return 20.0 * np.log10(mic_vrms_V) - mic["Gain"] - (mic["RefSig"] - 94.0)


def compute_mVPerPa(mic: dict) -> float:
    return 1000.0 * 10.0 ** (mic["dBPerVPa"] / 20.0)


def octave_calc(f0: float, oct: float, fs: float) -> list:
    """Low and high frequencies of the 1/oct octave band around f0, limited by Nyquist."""
    octfrac = 2.0 ** (1.0 / oct)
    flow = f0 / octfrac
    fhigh = f0 * octfrac
    if fhigh > fs / 2.0:
        fhigh = np.floor(fs / 2.2)
    return [flow, fhigh]


def spl_at_f(freqs, spls, f):
    """Linear interpolation of SPL on log2(frequency); NaN outside the measured range.

    MATLAB interp1 accepts unsorted x, so the points are sorted here for np.interp.
    """
    freqs = np.asarray(freqs, dtype=float).ravel()
    spls = np.asarray(spls, dtype=float).ravel()
    order = np.argsort(freqs)
    return np.interp(
        np.log2(f), np.log2(freqs[order]), spls[order], left=np.nan, right=np.nan
    )


def tonepip(
    amp: float,
    freq: float,
    delay: float,
    duration: float,
    rf: float,
    phase0: float,
    sampfreq: float,
    ipi: float,
    np_pips: int,
    alternate: int,
):
    """Tone pip with sin^2 ramps. Times in msec, freq in Hz (tonepip.m).

    Returns the waveform and the sample clock in usec. As in tonepip.m, both
    branches of the alternate test use sign = -1, so "alternate" has no effect.
    """
    clock = 1000 / sampfreq
    uclock = 1000 * clock
    phi = 2 * np.pi * phase0 / 360

    jdur = int(np.floor((duration - 2 * rf) / clock))
    np_ipi = int(np.floor(ipi / clock))
    nfilter_points = int(np.floor(rf / clock))
    npts = jdur + 2 * nfilter_points
    fil = np.ones(npts)
    if nfilter_points > 0:
        fo = 1 / (4 * rf)
        rise = np.sin(2 * np.pi * fo * np.arange(nfilter_points) * clock) ** 2
        fil[:nfilter_points] = rise
        fil[npts - nfilter_points :] = rise[::-1]

    tfil = np.arange(npts) * clock
    ws = amp * np.sin(phi + 2 * np.pi * freq / 1000 * tfil)
    wf = ws * fil if rf > 0.0 else ws
    nwf = len(wf)

    ipi_pts = int(np.floor(ipi / clock))
    jd = int(np.floor(delay / clock))
    if np_pips > 1:
        w = np.zeros(jd + np_pips * np_ipi)
    elif jd + nwf < np_ipi:
        w = np.zeros(np_ipi)
    else:
        w = np.zeros(jd + nwf)

    sign = -1
    for i in range(np_pips):
        j0 = jd + i * ipi_pts
        w[j0 : j0 + nwf] = sign * wf
    return w[: len(w) - jd], uclock


def cosinor(t: np.ndarray, y: np.ndarray, w: float):
    """Least squares fit of M + beta*cos(wt) + gamma*sin(wt) (cosinor.m).

    Solves the same normal equations as cosinor.m. Only Amp and M are used by the
    calibration, so the confidence interval calculations are not ported.
    """
    n = len(t)
    x = np.cos(w * t)
    z = np.sin(w * t)
    sx, sz = np.sum(x), np.sum(z)
    sxz = np.sum(x * z)
    a = np.array([[n, sx, sz], [sx, np.sum(x * x), sxz], [sz, sxz, np.sum(z * z)]])
    b = np.array([np.sum(y), np.sum(x * y), np.sum(z * y)])
    mesor, beta, gamma = np.linalg.solve(a, b)
    return float(np.sqrt(beta**2 + gamma**2)), float(mesor)


def compute_cosinors(fr, trec: np.ndarray, ch2: np.ndarray):
    """Cosinor amplitude at each frequency in fr (calibrations.m)."""
    fr = np.atleast_1d(np.asarray(fr, dtype=float))
    amp = np.array([cosinor(trec, ch2, 2.0 * np.pi * f)[0] for f in fr])
    return amp, fr


def design_notch(fs: float, low: float, high: float) -> np.ndarray:
    """designfilt('bandstopiir', 'FilterOrder', 20, ...): default Butterworth.

    A MATLAB bandstop order of 20 corresponds to N=10 in scipy.signal.butter.
    """
    return scipy.signal.butter(10, [low, high], btype="bandstop", fs=fs, output="sos")


def apply_notch(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Causal filtering, as with MATLAB filter(); not zero phase."""
    return scipy.signal.sosfilt(sos, x)


def bandpass(
    x: np.ndarray, wpass, fs: float, stopband_attenuation: float = 60.0, steepness: float = 0.85
) -> np.ndarray:
    """Equivalent of bandpass(x, wpass, fs, 'StopbandAttenuation', a, 'Steepness', s).

    Follows the documented MATLAB design: a minimum-order Kaiser FIR filter (chosen
    by 'auto' for signals of this length), with delay compensation. Stopband edges
    are s*f1 and f2 + (1-s)*(fN - f2). This is the one calculation that cannot be
    an exact transcription, because the MATLAB source is not available. It needs to
    be compared with MATLAB output on the same data.
    """
    fnyq = fs / 2.0
    f1, f2 = wpass
    stop1 = steepness * f1
    stop2 = f2 + (1.0 - steepness) * (fnyq - f2)
    width = min(f1 - stop1, stop2 - f2)
    numtaps, beta = scipy.signal.kaiserord(stopband_attenuation, width / fnyq)
    numtaps |= 1  # odd length, so the delay is a whole number of samples
    taps = scipy.signal.firwin(
        numtaps,
        [(stop1 + f1) / 2.0, (f2 + stop2) / 2.0],
        window=("kaiser", beta),
        pass_zero=False,
        fs=fs,
    )
    delay = (numtaps - 1) // 2
    return scipy.signal.oaconvolve(x, taps, mode="full")[delay : delay + len(x)]


def periodogram(x: np.ndarray, fs: float):
    """periodogram(x, rectwin(n), n, fs): returns (psd, freqs), as in MATLAB."""
    freqs, psd = scipy.signal.periodogram(
        x, fs=fs, window="boxcar", nfft=len(x), detrend=False, scaling="density"
    )
    return psd, freqs


def noise_spectrum(x: np.ndarray, fs: float):
    """Power spectrum of a noise recording, returned as (psd, freqs).

    MATLAB uses pspectrum here. This is for display only: its result is not saved,
    and the plot will differ in detail from pspectrum.
    """
    freqs, psd = scipy.signal.welch(
        x, fs=fs, window=("kaiser", 38.0), nperseg=min(len(x), 65536), detrend=False
    )
    return psd, freqs
