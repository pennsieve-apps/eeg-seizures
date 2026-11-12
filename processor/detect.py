"""
The line-length seizure detector and the event post-processing shared with the model path.

Line length (Esteller et al. 2001): the sum of absolute sample-to-sample differences in a
window, a cheap measure that grows with both amplitude and frequency, which is why rhythmic
ictal discharges stand out. Per channel, it is computed on the band-passed 256 Hz signal in
2 s windows every 1 s, log-transformed and normalised by a trailing 5-minute baseline (median
and MAD, so the detector adapts to each channel and to slow changes of state). The channel
z-scores are combined into one score per second (mean of the top-k channels, so a focal
seizure on a few channels counts), smoothed, and thresholded with hysteresis. Events then go
through SzCORE-style post-processing: events shorter than MIN_DURATION_S are dropped, events
closer than MERGE_GAP_S are merged, events longer than MAX_DURATION_S are split.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

FS = 256                 # SzCORE: every recording is analysed at 256 Hz
LL_WINDOW_S = 2.0        # line-length window
STEP_S = 1               # one value per second
MAD_TO_SD = 1.4826
MIN_SCALE = 0.02         # floor of the baseline spread, log10 units (a flat channel must not explode)
MIN_BASELINE_S = 60      # fewer valid seconds than this in the trailing window: use the recording's first minutes


def bandpass_sos(fs: float, low: float, high: float | None):
    from scipy.signal import butter

    if high is not None and high < fs / 2 * 0.95:
        return butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return butter(4, low, btype="highpass", fs=fs, output="sos")


def resample_ratio(fs: float, target: int = FS) -> tuple[int, int]:
    """(up, down) for scipy.signal.resample_poly, from fs to target Hz."""
    f = Fraction(target / fs).limit_denominator(1000)
    return f.numerator, f.denominator


def to_256(data: np.ndarray, fs: float, sos) -> np.ndarray:
    """Band-pass (zero phase) and resample a (channels, samples) block to 256 Hz."""
    from scipy.signal import resample_poly, sosfiltfilt

    x = sosfiltfilt(sos, data, axis=-1) if sos is not None else data
    if abs(fs - FS) > 1e-6:
        up, down = resample_ratio(fs)
        x = resample_poly(x, up, down, axis=-1)
    return x


def line_length_per_second(x: np.ndarray, t0_s: float, first_s: int, last_s: int) -> np.ndarray:
    """
    Line length (µV/s) of each channel of x (256 Hz, first sample at t0_s seconds) for the
    seconds first_s..last_s-1: second k is measured over [k - 0.5, k + 1.5) s, a 2 s window
    centred on the middle of second k, clipped to the data. Returns (channels, seconds).
    """
    n = x.shape[1]
    d = np.abs(np.diff(x, axis=1))
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)  # cs[:, i] = sum d[:, :i]
    ks = np.arange(first_s, last_s)
    a = np.clip(np.round((ks - 0.5 - t0_s) * FS).astype(int), 0, n - 1)
    b = np.clip(np.round((ks + 1.5 - t0_s) * FS).astype(int), 0, n - 1)
    span = np.maximum(b - a, 1)
    return (cs[:, b] - cs[:, a]) / (span / FS)


def normalise(ll: np.ndarray, valid: np.ndarray, baseline_s: int) -> np.ndarray:
    """
    z-scores of log10 line length against a trailing baseline: for second k, the median and MAD
    of the channel's previous `baseline_s` seconds (seconds in bad segments left out). Until
    MIN_BASELINE_S valid seconds have been seen, the median/MAD of the first `baseline_s`
    valid seconds are used. ll: (channels, seconds); valid: (seconds,) bool. NaN where invalid.
    """
    import pandas as pd

    logll = np.log10(np.maximum(ll, 1e-3))
    logll[:, ~valid] = np.nan
    z = np.full_like(logll, np.nan)
    for c in range(logll.shape[0]):
        s = pd.Series(logll[c])
        med = s.rolling(baseline_s, min_periods=MIN_BASELINE_S).median().shift(1)
        first = s.dropna().iloc[:baseline_s]
        med0 = float(first.median()) if len(first) else np.nan
        med = med.fillna(med0)
        dev = (s - med).abs()
        mad = dev.rolling(baseline_s, min_periods=MIN_BASELINE_S).median().shift(1)
        mad0 = float((first - med0).abs().median()) if len(first) else np.nan
        mad = mad.fillna(mad0)
        scale = np.maximum(MAD_TO_SD * mad.to_numpy(), MIN_SCALE)
        z[c] = (s.to_numpy() - med.to_numpy()) / scale
    return z


def combine(z: np.ndarray, top_k: int, smooth_s: int) -> tuple[np.ndarray, np.ndarray]:
    """(score per second: mean of the top_k channel z-scores, smoothed by a centred moving mean
    of smooth_s seconds; rank order of channels per second is not kept). NaN where no channel is
    valid. Also returns the per-channel z (unchanged) for naming the channels of an event."""
    k = max(1, min(top_k, z.shape[0]))
    zz = np.where(np.isnan(z), -np.inf, z)
    top = -np.sort(-zz, axis=0)[:k]
    top = np.where(np.isfinite(top), top, np.nan)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmean(top, axis=0)
    if smooth_s > 1:
        import pandas as pd

        score = np.array(pd.Series(score).rolling(smooth_s, center=True, min_periods=1).mean(), dtype=float)
    return score, z


def hysteresis(score: np.ndarray, on: float, off: float) -> list[tuple[int, int]]:
    """[start, end) second indices where score rises to >= on and stays >= off. NaN ends an event."""
    events = []
    inside = False
    start = 0
    for i, v in enumerate(score):
        if not inside:
            if v >= on:
                inside, start = True, i
        elif not (v >= off):
            events.append((start, i))
            inside = False
    if inside:
        events.append((start, len(score)))
    return events


def postprocess(events: list[tuple[float, float]], min_duration_s: float, merge_gap_s: float,
                max_duration_s: float, drop_first: bool = True) -> list[tuple[float, float]]:
    """
    SzCORE-style post-processing of (start, end) events in seconds, in this order:
    drop events shorter than min_duration_s, merge events separated by less than merge_gap_s,
    split events longer than max_duration_s into consecutive pieces (as timescoring does
    when it scores). drop_first=False merges before dropping (kept for the threshold sweep).
    """
    ev = sorted((a, b) for a, b in events if not drop_first or b - a >= min_duration_s)
    merged: list[list[float]] = []
    for a, b in ev:
        if merged and a - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not drop_first:
        merged = [m for m in merged if m[1] - m[0] >= min_duration_s]
    out = []
    for a, b in merged:
        while max_duration_s > 0 and b - a > max_duration_s:
            out.append((a, a + max_duration_s))
            a += max_duration_s
        out.append((a, b))
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))
