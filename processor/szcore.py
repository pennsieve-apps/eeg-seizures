"""
SzCORE event files and their scoring (Dan et al. 2024, Epilepsia, doi:10.1111/epi.18113).

The TSV is written exactly as SzCORE's `Annotations.saveTsv` (epilepsy2bids 0.0.7) writes it:
header `onset duration eventType confidence channels dateTime recordingDuration`, onset /
duration / confidence / recordingDuration with two decimals, channels comma-separated, `dateTime`
as `%Y-%m-%d %H:%M:%S` (or `n/a`), and a single `bckg` row spanning the recording when there
is no seizure event.

Scoring follows SzCORE's evaluation (szcore-evaluation 0.0.7): reference and hypothesis are
turned into 1 Hz masks (`Annotations.getMask`: samples int(onset)..int(onset + duration)) and
scored with timescoring's `SampleScoring` (1 Hz) and `EventScoring` with SzCORE's defaults
(30 s tolerance before, 60 s after, any overlap, events split at 5 min, events closer than 90 s
merged). Pooled scores add TP, FP, reference events and duration over recordings (SzCORE's
"cumulated"); subject means average the per-subject scores (SzCORE's default leaderboard rule).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

HEADER = ["onset", "duration", "eventType", "confidence", "channels", "dateTime", "recordingDuration"]
SCORE_FS = 1
EVENT_PARAMS = {"toleranceStart": 30.0, "toleranceEnd": 60.0, "minOverlap": 0.0,
                "maxEventDuration": 300.0, "minDurationBetweenEvents": 90.0}


def write_tsv(path: Path, events: list[dict], recording_duration: float, start: datetime | None) -> None:
    """events: dicts with onset, duration, confidence (float or None), channels (list)."""
    dt = start.strftime("%Y-%m-%d %H:%M:%S") if start is not None else "n/a"
    lines = ["\t".join(HEADER)]
    for e in events:
        conf = "{:.2f}".format(e["confidence"]) if e.get("confidence") is not None else "n/a"
        chans = ",".join(e["channels"]) if e.get("channels") else "n/a"
        lines.append("\t".join(["{:.2f}".format(e["onset"]), "{:.2f}".format(e["duration"]), "sz", conf, chans, dt,
                                "{:.2f}".format(recording_duration)]))
    if not events:
        lines.append("\t".join(["{:.2f}".format(0), "{:.2f}".format(recording_duration), "bckg", "n/a", "n/a", dt,
                                "{:.2f}".format(recording_duration)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask(events: list[tuple[float, float]], recording_duration: float, fs: int = SCORE_FS) -> np.ndarray:
    """1 Hz seizure mask as SzCORE's Annotations.getMask builds it. events: (onset, duration)."""
    m = np.zeros(int(recording_duration * fs), dtype=bool)
    for onset, duration in events:
        m[int(onset * fs): int((onset + duration) * fs)] = True
    return m


def score(reference: list[tuple[float, float]], hypothesis: list[tuple[float, float]],
          recording_duration: float) -> dict:
    """timescoring sample and event scores of one recording. Both lists are (onset, duration)."""
    from timescoring import scoring
    from timescoring.annotations import Annotation

    ref = Annotation(mask(reference, recording_duration), SCORE_FS)
    hyp = Annotation(mask(hypothesis, recording_duration), SCORE_FS)
    out = {}
    sample = scoring.SampleScoring(ref, hyp)
    event = scoring.EventScoring(ref, hyp, scoring.EventScoring.Parameters(**EVENT_PARAMS))
    for scope, s in (("sample", sample), ("event", event)):
        out[scope] = counts(s.tp, s.fp, s.refTrue, len(s.ref.mask) / s.ref.fs)
    return out


def counts(tp: float, fp: float, ref_true: float, duration_s: float) -> dict:
    """Metrics from counts, with timescoring's definitions (NaN when undefined)."""
    tp, fp, ref_true = float(tp), float(fp), float(ref_true)
    sens = tp / ref_true if ref_true > 0 else float("nan")
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + (ref_true - tp)) if ref_true + fp > 0 else float("nan")
    days = duration_s / 3600 / 24
    return {"sensitivity": sens, "precision": prec, "f1": f1, "fp_per_day": fp / days if days > 0 else float("nan"),
            "tp": tp, "fp": fp, "n_reference": ref_true, "hours": duration_s / 3600}


def pooled(results: list[dict], scope: str) -> dict:
    """SzCORE 'cumulated': counts and durations added over recordings."""
    tp = sum(r[scope]["tp"] for r in results)
    fp = sum(r[scope]["fp"] for r in results)
    ref = sum(r[scope]["n_reference"] for r in results)
    dur = sum(r[scope]["hours"] for r in results) * 3600
    return counts(tp, fp, ref, dur)


def subject_mean(results_by_subject: dict[str, list[dict]], scope: str) -> dict:
    """SzCORE's default aggregation: pool each subject's recordings, then average the subjects'
    metrics (NaN-aware, as np.nanmean)."""
    per = [pooled(rs, scope) for rs in results_by_subject.values() if rs]
    out = {}
    for m in ("sensitivity", "precision", "f1", "fp_per_day"):
        vals = [p[m] for p in per if np.isfinite(p[m])]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    out["n_subjects"] = float(len(per))
    return out
