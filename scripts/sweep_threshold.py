"""
Tune the line-length detector on annotated data: compute each recording's per-second line
length once (cached), then score every combination of the post-feature parameters with
timescoring (SzCORE event scoring), pooled over the recordings of the chosen subjects.

Usage (from the repo root, with the processor's dependencies installed):
  python -m scripts.sweep_threshold --input <BIDS or eeg-prep folder> --cache <dir> \
      [--subjects 00,01,...] [--lowpass 25] [--out sweep.csv]

Used to choose the default THRESHOLD on a tuning subset (see README, "Validation").
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--lowpass", default="25")
    ap.add_argument("--highpass", default="1")
    ap.add_argument("--out", default="")
    ap.add_argument("--thresholds", default="2,2.5,3,3.5,4,4.5,5,5.5,6,7,8")
    ap.add_argument("--top-k", default="2,4,6")
    ap.add_argument("--baseline", default="300")
    ap.add_argument("--hysteresis", default="0.75")
    ap.add_argument("--min-duration", default="10")
    ap.add_argument("--order", default="drop_first")
    a = ap.parse_args()
    os.environ["LOWPASS_HZ"] = a.lowpass
    os.environ["HIGHPASS_HZ"] = a.highpass
    from processor import detect as D
    from processor import eeg as E
    from processor import main as MAIN
    from processor import szcore as S

    p = MAIN.env_params()
    p["score"] = False
    subjects = {s.strip() for s in a.subjects.split(",") if s.strip()}
    recs, _, _ = E.find_recordings(Path(a.input))
    recs = [r for r in recs if r.labels_known and (not subjects or r.subject in subjects)]
    cache = Path(a.cache) / f"hp{a.highpass}_lp{a.lowpass}"
    cache.mkdir(parents=True, exist_ok=True)
    data = []
    for r in recs:
        f = cache / f"{r.recording_id}.npz"
        if not f.exists():
            res = MAIN.process_recording(r, p, None)
            np.savez(f, ll=res["ll"], valid=res["valid"], duration=res["duration"])
            print(f"cached {r.recording_id}", file=sys.stderr, flush=True)
        z = np.load(f)
        data.append((r, z["ll"], z["valid"], float(z["duration"])))
    print(f"{len(data)} recordings, {sum(len(r.seizures) for r, *_ in data)} seizures, "
          f"{sum(d for *_, d in data) / 3600:.1f} h", file=sys.stderr)
    rows = []
    fl = lambda s: [float(x) for x in s.split(",")]  # noqa: E731
    zcache = {}
    for base, k, hyst, mind in itertools.product(fl(a.baseline), fl(a.top_k), fl(a.hysteresis), fl(a.min_duration)):
        scores = []
        for i, (r, ll, valid, dur) in enumerate(data):
            key = (i, int(base))
            if key not in zcache:
                zcache = {kk: v for kk, v in zcache.items() if kk[1] == int(base)}
                zcache[key] = D.normalise(ll, valid, int(base))
            sc, _ = D.combine(zcache[key], int(k), MAIN.SMOOTH_S)
            sc[~valid] = np.nan
            scores.append(sc)
        for thr in fl(a.thresholds):
            res = []
            by_sub = {}
            for (r, ll, valid, dur), sc in zip(data, scores):
                ev = D.hysteresis(sc, thr, thr * hyst)
                ev = D.postprocess([(float(x), float(y)) for x, y in ev], mind, 90, 300,
                                   drop_first=a.order == "drop_first")
                ref = [(on, d if d is not None else 10.0) for on, d in r.seizures]
                s = S.score(ref, [(x, y - x) for x, y in ev], dur)
                res.append(s)
                by_sub.setdefault(r.subject, []).append(s)
            pe = S.pooled(res, "event")
            sm = S.subject_mean(by_sub, "event")
            ps = S.pooled(res, "sample")
            rows.append({"baseline": base, "top_k": k, "hysteresis": hyst, "min_duration": mind, "threshold": thr,
                         "sens": pe["sensitivity"], "prec": pe["precision"], "f1": pe["f1"],
                         "fp_day": pe["fp_per_day"], "tp": pe["tp"], "fp": pe["fp"], "ref": pe["n_reference"],
                         "subj_f1": sm["f1"], "subj_sens": sm["sensitivity"], "subj_fp_day": sm["fp_per_day"],
                         "sample_f1": ps["f1"]})
            print(" ".join(f"{k_}={v:.3g}" for k_, v in rows[-1].items()), flush=True)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
