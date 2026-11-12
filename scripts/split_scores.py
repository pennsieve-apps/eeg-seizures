"""
Pool an eeg-seizures scores.csv over a subset of subjects (SzCORE 'cumulated' and the subject
mean), e.g. to report the tuning and held-out halves of a dataset separately.

Usage: python -m scripts.split_scores <scores.csv> <subject,subject,...> [<label>]
Subjects are the `sub-` labels (00, 03, ...).
"""
from __future__ import annotations

import csv
import sys

from processor import szcore as S


def main() -> int:
    path, subjects = sys.argv[1], {s.strip() for s in sys.argv[2].split(",") if s.strip()}
    label = sys.argv[3] if len(sys.argv) > 3 else ",".join(sorted(subjects))
    recs: dict[str, dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rid = row["recording_id"]
            if rid in ("pooled", "subject_mean"):
                continue
            sub = rid.split("_")[0].removeprefix("sub-")
            if sub not in subjects:
                continue
            recs.setdefault(rid, {"subject": sub, "sample": {}, "event": {}})[row["scope"]][row["metric"]] = \
                float(row["value"]) if row["value"] else float("nan")
    by_sub: dict[str, list] = {}
    for r in recs.values():
        by_sub.setdefault(r["subject"], []).append(r)
    for scope in ("event", "sample"):
        pl = S.pooled(list(recs.values()), scope)
        sm = S.subject_mean(by_sub, scope)
        print(f"{label} {scope}: {len(by_sub)} subjects, {len(recs)} recordings, {pl['hours']:.1f} h, "
              f"{pl['n_reference']:.0f} reference events | pooled: sensitivity {pl['sensitivity']:.3f} precision "
              f"{pl['precision']:.3f} F1 {pl['f1']:.3f} FP/day {pl['fp_per_day']:.2f} (TP {pl['tp']:.0f}, FP "
              f"{pl['fp']:.0f}) | subject mean: sensitivity {sm['sensitivity']:.3f} precision {sm['precision']:.3f} "
              f"F1 {sm['f1']:.3f} FP/day {sm['fp_per_day']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
