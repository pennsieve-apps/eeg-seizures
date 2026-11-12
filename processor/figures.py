"""
Figures for summary.json (rendered in order, with captions, by eeg-report).

Colours: annotated seizures blue, detections orange (fixed roles in every figure); text in
neutral ink; one y-axis per panel.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ANNOTATED = "#2a78d6"
DETECTED = "#eb6834"
MISSED = "#c9c8c2"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
MAX_RECORDING_FIGURES = 12


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.titleweight": "bold"})
    return plt


def _time_unit(duration_s: float) -> tuple[float, str]:
    return (3600.0, "hours") if duration_s > 3 * 3600 else (60.0, "minutes")


def recording_figure(path: Path, rid: str, r: dict, p: dict) -> None:
    plt = _plt()
    score, events = r["score"], r["events"]
    rec = r["rec"]
    unit, uname = _time_unit(r["duration"])
    t = (np.arange(len(score)) + 0.5) / unit
    fig, (ax_lane, ax) = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
    for on, dur in rec.seizures:
        d = dur if dur is not None else 10.0
        ax_lane.add_patch(plt.Rectangle((on / unit, 0.55), d / unit, 0.35, color=ANNOTATED, lw=0))
        ax.axvspan(on / unit, (on + d) / unit, color=ANNOTATED, alpha=0.12, lw=0)
    for e in events:
        ax_lane.add_patch(plt.Rectangle((e["onset"] / unit, 0.1), e["duration"] / unit, 0.35, color=DETECTED, lw=0))
    for a, b, _ in rec.excluded:
        ax.axvspan(a / unit, b / unit, facecolor="none", edgecolor=MISSED, hatch="///", lw=0)
    min_w = r["duration"] / unit / 400  # keep short events visible
    for on, dur in rec.seizures:
        ax_lane.add_patch(plt.Rectangle((on / unit, 0.55), min_w, 0.35, color=ANNOTATED, lw=0))
    for e in events:
        ax_lane.add_patch(plt.Rectangle((e["onset"] / unit, 0.1), min_w, 0.35, color=DETECTED, lw=0))
    ax_lane.set_ylim(0, 1)
    ax_lane.set_yticks([0.275, 0.725], ["detected", "annotated"])
    ax_lane.tick_params(axis="y", length=0)
    ax_lane.spines["left"].set_visible(False)
    ax_lane.spines["bottom"].set_visible(False)
    ax_lane.tick_params(axis="x", length=0)
    ax.plot(t, score, color=INK2, lw=0.6)
    ax.axhline(r["threshold"], color=DETECTED, lw=1.0, ls="--")
    ylab = "line-length score (z)" if r["detector"] == "linelength" else "seizure probability"
    ax.set_ylabel(ylab)
    ax.set_xlabel(f"time from recording start ({uname})")
    ax.set_xlim(0, r["duration"] / unit)
    finite = score[np.isfinite(score)]
    if finite.size and r["detector"] == "linelength":
        ax.set_ylim(min(-2.0, float(np.percentile(finite, 1))), max(r["threshold"] * 2, float(finite.max()) * 1.05))
    ax.grid(axis="y", color=GRID, lw=0.6)
    title = f"{rid}: {len(events)} detection(s), {len(rec.seizures)} annotated seizure(s)"
    if r["scores"]:
        ev = r["scores"]["event"]
        title += f"; event sensitivity {_pct(ev['sensitivity'])}, {ev['fp_per_day']:.1f} false/day"
    ax_lane.set_title(title, loc="left", color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pct(v: float) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{100 * v:.0f}%"


def overview_figure(path: Path, results: dict) -> None:
    plt = _plt()
    rids = sorted(results)
    n = len(rids)
    unit = 3600.0 if max(r["duration"] for r in results.values()) > 3 * 3600 else 60.0
    uname = "hours" if unit == 3600.0 else "minutes"
    fig, ax = plt.subplots(figsize=(10, max(2.0, 0.28 * n + 1.2)))
    for i, rid in enumerate(rids):
        r = results[rid]
        y = n - 1 - i
        ax.add_patch(plt.Rectangle((0, y - 0.04), r["duration"] / unit, 0.08, color=GRID, lw=0))
        w = max(r["duration"] for r in results.values()) / unit / 300
        for on, dur in r["rec"].seizures:
            ax.add_patch(plt.Rectangle((on / unit, y + 0.04), max((dur or 10) / unit, w), 0.3, color=ANNOTATED, lw=0))
        for e in r["events"]:
            ax.add_patch(plt.Rectangle((e["onset"] / unit, y - 0.34), max(e["duration"] / unit, w), 0.3,
                                       color=DETECTED, lw=0))
    ax.set_yticks(range(n), [rids[n - 1 - i] for i in range(n)], fontsize=7 if n > 20 else 8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, max(r["duration"] for r in results.values()) / unit * 1.01)
    ax.set_xlabel(f"time from recording start ({uname})")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=ANNOTATED, label="annotated seizure (above the line)"),
                       Patch(color=DETECTED, label="detection (below the line)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12 if n > 6 else -0.3), ncol=2, frameon=False)
    ax.set_title("Detections and annotated seizures, every recording", loc="left", color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scores_figure(path: Path, scored: dict, pooled: dict) -> None:
    plt = _plt()
    rids = sorted(scored)
    n = len(rids)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, max(2.0, 0.26 * n + 1.4)), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    y = np.arange(n)[::-1]
    tp = np.array([scored[r]["scores"]["event"]["tp"] for r in rids])
    ref = np.array([scored[r]["scores"]["event"]["n_reference"] for r in rids])
    fpd = np.array([scored[r]["scores"]["event"]["fp_per_day"] for r in rids])
    a1.barh(y, tp, color=ANNOTATED, height=0.7, label="detected")
    a1.barh(y, ref - tp, left=tp, color=MISSED, height=0.7, label="missed")
    a1.set_yticks(y, rids, fontsize=7 if n > 20 else 8)
    a1.set_xlabel("annotated seizures")
    a1.xaxis.get_major_locator().set_params(integer=True)
    a1.legend(loc="lower right", frameon=False, fontsize=8)
    a2.barh(y, fpd, color=DETECTED, height=0.7)
    a2.set_xlabel("false detections per day")
    for ax in (a1, a2):
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    ev = pooled.get("event", {}).get("pooled")
    t = "Event scores per recording"
    if ev:
        t += (f" (pooled: sensitivity {_pct(ev['sensitivity'])}, precision {_pct(ev['precision'])}, "
              f"F1 {_pct(ev['f1'])}, {ev['fp_per_day']:.1f} false/day)")
    fig.suptitle(t, x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_all(out: Path, results: dict, pooled: dict, p: dict) -> list[dict]:
    figs: list[dict] = []
    if not results:
        return figs
    fdir = out / "figures"
    det = "line-length score" if p["detector"] == "linelength" else "model probability"
    overview_figure(fdir / "overview_detections.png", results)
    figs.append({"path": "figures/overview_detections.png", "caption": (
        "Every recording on one time axis (from its own start): annotated seizures in blue above each line, "
        "the detector's candidate seizures in orange below it. A blue mark with an orange mark under it is a "
        "detected seizure; orange marks alone are false detections, blue alone a missed seizure. Candidates "
        "are for review, not a diagnosis.")})
    scored = {rid: r for rid, r in results.items() if r["scores"]}
    if scored:
        scores_figure(fdir / "scores_per_recording.png", scored, pooled)
        figs.append({"path": "figures/scores_per_recording.png", "caption": (
            "SzCORE event scoring (timescoring; a detection counts when it overlaps an annotated seizure "
            "widened by 30 s before and 60 s after). Left: annotated seizures detected (blue) and missed "
            "(grey) per recording. Right: false detections per day, always to be read next to sensitivity: "
            "a detector that finds every seizure but raises dozens of false alarms a day is not useful for "
            "review.")})
    order = sorted(results, key=lambda rid: (-len(results[rid]["rec"].seizures), -len(results[rid]["events"]), rid))
    order = [rid for rid in order if results[rid]["rec"].seizures or results[rid]["events"]]
    for i, rid in enumerate(order):
        name = f"{rid}_detections.png"
        recording_figure(fdir / name, rid, results[rid], p)
        if i < MAX_RECORDING_FIGURES:
            figs.append({"path": f"figures/{name}", "caption": (
                f"{rid}: the per-second {det} (grey; dashed orange line = threshold) with annotated seizures "
                "(blue band and top lane) and detections (orange, bottom lane); hatched grey = bad segments left out. Look for whether the score "
                "rises at the annotated onset and what else crosses the threshold (artifacts, arousals).")})
    return figs
