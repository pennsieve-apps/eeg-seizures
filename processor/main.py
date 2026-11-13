"""
eeg-seizures: seizure detection on clinical scalp EEG in the SzCORE event format.

Reads the BIDS-EEG derivative of eeg-prep (or any BIDS-EEG dataset, or loose EDF/BDF/
BrainVision/FIF/EEGLAB files) from INPUT_DIR and writes OUTPUT_DIR/eeg-seizures/[LABEL/]:

  seizures.csv          recording_id, onset_s, duration_s, confidence, channels, detector
  szcore/<id>.tsv       the same events in SzCORE's HED-SCORE event format
  probability.csv       recording_id, time_s, probability (one row per second)
  scores.csv            recording_id, scope, metric, value (timescoring against annotations)
  figures/*.png, summary.json, provenance.json

Every detection is a candidate for review, not a diagnosis. One recording at a time, read in
WINDOW_S windows, so memory does not grow with the recording's length.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import resource
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from processor import resources as R

APP_NAME = "eeg-seizures"
REPO_ROOT = Path(__file__).resolve().parent.parent
# Memory: the main process holds one window (19-23 channels x WINDOW_S at the native rate,
# float64, plus the filtered copy) and per-second arrays of the recording; measured < 0.5 GB.
# Threads only speed up the ONNX model and BLAS, so one "worker" is one thread.
WORKER_MB = 100
MAIN_MB = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(APP_NAME)
for noisy in ("mne", "matplotlib", "PIL"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _setup_runtime() -> tuple[int, int, str]:
    """HOME/TMPDIR exist and are writable (Lambda-safe), Matplotlib's cache goes to TMPDIR, and
    BLAS/OpenMP threads are set from THREADS before numpy is imported. Returns
    (requested, used, why)."""
    import tempfile

    for var in ("TMPDIR", "HOME"):
        p = os.environ.get(var)
        if p:
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
    tmp = Path(tempfile.gettempdir())
    mpl = tmp / "mpl"
    try:
        mpl.mkdir(parents=True, exist_ok=True)
        baked = Path(os.environ.get("MPL_FONT_CACHE", "/opt/eeg-seizures/matplotlib"))
        if baked.is_dir():
            for f in baked.glob("fontlist-*.json"):
                if not (mpl / f.name).exists():
                    shutil.copy(f, mpl / f.name)
        os.environ["MPLCONFIGDIR"] = str(mpl)
    except OSError:
        pass
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_USE_CUDA", "false")
    try:
        requested = int((os.environ.get("THREADS") or "0").strip())
    except ValueError:
        requested = -1
    used, why = R.resolve_workers(max(requested, 0), WORKER_MB, MAIN_MB)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(used)
    return requested, used, why


THREADS_REQUESTED, THREADS_USED, THREADS_WHY = _setup_runtime()

import numpy as np  # noqa: E402  (after the thread variables are set)

from processor import detect as D  # noqa: E402
from processor import eeg as E  # noqa: E402
from processor import model as M  # noqa: E402
from processor import szcore as S  # noqa: E402

HYSTERESIS = 0.75     # an event ends when the score falls below HYSTERESIS x the threshold
SMOOTH_S = 5          # centred moving mean of the per-second line-length score
PAD_S = 5.0           # context read on each side of a window (filter edges)
PROGRESS_S = 60.0
UNKNOWN_DURATION_S = 10.0  # a reference seizure without a known end is scored as onset + 10 s

CITATIONS = ["10.1111/epi.18113", "10.1109/IEMBS.2001.1020545", "10.3389/fnins.2013.00267"]


def version_of(root: Path) -> str:
    try:
        return (root / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("mne", "numpy", "scipy", "pandas", "timescoring", "onnxruntime", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be true or false (got {value!r})")


DEFAULTS = {
    "DETECTOR": "linelength", "THRESHOLD": "4.5", "MODEL_THRESHOLD": "auto", "TOP_K": "8",
    "MIN_DURATION_S": "20", "MERGE_GAP_S": "90", "MAX_DURATION_S": "300", "BASELINE_S": "300",
    "HIGHPASS_HZ": "1", "LOWPASS_HZ": "20", "WINDOW_S": "600", "EXCLUDE_BAD_SEGMENTS": "false", "SCORE": "true", "THREADS": "0", "LABEL": "",
}


def env_params() -> dict:
    """Parameters from the environment; raises ValueError on a bad value."""
    raw = {k: (os.environ.get(k) or d).strip() for k, d in DEFAULTS.items()}
    p: dict = {"raw": raw}
    p["detector"] = raw["DETECTOR"].lower()
    if p["detector"] not in ("linelength", "model"):
        raise ValueError(f"DETECTOR must be linelength or model (got {raw['DETECTOR']!r})")

    def num(name, lo=None, hi=None, integer=False):
        try:
            v = float(raw[name])
        except ValueError:
            raise ValueError(f"{name} must be a number (got {raw[name]!r})") from None
        if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(f"{name} must be between {lo} and {hi} (got {raw[name]!r})")
        if integer:
            if v != int(v):
                raise ValueError(f"{name} must be a whole number (got {raw[name]!r})")
            return int(v)
        return v

    p["threshold"] = num("THRESHOLD", 0.5, 50)
    p["model_threshold"] = None if raw["MODEL_THRESHOLD"].lower() == "auto" else num("MODEL_THRESHOLD", 0.01, 0.99)
    p["top_k"] = num("TOP_K", 1, 23, integer=True)
    p["min_duration"] = num("MIN_DURATION_S", 0, 600)
    p["merge_gap"] = num("MERGE_GAP_S", 0, 3600)
    p["max_duration"] = num("MAX_DURATION_S", 0, 86400)
    p["baseline"] = num("BASELINE_S", 60, 3600, integer=True)
    p["highpass"] = num("HIGHPASS_HZ", 0.1, 10)
    p["lowpass"] = None if raw["LOWPASS_HZ"].lower() in ("none", "0", "") else num("LOWPASS_HZ", 5, 120)
    if p["lowpass"] is not None and p["lowpass"] <= p["highpass"]:
        raise ValueError("LOWPASS_HZ must be above HIGHPASS_HZ")
    p["window"] = num("WINDOW_S", 60, 7200, integer=True)
    p["score"] = _bool("SCORE", raw["SCORE"])
    p["exclude_bad"] = _bool("EXCLUDE_BAD_SEGMENTS", raw["EXCLUDE_BAD_SEGMENTS"])
    if THREADS_REQUESTED < 0:
        raise ValueError("THREADS must be a whole number >= 0 (0 = every vCPU)")
    p["threads"], p["threads_why"] = THREADS_USED, THREADS_WHY
    if raw["LABEL"] and not all(c.isalnum() or c in "-_" for c in raw["LABEL"]):
        raise ValueError("LABEL may hold letters, digits, - and _ only")
    p["label"] = raw["LABEL"]
    return p


class Skipped(Exception):
    """A designed reason not to analyse a recording (not a failure)."""


def _excluded_mask(rec: E.Recording, n_sec: int, exclude_bad_segments: bool) -> tuple[np.ndarray, float]:
    valid = np.ones(n_sec, dtype=bool)
    for a, b, _ in E.excluded_intervals(rec, exclude_bad_segments):
        valid[max(0, int(math.floor(a))): max(0, min(n_sec, int(math.ceil(b))))] = False
    return valid, float((~valid).sum())


def _model_input(x: np.ndarray, names: list[str], model: M.Model) -> tuple[np.ndarray | None, list[str]]:
    """(channels in the model's order, missing channel names). Pairs a referential recording
    lacks are computed from their electrodes."""
    index = {n: i for i, n in enumerate(names)}
    rows, missing = [], []
    for ch in model.channels:
        key = E.clean_label(ch)
        el = E.electrode(key)
        if el and el in index:
            rows.append(x[index[el]])
            continue
        parts = key.split("-")
        if len(parts) == 2 and E.electrode(parts[0]) and E.electrode(parts[1]):
            a, b = E.electrode(parts[0]), E.electrode(parts[1])
            if f"{a}-{b}" in index:
                rows.append(x[index[f"{a}-{b}"]])
                continue
            if f"{b}-{a}" in index:
                rows.append(-x[index[f"{b}-{a}"]])
                continue
            if a in index and b in index:
                rows.append(x[index[a]] - x[index[b]])
                continue
        rows.append(np.zeros(x.shape[1]))
        missing.append(ch)
    return np.vstack(rows), missing


def process_recording(rec: E.Recording, p: dict, model: M.Model | None) -> dict:
    t_start = time.monotonic()
    raw = E.open_raw(rec.path)
    fs = float(raw.info["sfreq"])
    if fs < 200:
        raise Skipped(f"skipped: sampling rate {fs:g} Hz is below 200 Hz (SzCORE analyses 256 Hz)")
    n_times = int(raw.n_times)
    duration = n_times / fs
    n_sec = int(math.floor(duration))
    if n_sec < 60:
        raise Skipped(f"skipped: recording is {duration:.0f} s long; at least 60 s are needed")
    bad, types = E.channel_table(rec)
    mont = E.montage_for(list(raw.ch_names), list(raw.get_channel_types()), bad, types)
    if mont.kind == "none":
        raise Skipped(f"skipped: {mont.note}")
    if model is not None and model.montage == "average" and mont.kind != "average":
        raise Skipped("skipped: the model expects average-referenced 10-20 electrodes and this recording is "
                      "bipolar-only (no way back to a referential montage)")
    index = {n: i for i, n in enumerate(raw.ch_names)}
    picks = [index[s[0]] for s in mont.sources]
    nch = len(picks)
    start_dt = E.start_datetime(rec, raw)
    valid, n_excluded = _excluded_mask(rec, n_sec, p["exclude_bad"])
    n_bad_kept = sum(1 for e in rec.excluded if e[2] == "bad_segment") if not p["exclude_bad"] else 0

    if model is None:
        sos = D.bandpass_sos(fs, p["highpass"], p["lowpass"])
    else:
        sos = D.bandpass_sos(fs, *model.bandpass) if model.bandpass else None
    ll = np.full((nch, n_sec), np.nan)
    prob_sum = np.zeros(n_sec)
    prob_n = np.zeros(n_sec)
    model_missing: list[str] = []
    t_read = t_compute = 0.0
    last_log = time.monotonic()
    extra = model.window_s if model is not None else 0
    win = p["window"]
    for c0 in range(0, n_sec, win):
        c1 = min(n_sec, c0 + win)
        r0 = max(0.0, c0 - PAD_S)
        r1 = min(duration, c1 + extra + PAD_S)
        s, e = int(math.floor(r0 * fs)), min(n_times, int(math.ceil(r1 * fs)))
        ta = time.monotonic()
        data = E.read_window(raw, picks, s, e)
        tb = time.monotonic()
        if mont.kind == "average":
            data -= data.mean(axis=0, keepdims=True)
        x = D.to_256(data, fs, sos) if (sos is not None or abs(fs - D.FS) > 1e-6) else data
        del data
        t0 = s / fs
        if model is None:
            ll[:, c0:c1] = D.line_length_per_second(x, t0, c0, c1)
        else:
            xin, miss = _model_input(x, mont.names, model)
            if miss and not model_missing:
                model_missing = miss
                if len(miss) > len(model.channels) / 4:
                    raise Skipped(f"skipped: the recording lacks {len(miss)} of the model's {len(model.channels)} "
                                  f"channels ({', '.join(miss)})")
            w = model.window_s
            starts = list(range(int(math.ceil(c0 / model.step_s)) * model.step_s, c1, model.step_s))
            if c1 == n_sec and (not starts or starts[-1] + w < n_sec) and n_sec >= w:
                starts.append(n_sec - w)
            starts = [t for t in starts if t + w <= n_sec]
            batch, kept = [], []
            for t in starts:
                i0 = int(round((t - t0) * D.FS))
                seg = xin[:, i0:i0 + w * D.FS]
                if i0 < 0 or seg.shape[1] < w * D.FS:
                    continue
                batch.append(seg)
                kept.append(t)
            for b0 in range(0, len(batch), 32):
                y = model.predict(np.stack(batch[b0:b0 + 32]))
                for t, row in zip(kept[b0:b0 + 32], y):
                    prob_sum[t:t + w] += row
                    prob_n[t:t + w] += 1
        del x
        t_read += tb - ta
        t_compute += time.monotonic() - tb
        if time.monotonic() - last_log > PROGRESS_S:
            log.info("%s: %.0f of %.0f min analysed", rec.recording_id, c1 / 60, duration / 60)
            last_log = time.monotonic()

    tc = time.monotonic()
    if model is None:
        z = D.normalise(ll, valid, p["baseline"])
        score, _ = D.combine(z, p["top_k"], SMOOTH_S)
        score[~valid] = np.nan
        on = p["threshold"]
        prob = D.sigmoid(score - on)
        detector = "linelength"
    else:
        z = None
        with np.errstate(invalid="ignore"):
            prob = np.where(prob_n > 0, prob_sum / np.maximum(prob_n, 1), np.nan)
        prob[~valid] = np.nan
        score = prob
        on = p["model_threshold"] if p["model_threshold"] is not None else model.threshold
        detector = f"model:{model.name}"
    raw_events = D.hysteresis(score, on, on * HYSTERESIS)
    events = D.postprocess([(float(a), float(b)) for a, b in raw_events], p["min_duration"], p["merge_gap"],
                           p["max_duration"])
    out_events = []
    for a, b in events:
        ia, ib = int(a), max(int(a) + 1, int(math.ceil(b)))
        conf = float(np.nanmean(prob[ia:ib])) if np.isfinite(prob[ia:ib]).any() else None
        chans: list[str] = []
        if z is not None:
            head = z[:, ia:min(ib, ia + 20)]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mz = np.nanmean(head, axis=1)
            order = [i for i in np.argsort(-np.nan_to_num(mz, nan=-np.inf)) if np.isfinite(mz[i])]
            chans = [mont.names[i] for i in order[:p["top_k"]] if mz[i] >= on] or \
                    [mont.names[i] for i in order[:1]]
        out_events.append({"onset": float(a), "duration": float(b - a), "confidence": conf, "channels": chans})
    t_compute += time.monotonic() - tc

    scores = None
    if p["score"] and rec.labels_known:
        ref = [(on_, d if d is not None else UNKNOWN_DURATION_S) for on_, d in rec.seizures]
        hyp = [(e["onset"], e["duration"]) for e in out_events]
        scores = S.score(ref, hyp, duration)
        if any(d is None for _, d in rec.seizures):
            rec.notes.append(f"a reference seizure has no end; scored as onset + {UNKNOWN_DURATION_S:g} s")

    summary = {
        "status": "ok", "montage": mont.kind, "montage_note": mont.note, "sfreq": fs,
        "resampled_to": D.FS, "duration_s": round(duration, 3), "n_channels": nch, "channels": mont.names,
        "channels_missing": mont.missing, "channels_ignored": mont.ignored,
        "excluded_s": n_excluded,
        "excluded": [{"start_s": a, "end_s": b, "reason": k} for a, b, k in E.excluded_intervals(rec, p["exclude_bad"])],
        "bad_segments_not_excluded": n_bad_kept,
        "start_time": start_dt.strftime("%Y-%m-%dT%H:%M:%S") if start_dt else None,
        "annotations": {"known": rec.labels_known, "sources": rec.label_sources, "n_seizures": len(rec.seizures)},
        "n_detections": len(out_events), "detector": detector, "threshold": on,
        "model_missing_channels": model_missing, "notes": rec.notes,
        "seconds": round(time.monotonic() - t_start, 2), "read_seconds": round(t_read, 2),
        "compute_seconds": round(t_compute, 2),
    }
    return {"events": out_events, "prob": prob, "score": score, "scores": scores, "summary": summary,
            "start": start_dt, "duration": duration, "detector": detector, "threshold": on,
            "ll": ll if model is None else None, "valid": valid, "channels": mont.names}


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _json_num(v):
    if v is None:
        return None
    v = float(v)
    return round(v, 4) if math.isfinite(v) else None


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1 << 20) if sys.platform == "darwin" else r / 1024, 1)


def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    try:
        p = env_params()
    except ValueError as exc:
        log.error("bad parameter: %s", exc)
        return 2
    out = output_dir / APP_NAME
    if p["label"]:
        out = out / p["label"]
    (out / "szcore").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    model = None
    model_folder = M.find_model_folder(input_dir)
    skip = ({os.path.realpath(model_folder)} if model_folder is not None
            and os.path.realpath(model_folder) != os.path.realpath(input_dir) else set())
    if p["detector"] == "model":
        if model_folder is None:
            log.error("DETECTOR=model but no model.json was found in the input (link a model folder to the "
                      "`model` port; see the README)")
            return 1
        try:
            model = M.read_model(model_folder)
            model.load(p["threads"])
        except M.ModelError as exc:
            log.error("model: %s", exc)
            return 1
        log.info("model %s (%s, %d s windows every %d s, %d channels, %s, threshold %s), licence: %s",
                 model.name, model.format, model.window_s, model.step_s, len(model.channels), model.montage,
                 p["model_threshold"] or model.threshold, model.licence)
    elif model_folder is not None:
        warnings.append("a model was linked but DETECTOR=linelength; the model was not used (set DETECTOR=model)")
        log.warning(warnings[-1])

    recs, seen, problems = E.find_recordings(input_dir, skip)
    log.info("input: %d recording(s), %d file(s); detector %s, threshold %s, %d thread(s) (%s)", len(recs),
             len(seen), p["detector"], p["threshold"] if model is None else (p["model_threshold"] or model.threshold),
             p["threads"], p["threads_why"])
    for pr in problems:
        log.warning("%s: %s", pr["input"], pr["reason"])

    sz_fh = open(out / "seizures.csv", "w", newline="", encoding="utf-8")
    sz_w = csv.writer(sz_fh, lineterminator="\n")
    sz_w.writerow(["recording_id", "onset_s", "duration_s", "confidence", "channels", "detector"])
    pr_fh = open(out / "probability.csv", "w", newline="", encoding="utf-8")
    pr_w = csv.writer(pr_fh, lineterminator="\n")
    pr_w.writerow(["recording_id", "time_s", "probability"])

    prov_recs: dict[str, dict] = {}
    for pr in problems:
        if pr.get("recording_id") and pr.get("status", "failed") == "failed":
            prov_recs[pr["recording_id"]] = {"status": "failed", "input": pr["input"], "seconds": 0.0,
                                             "montage": None, "error": pr["reason"]}
        else:
            warnings.append(f"{pr['input']}: {pr['reason']}")
    n_ok = n_skipped = 0
    n_failed = sum(1 for pr in problems if pr.get("recording_id") and pr.get("status", "failed") == "failed")
    results: dict[str, dict] = {}
    for rec in recs:
        try:
            res = process_recording(rec, p, model)
        except Skipped as why:
            log.warning("%s: %s", rec.recording_id, why)
            prov_recs[rec.recording_id] = {"status": str(why), "input": rec.rel, "seconds": 0.0, "montage": None,
                                           "notes": rec.notes}
            n_skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001 - one recording's failure must not sink the run
            log.error("%s: FAILED: %s\n%s", rec.recording_id, exc, traceback.format_exc())
            prov_recs[rec.recording_id] = {"status": "failed", "input": rec.rel, "seconds": 0.0, "montage": None,
                                           "error": f"{type(exc).__name__}: {exc}"}
            n_failed += 1
            continue
        n_ok += 1
        res.pop("ll", None)
        rid = rec.recording_id
        for e in res["events"]:
            sz_w.writerow([rid, _fmt(e["onset"], 2), _fmt(e["duration"], 2), _fmt(e["confidence"], 3),
                           ",".join(e["channels"]), res["detector"]])
        S.write_tsv(out / "szcore" / f"{rid}.tsv", res["events"], res["duration"], res["start"])
        pr_w.writerows([rid, t, "" if not math.isfinite(v) else f"{v:.4f}"] for t, v in enumerate(res["prob"]))
        s = res["summary"]
        results[rid] = {"rec": rec, **res}
        sc = res["scores"]
        sc_txt = ""
        if sc:
            ev = sc["event"]
            sc_txt = (f"; annotated {int(ev['n_reference'])}, detected {int(ev['tp'])}, false {int(ev['fp'])} "
                      f"({ev['fp_per_day']:.1f}/day)")
        log.info("%s: %s, %d channels, %.0f Hz -> 256 Hz, %.2f h, %d detection(s)%s; %.1f s (read %.1f, compute %.1f)",
                 rid, s["montage"], s["n_channels"], s["sfreq"], s["duration_s"] / 3600, s["n_detections"], sc_txt,
                 s["seconds"], s["read_seconds"], s["compute_seconds"])
        for note in s["notes"]:
            log.info("%s: %s", rid, note)
        prov_recs[rid] = {"input": rec.rel, **s, "scores": _scores_json(sc)}
    sz_fh.close()
    pr_fh.close()

    scored = {rid: r for rid, r in results.items() if r["scores"]}
    pooled_rows = _write_scores(out / "scores.csv", scored)

    figures = []
    try:
        from processor import figures as F

        figures = F.make_all(out, results, pooled_rows, p)
    except Exception as exc:  # noqa: BLE001 - figures must not sink the tables
        log.error("figures failed: %s\n%s", exc, traceback.format_exc())
        warnings.append(f"figures failed: {exc}")

    status = "ok"
    if not recs and not problems:
        log.error("no recording found under INPUT_DIR (looked for recordings.csv and %s files); it holds %d file(s): "
                  "%s", "/".join(E.SIGNAL_EXTENSIONS), len(seen), ", ".join(seen[:40]) + (" …" if len(seen) > 40 else ""))
        status = "failed: no input"
    elif n_ok == 0 and n_failed > 0 and n_skipped == 0:
        status = "failed: every recording failed"
    elif n_ok == 0:
        status = f"failed: nothing analysed ({n_skipped} skipped, {n_failed} failed)"

    licence_notice = None
    if model is not None:
        licence_notice = (f"Detections by the user-supplied model '{model.name}' (licence: {model.licence}). "
                          "Its terms apply to its use; Pennsieve did not train or validate it.")
    params_out = {k: p["raw"][k] for k in DEFAULTS}
    resolved = {"hysteresis_fraction": HYSTERESIS, "smooth_s": SMOOTH_S, "line_length_window_s": D.LL_WINDOW_S,
                "step_s": D.STEP_S, "sfreq": D.FS, "scoring": {"sample_fs_hz": S.SCORE_FS, **S.EVENT_PARAMS}}
    if model is not None:
        resolved["model"] = {"name": model.name, "format": model.format, "file": model.file.name,
                             "window_s": model.window_s, "step_s": model.step_s, "channels": model.channels,
                             "montage": model.montage, "threshold": p["model_threshold"] or model.threshold,
                             "bandpass": model.bandpass, "licence": model.licence, "citation": model.citation}
    citations = list(CITATIONS)
    if model is not None and model.citation:
        citations.append(model.citation)
    finished = datetime.now(timezone.utc)
    prov = {
        "app": APP_NAME, "version": version_of(REPO_ROOT), "variant": None, "tools": tool_versions(),
        "parameters": params_out, "resolved": resolved, "inputs": seen,
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": max(THREADS_REQUESTED, 0), "used": p["threads"], "why": p["threads_why"]},
        "status": status, "recordings": prov_recs, "citations": citations, "licence_notice": licence_notice,
        "warnings": warnings,
    }
    with open(out / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    _write_summary(out, results, pooled_rows, figures, warnings, p, model)
    log.info("done: %d recording(s) ok, %d skipped, %d failed, %.1f s, peak RSS %.0f MB -> %s", n_ok, n_skipped,
             n_failed, prov["wall_seconds"], prov["peak_rss_mb"], out)
    return 1 if status.startswith("failed") else 0


def _scores_json(sc: dict | None) -> dict | None:
    if not sc:
        return None
    return {scope: {k: _json_num(v) for k, v in vals.items()} for scope, vals in sc.items()}


METRICS = ("sensitivity", "precision", "f1", "fp_per_day", "tp", "fp", "n_reference", "hours")


def _write_scores(path: Path, scored: dict[str, dict]) -> dict:
    """scores.csv: one block per scored recording, then `pooled` (counts added over recordings,
    SzCORE's 'cumulated') and `subject_mean` (SzCORE's default: mean over subjects)."""
    rows = []
    for rid, r in scored.items():
        for scope in ("sample", "event"):
            for m in METRICS:
                rows.append([rid, scope, m, _fmt(r["scores"][scope][m], 6)])
    pooled_out: dict = {}
    if scored:
        by_subject: dict[str, list] = {}
        for r in scored.values():
            by_subject.setdefault(r["rec"].subject, []).append(r["scores"])
        all_scores = [r["scores"] for r in scored.values()]
        for scope in ("sample", "event"):
            pl = S.pooled(all_scores, scope)
            sm = S.subject_mean(by_subject, scope)
            pooled_out[scope] = {"pooled": pl, "subject_mean": sm}
            for m in METRICS:
                rows.append(["pooled", scope, m, _fmt(pl[m], 6)])
            for m in ("sensitivity", "precision", "f1", "fp_per_day", "n_subjects"):
                rows.append(["subject_mean", scope, m, _fmt(sm[m], 6)])
        pooled_out["n_recordings"] = len(scored)
        pooled_out["n_subjects"] = len(by_subject)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "scope", "metric", "value"])
        w.writerows(rows)
    return pooled_out


def _write_summary(out: Path, results: dict, pooled_rows: dict, figures: list, warnings: list, p: dict,
                   model) -> None:
    recs = {}
    for rid, r in results.items():
        s = r["summary"]
        d = {"duration_h": _json_num(s["duration_s"] / 3600), "montage": s["montage"], "n_channels": s["n_channels"],
             "n_detections": s["n_detections"],
             "detections_per_day": _json_num(s["n_detections"] / (s["duration_s"] / 86400)),
             "n_annotated_seizures": s["annotations"]["n_seizures"] if s["annotations"]["known"] else None}
        if r["scores"]:
            ev, sa = r["scores"]["event"], r["scores"]["sample"]
            d.update({"event_sensitivity": _json_num(ev["sensitivity"]), "event_precision": _json_num(ev["precision"]),
                      "event_f1": _json_num(ev["f1"]), "event_fp_per_day": _json_num(ev["fp_per_day"]),
                      "seizures_detected": int(ev["tp"]), "false_detections": int(ev["fp"]),
                      "sample_f1": _json_num(sa["f1"])})
        recs[rid] = d
    summary = {
        "app": APP_NAME, "version": version_of(REPO_ROOT),
        "detector": p["detector"] if model is None else f"model:{model.name}",
        "note": "Detections are research outputs: candidates for an expert to review, not a diagnosis. "
                "False detections per day are always shown next to sensitivity.",
        "recordings": recs,
        "pooled": {scope: {agg: {k: _json_num(v) for k, v in vals.items()} for agg, vals in d.items()}
                   for scope, d in pooled_rows.items() if scope in ("sample", "event")},
        "figures": figures,
        "tables": [
            {"path": "seizures.csv", "caption": "Detected seizure candidates: onset and duration (s from the recording "
                                                "start), confidence (mean per-second probability) and the channels "
                                                "where line length rose most at onset."},
            {"path": "scores.csv", "caption": "SzCORE scores (timescoring) against the annotated seizures, per "
                                              "recording and pooled: event-based sensitivity, precision, F1 and "
                                              "false detections per day; sample-based (1 s) the same."},
            {"path": "probability.csv", "caption": "Per-second detector output between 0 and 1 (0.5 = the threshold)."},
        ],
        "warnings": warnings,
    }
    if pooled_rows:
        summary["pooled"]["n_recordings"] = pooled_rows.get("n_recordings")
        summary["pooled"]["n_subjects"] = pooled_rows.get("n_subjects")
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
