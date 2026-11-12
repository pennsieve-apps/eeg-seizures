"""
Finding and reading scalp EEG recordings: the BIDS-EEG derivative of eeg-prep (its `eeg`
port), any plain BIDS-EEG dataset (e.g. the SzCORE BIDS conversions of Siena and CHB-MIT), or
loose EDF/BDF/BrainVision/FIF/EEGLAB files (e.g. CHB-MIT as PhysioNet ships it).

The contract is workflow-scalp-eeg docs/conventions.md:

- `recordings.csv` (anywhere under the input) lists the recordings, `path` relative to the
  folder the CSV is in, plus `start_time`, `montage` and `n_seizures`;
- `<stem>[_desc-preproc]_channels.tsv` next to the signal says which channels are good EEG;
- `<stem>[_desc-preproc]_events.tsv` holds `seizure`, `bad_segment` and `artifact` rows, or,
  in SzCORE's HED-SCORE files, `eventType` `sz...` (seizure) and `bckg` (no seizure) rows.

Signal files no recordings.csv lists are picked up as well, with whatever sidecars they have.
Seizure labels also come from CHB-MIT `*-summary.txt` files and an optional `labels.csv`.
Nothing here loads signal data; the callers read one time window at a time.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SIGNAL_EXTENSIONS = (".edf", ".bdf", ".vhdr", ".fif", ".set")
SEIZURE_TYPES = {"seizure", "ictal", "sz"}
EXCLUDED_TYPES = {"bad_segment", "artifact"}

# The 19 electrodes of the international 10-20 system in SzCORE's channel order (Dan et al.
# 2024), with the modern names MNE uses (T7/T8/P7/P8 for the older T3/T4/T5/T6).
SZCORE_CHANNELS = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "Fz", "Cz", "Pz",
                   "Fp2", "F4", "C4", "P4", "O2", "F8", "T8", "P8"]
OLD_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
# 10-20 and the common 10-10 electrodes (for recognising bipolar pairs such as FT9-FT10).
_TEN_TEN = ("Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F9 F7 F5 F3 F1 Fz F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 "
            "FCz FC2 FC4 FC6 FT8 FT10 T9 T7 C5 C3 C1 Cz C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPz "
            "CP2 CP4 CP6 TP8 TP10 P9 P7 P5 P3 P1 Pz P2 P4 P6 P8 P10 PO7 PO3 POz PO4 PO8 O1 Oz O2 "
            "Iz A1 A2 M1 M2").split()
ELECTRODES = {n.upper(): n for n in _TEN_TEN}
for _old, _new in OLD_NAMES.items():
    ELECTRODES[_old] = _new
NON_EEG_NAME = re.compile(r"^(ECG|EKG|EMG|EOG|RESP|PHOTIC|EVENT|STATUS|TRIG|DC\d*|SPO2|PULSE|PR|OSAT|ANNOT|EDF ANNOT|"
                          r"LOC|ROC|VNS|\.|-)", re.IGNORECASE)


def walk_files(root: Path, skip: set[str] | None = None):
    """Every file under root, following symlinked folders (INPUT_DIR entries are symlinks), once.
    Folders whose real path is in `skip` (the linked model) are not entered."""
    seen: set[str] = set(skip or ())
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def _float(value) -> float | None:
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_table(path: Path, delimiter: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh, delimiter=delimiter)]


@dataclass
class Recording:
    recording_id: str
    subject: str
    path: Path                         # the signal file
    rel: str                           # relative to INPUT_DIR, for logs and provenance
    source: str                        # "recordings.csv" or "scan"
    start_time: str = ""               # ISO 8601 wall clock from recordings.csv
    montage: str = ""                  # from recordings.csv
    n_seizures: int | None = None      # from recordings.csv
    channels_tsv: Path | None = None
    events_tsv: Path | None = None
    sidecar_json: Path | None = None
    seizures: list[tuple[float, float | None]] = field(default_factory=list)  # (onset, duration or None)
    labels_known: bool = False         # an annotation source covers this recording (0 seizures is meaningful)
    label_sources: list[str] = field(default_factory=list)
    excluded: list[tuple[float, float, str]] = field(default_factory=list)  # bad_segment / artifact
    event_datetime: str = ""           # SzCORE `dateTime` of the source events.tsv
    notes: list[str] = field(default_factory=list)


def signal_prefix(path: Path) -> str:
    """`sub-01_task-x_desc-preproc_eeg.edf` -> `sub-01_task-x_desc-preproc`; `foo.edf` -> `foo`."""
    stem = path.name[: -len(path.suffix)]
    return stem[: -len("_eeg")] if stem.endswith("_eeg") else stem


def recording_id_for(path: Path) -> tuple[str, str]:
    """(recording id, subject). BIDS names: the entity chain without `_desc-`; loose files get
    a BIDS-like id as eeg-prep makes one (conventions.md): `chb01_03.edf` ->
    `sub-chb01_task-eeg_run-03`, `patient7.edf` -> `sub-patient7_task-eeg`."""
    prefix = signal_prefix(path)
    if prefix.startswith("sub-"):
        rid = re.sub(r"_desc-[A-Za-z0-9]+", "", prefix)
        return rid, rid.split("_")[0][len("sub-"):]
    m = re.match(r"^([A-Za-z0-9]+)[_-]0*(\d+)$", prefix)
    if m:
        return f"sub-{m.group(1)}_task-eeg_run-{int(m.group(2)):02d}", m.group(1)
    sub = re.sub(r"[^A-Za-z0-9]", "", prefix) or "unknown"
    return f"sub-{sub}_task-eeg", sub


def _sidecars(rec: Recording) -> None:
    prefix = signal_prefix(rec.path)
    folder = rec.path.parent
    for attr, suffix in (("channels_tsv", "_channels.tsv"), ("events_tsv", "_events.tsv"),
                         ("sidecar_json", "_eeg.json")):
        if getattr(rec, attr) is None:
            p = folder / f"{prefix}{suffix}"
            if p.is_file():
                setattr(rec, attr, p)


def parse_chbmit_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    """CHB-MIT `chbNN-summary.txt`: {file name: [(start_s, end_s), ...]} for every file it lists
    (an empty list = listed with no seizure)."""
    out: dict[str, list[tuple[float, float]]] = {}
    current = None
    start = None
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            current = m.group(1)
            out[current] = []
            continue
        if current is None:
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)\s*seconds", line, re.I)
        if m:
            start = float(m.group(1))
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)\s*seconds", line, re.I)
        if m and start is not None:
            out[current].append((start, float(m.group(1))))
            start = None
    return out


def _events_from_tsv(rec: Recording) -> None:
    try:
        rows = read_table(rec.events_tsv, "\t")
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        rec.notes.append(f"events.tsv unreadable: {exc}")
        return
    rec.labels_known = True
    rec.label_sources.append(rec.events_tsv.name)
    for e in rows:
        kind = (e.get("trial_type") or "").strip().lower()
        etype = (e.get("eventType") or "").strip().lower()
        on, dur = _float(e.get("onset")), _float(e.get("duration"))
        if on is None:
            continue
        if not rec.event_datetime and e.get("dateTime") not in (None, "", "n/a"):
            rec.event_datetime = e["dateTime"]
        if kind in SEIZURE_TYPES or etype.startswith("sz"):
            rec.seizures.append((on, dur if dur is not None and dur > 0 else None))
        elif kind in EXCLUDED_TYPES and dur is not None and dur > 0:
            rec.excluded.append((on, on + dur, kind))


def find_recordings(input_dir: Path, skip: set[str] | None = None
                    ) -> tuple[list[Recording], list[str], list[dict]]:
    """(recordings, every input file seen (relative), problems). A problem is a listed
    recording whose file is missing, or a duplicate recording id."""
    files = list(walk_files(input_dir, skip))
    rel = lambda p: os.path.relpath(p, input_dir)  # noqa: E731
    seen = [rel(p) for p in files]
    problems: list[dict] = []
    recs: list[Recording] = []
    listed: set[str] = set()

    for csv_path in (p for p in files if p.name == "recordings.csv"):
        try:
            rows = read_table(csv_path, ",")
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            problems.append({"recording_id": "", "input": rel(csv_path), "reason": f"unreadable recordings.csv: {exc}"})
            continue
        for row in rows:
            rid, path = row.get("recording_id", ""), row.get("path", "")
            if not rid or not path:
                continue
            sig = csv_path.parent / path
            if not sig.is_file():
                problems.append({"recording_id": rid, "input": rel(csv_path),
                                 "reason": f"recordings.csv lists {path}, which is not in the input"})
                continue
            listed.add(os.path.realpath(sig))
            nsz = _float(row.get("n_seizures"))
            recs.append(Recording(
                recording_id=rid, subject=row.get("subject") or rid.split("_")[0].removeprefix("sub-"),
                path=sig, rel=rel(sig), source="recordings.csv", start_time=row.get("start_time", ""),
                montage=row.get("montage", ""), n_seizures=int(nsz) if nsz is not None else None))

    for p in files:
        if p.suffix.lower() not in SIGNAL_EXTENSIONS or os.path.realpath(p) in listed:
            continue
        rid, sub = recording_id_for(p)
        recs.append(Recording(recording_id=rid, subject=sub, path=p, rel=rel(p), source="scan"))

    summaries: dict[str, tuple[Path, list]] = {}
    for p in files:
        if p.name.lower().endswith("-summary.txt"):
            try:
                for fname, sz in parse_chbmit_summary(p).items():
                    summaries[fname] = (p, sz)
            except OSError as exc:
                problems.append({"recording_id": "", "input": rel(p), "status": "warning",
                                 "reason": f"unreadable summary: {exc}"})
    labels_rows: list[dict] = []
    for p in files:
        if p.name == "labels.csv":
            try:
                labels_rows.extend(read_table(p, ","))
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                problems.append({"recording_id": "", "input": rel(p), "status": "warning",
                                 "reason": f"unreadable labels.csv: {exc}"})

    for r in recs:
        _sidecars(r)
        if r.events_tsv is not None:
            _events_from_tsv(r)
        if r.path.name in summaries and not r.seizures:
            sp, sz = summaries[r.path.name]
            r.labels_known = True
            r.label_sources.append(sp.name)
            r.seizures.extend((a, b - a) for a, b in sz)
        for row in labels_rows:
            if row.get("subject", "").removeprefix("sub-") not in ("", r.subject):
                continue
            if row.get("recording_id") not in ("", None, r.recording_id):
                continue
            on, dur = _float(row.get("onset_s")), _float(row.get("duration_s"))
            if on is None or (row.get("trial_type") or "seizure").lower() not in SEIZURE_TYPES:
                continue
            r.labels_known = True
            if "labels.csv" not in r.label_sources:
                r.label_sources.append("labels.csv")
            if not any(abs(on - s) < 1.0 for s, _ in r.seizures):
                r.seizures.append((on, dur if dur and dur > 0 else None))
        if r.n_seizures is not None and not r.labels_known:
            r.labels_known = True
            r.label_sources.append("recordings.csv n_seizures")
            if r.n_seizures > 0:
                r.notes.append(f"recordings.csv says {r.n_seizures} seizure(s) but none has a time; not scored")
                r.labels_known = False
        r.seizures.sort()

    out: list[Recording] = []
    by_id: dict[str, Recording] = {}
    for r in sorted(recs, key=lambda r: (r.source != "recordings.csv", r.rel)):
        if r.recording_id in by_id:
            keep = by_id[r.recording_id]
            problems.append({"recording_id": r.recording_id, "input": r.rel, "status": "skipped",
                             "reason": f"duplicate recording id; {keep.rel} is used"})
            continue
        by_id[r.recording_id] = r
        out.append(r)
    out.sort(key=lambda r: r.recording_id)
    return out, seen, problems


def excluded_intervals(rec: Recording, exclude_bad_segments: bool) -> list[tuple[float, float, str]]:
    """Intervals left out of detection: annotated `artifact` rows always; eeg-prep's automatic
    `bad_segment` rows only with EXCLUDE_BAD_SEGMENTS=true, because high-amplitude ictal activity
    exceeds eeg-prep's amplitude criterion and its bad segments can cover the very seizures this
    step must find."""
    return [e for e in rec.excluded if e[2] == "artifact" or (exclude_bad_segments and e[2] == "bad_segment")]


# --- channels and montage ------------------------------------------------------------------

def electrode(name: str) -> str | None:
    """The canonical 10-20/10-10 name of one electrode label, or None."""
    return ELECTRODES.get(name.strip().upper())


def clean_label(name: str) -> str:
    """Strip recording-system decorations: `EEG Fp1-Ref` -> `Fp1`, `Fp1-Avg` -> `Fp1`,
    `T8-P8-1` (MNE's de-duplication) -> `T8-P8`."""
    n = re.sub(r"^(EEG|eeg)[\s_]+", "", name.strip())
    n = re.sub(r"[-_ ](Avg|AVG|avg|Ref|REF|ref|CAR|AR|LE|M1|M2|A1A2)$", "", n)
    parts = n.split("-")
    if len(parts) == 3 and parts[2].isdigit():
        n = "-".join(parts[:2])
    return n


@dataclass
class Montage:
    kind: str                      # "average" | "bipolar-longitudinal"
    names: list[str]               # canonical output names (Fp1, or Fp1-F7)
    sources: list[list[str]]       # raw channel names: [name] or [anode, cathode] of a derived pair
    missing: list[str]             # SzCORE electrodes not available (referential)
    ignored: dict[str, str]        # raw channel -> why it is not used
    note: str = ""


def montage_for(raw_names: list[str], raw_types: list[str], bad: set[str], tsv_types: dict[str, str]) -> Montage:
    """
    Which channels are analysed and how. Referential recordings (single electrodes): the 19
    SzCORE electrodes that are present and good, re-referenced to their common average.
    Bipolar-only recordings (channels are pairs such as `FP1-F7`): the pairs whose two
    contacts are 10-20/10-10 electrodes, de-duplicated (a pair and its reverse count once).
    """
    ignored: dict[str, str] = {}
    referential: dict[str, str] = {}
    pairs: dict[frozenset, tuple[str, str]] = {}
    for name, mtype in zip(raw_names, raw_types):
        ttype = tsv_types.get(name, "").upper()
        if name in bad:
            ignored[name] = "marked bad"
            continue
        if ttype and ttype != "EEG":
            ignored[name] = f"type {ttype}"
            continue
        if not ttype and mtype.upper() not in ("EEG", "MISC") or NON_EEG_NAME.match(name):
            ignored[name] = f"not EEG ({mtype})"
            continue
        label = clean_label(name)
        el = electrode(label)
        if el:
            if el in referential:
                ignored[name] = f"duplicate of {referential[el]}"
            else:
                referential[el] = name
            continue
        parts = label.split("-")
        if len(parts) == 2 and electrode(parts[0]) and electrode(parts[1]):
            a, b = electrode(parts[0]), electrode(parts[1])
            key = frozenset((a, b))
            if key in pairs:
                ignored[name] = f"duplicate of {pairs[key][0]}"
            else:
                pairs[key] = (name, f"{a}-{b}")
            continue
        ignored[name] = "not a 10-20/10-10 electrode or pair"
    present = [c for c in SZCORE_CHANNELS if c in referential]
    if len(present) >= 4:
        for el, name in referential.items():
            if el not in SZCORE_CHANNELS:
                ignored[name] = "not one of the 19 SzCORE 10-20 electrodes"
        for name, _ in pairs.values():
            ignored[name] = "bipolar pair in a referential recording (the referential channels are used)"
        missing = [c for c in SZCORE_CHANNELS if c not in referential]
        return Montage("average", present, [[referential[c]] for c in present], missing, ignored,
                       "the 19 10-20 electrodes in common average reference (SzCORE)" if not missing else
                       f"{len(present)} of the 19 10-20 electrodes in common average reference; missing "
                       + ", ".join(missing))
    if len(pairs) >= 2:
        for el, name in referential.items():
            ignored[name] = "single electrode in a bipolar-only recording"
        items = sorted(pairs.values(), key=lambda t: _pair_order(t[1]))
        return Montage("bipolar-longitudinal", [n for _, n in items], [[s] for s, _ in items], [], ignored,
                       f"bipolar-only recording: {len(items)} pairs kept as recorded (no referential data to "
                       "re-reference; SzCORE expects average reference)")
    return Montage("none", [], [], list(SZCORE_CHANNELS), ignored,
                   f"too few usable EEG channels ({len(referential)} electrodes, {len(pairs)} pairs)")


_DOUBLE_BANANA = ["Fp1-F7", "F7-T7", "T7-P7", "P7-O1", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F4", "F4-C4",
                  "C4-P4", "P4-O2", "Fp2-F8", "F8-T8", "T8-P8", "P8-O2", "Fz-Cz", "Cz-Pz"]


def _pair_order(name: str) -> tuple[int, str]:
    return (_DOUBLE_BANANA.index(name) if name in _DOUBLE_BANANA else len(_DOUBLE_BANANA), name)


def channel_table(rec: Recording) -> tuple[set[str], dict[str, str]]:
    """(bad channel names, {name: BIDS type}) from the channels.tsv sidecar, when there is one."""
    bad: set[str] = set()
    types: dict[str, str] = {}
    if rec.channels_tsv is None:
        return bad, types
    try:
        rows = read_table(rec.channels_tsv, "\t")
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        rec.notes.append(f"channels.tsv unreadable: {exc}")
        return bad, types
    for row in rows:
        name = row.get("name", "")
        if not name:
            continue
        types[name] = row.get("type", "")
        if (row.get("status") or "good").lower() == "bad":
            bad.add(name)
    return bad, types


# --- wall clock --------------------------------------------------------------------------

def start_datetime(rec: Recording, raw) -> datetime | None:
    """The recording's wall-clock start: recordings.csv `start_time`, else the SzCORE
    `dateTime` of its events.tsv, else the file header (MNE `meas_date`)."""
    for value in (rec.start_time, rec.event_datetime):
        if value:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    return datetime.strptime(value.replace("Z", "").split("+")[0], fmt)
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(value).replace(tzinfo=None)
            except ValueError:
                rec.notes.append(f"start time {value!r} not understood")
    md = raw.info.get("meas_date")
    if md is not None:
        if md.tzinfo is not None:
            md = md.astimezone(timezone.utc).replace(tzinfo=None)
        return md.replace(microsecond=0)
    return None


def sidecar(rec: Recording) -> dict:
    if rec.sidecar_json is None:
        return {}
    try:
        return json.loads(rec.sidecar_json.read_text())
    except (OSError, ValueError):
        return {}


# --- signal ------------------------------------------------------------------------------

def open_raw(path: Path):
    """The recording, lazily (nothing loaded)."""
    import mne

    return mne.io.read_raw(str(path), preload=False, verbose="error")


def read_window(raw, picks: list[int], start: int, stop: int):
    """Samples [start, stop) of the picked channels as float64 microvolts (channels, samples)."""
    return raw.get_data(picks=picks, start=start, stop=stop) * 1e6  # MNE returns volts
