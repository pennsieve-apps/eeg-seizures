"""
A small synthetic scalp EEG input for the tests, `make run` and the app.yml review block.

  <out>/eeg/                          the `eeg` port: a BIDS-EEG derivative as eeg-prep writes it
    dataset_description.json, recordings.csv
    sub-01/ses-01/eeg/sub-01_ses-01_task-szMonitoring_run-01_desc-preproc_{eeg.edf,eeg.json,channels.tsv,events.tsv}
        19 channels, 10-20, common average reference, 256 Hz, 15 min. Two planted seizures:
        a focal left-temporal rhythmic discharge (360-405 s, 6 Hz slowing to 3.5 Hz, building
        up to ~150 µV on F7/T7/P7) and a generalised, frontally dominant 3 Hz spike-wave discharge (690-720 s).
        Distractors that must NOT be detected: a 3 s high-amplitude artifact at 560 s (shorter
        than MIN_DURATION_S) and a 30 s burst of large noise at 800-830 s marked as a
        `bad_segment` in events.tsv. Fz is marked bad in channels.tsv.
    sub-02/ses-01/eeg/sub-02_ses-01_task-szMonitoring_run-01_desc-preproc_*  512 Hz (resampled in the
        app), 8 min, no seizure (a recording with no detection must give a `bckg` SzCORE row).
    loose/chb99/chb99_01.edf + chb99-summary.txt
        CHB-MIT style: 23 longitudinal bipolar pairs only (FP1-F7 ..., T8-P8 twice), 256 Hz,
        12 min, one seizure 400-440 s listed in the summary file (recording id
        sub-chb99_task-eeg_run-01).
  <out>/truth.json                    the planted seizures per recording id.

Written with numpy and edfio only (the review generator image has no MNE).
Usage: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SEED = 42
CH19 = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "Fz", "Cz", "Pz",
        "Fp2", "F4", "C4", "P4", "O2", "F8", "T8", "P8"]
BIPOLAR = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
           "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
           "FT9-FT10", "FT10-T8", "T8-P8"]
EXTRA = ["FT9", "FT10"]


def pink(n: int, rng: np.random.Generator, std: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std() * std


def background(names: list[str], fs: float, dur: float, rng) -> np.ndarray:
    n = int(fs * dur)
    common = pink(n, rng, 8.0)
    x = np.vstack([pink(n, rng, 12.0) + 0.5 * common for _ in names])
    t = np.arange(n) / fs
    for i, name in enumerate(names):
        if name in ("O1", "O2", "P3", "P4", "Pz", "P7", "P8"):
            env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t + i)
            x[i] += 15.0 * env * np.sin(2 * np.pi * 10.0 * t + i)
    return x


def rhythmic_seizure(fs: float, dur: float, f0: float, f1: float, amp: float) -> np.ndarray:
    """A discharge whose frequency slows from f0 to f1 Hz and whose amplitude builds up over
    the first third, with sharp (harmonic-rich) waveforms."""
    t = np.arange(int(fs * dur)) / fs
    freq = f0 + (f1 - f0) * t / dur
    phase = 2 * np.pi * np.cumsum(freq) / fs
    wave = np.sin(phase) + 0.45 * np.sin(2 * phase + 0.6) + 0.2 * np.sin(3 * phase + 1.1)
    env = np.minimum(1.0, t / (dur / 3)) * np.minimum(1.0, (dur - t) / 3.0)
    return amp * env * wave / np.abs(wave).max()


def spike_wave(fs: float, dur: float, amp: float) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    ph = (t * 3.0) % 1.0
    spike = np.exp(-((ph - 0.1) / 0.025) ** 2)
    wave = -0.6 * np.exp(-((ph - 0.45) / 0.15) ** 2)
    env = np.minimum(1.0, t / 2.0) * np.minimum(1.0, (dur - t) / 2.0)
    return amp * env * (spike + wave)


# 3 Hz spike-wave is bilateral and frontally dominant (a field identical on every electrode
# would vanish in the common average reference).
SPIKE_WAVE = {"Fp1": 1.0, "Fp2": 1.0, "F3": 1.0, "F4": 1.0, "Fz": 1.0, "F7": 0.8, "F8": 0.8, "C3": 0.6, "C4": 0.6,
              "Cz": 0.7, "T7": 0.4, "T8": 0.4, "P3": 0.3, "P4": 0.3, "Pz": 0.3, "P7": 0.2, "P8": 0.2, "O1": 0.1,
              "O2": 0.1}
FOCAL = {"F7": 1.0, "T7": 1.0, "P7": 0.8, "F3": 0.5, "C3": 0.5, "Fp1": 0.4, "O1": 0.3, "P3": 0.4}


def add(x: np.ndarray, names: list[str], fs: float, t0: float, sig: np.ndarray, gains: dict[str, float]) -> None:
    i0 = int(t0 * fs)
    for i, n in enumerate(names):
        g = gains.get(n, 0.0)
        if g:
            x[i, i0:i0 + len(sig)] += g * sig


def write_edf(path: Path, data: np.ndarray, names: list[str], fs: float, startdate=None, starttime=None) -> None:
    import datetime as dt

    from edfio import Edf, EdfSignal

    sigs = []
    for x, name in zip(data, names):
        r = max(1.0, float(np.max(np.abs(x))) * 1.1)
        sigs.append(EdfSignal(x, sampling_frequency=fs, label=name, physical_dimension="uV", physical_range=(-r, r)))
    edf = Edf(sigs, starttime=starttime or dt.time(0, 0, 0))
    if startdate is not None:
        edf.startdate = startdate
    path.parent.mkdir(parents=True, exist_ok=True)
    edf.write(path)


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


REC_COLS = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good",
            "duration_s", "montage", "reference", "kind", "start_time", "n_seizures", "source_format", "age", "sex"]


def derivative_recording(root: Path, sub: str, fs: float, data: np.ndarray, start: str, events: list[dict],
                         bad=()) -> dict:
    import datetime as dt

    stem = f"sub-{sub}_ses-01_task-szMonitoring_run-01"
    d = root / f"sub-{sub}" / "ses-01" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    when = dt.datetime.fromisoformat(start)
    write_edf(d / f"{stem}_desc-preproc_eeg.edf", data, CH19, fs, when.date(), when.time())
    write_tsv(d / f"{stem}_desc-preproc_channels.tsv",
              ["name", "type", "units", "sampling_frequency", "status", "status_description", "contacts"],
              [{"name": n, "type": "EEG", "units": "µV", "sampling_frequency": f"{fs:g}",
                "status": "bad" if n in bad else "good", "status_description": "flat" if n in bad else "n/a",
                "contacts": n} for n in CH19])
    write_tsv(d / f"{stem}_desc-preproc_events.tsv", ["onset", "duration", "trial_type", "source", "channels"], events)
    (d / f"{stem}_desc-preproc_eeg.json").write_text(json.dumps(
        {"SamplingFrequency": fs, "EEGReference": "average", "PowerLineFrequency": 50,
         "RecordingDuration": data.shape[1] / fs, "EEGChannelCount": len(CH19)}, indent=2))
    n_sz = sum(1 for e in events if e["trial_type"] == "seizure")
    return {"recording_id": stem, "subject": sub, "session": "01", "task": "szMonitoring", "run": "01",
            "path": f"sub-{sub}/ses-01/eeg/{stem}_desc-preproc_eeg.edf", "sfreq": f"{fs:g}", "n_channels": len(CH19),
            "n_good": len(CH19) - len(bad), "duration_s": f"{data.shape[1] / fs:g}", "montage": "average",
            "reference": "average", "kind": "routine", "start_time": start, "n_seizures": n_sz,
            "source_format": "bids-edf", "age": "", "sex": ""}


def make(out: Path) -> Path:
    rng = np.random.default_rng(SEED)
    root = out / "eeg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "eeg-seizures synthetic fixture", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "eeg-prep", "Description": "tests/make_fixture.py of eeg-seizures"}]}, indent=2))
    truth: dict = {}
    recs = []

    # sub-01: two seizures, an artifact, a bad segment
    fs, dur = 256.0, 900.0
    x = background(CH19, fs, dur, rng)
    add(x, CH19, fs, 360.0, rhythmic_seizure(fs, 45.0, 6.0, 3.5, 150.0), FOCAL)
    add(x, CH19, fs, 690.0, spike_wave(fs, 30.0, 180.0), SPIKE_WAVE)
    add(x, CH19, fs, 560.0, 400.0 * np.hanning(int(3 * fs)) * np.sign(rng.standard_normal(int(3 * fs))),
        {"Fp1": 1.0, "Fp2": 1.0, "F7": 0.6, "F8": 0.6})
    i0 = int(800.0 * fs)
    x[:, i0:i0 + int(30 * fs)] += 150.0 * rng.standard_normal((len(CH19), int(30 * fs)))
    x[CH19.index("Fz")] = 0.05 * rng.standard_normal(x.shape[1])  # flat, marked bad
    good = [i for i, n in enumerate(CH19) if n != "Fz"]
    x[good] -= x[good].mean(axis=0)  # common average of the good channels, as eeg-prep writes it
    ev = [{"onset": "360.0", "duration": "45.0", "trial_type": "seizure", "source": "annotation", "channels": ""},
          {"onset": "690.0", "duration": "30.0", "trial_type": "seizure", "source": "annotation", "channels": ""},
          {"onset": "800.0", "duration": "30.0", "trial_type": "bad_segment", "source": "eeg-prep", "channels": ""}]
    recs.append(derivative_recording(root, "01", fs, x, "2024-03-05T22:14:03", ev, bad=("Fz",)))
    truth[recs[-1]["recording_id"]] = {"seizures": [[360.0, 405.0], [690.0, 720.0]], "artifact": [560.0, 563.0],
                                       "bad_segment": [800.0, 830.0]}

    # sub-02: 512 Hz, no seizure
    fs, dur = 512.0, 480.0
    x = background(CH19, fs, dur, rng)
    x -= x.mean(axis=0)
    recs.append(derivative_recording(root, "02", fs, x, "2024-03-06T09:00:00", []))
    truth[recs[-1]["recording_id"]] = {"seizures": []}

    with open(root / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REC_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)

    # CHB-MIT style bipolar-only loose EDF
    fs, dur = 256.0, 720.0
    names = CH19 + EXTRA
    x = background(names, fs, dur, rng)
    add(x, names, fs, 400.0, rhythmic_seizure(fs, 40.0, 5.0, 3.0, 150.0),
        {"F8": 1.0, "T8": 1.0, "P8": 0.8, "F4": 0.5, "C4": 0.5, "Fp2": 0.4, "FT10": 0.7})
    idx = {n.upper(): i for i, n in enumerate(names)}
    pairs = np.vstack([x[idx[a]] - x[idx[b]] for a, b in (p.split("-") for p in BIPOLAR)])
    loose = root / "loose" / "chb99"
    import datetime as dt

    write_edf(loose / "chb99_01.edf", pairs, BIPOLAR, fs, dt.date(2075, 1, 1), dt.time(11, 42, 54))
    (loose / "chb99-summary.txt").write_text(
        "Data Sampling Rate: 256 Hz\n*************************\n\n"
        "File Name: chb99_01.edf\nFile Start Time: 11:42:54\nFile End Time: 11:54:54\n"
        "Number of Seizures in File: 1\nSeizure Start Time: 400 seconds\nSeizure End Time: 440 seconds\n\n")
    truth["sub-chb99_task-eeg_run-01"] = {"seizures": [[400.0, 440.0]]}

    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return out


if __name__ == "__main__":
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture")))
