# eeg-seizures

Seizure detection on clinical scalp EEG in the **SzCORE** format, built on the work of others:
the **SzCORE** framework and its **timescoring** library, developed by Jonathan Dan, Una Pale and
colleagues at the Embedded Systems Laboratory of EPFL with the Pedesite consortium
([SzCORE](https://github.com/esl-epfl/szcore), Dan et al. 2024, *Epilepsia*,
doi:10.1111/epi.18113; [timescoring](https://github.com/esl-epfl/timescoring)), which define the
event file format and the scoring rules used here; the **line-length** feature of R. Esteller,
J. Echauz, T. Tcheng, B. Litt and B. Pless (2001, Proc. 23rd IEEE EMBS,
doi:10.1109/IEMBS.2001.1020545); and **MNE-Python**, developed by Alexandre Gramfort and the MNE
community ([mne-python](https://github.com/mne-tools/mne-python), doi:10.3389/fnins.2013.00267),
which reads the recordings. Pennsieve packages these methods and did not develop them. The demo
results below use the Siena Scalp EEG Database (Paolo Detti and colleagues, University of Siena;
BIDS conversion by Jonathan Dan et al., EPFL) and the CHB-MIT Scalp EEG Database (Ali Shoeb and
John Guttag, MIT, with Children's Hospital Boston).

A step of the **Scalp EEG Seizure Detection & Monitoring** workflow
(`eeg-prep → {eeg-seizures, eeg-sleep → eeg-spikes} → eeg-report`).

> **Detections are research outputs: candidates for an expert to review, not a diagnosis.**
> False detections per day are always reported next to sensitivity. On held-out data the default
> detector finds roughly half of the annotated seizures with about a dozen false detections per
> day (see *Validation*).

**For the epileptologist.** The app marks stretches of EEG where the signal suddenly becomes
larger and more rhythmic than in the preceding five minutes on several channels at once, the
way most electrographic seizures look, and lists them as candidate seizures with their onset,
duration and the channels where the change was largest. On long-term monitoring it gives you a
short list of places to look at first. It does not replace reading the EEG: it misses seizures
with little rhythmic build-up (e.g. subtle focal-aware seizures, seizures obscured by artifact)
and it raises false alarms on artifacts (chewing, movement, electrode pops) and on some normal
transients. When the recording has annotated seizures, it reports how many it found and how many
false detections per day it made, so you know how far to trust it on that patient.

**For the researcher.** Input follows the SzCORE contract: the 19 10-20 electrodes in common
average reference at 256 Hz (resampled in-app; bipolar-only recordings such as CHB-MIT are kept
bipolar and recorded as such). Output is one SzCORE event file per recording (`szcore/*.tsv`,
exactly as epilepsy2bids' `Annotations.saveTsv` writes it), plus long-format CSV
(`seizures.csv`, `probability.csv`, `scores.csv`) keyed by `recording_id`. Scores are computed
with timescoring's `SampleScoring` and `EventScoring` at SzCORE's defaults and aggregated as
szcore-evaluation does. The default detector needs no training; an **ONNX model of your own** can
be plugged in through the optional `model` port without rebuilding the app.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `eeg` | input | subfolder | The BIDS-EEG derivative of eeg-prep (`recordings.csv`, `*_desc-preproc_eeg.edf` with `_channels.tsv` and `_events.tsv`), any BIDS-EEG dataset (e.g. SzCORE's BIDS Siena / CHB-MIT with HED-SCORE `_events.tsv`), or loose EDF/BDF/BrainVision/FIF/EEGLAB files. Seizure times come from `_events.tsv` (`trial_type` `seizure`, or HED-SCORE `eventType` `sz…`), CHB-MIT `*-summary.txt` next to the EDFs, or a `labels.csv` (`subject, recording_id, onset_s, duration_s, trial_type`). Found anywhere, folders and symlinks followed. |
| `model` | input, optional | subfolder | A folder with `model.json` and an ONNX model (*Bring your own model*). Used only with `DETECTOR=model`. No model is bundled. |
| `seizures` | output | `OUTPUT_DIR/eeg-seizures/` (or `…/eeg-seizures/<LABEL>/`) | The tables, `szcore/`, `figures/`, `summary.json`, `provenance.json` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `DETECTOR` | `linelength` | `linelength` (no training, no model) or `model` (the ONNX model on the `model` port). |
| `THRESHOLD` | `4.5` | Line-length detector: an event starts when the combined score (a robust z-score against the previous 5 minutes) reaches this and ends when it drops below 0.75 × this. Lower finds more seizures and raises more false alarms. Tuned on 7 Siena subjects (*Validation*). |
| `MODEL_THRESHOLD` | `auto` | Model detector: `auto` = the model's own `threshold` (else 0.5), or a probability 0.01–0.99. |
| `TOP_K` | `8` | The per-second score is the mean z-score of the `TOP_K` channels that rose most, so a focal seizure on a few channels still counts. |
| `MIN_DURATION_S` | `20` | Candidate events shorter than this are dropped (before merging). |
| `MERGE_GAP_S` | `90` | Events closer than this are merged (SzCORE merges at 90 s when scoring). |
| `MAX_DURATION_S` | `300` | Longer events are split into 5-minute pieces, as SzCORE's scoring does (`0` = never). |
| `BASELINE_S` | `300` | Trailing window whose median and MAD normalise each channel. |
| `HIGHPASS_HZ`, `LOWPASS_HZ` | `1`, `20` | Zero-phase 4th-order Butterworth band-pass before line length (`LOWPASS_HZ=0`: high-pass only). 20 Hz keeps rhythmic ictal activity and suppresses muscle artifact. Not applied to a model's input (the model says what it wants, below). |
| `WINDOW_S` | `600` | Windows the recording is read in (plus 5 s of context each side): memory depends on this, not on the recording's length. |
| `EXCLUDE_BAD_SEGMENTS` | `false` | Whether eeg-prep's automatic `bad_segment` intervals are left out of detection. **Off by default**: eeg-prep marks a segment bad when its amplitude exceeds `AMPLITUDE_UV` (500 µV), and high-amplitude ictal activity does exactly that, so its bad segments can cover the seizures this step must find (on chb01 they cover all 7; see *Validation*). Set `true` to trade sensitivity for far fewer false detections from artifact. Annotated `artifact` intervals of the source recording are always left out. |
| `SCORE` | `true` | Score against annotated seizures when a recording has an annotation source. Recordings without one are not scored (no annotation is not the same as "no seizure"). |
| `THREADS` | `0` | Threads for ONNX Runtime and BLAS; `0` = one per vCPU of the container's cgroup quota (not the host's count). Logged and kept in `provenance.json`. |
| `LABEL` | (empty) | Write to `eeg-seizures/<LABEL>/` (e.g. one run per detector). |

Fixed (documented, not parameters): line length over 2 s windows every 1 s, a 5 s centred
moving mean of the combined score, hysteresis at 0.75 × the threshold, and SzCORE's scoring
parameters (30 s before, 60 s after, any overlap, 5 min split, 90 s merge). There is no random
step, so there is no `SEED`.

## Outputs

All under `OUTPUT_DIR/eeg-seizures/`. Times are seconds from the recording start.

**`seizures.csv`**: one row per candidate seizure: `recording_id`, `onset_s`, `duration_s`,
`confidence` (mean per-second probability over the event), `channels` (line-length detector: up
to `TOP_K` channels whose z-score over the first 20 s of the event reached the threshold,
largest first; empty for a model), `detector` (`linelength` or `model:<name>`).

**`szcore/<recording_id>.tsv`**: the same events in SzCORE's HED-SCORE format: `onset, duration,
eventType (sz), confidence, channels, dateTime, recordingDuration`, two decimals, `dateTime` the
recording's wall-clock start (`recordings.csv` `start_time`, else the SzCORE `dateTime` of the
source events, else the EDF header; `n/a` when unknown); a recording without detections gets one
`bckg` row spanning it, as SzCORE writes. These files load with `epilepsy2bids`'
`Annotations.loadTsv`, so they can go straight into szcore-evaluation.

**`probability.csv`**: `recording_id, time_s, probability`, one row per second. For the
line-length detector this is the score mapped to 0–1 with 0.5 at the threshold (a logistic of
score − threshold): an ordering, **not a calibrated probability**. Empty in the intervals left out (annotated artifacts; eeg-prep bad segments only with `EXCLUDE_BAD_SEGMENTS=true`).

**`scores.csv`**: `recording_id, scope (sample | event), metric, value` for every scored
recording, then `pooled` (counts and hours added over recordings: SzCORE's "cumulated") and
`subject_mean` (each subject's recordings pooled, then the subjects averaged: SzCORE's default).
Metrics: `sensitivity`, `precision`, `f1`, `fp_per_day`, `tp`, `fp`, `n_reference`, `hours`.
For the sample scope the counts are seconds (so `fp_per_day` is false-positive seconds per day).

**`figures/`** (in `summary.json`, with captions): `overview_detections.png` (every recording on
one time axis, annotated seizures above, detections below), `scores_per_recording.png` (seizures
detected/missed and false detections per day, per recording), and
`<recording_id>_detections.png` (the per-second score with threshold, annotations, detections and
the intervals left out; the first 12 recordings with seizures or detections are listed in `summary.json`,
all are written).

**`summary.json`**: per recording `duration_h, montage, n_channels, n_detections,
detections_per_day, n_annotated_seizures` and, when scored, `event_sensitivity,
event_precision, event_f1, event_fp_per_day, seizures_detected, false_detections, sample_f1`;
`pooled` scores; figures and tables with captions; warnings.

**`provenance.json`**: the conventions' keys (`app, version, variant, tools, parameters, inputs,
started, finished, wall_seconds, peak_rss_mb, threads, recordings, citations,
licence_notice`), plus `resolved` (fixed constants, the model's interface) and per recording the
montage and why (channels used, missing, ignored with the reason), the intervals left out and the number of eeg-prep bad segments kept,
annotation sources, timings and scores. `licence_notice` names a user-supplied model's licence.

## How it works

1. **Finds the recordings** (every `recordings.csv`, plus signal files it does not list, with
   their sidecars), one at a time, opened lazily with MNE. Recording ids follow the conventions:
   the BIDS entity chain, or for loose files `sub-<name>_task-eeg_run-<nn>` (`chb01_03.edf` →
   `sub-chb01_task-eeg_run-03`), as eeg-prep names them.
2. **Montage.** Channel names are cleaned (`Fp1-Avg`, `EEG Fp1-Ref` → `Fp1`; `T3/T4/T5/T6` →
   `T7/T8/P7/P8`). A **referential** recording uses the 19 SzCORE electrodes that are present and
   good (`_channels.tsv` status), re-referenced to their common average (idempotent on data that
   already is). A **bipolar-only** recording (channels are pairs such as `FP1-F7`, as in CHB-MIT)
   keeps its 10-20/10-10 pairs, a pair and its reverse counted once; there is no way back to a
   referential montage, and `provenance.json` says so.
3. **Per window** of `WINDOW_S`: band-pass, resample to 256 Hz (polyphase), and line length of
   each channel over 2 s windows centred every second.
4. **Normalise** each channel's log line length by the median and MAD (× 1.4826) of its
   previous `BASELINE_S` seconds (left-out intervals excluded; the first minutes use the recording's
   first `BASELINE_S`), giving a robust z-score; **combine** as the mean of the `TOP_K` largest
   channel z-scores per second, smoothed over 5 s.
5. **Events**: hysteresis (start at `THRESHOLD`, end below 0.75 × it), then events shorter than
   `MIN_DURATION_S` dropped, events closer than `MERGE_GAP_S` merged, events longer than
   `MAX_DURATION_S` split. Annotated `artifact` intervals of `_events.tsv` are never detected in;
   eeg-prep's `bad_segment` intervals only when `EXCLUDE_BAD_SEGMENTS=true`.
6. **Score** (when annotated): reference and detections as 1 Hz masks (SzCORE's
   `Annotations.getMask`), timescoring `SampleScoring` and `EventScoring` with SzCORE's defaults.
   A reference seizure without an end is scored as onset + 10 s (noted).

## Bring your own model

`DETECTOR=model` runs a model you link to the `model` port, so open-weight detectors can be used
without rebuilding the app. The folder holds `model.json` and the model file:

```json
{
  "name": "my-detector",
  "format": "onnx",
  "file": "model.onnx",
  "sfreq": 256,
  "window_s": 10,
  "step_s": 10,
  "channels": ["Fp1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "Fz", "Cz", "Pz",
               "Fp2", "F4", "C4", "P4", "O2", "F8", "T8", "P8"],
  "montage": "average",
  "units": "uV",
  "bandpass": null,
  "threshold": 0.5,
  "licence": "MIT",
  "citation": "10.xxxx/your-paper"
}
```

- **Input**: float32 tensor `(batch, n_channels, 256 × window_s)`: the 256 Hz signal of each
  channel in the order of `channels` (default: the 19 SzCORE electrodes), in the model's
  `montage` (`average`, or `bipolar-longitudinal` with pair names like `"Fp1-F7"`; a bipolar model
  on referential data gets the pairs computed from the electrodes), in µV (or V with
  `"units": "V"`), band-passed only if `bandpass` is `[low, high]`.
- **Output**: per-second seizure probabilities `(batch, window_s)` (or `(batch, window_s, 1)`) in
  [0, 1]. Windows start every `step_s` seconds; overlapping predictions are averaged. Events are
  made with the same hysteresis and post-processing as above, from `MODEL_THRESHOLD`.
- A channel the recording lacks is filled with zeros (listed in `provenance.json`); more than a
  quarter missing skips the recording. An `average` model skips bipolar-only recordings.
- **ONNX** runs with ONNX Runtime 1.31 (CPU). **TorchScript** (`"format": "torchscript"`) is
  supported by the code but needs PyTorch, which the image does not include (it would add about
  700 MB); export the model with `torch.onnx.export` instead.
- The model's `licence` is shown in `provenance.json` (`licence_notice`) and in the report. Use
  only models whose licence allows it; the app does not check.

## Runtime

`computeTypes: [standard]`, **1 vCPU / 2 GiB**, 2 h timeout. One recording at a time, read in
10-minute windows; memory does not grow with the recording's length beyond a few per-second
arrays (≈ 20 MB for 24 h). Measured in Docker, **linux/amd64 emulated on an M-series Mac**:

| Run | Data | `--cpus` / `--memory` | Wall time | Peak memory (cgroup `memory.peak`, incl. page cache) | Peak RSS |
|---|---|---|---|---|---|
| Siena, all 41 recordings, source data | 141.0 h, 19 ch, 256 Hz, 4.7 GB EDF | 2 / 3g | 93 s in the app | 0.87 GB | 360 MB |
| Siena, all 41 recordings, eeg-prep output | 141.0 h | 2 / 3g | 95 s in the app | 0.99 GB | 360 MB |
| CHB-MIT chb01, 42 files, source data | 40.6 h, 23 bipolar ch, 256 Hz | 2 / 3g | 25 s in the app | 0.62 GB | 349 MB |
| CHB-MIT chb01, 42 files, eeg-prep output | 40.6 h, 21 pairs | 2 / 3g | 38 s in the app | 0.60 GB | 348 MB |
| Siena sub-01 run-00, one long recording | 13.5 h | 1 / 3g | 15 s (11.5 s for the recording: read 1.2 s, compute 6.7 s) | 0.76 GB | 330 MB |
| Synthetic fixture, Lambda-style (`--read-only --user 65534:65534 --tmpfs /tmp`) | 3 recordings, 35 min | 2 / 3g | 6 s | — | 394 MB |

So about a second per 1.5 hours of EEG under emulation. A user-supplied model is slower by
however much its inference costs; the 2 h timeout leaves room. The image is 659 MB unpacked,
≈ 160 MB compressed.

## Validation

**Synthetic fixture** (`tests/make_fixture.py`, `make test`): a focal left-temporal rhythmic
discharge and a frontally dominant 3 Hz spike-wave discharge in a 19-channel average-referenced
recording, a seizure in a CHB-MIT-style bipolar-only recording, and distractors (a 3 s
high-amplitude artifact, an annotated 30 s artifact, a seizure-free 512 Hz recording). The
spike-wave seizure is covered by an eeg-prep-style `bad_segment`. All three seizures are found at
their planted times (focal one on F7/T7), no distractor is detected, and with
`EXCLUDE_BAD_SEGMENTS=true` the covered seizure is lost, as expected.

**Real data**, run with the image above in two ways: on **eeg-prep's output** (its BIDS-EEG
derivative: bad channels, ICA artifact removal, filters; the workflow's path; outputs in
`data/work/siena/eeg-seizures/` and `data/work/chbmit/eeg-seizures/` of workflow-scalp-eeg) and on
the **source datasets** directly (BIDS Siena, CHB-MIT as PhysioNet ships it;
`data/work/<demo>/eeg-seizures-on-source/eeg-seizures/`). SzCORE event scoring with its
defaults, against the same annotations. The defaults were **tuned on 7 of the 14 Siena
subjects of the source data** (sub-00, 03, 06, 09, 11, 13, 16: 21 seizures, 62.3 h), choosing the
low-pass (20/25/30 Hz), `TOP_K`, `MIN_DURATION_S` (10/15/20 s; 30 s was not allowed so that
20–30 s seizures stay detectable) and `THRESHOLD` with the best pooled event F1
(`scripts/sweep_threshold.py`). The other 7 Siena subjects and CHB-MIT chb01 were not looked at
while tuning; nothing was re-tuned on eeg-prep's output.

Event sensitivity (seizures found), precision, F1 and false detections per day:

| Set | Seizures | Hours | Source data: sens. / prec. / **F1** / **FP/day** | eeg-prep output: sens. / prec. / **F1** / **FP/day** |
|---|---|---|---|---|
| Siena, tuning half (7 subjects) | 21 | 62.3 | 0.62 (13) / 0.23 / 0.33 / 16.9 | 0.67 (14) / 0.21 / 0.32 / 20.0 |
| **Siena, held-out half (7 subjects)** | 26 | 78.7 | **0.38 (10) / 0.21 / 0.27 / 11.6** | **0.50 (13) / 0.27 / 0.35 / 10.7** |
| Siena, all 14 subjects (pooled) | 47 | 141.0 | 0.49 (23) / 0.22 / 0.30 / 14.0 | 0.57 (27) / 0.24 / 0.34 / 14.8 |
| Siena, all 14 (subject mean, SzCORE's default) | 47 | 141.0 | 0.54 / 0.40 / 0.33 / 16.1 | 0.63 / 0.42 / 0.40 / 16.8 |
| **CHB-MIT chb01 (held out, bipolar-only, paediatric)** | 7 | 40.6 | **0.86 (6) / 0.75 / 0.80 / 1.2** | **0.71 (5) / 0.71 / 0.71 / 1.2** |

Sample-based (1 s) F1, source / eeg-prep: Siena 0.29 / 0.32 pooled, chb01 0.57 / 0.51. The
Siena held-out subject mean on eeg-prep's output is sensitivity 0.59, precision 0.44, F1 0.39,
14.1 false detections/day. Both Siena runs were re-derived independently from the
`szcore/*.tsv` files with epilepsy2bids' `Annotations.loadTsv` and timescoring against the
dataset's own HED-SCORE files: identical (source 23 TP / 82 FP, eeg-prep 27 TP / 87 FP, 47
reference). chb01's seizures are large and well known to be easy; Siena (adults with focal
epilepsy; 33 of its 47 seizures are focal with impaired awareness) is the harder and more
representative test.

**Why eeg-prep's bad segments are not left out by default.** With `EXCLUDE_BAD_SEGMENTS=true` on
the same eeg-prep output, false detections nearly vanish but so do the seizures: Siena all 14
subjects 0.34 (16 of 47) / 0.89 / 0.49 / 0.34 per day (held-out half 0.27 / 0.88 / 0.41 / 0.31),
and chb01 **0 of 7** seizures, because eeg-prep's amplitude criterion (`AMPLITUDE_UV` 500 µV)
marks every chb01 seizure as a bad segment. The option is there for review settings where a few
false alarms matter more than missed seizures; it is not a fix for artifacts.

For context, the SzCORE benchmark's published Siena event scores
(`results/*/siena.json` in [esl-epfl/szcore](https://github.com/esl-epfl/szcore), read
2026-11-12) put the EPFL reimplementation of Gotman's 1982 classical detector at F1 0.07 with 286
false detections/day; the other entries range from F1 0.00 to 0.82, the strongest being trained
deep models (DeepSOZ-HEM 0.82, SeizureTransformer 0.71, EventNet 0.53), some trained partly on
Siena itself. This app's line-length default is a transparent classical baseline, not a
state-of-the-art detector; plug in a model for more.

## Limitations

- **Review candidates only.** On held-out adult focal epilepsy (Siena) the default finds about
  4–5 in 10 annotated seizures (source data; half on eeg-prep's output) at about 11–12 false
  detections per day; performance differs a lot
  between patients (0–100 % per subject). Do not use it to count seizures or to decide treatment.
- Line length rises with any large, fast, sustained signal: chewing, movement, electrode
  artifact and high-amplitude sleep activity cause false detections; low-amplitude or
  non-rhythmic seizures (many focal-aware seizures) are missed. Annotated `artifact` intervals are
  left out; eeg-prep's automatic `bad_segment` intervals are not, by default (`EXCLUDE_BAD_SEGMENTS`).
- The common average spreads a large focal discharge into every channel with opposite sign, so
  the `channels` column is a hint about where the change was largest, not a localisation.
- The trailing 5-minute baseline adapts: a seizure starting within minutes of a previous one, or
  a gradual evolution over many minutes, may not stand out.
- Defaults were tuned on 21 seizures of 7 Siena subjects; on other populations (neonates, ICU
  EEG, other montages) re-tune `THRESHOLD` on annotated data of that population
  (`scripts/sweep_threshold.py`) or link a model.
- Bipolar-only recordings are analysed bipolar; SzCORE models expect average reference and a
  model with `"montage": "average"` skips them.
- `probability.csv` of the line-length detector is not a calibrated probability.

## Development

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python      # units + the fixture end to end (incl. a toy ONNX model)
make run                               # Docker over the synthetic fixture -> data/output/
make run-demo DEMO=../workflow-scalp-eeg/data/siena/BIDS_Siena
python -m scripts.sweep_threshold --input <annotated data> --cache <dir> --subjects 00,03 --out sweep.csv
python -m scripts.split_scores <scores.csv> 01,05,07 held-out
```

## Licences

Apache-2.0 for this app's code (`LICENSE`). The image bundles timescoring (MIT), MNE-Python,
NumPy, SciPy, pandas (BSD-3-Clause), Matplotlib (Matplotlib License), ONNX Runtime (MIT) and
their dependencies: see `THIRD_PARTY.md`. No model weights are bundled. The SzCORE event format
and scoring rules are re-implemented from the published format (SzCORE, MIT); no GPL
reimplementation (gotman_1982, eventnet_2024, SeizUnet) is used. The demo datasets are not part
of the app: BIDS Siena (CC BY 4.0), Siena on PhysioNet and CHB-MIT (ODC-By 1.0).

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in CITATION.cff.

- Dan J, Pale U, Amirshahi A, Cappelletti W, Ingolfsson TM, Wang X, Cossettini A, Bernini A,
  Benini L, Beniczky S, Atienza D, Ryvlin P. SzCORE: Seizure Community Open‐Source Research
  Evaluation framework for the validation of electroencephalography‐based automated seizure
  detection algorithms. *Epilepsia* 2025;66(S3):14–24 (online 2024). doi:10.1111/epi.18113,
  PMID 39292446.
- Esteller R, Echauz J, Tcheng T, Litt B, Pless B. Line length: an efficient feature for seizure
  onset detection. *2001 Conference Proceedings of the 23rd Annual International Conference of
  the IEEE Engineering in Medicine and Biology Society*, 2001:1707–1710.
  doi:10.1109/IEMBS.2001.1020545.
- Dan J, Pale U. timescoring 0.0.7: library for measuring performance of time series
  classification. https://github.com/esl-epfl/timescoring (MIT). No DOI.
- Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python. *Frontiers in
  Neuroscience* 2013;7:267. doi:10.3389/fnins.2013.00267, PMID 24431986.
- Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. *Nature*
  2020;585:357–362. doi:10.1038/s41586-020-2649-2, PMID 32939066.
- Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific
  computing in Python. *Nature Methods* 2020;17:261–272. doi:10.1038/s41592-019-0686-2,
  PMID 32015543.
- McKinney W. Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python
  in Science Conference* 2010:56–61. doi:10.25080/Majora-92bf1922-00a.
- Hunter JD. Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*
  2007;9(3):90–95. doi:10.1109/MCSE.2007.55.
- ONNX Runtime developers. ONNX Runtime 1.31.1. https://github.com/microsoft/onnxruntime (MIT).

Demo data:

- Dan J, Detti P. BIDS Siena Scalp EEG Database. EPFL, Zenodo 2024. doi:10.5281/zenodo.10640762.
- Detti P. Siena Scalp EEG Database. PhysioNet 2020. doi:10.13026/5d4a-j060.
- Detti P, Vatti G, Zabalo Manrique de Lara G. EEG Synchronization Analysis for Seizure
  Prediction: A Study on Data of Noninvasive Recordings. *Processes* 2020;8(7):846.
  doi:10.3390/pr8070846.
- Shoeb A. CHB-MIT Scalp EEG Database. PhysioNet 2010. doi:10.13026/C2K01R.
- Shoeb AH. Application of machine learning to epileptic seizure onset detection and treatment.
  PhD thesis, Massachusetts Institute of Technology, 2009. http://hdl.handle.net/1721.1/54669.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB,
  Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*
  2000;101(23):e215. doi:10.1161/01.CIR.101.23.e215, PMID 10851218.

## Acknowledgements

SzCORE and timescoring come from the Embedded Systems Laboratory (ESL) of EPFL; SzCORE
originated in the Pedesite consortium (EPFL-ESL, ETH-IIS, CHUV, EPFL-LTS4; Swiss National Science
Foundation grant 193813), and its benchmark platform was started at the ORD for the Sciences
hackathon (EPFL Open Science and SDSC), as its README records. MNE-Python is developed by the MNE
community. Thanks to the Siena and CHB-MIT teams and to PhysioNet for open data.
