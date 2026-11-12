# Third-party components

The image bundles the packages below (versions from `requirements.txt`, licences from each
package's metadata in the image and its repository, checked 2026-11-12). This app's own code is
Apache-2.0 (`LICENSE`). **No model weights are bundled**: a model linked to the optional
`model` port is the user's, under its own licence (its `licence` field is shown in
`provenance.json` and the report).

| Component | Version | What it does here | Authors / copyright | Licence | URL |
|---|---|---|---|---|---|
| timescoring | 0.0.7 | SzCORE sample- and event-based scoring | Jonathan Dan, Una Pale (ESL-EPFL) | MIT (LICENSE in the repository; the wheel's metadata leaves the field empty) | https://github.com/esl-epfl/timescoring |
| MNE-Python | 1.13.2 | Reads EDF/BDF/BrainVision/FIF/EEGLAB lazily, window by window | MNE-Python developers | BSD-3-Clause | https://github.com/mne-tools/mne-python |
| NumPy | 2.5.4 | Arrays | NumPy developers | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | https://github.com/numpy/numpy |
| SciPy | 1.18.1 | Band-pass filters (`sosfiltfilt`), polyphase resampling | SciPy developers | BSD-3-Clause | https://github.com/scipy/scipy |
| pandas | 3.1.1 | Rolling median / MAD baselines | pandas developers | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| Matplotlib | 3.11.2 | Figures | Matplotlib Development Team | Matplotlib License (PSF-based, BSD-compatible) | https://github.com/matplotlib/matplotlib |
| ONNX Runtime | 1.31.1 | Runs a user-supplied ONNX model (`DETECTOR=model`) | Microsoft Corporation | MIT | https://github.com/microsoft/onnxruntime |
| protobuf | 7.36.2 | ONNX Runtime dependency | Google LLC | BSD-3-Clause | https://github.com/protocolbuffers/protobuf |
| flatbuffers | 25.12.19 | ONNX Runtime dependency | Google LLC | Apache-2.0 | https://github.com/google/flatbuffers |
| contourpy, cycler, kiwisolver, fonttools, pillow, pyparsing | see requirements.txt | Matplotlib dependencies | their authors | BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, MIT, MIT-CMU, MIT | PyPI |
| python-dateutil, tzdata | 4.0.0, 2026.6 | pandas dependencies | their authors | BSD-3-Clause, Apache-2.0 | PyPI |
| decorator, lazy-loader, packaging, Jinja2, MarkupSafe, tqdm, pooch, platformdirs | see requirements.txt | MNE-Python dependencies (pooch's downloads are never used: no network at run time) | their authors | BSD-2-Clause, BSD-3-Clause, Apache-2.0 OR BSD-2-Clause, BSD-3-Clause, BSD-3-Clause, MPL-2.0 AND MIT, BSD-3-Clause, MIT | PyPI |
| requests, urllib3, idna, charset-normalizer, certifi | see requirements.txt | pooch dependencies | their authors | Apache-2.0, MIT, BSD-3-Clause, MIT, MPL-2.0 | PyPI |
| Python | 3.12.16 | Runtime (`python:3.12.16-slim-bookworm`, pinned by digest) | Python Software Foundation | PSF-2.0 | https://www.python.org |

## Formats and methods used, not bundled

| What | Used for | Licence |
|---|---|---|
| SzCORE (esl-epfl/szcore) | The event file format (`Annotations.saveTsv` of epilepsy2bids 0.0.7) and scoring rules, re-implemented in `processor/szcore.py` from the published format; no code copied | MIT |
| szcore-evaluation | Its aggregation rules (summed counts; subject means) are followed; no code copied (the repository has no licence file) | none stated |
| Line length (Esteller et al. 2001) | The detector's feature, written from the paper | (method) |

Not used, on purpose (builder brief): the GPL-3.0 `gotman_1982`, `eventnet_2024` and SeizUnet
reimplementations; SeizureTransformer weights (only inside a Docker image, no licence); DynaSD
(CC BY-NC).
