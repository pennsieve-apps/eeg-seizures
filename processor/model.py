"""
A user-supplied seizure detection model (the optional `model` input port, DETECTOR=model).

No model is bundled. A model folder holds `model.json` and the model file it names:

    {
      "name": "my-detector",                 # shown in provenance and the report
      "format": "onnx",                      # "onnx" (run with onnxruntime, in the image) or
                                             # "torchscript" (needs PyTorch, not in the image)
      "file": "model.onnx",
      "sfreq": 256,                          # Hz; only 256 is accepted (SzCORE)
      "window_s": 10,                        # W: seconds per input window
      "step_s": 10,                          # seconds between window starts (default W)
      "channels": ["Fp1", "F3", ...],        # input channel order; default the 19 SzCORE 10-20
                                             # electrodes; bipolar names ("Fp1-F7") for a
                                             # model trained on bipolar data
      "montage": "average",                  # "average" or "bipolar-longitudinal"
      "units": "uV",                         # "uV" (default) or "V"
      "threshold": 0.5,                      # default decision threshold on the probability
      "licence": "MIT",                      # shown to the user (provenance licence_notice)
      "citation": "10.xxxx/...",             # DOI to cite, optional
      "bandpass": null,                      # [low, high] Hz zero-phase band-pass applied
                                             # before the model, or null (the default: the
                                             # signal as SzCORE gives it, only resampled)
      "input_name": null, "output_name": null   # graph names when the model has several
    }

The model gets float32 tensors of shape (batch, n_channels, 256 * W): the 256 Hz signal of
each channel in the order of `channels`, in the model's montage (a bipolar model on a
referential recording gets the pairs computed from the electrodes), µV unless `units` says V. It must return per-second seizure probabilities of
shape (batch, W) (or (batch, W, 1)) in [0, 1]. Where windows overlap (step < W), the
probabilities of a second are averaged. A channel the recording lacks is filled with zeros
(listed in provenance); more than a quarter missing skips the recording.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ModelError(Exception):
    pass


@dataclass
class Model:
    folder: Path
    name: str
    format: str
    file: Path
    window_s: int
    step_s: int
    channels: list[str]
    montage: str
    scale: float              # multiply µV by this before feeding
    threshold: float
    licence: str
    citation: str
    bandpass: tuple[float, float] | None = None
    input_name: str | None = None
    output_name: str | None = None
    _session: object = field(default=None, repr=False)

    def load(self, threads: int) -> None:
        if self.format == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - always in the image
                raise ModelError("onnxruntime is not installed") from exc
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, threads)
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(self.file), sess_options=so, providers=["CPUExecutionProvider"])
            self.input_name = self.input_name or sess.get_inputs()[0].name
            self.output_name = self.output_name or sess.get_outputs()[0].name
            self._session = sess
        elif self.format == "torchscript":
            try:
                import torch
            except ImportError as exc:
                raise ModelError("TorchScript models need PyTorch, which this image does not include (it would "
                                 "add ~700 MB). Export the model to ONNX (torch.onnx.export) and set "
                                 "\"format\": \"onnx\".") from exc
            torch.set_num_threads(max(1, threads))
            self._session = torch.jit.load(str(self.file), map_location="cpu").eval()
        else:
            raise ModelError(f"model.json format must be onnx or torchscript (got {self.format!r})")

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """(batch, channels, 256 W) float32 -> (batch, W) probabilities."""
        x = (batch * self.scale).astype(np.float32, copy=False)
        if self.format == "onnx":
            y = self._session.run([self.output_name], {self.input_name: x})[0]
        else:
            import torch

            with torch.no_grad():
                y = self._session(torch.from_numpy(x)).cpu().numpy()
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y[..., 0]
        if y.shape != (batch.shape[0], self.window_s):
            raise ModelError(f"model returned shape {tuple(y.shape)}, expected ({batch.shape[0]}, {self.window_s})")
        return np.clip(np.nan_to_num(y, nan=0.0), 0.0, 1.0)


def find_model_folder(root: Path) -> Path | None:
    """The folder of the first model.json under root (following symlinks), or None."""
    import os

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if "model.json" in filenames:
            return Path(dirpath)
    return None


def read_model(folder: Path) -> Model:
    from processor.eeg import SZCORE_CHANNELS

    try:
        spec = json.loads((folder / "model.json").read_text())
    except (OSError, ValueError) as exc:
        raise ModelError(f"model.json unreadable: {exc}") from exc
    fmt = str(spec.get("format", "onnx")).lower()
    default_file = "model.onnx" if fmt == "onnx" else "model.pt"
    file = folder / str(spec.get("file", default_file))
    if not file.is_file():
        raise ModelError(f"model file {file.name} not found next to model.json")
    if int(spec.get("sfreq", 256)) != 256:
        raise ModelError("model.json sfreq must be 256 (SzCORE)")
    w = int(spec.get("window_s", 10))
    step = int(spec.get("step_s", w))
    if w < 1 or step < 1 or step > w:
        raise ModelError("model.json window_s and step_s must be >= 1 with step_s <= window_s")
    montage = str(spec.get("montage", "average"))
    if montage not in ("average", "bipolar-longitudinal"):
        raise ModelError("model.json montage must be average or bipolar-longitudinal")
    chans = list(spec.get("channels") or SZCORE_CHANNELS)
    units = str(spec.get("units", "uV")).lower()
    if units not in ("uv", "µv", "v"):
        raise ModelError("model.json units must be uV or V")
    bp = spec.get("bandpass")
    if bp is not None:
        try:
            bp = (float(bp[0]), float(bp[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ModelError("model.json bandpass must be [low, high] in Hz or null") from exc
        if not 0 < bp[0] < bp[1] < 128:
            raise ModelError("model.json bandpass must satisfy 0 < low < high < 128 Hz")
    thr = float(spec.get("threshold", 0.5))
    if not 0 < thr < 1:
        raise ModelError("model.json threshold must be between 0 and 1")
    return Model(folder=folder, name=str(spec.get("name", folder.name)), format=fmt, file=file, window_s=w,
                 step_s=step, channels=chans, montage=montage, scale=1e-6 if units == "v" else 1.0, threshold=thr,
                 licence=str(spec.get("licence", spec.get("license", "not stated"))),
                 citation=str(spec.get("citation", "")), bandpass=bp, input_name=spec.get("input_name"),
                 output_name=spec.get("output_name"))
