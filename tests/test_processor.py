"""The processor end to end on the synthetic fixture (tests/make_fixture.py): planted seizures
are recovered on the referential and the bipolar-only recording, distractors are not
detected, and every output follows the contract (docs/conventions.md of workflow-scalp-eeg)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests import make_fixture

REPO = Path(__file__).resolve().parent.parent


def run(inp: Path, out: Path, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("THREADS", "DETECTOR", "LABEL")}
    e.update({"INPUT_DIR": str(inp), "OUTPUT_DIR": str(out), "PYTHONPATH": str(REPO), "THREADS": "1"}, **env)
    return subprocess.run([sys.executable, "-m", "processor.main"], env=e, cwd=REPO, capture_output=True, text=True,
                          timeout=600)


@pytest.fixture(scope="session")
def fixture(tmp_path_factory):
    return make_fixture.make(tmp_path_factory.mktemp("fixture"))


@pytest.fixture(scope="session")
def output(fixture, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    r = run(fixture, out)
    assert r.returncode == 0, r.stdout + r.stderr
    return out / "eeg-seizures"


def rows(path: Path, delimiter=","):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def test_planted_seizures_are_detected_and_distractors_are_not(fixture, output):
    truth = json.loads((fixture / "truth.json").read_text())
    det = rows(output / "seizures.csv")
    by_rec: dict[str, list] = {}
    for d in det:
        by_rec.setdefault(d["recording_id"], []).append((float(d["onset_s"]), float(d["onset_s"]) + float(d["duration_s"])))
    for rid, t in truth.items():
        found = by_rec.get(rid, [])
        assert len(found) == len(t["seizures"]), (rid, found)
        for (a, b), (fa, fb) in zip(t["seizures"], sorted(found)):
            assert a - 30 <= fa <= b and fb >= a, (rid, a, b, fa, fb)
    sub01 = by_rec["sub-01_ses-01_task-szMonitoring_run-01"]
    assert not any(fa < 563 and fb > 560 for fa, fb in sub01), "3 s artifact detected"
    assert not any(fa < 830 and fb > 800 for fa, fb in sub01), "bad segment detected"
    assert all(d["detector"] == "linelength" for d in det)


def test_focal_seizure_channels(output):
    det = rows(output / "seizures.csv")
    first = [d for d in det if d["recording_id"].startswith("sub-01")][0]
    chans = first["channels"].split(",")
    assert {"F7", "T7"} & set(chans[:2])


def test_szcore_files(output):
    files = sorted(p.name for p in (output / "szcore").glob("*.tsv"))
    assert files == ["sub-01_ses-01_task-szMonitoring_run-01.tsv", "sub-02_ses-01_task-szMonitoring_run-01.tsv",
                     "sub-chb99_task-eeg_run-01.tsv"]
    lines = (output / "szcore" / files[0]).read_text().splitlines()
    assert lines[0] == "onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration"
    f = lines[1].split("\t")
    assert f[2] == "sz" and f[5] == "2024-03-05 22:14:03" and f[6] == "900.00"
    bckg = (output / "szcore" / files[1]).read_text().splitlines()
    assert bckg[1].split("\t")[:3] == ["0.00", "480.00", "bckg"]


def test_scores(output):
    sc = {(r["recording_id"], r["scope"], r["metric"]): r["value"] for r in rows(output / "scores.csv")}
    assert float(sc[("pooled", "event", "sensitivity")]) == 1.0
    assert float(sc[("pooled", "event", "fp")]) == 0
    assert float(sc[("pooled", "event", "n_reference")]) == 3
    assert float(sc[("sub-02_ses-01_task-szMonitoring_run-01", "event", "fp_per_day")]) == 0
    assert ("subject_mean", "event", "f1") in sc


def test_probability_one_row_per_second(output):
    pr = rows(output / "probability.csv")
    n = {}
    for r in pr:
        n[r["recording_id"]] = n.get(r["recording_id"], 0) + 1
    assert n == {"sub-01_ses-01_task-szMonitoring_run-01": 900, "sub-02_ses-01_task-szMonitoring_run-01": 480,
                 "sub-chb99_task-eeg_run-01": 720}
    bad = [r for r in pr if r["recording_id"].startswith("sub-01") and 805 <= int(r["time_s"]) < 825]
    assert all(r["probability"] == "" for r in bad)  # bad segment left out


def test_provenance_and_summary(output):
    prov = json.loads((output / "provenance.json").read_text())
    for k in ("app", "version", "variant", "tools", "parameters", "inputs", "started", "finished", "wall_seconds",
              "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"):
        assert k in prov
    assert prov["app"] == "eeg-seizures" and prov["status"] == "ok"
    assert set(prov["threads"]) == {"requested", "used", "why"}
    recs = prov["recordings"]
    assert recs["sub-chb99_task-eeg_run-01"]["montage"] == "bipolar-longitudinal"
    assert recs["sub-01_ses-01_task-szMonitoring_run-01"]["montage"] == "average"
    assert recs["sub-01_ses-01_task-szMonitoring_run-01"]["channels_missing"] == ["Fz"]
    assert recs["sub-02_ses-01_task-szMonitoring_run-01"]["sfreq"] == 512
    assert "10.1111/epi.18113" in prov["citations"]
    summ = json.loads((output / "summary.json").read_text())
    assert summ["figures"] and all((output / f["path"]).is_file() and f["caption"] for f in summ["figures"])
    assert all((output / t["path"]).is_file() for t in summ["tables"])
    assert summ["recordings"]["sub-01_ses-01_task-szMonitoring_run-01"]["seizures_detected"] == 2
    assert summ["pooled"]["event"]["pooled"]["fp_per_day"] == 0


def test_bad_parameter_exits_2(fixture, tmp_path):
    assert run(fixture, tmp_path, THRESHOLD="abc").returncode == 2
    assert run(fixture, tmp_path, DETECTOR="gotman").returncode == 2


def test_no_input_exits_1(tmp_path):
    (tmp_path / "in").mkdir()
    assert run(tmp_path / "in", tmp_path / "out").returncode == 1


def test_label_and_flat_layout(fixture, tmp_path):
    r = run(fixture / "eeg" / "loose", tmp_path, LABEL="bipolar")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "eeg-seizures" / "bipolar" / "szcore" / "sub-chb99_task-eeg_run-01.tsv").is_file()


def _toy_onnx(path: Path, window_s: int = 10) -> None:
    """A stand-in model: per second, the sigmoid of (mean |x| over channels - 15 µV) / 1.5."""
    import onnx
    from onnx import TensorProto, helper

    n = window_s * 256
    nodes = [
        helper.make_node("Abs", ["x"], ["a"]),
        helper.make_node("ReduceMean", ["a"], ["m"], axes=[1], keepdims=0),
        helper.make_node("Reshape", ["m", "shape"], ["r"]),
        helper.make_node("ReduceMean", ["r"], ["s"], axes=[2], keepdims=0),
        helper.make_node("Sub", ["s", "c"], ["d"]),
        helper.make_node("Div", ["d", "k"], ["q"]),
        helper.make_node("Sigmoid", ["q"], ["y"]),
    ]
    inits = [helper.make_tensor("shape", TensorProto.INT64, [3], [-1, window_s, 256]),
             helper.make_tensor("c", TensorProto.FLOAT, [], [15.0]),
             helper.make_tensor("k", TensorProto.FLOAT, [], [1.5])]
    g = helper.make_graph(nodes, "toy", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b", 19, n])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["b", window_s])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_model_port(fixture, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    inp = tmp_path / "in"
    (inp / "model").mkdir(parents=True)
    os.symlink(fixture / "eeg", inp / "eeg")
    _toy_onnx(inp / "model" / "model.onnx")
    (inp / "model" / "model.json").write_text(json.dumps(
        {"name": "toy", "format": "onnx", "file": "model.onnx", "window_s": 10, "step_s": 5, "licence": "MIT",
         "threshold": 0.5}))
    r = run(inp, tmp_path / "out", DETECTOR="model")
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "out" / "eeg-seizures"
    det = [d for d in rows(out / "seizures.csv") if d["recording_id"].startswith("sub-01")]
    on = sorted(float(d["onset_s"]) for d in det)
    assert len(on) == 2 and 330 <= on[0] <= 405 and 660 <= on[1] <= 720, on
    prov = json.loads((out / "provenance.json").read_text())
    assert "toy" in prov["licence_notice"] and "MIT" in prov["licence_notice"]
    assert prov["recordings"]["sub-chb99_task-eeg_run-01"]["status"].startswith("skipped: the model expects")
    assert all(d["detector"] == "model:toy" for d in det)


def test_model_missing_exits_1(fixture, tmp_path):
    assert run(fixture, tmp_path, DETECTOR="model").returncode == 1
