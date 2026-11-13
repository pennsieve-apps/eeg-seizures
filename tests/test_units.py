"""Unit tests: line length, normalisation, hysteresis, post-processing, channels and montage,
recording ids, CHB-MIT summaries, the SzCORE file format and scoring."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from processor import detect as D
from processor import eeg as E
from processor import szcore as S


def test_line_length_per_second_matches_definition():
    fs = D.FS
    t = np.arange(10 * fs) / fs
    x = np.vstack([10 * np.sin(2 * np.pi * 5 * t), np.zeros_like(t)])
    ll = D.line_length_per_second(x, 0.0, 1, 9)
    # a 5 Hz sine of amplitude A has line length 4 A f per second
    assert ll.shape == (2, 8)
    assert np.allclose(ll[0], 4 * 10 * 5, rtol=0.02)
    assert np.allclose(ll[1], 0)


def test_normalise_flags_a_rise_against_the_trailing_baseline():
    rng = np.random.default_rng(0)
    ll = 50 * np.exp(0.1 * rng.standard_normal((3, 900)))
    ll[:, 600:640] *= 5
    z = D.normalise(ll, np.ones(900, bool), 300)
    assert np.all(np.nanmedian(z[:, 600:640], axis=1) > 10)
    assert np.all(np.abs(np.nanmedian(z[:, 300:600], axis=1)) < 1)
    valid = np.ones(900, bool)
    valid[100:200] = False
    z = D.normalise(ll, valid, 300)
    assert np.isnan(z[:, 100:200]).all()


def test_combine_uses_the_top_k_channels():
    z = np.zeros((6, 10))
    z[0, 3:6] = 12.0
    score, _ = D.combine(z, 2, 1)
    assert score[4] == pytest.approx(6.0)
    assert score[0] == 0.0


def test_hysteresis_and_postprocess():
    s = np.array([0, 5, 5, 3, 3, 1, 0, 6, 0])
    assert D.hysteresis(s, 4.5, 2.5) == [(1, 5), (7, 8)]
    assert D.hysteresis(np.array([5.0, 5.0]), 4.5, 3) == [(0, 2)]
    # drop short, then merge closer than 90 s, then split at 300 s
    ev = [(0, 5), (100, 130), (150, 170), (400, 1100)]
    assert D.postprocess(ev, 10, 90, 300) == [(100, 170), (400, 700), (700, 1000), (1000, 1100)]
    assert D.postprocess([(0, 5), (50, 58)], 10, 90, 300, drop_first=False) == [(0, 58)]


def test_resample_ratio():
    assert D.resample_ratio(512) == (1, 2)
    assert D.resample_ratio(500) == (64, 125)
    assert D.resample_ratio(256) == (1, 1)


@pytest.mark.parametrize("label,expected", [("Fp1-Avg", "Fp1"), ("EEG T3-Ref", "T3"), ("T8-P8-1", "T8-P8"),
                                            ("FP1-F7", "FP1-F7"), ("C3-M2", "C3")])
def test_clean_label(label, expected):
    assert E.clean_label(label) == expected


def test_montage_referential_old_names_and_bad_channels():
    names = [f"{n}-Avg" for n in ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz", "Cz", "Pz",
                                  "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6"]] + ["EKG", "EEG Fpz-Ref"]
    m = E.montage_for(names, ["eeg"] * len(names), {"Cz-Avg"}, {})
    assert m.kind == "average"
    assert m.names == [c for c in E.SZCORE_CHANNELS if c != "Cz"]
    assert m.missing == ["Cz"]
    assert m.sources[E.SZCORE_CHANNELS.index("T7")] == ["T3-Avg"]
    assert "EKG" in m.ignored and "EEG Fpz-Ref" in m.ignored


def test_montage_bipolar_only_chbmit():
    names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
             "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
             "FT9-FT10", "FT10-T8", "T8-P8-1", "-", "ECG"]
    m = E.montage_for(names, ["eeg"] * len(names), set(), {})
    assert m.kind == "bipolar-longitudinal"
    assert m.names[:3] == ["Fp1-F7", "F7-T7", "T7-P7"]
    assert len(m.names) == 21  # 23 pairs minus the duplicate T8-P8 and the reversed P7-T7
    assert "duplicate" in m.ignored["T8-P8-1"] and "duplicate" in m.ignored["P7-T7"]


def test_montage_too_few_channels():
    m = E.montage_for(["ECG", "EMG"], ["ecg", "emg"], set(), {})
    assert m.kind == "none"


@pytest.mark.parametrize("name,rid,sub", [
    ("sub-00_ses-01_task-szMonitoring_run-01_eeg.edf", "sub-00_ses-01_task-szMonitoring_run-01", "00"),
    ("sub-01_ses-01_task-x_run-01_desc-preproc_eeg.edf", "sub-01_ses-01_task-x_run-01", "01"),
    ("chb01_03.edf", "sub-chb01_task-eeg_run-03", "chb01"),
    ("patient 7.edf", "sub-patient7_task-eeg", "patient7")])
def test_recording_ids(tmp_path, name, rid, sub):
    assert E.recording_id_for(tmp_path / name) == (rid, sub)


def test_chbmit_summary(tmp_path):
    p = tmp_path / "chb01-summary.txt"
    p.write_text("File Name: chb01_01.edf\nNumber of Seizures in File: 0\n\nFile Name: chb01_03.edf\n"
                 "Number of Seizures in File: 2\nSeizure 1 Start Time: 2996 seconds\nSeizure 1 End Time: 3036 seconds\n"
                 "Seizure 2 Start Time: 100 seconds\nSeizure 2 End Time: 130 seconds\n")
    assert E.parse_chbmit_summary(p) == {"chb01_01.edf": [], "chb01_03.edf": [(2996.0, 3036.0), (100.0, 130.0)]}


def test_szcore_tsv_matches_epilepsy2bids_savetsv(tmp_path):
    p = tmp_path / "a.tsv"
    S.write_tsv(p, [{"onset": 296.0, "duration": 40.0, "confidence": 0.8765, "channels": ["T7", "P7"]}],
                3600.0, datetime(2016, 11, 6, 13, 43, 4))
    assert p.read_text() == ("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"
                             "296.00\t40.00\tsz\t0.88\tT7,P7\t2016-11-06 13:43:04\t3600.00\n")
    S.write_tsv(p, [], 120.5, None)
    assert p.read_text().splitlines()[1] == "0.00\t120.50\tbckg\tn/a\tn/a\tn/a\t120.50"


def test_scoring_szcore_rules():
    # one 60 s seizure; a detection 20 s before it counts (30 s tolerance), one 2 min later is false
    sc = S.score([(1000.0, 60.0)], [(985.0, 10.0), (1300.0, 30.0)], 3600.0)["event"]
    assert (sc["tp"], sc["fp"], sc["n_reference"]) == (1, 1, 1)
    assert sc["fp_per_day"] == pytest.approx(24.0)
    assert sc["f1"] == pytest.approx(2 / 3)
    none = S.score([], [], 3600.0)["event"]
    assert np.isnan(none["sensitivity"]) and none["fp"] == 0
    pooled = S.pooled([S.score([(1000.0, 60.0)], [(1000.0, 60.0)], 3600.0),
                       S.score([], [(10.0, 30.0)], 3600.0)], "event")
    assert (pooled["tp"], pooled["fp"], pooled["fp_per_day"]) == (1, 1, pytest.approx(12.0))


@pytest.mark.parametrize("value,expected", [
    ("2016-01-01T19:39:33", datetime(2016, 1, 1, 19, 39, 33)),
    ("2016-01-01T19:39:33+00:00", datetime(2016, 1, 1, 19, 39, 33)),
    ("2016-01-01T19:39:33-05:00", datetime(2016, 1, 1, 19, 39, 33)),
    ("2016-01-01 19:39:33", datetime(2016, 1, 1, 19, 39, 33)),
    ("n/a", None), ("", None), ("yesterday", None)])
def test_parse_wall_clock(value, expected):
    assert E.parse_wall_clock(value) == expected


def test_events_tsv_without_seizure_information_is_not_a_label_source(tmp_path):
    d = tmp_path / "sub-01" / "eeg"
    d.mkdir(parents=True)
    (d / "sub-01_task-rest_eeg.edf").write_bytes(b"")
    (d / "sub-01_task-rest_events.tsv").write_text("onset\tduration\ttrial_type\n10\t2\teyes_closed\n")
    recs, _, _ = E.find_recordings(tmp_path)
    assert not recs[0].labels_known
    (d / "sub-01_task-rest_events.tsv").write_text("onset\tduration\teventType\n0\t600\tbckg\n")
    recs, _, _ = E.find_recordings(tmp_path)
    assert recs[0].labels_known and recs[0].seizures == []
