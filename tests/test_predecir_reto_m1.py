"""src/predecir_reto_m1.py: shape and values of the CSV of the challenge (synthetic labels at the real frames; the real CSV when it exists)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("src", "bancos"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import features_reto as fr  # noqa: E402
import possession_metrics as pm  # noqa: E402
import predecir_reto_m1 as pr  # noqa: E402
import reto_periodos as rpd  # noqa: E402

N = 199_319


def fake_labels():
    """Cut rows with a repeating H/H/A/D pattern, pushed to the video index like the script does."""
    frames = rpd.play_frames(N)
    letters = np.array(list("HHAD"))[(np.arange(len(frames)) // 50) % 4]
    info = {"frames": frames, "n_video": N}
    return fr.to_video(letters, info), info


def test_to_video_has_the_layout_of_the_challenge():
    labels, _ = fake_labels()
    pr.check_labels(labels)
    assert len(labels) == N
    g = pr.graded_mask(N)
    assert int(g.sum()) == 158_857                          # DATA.md:262-263
    assert (labels[:11_185] == "X").all() and (labels[90_652:110_808] == "X").all() and (labels[190_198:] == "X").all()
    assert (labels[110_808:113_742] == "D").all()           # FAQ.md:38-39: graded and all D before the kick-off
    assert set(labels[g].tolist()) == {"H", "A", "D"}
    assert (labels[113_742:190_198] != "X").all() and (labels[11_185:90_652] != "X").all()


def test_a_cut_row_lands_on_its_video_frame():
    frames = rpd.play_frames(N)
    letters = np.full(len(frames), "H")
    letters[0], letters[-1] = "A", "A"
    labels = fr.to_video(letters, {"frames": frames, "n_video": N})
    assert labels[11_185] == "A" and labels[190_197] == "A"
    assert labels[11_186] == "H" and labels[113_742] == "H"


def test_check_labels_rejects_what_the_scorer_would_reject():
    labels, _ = fake_labels()
    for bad in (labels[:-1], np.where(np.arange(N) == 50_000, "Z", labels), np.where(np.arange(N) == 50_000, "X", labels),
                np.where(np.arange(N) == 100, "H", labels), np.where(np.arange(N) == 111_000, "H", labels)):
        with pytest.raises(AssertionError):
            pr.check_labels(bad)


def test_written_csv_passes_the_official_validation(tmp_path):
    labels, _ = fake_labels()
    path = pr.write_csv(str(tmp_path / "x.csv"), labels)
    t = pm.read_label_csv(path)
    assert list(t.columns) == ["frame", "label"]
    assert pm.validate_submission(t, N) is None
    assert (t["frame"].values == np.arange(N)).all()


def test_counters_count_spells_per_graded_minute():
    labels = np.array(["X"] * 10 + ["H"] * 25 * 60 + ["D"] * 25 * 60 + ["A"] * 25 * 60 + ["X"] * 5)
    d, n_sp, per_min, minutes = pr.counters(labels)
    assert (n_sp, round(minutes, 6), round(per_min, 6)) == (2, 3.0, round(2 / 3, 6)) and round(d, 6) == round(1 / 3, 6)


def test_lambda_file_is_read_as_a_pair(tmp_path):
    p = tmp_path / "l.json"
    p.write_text('{"lambda": [8, 24]}')
    assert pr.lambda_final(str(p)) == (8, 24)


@pytest.mark.skipif(not os.path.exists(pr.OUT), reason="src/out/reto_m1_t1.csv not generated yet (./run.sh predecir-reto-m1)")
def test_generated_csv_has_the_challenge_shape_and_values():
    t = pm.read_label_csv(pr.OUT)
    assert pm.validate_submission(t, N) is None
    labels = t["label"].values
    pr.check_labels(labels)
    assert set(np.unique(labels).tolist()) <= {"H", "A", "D", "X"}


def test_models_table_keeps_t1_as_before_and_t1wm_apart():
    cache, lam_file, out, report = pr.MODELS["T1"]
    assert (cache, lam_file, out, report) == (pr.mt.CACHE, pr.LAMBDA_FILE, pr.OUT, pr.REPORT)
    _, lam_wm, out_wm, report_wm = pr.MODELS["T1wm"]
    assert len({out, out_wm}) == 2 and len({lam_file, lam_wm}) == 2 and len({report, report_wm}) == 2
    assert pr.probs_file("T1", "g6h", "3826") == os.path.join(pr.mt.PROBS, "T1__g6h__3826.npy")
    assert pr.probs_file("T1wm", "g6h", "3826") == os.path.join(pr.RONDA5, "probs", "T1wm__g6h__3826.npy")


def test_t1w_reads_the_committed_files_unless_cache():
    files = pr.booster_files("T1w")
    assert [os.path.basename(f) for f in files] == [f"T1w_k{k}.txt" for k in range(5)]
    assert {os.path.dirname(f) for f in files} == {pr.onside_paths.T1W_DIR}
    assert pr.lambda_path("T1w") == os.path.join(pr.onside_paths.T1W_DIR, "lambda_final_t1w.json")
    assert pr.booster_files("T1w", cache=True)[0] == os.path.join(pr.RONDA5, "modelos", "T1w_k0.txt")
    assert pr.lambda_path("T1w", cache=True) == pr.MODELS["T1w"][1]
    assert pr.booster_files("T1wm")[0] == os.path.join(pr.RONDA5, "modelos", "T1wm_k0.txt")     # the other models stay in the cache
    assert pr.lambda_path("T1wm") == pr.MODELS["T1wm"][1]


def test_committed_t1w_files_match_their_sums():
    import hashlib
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "t1w")
    sums = dict(line.split()[::-1] for line in open(os.path.join(root, "SHA256SUMS")).read().splitlines() if line.strip())
    assert sorted(sums) == sorted([f"T1w_k{k}.txt" for k in range(5)] + ["feature_names.json", "lambda_final_t1w.json"])
    for name, want in sums.items():
        assert hashlib.sha256(open(os.path.join(root, name), "rb").read()).hexdigest() == want, name


def test_expected_hashes_cover_both_csvs():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = dict(line.split()[::-1] for line in open(os.path.join(root, "scripts", "expected_sha256.txt")).read().splitlines() if line.strip())
    assert set(rows) == {"reto_m1_t1w.csv", "reto_m2_t1w_bordes.csv"} and all(len(v) == 64 for v in rows.values())


def test_check_feature_names_detects_a_drift_against_the_detector_metadata(monkeypatch):
    import json
    meta = json.load(open(os.path.join(pr.onside_paths.MODELS_DIR, "detector_bordes.json")))
    border = meta["feature_names"]
    monkeypatch.setattr(pr.mt, "path_meta_t", lambda *a: "/nonexistent")
    full = json.load(open(os.path.join(pr.onside_paths.T1W_DIR, "feature_names.json")))
    assert [n for n in full if n in set(border)] == border                 # the committed list contains the detector's border columns in order
    assert pr.check_feature_names(full, "T1w", len(full)) == "models/t1w/feature_names.json (all columns)"
    with pytest.raises(ValueError):                                         # a column shifted in front is a drift, not a match
        pr.check_feature_names(["ball_present"] + border, "T1w")
    with pytest.raises(ValueError):
        pr.check_feature_names(full[:1] + full[1:][::-1], "T1w")
    with pytest.raises(ValueError):                                         # a presence column swapped with another: border columns intact
        i, j = full.index("ball_present"), full.index("period")
        perm = list(full)
        perm[i], perm[j] = perm[j], perm[i]
        pr.check_feature_names(perm, "T1w")
    with pytest.raises(ValueError, match="columns"):
        pr.check_feature_names(full, "T1w", len(full) + 1)
