"""Guard external calibration, match isolation and selection without challenge data."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ("src", "bancos"):
    sys.path.insert(0, str(ROOT / directory))
pytest.importorskip("torch")
import a30_external as external
import a30_protocol as protocol
import balon_yolo as tracker
import posesion_rnn as pr


def test_gap_summary_resamples_only_covered_external_intervals() -> None:
    ball = np.ones((900, 2))
    ball[60:180] = np.nan
    ball[600:] = np.nan
    covered = np.ones(900, bool)
    covered[600:] = False
    result = external.gap_summary([(ball, covered)], 30.0)
    assert result["share"] == pytest.approx(0.2)
    assert result["fps"] == 25
    assert sum(item["n_runs"] for item in result["classes"]) == 1


def test_wrong_tracker_point_is_a_false_positive_and_false_negative() -> None:
    ball = np.array([[0.2, 0.3], [0.9, 0.9], [0.5, 0.5]])
    source = {"covered": np.ones(3, bool), "homography": np.tile(np.eye(3), (3, 1, 1)),
              "reviews": {"0": {"status": "accepted", "bbox": [0.1, 0.2, 0.3, 0.4]},
                          "1": {"status": "accepted", "bbox": [0.1, 0.2, 0.3, 0.4]},
                          "2": {"status": "absent"}}}
    result = external.point_f1(ball, source)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["f1"] == pytest.approx(0.4)


def test_tracker_with_no_valid_candidate_returns_absence() -> None:
    tid = np.full((4, 2), -1)
    ball, confidence = tracker.choose(tid, np.zeros((4, 2)), np.zeros((4, 2, 2)), {}, np.array([], bool))
    assert np.isnan(ball).all()
    assert not confidence.any()


def test_external_reader_rejects_reserved_match_before_opening_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a PFF development"):
        external.read_clip(tmp_path, tmp_path, {"id": "3854_T1", "match_id": "3854"}, "unused", "unused")


def test_training_does_not_open_challenge_track_or_inherit_anchor_probability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "YOLO_TRACK", str(tmp_path / "forbidden-challenge-track.npz"))
    ball = np.ones((1000, 2)) * 0.5
    mask = np.ones(1000, bool)
    mask[100:300] = False
    observed = ball.copy()
    observed[~mask] = np.nan
    calibration = {"source": "PFF development videos only", "anchor_probability": 0.0,
                   **external.gap_summary([(observed, np.ones(1000, bool))], 25.0)}
    path = tmp_path / "external.json"
    path.write_text(json.dumps(calibration))
    loaded = pr.external_gap_model(str(path))
    called = []
    original = pr.ih.inject_holes

    def record_injection(ball: np.ndarray, label: np.ndarray, per: np.ndarray, mid: str,
                         target: float, draw: object, anchor_p: float) -> tuple:
        called.append(anchor_p)
        return original(ball, label, per, mid, target=target, draw=draw, anchor_p=anchor_p)

    monkeypatch.setattr(pr.ih, "inject_holes", record_injection)
    result = pr.training_ball({"ball_xy": ball, "label": np.full(1000, "H"), "period": np.ones(1000)}, "3816", "g6h", loaded)
    assert called == [0.0]
    assert np.isnan(result).any()
    assert np.isfinite(ball).all()


def test_split_excludes_outer_matches_and_reserved_data() -> None:
    held = set(pr.rp.FOLDS[2])
    available = [mid for mid in pr.rp.pff_split.DESARROLLO if mid not in held]
    fit, validation = protocol.split_training(available)
    assert len(fit) == 28 and len(validation) == 4
    assert not (set(fit) | set(validation)) & held
    with pytest.raises(ValueError, match="only on PFF development"):
        protocol.split_training(["3854"])


def test_missing_t1w_cannot_produce_comparison_or_acceptance(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    with pytest.raises(FileNotFoundError, match="Missing historical T1w boosters"):
        protocol.compare_t1w(tmp_path, tmp_path / "missing-reference", output)
    assert not output.exists()


def test_fingerprint_names_include_separate_official_evaluator(tmp_path: Path) -> None:
    repository, challenge = tmp_path / "lab", tmp_path / "official"
    assert protocol.source_name(challenge / "evaluation/possession_metrics.py", repository, challenge) == "challenge/evaluation/possession_metrics.py"
    assert protocol.source_name(repository / "bancos/posesion_rnn.py", repository, challenge) == "bancos/posesion_rnn.py"
    with pytest.raises(ValueError, match="outside the declared repositories"):
        protocol.source_name(tmp_path / "other.py", repository, challenge)


def test_lambda_selection_optimises_worst_view(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "LAM_HA", (0.0, 1.0))
    monkeypatch.setattr(pr, "LAM_D", (0.0,))
    monkeypatch.setattr(pr.mo, "viterbi", lambda *args: [np.array([0]), np.array([1])])
    scores = {("g6h", 0): 0.95, ("g6h", 1): 0.7, ("g8h", 0): 0.5, ("g8h", 1): 0.7}
    monkeypatch.setattr(pr, "spell_f1", lambda gt, path: scores[(gt, int(path[0]))])
    entries = [{"view": view, "gt": view, "per": np.ones(1)} for view in pr.VIEWS]
    lam, objective = pr.choose_lambda([np.full((1, 3), 1 / 3)] * 2, entries)
    assert lam == (1.0, 0.0) and objective == pytest.approx(0.7)
