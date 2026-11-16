"""PFF ladder (bancos/evaluar_pff.py): no event-derived input reaches a method, extra time is cut, statistics are exact."""
import inspect
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import evaluar_pff as ep  # noqa: E402
import reglas  # noqa: E402

FORBIDDEN = re.compile(r"ball_status|is_touch|uncertain|video_missing|ball_visible|ball_z|players_visible|label_poss|label_ut"
                       r"|attack_right|\[\"label\"\]|\['label'\]")


def test_prediction_modules_never_read_event_derived_fields():
    for name in ("reglas.py", "toques.py", "equipo_toque.py"):
        src = open(os.path.join(ROOT, "src", name)).read()
        assert not FORBIDDEN.search(src), name
    import baseline
    for fn in (baseline.smooth,):                       # the only baseline code the ladder calls
        assert not FORBIDDEN.search(inspect.getsource(fn))
    assert list(inspect.signature(ep.ladder).parameters) == ["ball", "slots", "per", "pitch"]
    assert not FORBIDDEN.search(inspect.getsource(ep.ladder))


def test_evaluation_loads_matches_only_through_the_guard():
    src = open(os.path.join(ROOT, "bancos", "evaluar_pff.py")).read()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "pff_split.load_dev(" in code
    assert not re.search(r"^\s*import pff\b|\bpff\.load\(|from pff import", code, re.M)
    assert "smoothed=True" not in code


def _match(n=600):
    rng = np.random.default_rng(0)
    per = np.where(np.arange(n) < n // 2, 1, 3)            # period 3 = extra time
    ball = np.cumsum(rng.normal(0, 0.002, (n, 2)), 0) + 0.5
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    H[:, :11] = rng.uniform(0, 1, (n, 11, 2))
    A[:, :11] = rng.uniform(0, 1, (n, 11, 2))
    lab = np.array(["H"] * n, dtype="<U1")
    return {"ball_xy": ball, "players_slots": {"H": H, "A": A}, "period": per, "label": lab, "label_poss": lab.copy(),
            "uncertain": np.zeros(n, bool), "pitch": (105.0, 68.0)}


def test_clip_periods_drops_extra_time_and_keeps_alignment():
    d = _match()
    c = ep.clip_periods(d)
    assert len(c["period"]) == 300 and c["clipped"] == 300 and set(c["period"]) == {1}
    for k in ("ball_xy", "label", "label_poss", "uncertain"):
        assert len(c[k]) == 300
    assert all(len(v) == 300 for v in c["players_slots"].values())
    assert np.array_equal(c["ball_xy"], d["ball_xy"][:300], equal_nan=True)
    assert ep.clip_periods({**d, "period": np.ones(600, int)})["clipped"] == 0


def test_ladder_is_deterministic_and_b1_is_the_capped_carry_over():
    d = ep.clip_periods(_match())
    d["ball_xy"][100:130] = np.nan                         # a 1.2 s hole
    a = ep.ladder(d["ball_xy"], d["players_slots"], d["period"], d["pitch"])
    b = ep.ladder(d["ball_xy"].copy(), d["players_slots"], d["period"], d["pitch"])
    assert list(a) == list(ep.STEPS)
    for s in ep.STEPS:
        assert np.array_equal(a[s], b[s]) and set(np.unique(a[s])) <= {"H", "A", "D"} and len(a[s]) == 300
    present = ~np.isnan(d["ball_xy"][:, 0])
    assert np.array_equal(a["B1"], reglas.fill_gaps(a["B0"], present, d["period"], "F", 5))
    assert np.array_equal(a["B1b"], reglas.fill_gaps(a["B0"], present, d["period"], "F", 10))
    assert ep.CAP == 5 and ep.MODE == "dir" and ep.V2["r"] == 3.5 and ep.V2["dv"] == 2.0 and ep.V2["win"] == 5


def test_delta_stats():
    s = ep.delta_stats([0.01, 0.03, -0.02, 0.0, 0.02], n_boot=2000)
    assert (s["wins"], s["losses"], s["ties"], s["n"]) == (3, 1, 1, 5)
    assert abs(s["mean"] - 0.008) < 1e-12 and abs(s["sd"] - np.std([0.01, 0.03, -0.02, 0.0, 0.02], ddof=1)) < 1e-12
    assert s["lo"] <= s["mean"] <= s["hi"]
    assert s == ep.delta_stats([0.01, 0.03, -0.02, 0.0, 0.02], n_boot=2000)            # fixed seed
    c = ep.delta_stats(np.full(8, 0.02), n_boot=500)
    assert abs(c["lo"] - 0.02) < 1e-12 and abs(c["hi"] - 0.02) < 1e-12
