"""Ablation of the ball-presence family (bancos/ablacion_presencia.py), on synthetic data: the `without` variant reads exactly the columns of the shipped detector,
the `with` variant reads every column, and the training rows are the ones of `ronda_bordes` (same sampling and seed), so the `without` detectors are the shipped ones."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src", "tests"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import ablacion_presencia as ap  # noqa: E402
import borde  # noqa: E402
import modelo as mo  # noqa: E402
import ronda_bordes as rb  # noqa: E402
from test_modelo import toy_world  # noqa: E402


def test_the_variants_differ_only_in_the_presence_family():
    X, names = mo.build_features(*toy_world())
    cols = ap.variant_columns(names)
    assert np.array_equal(cols["without"], borde.border_columns(names))
    assert np.array_equal(cols["with"], np.arange(len(names)))
    extra = sorted(set(cols["with"].tolist()) - set(cols["without"].tolist()))
    assert extra and all(names[i].split("@")[0] in borde.PRESENCE for i in extra)


def fake_match(n, names, seed):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 4, n).astype(np.int8)
    return {"names": names, "calif": gt != borde.X, "y": rng.random(n) < 0.1, "F": None, "X": rng.normal(size=(n, len(names))).astype(np.float32)}


def test_the_training_rows_without_presence_are_the_ones_of_ronda_bordes(monkeypatch):
    X, names = mo.build_features(*toy_world())
    d = fake_match(5000, names, 3)
    d["F"] = d["X"][:, borde.border_columns(names)]
    monkeypatch.setattr(rb, "load_match", lambda mid, balon, lam, base=True: dict(d))
    monkeypatch.setattr(ap, "load_full", lambda mid, balon, lam, base=True: dict(d, F_all=d["X"]))
    for balon in ("g6h", "g8h"):
        Fr, yr, wr, nr = rb.training_rows(("3812", balon, (8.0, 16.0)))
        Fa, ya, wa, na = ap.training_rows(("3812", balon, (8.0, 16.0)))
        assert np.array_equal(Fa[:, ap.variant_columns(names)["without"]], Fr)
        assert np.array_equal(ya, yr) and np.array_equal(wa, wr) and na == nr
        assert Fa.shape[1] == len(names)
