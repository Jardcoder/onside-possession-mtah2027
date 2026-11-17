"""Cache fingerprints of bancos/modelo_pff.py and bancos/modelo_transfer.py: a change of logic invalidates old results, --rehacer forces
inner / outer / final, importances carry the fingerprint, cache writes are atomic. Nothing here trains a model or touches the real cache."""
import ast
import inspect
import os
import sys
import textwrap

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import ronda_pff as rp  # noqa: E402


class Trained(Exception):
    """Raised by the fake trainer: proves the result was recomputed instead of read from the cache."""


def fake_train(*a, **k):
    raise Trained


def other_version(f):
    def g(*a, **k):
        return "changed logic"
    g.__name__ = f.__name__
    return g


# ---------------------------------------------------------------------------------------------- (a) logic changes the fingerprint

@pytest.mark.parametrize("name", ["pick_lambda", "run_inner", "run_outer", "predict_matches", "task_grid", "task_final", "eval_stats"])
@pytest.mark.needs_pff
def test_m1_models_fp_covers_the_selection_and_scoring_code(monkeypatch, name):
    before = mp_.fp_models(5)
    monkeypatch.setattr(mp_, name, other_version(getattr(mp_, name)))
    assert mp_.fp_models(5) != before


@pytest.mark.parametrize("name", ["run_stage", "task_rows", "train_model_t"])
@pytest.mark.needs_pff_and_calibration
def test_transfer_models_fp_covers_its_code(monkeypatch, name):
    before = mt.fp_models()
    monkeypatch.setattr(mt, name, other_version(getattr(mt, name)))
    assert mt.fp_models() != before


@pytest.mark.needs_pff
def test_shared_helper_outside_the_file_is_covered(monkeypatch):
    b1, b2 = mp_.fp_models(5), mt.fp_models()
    monkeypatch.setattr(rp, "spell_row", other_version(rp.spell_row))
    assert mp_.fp_models(5) != b1 and mt.fp_models() != b2


@pytest.mark.needs_pff
def test_env_and_loader_in_fp(monkeypatch):
    b = mp_.fp_models(5)
    monkeypatch.setattr(mp_.lightgbm, "__version__", "0.0.test")
    assert mp_.fp_models(5) != b
    monkeypatch.undo()
    monkeypatch.setattr(mp_, "loader_hash", lambda: "another loader")
    assert mp_.fp_models(5) != b


@pytest.mark.needs_pff
def test_features_fp_gets_loader_and_numpy_only(monkeypatch):
    for fn in (mp_.fp_features, mt.fp_features):
        monkeypatch.setattr(mp_, "_FP", {})
        monkeypatch.setattr(mt, "_FP", {})
        base = fn()
        monkeypatch.setattr(mp_, "loader_hash", lambda: "another loader")
        monkeypatch.setattr(mp_, "_FP", {})
        monkeypatch.setattr(mt, "_FP", {})
        assert fn() != base
        monkeypatch.undo()
    # the features do not depend on LightGBM or on the training code: changing them must NOT force the 16 GB rebuild
    monkeypatch.setattr(mp_, "_FP", {})
    base = mp_.fp_features()
    monkeypatch.setattr(mp_.lightgbm, "__version__", "0.0.test")
    monkeypatch.setattr(mp_, "pick_lambda", other_version(mp_.pick_lambda))
    monkeypatch.setattr(mp_, "_FP", {})
    assert mp_.fp_features() == base


def _called_names(func, module_funcs):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in module_funcs}


def test_m1_call_graph_is_inside_the_fingerprint():
    """Every function of the file reachable from run_outer must be hashed by fp_models, or be a known non-result helper."""
    funcs = {n: f for n, f in vars(mp_).items() if inspect.isfunction(f) and f.__module__ == mp_.__name__}
    seen, todo = set(), ["run_outer"]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += _called_names(funcs[n], funcs)
    covered = {f.__name__ for f in mp_.model_functions()}
    not_results = {"load_if", "add_time", "inner_path", "outer_path", "importance_path", "times_path", "atomic_write", "atomic_save",
                   "atomic_savez", "atomic_json", "atomic_csv"}
    assert seen - covered - not_results == set()


# ---------------------------------------------------------------------------------------------- (b) --rehacer reaches the results

@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(mp_, "CACHE", str(tmp_path))
    return tmp_path


def test_rehacer_forces_inner(cache, monkeypatch):
    fp = "fp-test"
    mp_.atomic_savez(mp_.inner_path(0, 1), fp=fp, f1=np.zeros((8, 2, 30)), mids=np.array(["a"]))
    monkeypatch.setattr(mp_, "train_model", fake_train)
    f1, _ = mp_.run_inner(None, 0, 1, 5, fp, 1, [])                      # valid cache: read, no training
    assert f1.shape == (8, 2, 30)
    with pytest.raises(Trained):
        mp_.run_inner(None, 0, 1, 5, fp, 1, [], rehacer=True)


def test_rehacer_forces_outer_and_importance_needs_fp(cache, monkeypatch):
    fp = "fp-test"
    mp_.atomic_csv(mp_.outer_path(0), fp, mp_.FIELDS, [])
    mp_.atomic_savez(mp_.importance_path(0), fp=fp, gain=np.zeros(1))
    monkeypatch.setattr(mp_, "train_model", fake_train)
    monkeypatch.setattr(mp_, "run_inner", lambda *a, **k: (np.zeros((8, 2, len(mp_.GRID))), None))
    assert mp_.run_outer(None, 0, 5, fp, 1, []) is None                  # complete and fingerprinted: nothing to do
    with pytest.raises(Trained):
        mp_.run_outer(None, 0, 5, fp, 1, [], rehacer=True)
    np.savez(mp_.importance_path(0), gain=np.zeros(1))                   # importance of an old run: no fingerprint -> stale
    with pytest.raises(Trained):
        mp_.run_outer(None, 0, 5, fp, 1, [])


def test_rehacer_reaches_inner_from_outer(cache, monkeypatch):
    seen = []

    def spy(*a, **k):
        seen.append(a[7] if len(a) > 7 else k.get("rehacer"))
        raise Trained
    monkeypatch.setattr(mp_, "run_inner", spy)
    with pytest.raises(Trained):
        mp_.run_outer(None, 0, 5, "fp", 1, [], rehacer=True)
    assert seen == [True]


def test_rehacer_forces_transfer_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "CACHE", str(tmp_path))
    fp = "fp-test"
    mp_.atomic_csv(mt.rows_path("T0", 0), fp, mt.FIELDS, [])
    monkeypatch.setattr(mt, "train_model_t", fake_train)
    assert mt.run_stage(None, "T0", 0, fp, 1, []) is None
    with pytest.raises(Trained):
        mt.run_stage(None, "T0", 0, fp, 1, [], rehacer=True)


# ---------------------------------------------------------------------------------------------- (c) atomic writes

def test_atomic_write_ok_leaves_only_the_file(tmp_path):
    p = str(tmp_path / "a.npz")
    mp_.atomic_savez(p, x=np.arange(3))
    mp_.atomic_save(str(tmp_path / "b.npy"), np.arange(3))
    mp_.atomic_csv(str(tmp_path / "c.csv"), "fp", ["u"], [{"u": 1}])
    mp_.atomic_json(str(tmp_path / "d.json"), {"k": 1})
    assert sorted(os.listdir(tmp_path)) == ["a.npz", "b.npy", "c.csv", "d.json"]
    assert open(tmp_path / "c.csv").readline() == "# fp\n"
    assert list(np.load(p)["x"]) == [0, 1, 2]


def test_atomic_write_failure_keeps_the_old_file_and_no_leftovers(tmp_path):
    p = str(tmp_path / "res.csv")
    mp_.atomic_csv(p, "old", ["u"], [{"u": 1}])

    def broken(f):
        f.write("# new\npartial")
        raise RuntimeError("killed halfway")
    with pytest.raises(RuntimeError):
        mp_.atomic_write(p, broken, "w")
    assert open(p).readline() == "# old\n"                               # the previous complete file survives
    assert os.listdir(tmp_path) == ["res.csv"]                           # and no temp file stays behind
    q = str(tmp_path / "new.csv")
    with pytest.raises(RuntimeError):
        mp_.atomic_write(q, broken, "w")
    assert not os.path.exists(q) and os.listdir(tmp_path) == ["res.csv"]
    with pytest.raises(ValueError):                                      # a failing serializer too (rows with a field outside the header)
        mp_.atomic_csv(q, "fp", ["u"], [{"zz": 1}])
    assert not os.path.exists(q) and os.listdir(tmp_path) == ["res.csv"]
