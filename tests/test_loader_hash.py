"""Content fingerprint of the PFF loader (modelo_pff.loader_hash / content_hash): it hashes what the pipeline consumes, not the file pff.py.
Synthetic matches and a fake pff.py only: no PFF data is loaded."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo_pff as mp_  # noqa: E402
import pff_split  # noqa: E402

MIDS = list(pff_split.DESARROLLO[:3])


def fake_match(mid, extra=None):
    n = 50 + int(mid[-1])
    rng = np.random.default_rng(int(mid))
    d = {"ball_xy": rng.random((n, 2)), "players_slots": {"H1": rng.random((n, 2)), "A1": rng.random((n, 2))},
         "period": np.ones(n, np.int8), "label": np.array(["H"] * n), "label_poss": np.array(["A"] * n), "pitch": np.array([105.0, 68.0]),
         "uncertain": np.zeros(n, bool), "clipped": 0, "event_f": np.arange(3)}
    d["ball_xy"][3, 0] = np.nan
    d.update(extra or {})
    return d


def hash_of(mutate=None, mids=MIDS):
    def load(mid):
        d = fake_match(mid)
        if mutate:
            mutate(mid, d)
        return d
    return mp_.content_hash(mids, load=load)


def test_changing_a_consumed_value_changes_the_hash():
    base = hash_of()
    assert hash_of(lambda m, d: d["ball_xy"].__setitem__((10, 1), 0.5)) != base
    assert hash_of(lambda m, d: d["players_slots"]["A1"].__setitem__((0, 0), -1.0)) != base
    assert hash_of(lambda m, d: d["label"].__setitem__(5, "D")) != base
    assert hash_of(lambda m, d: d.__setitem__("pitch", np.array([105.0, 67.0]))) != base
    assert hash_of(lambda m, d: d.__setitem__("period", d["period"].astype(np.int32))) != base       # dtype counts
    assert hash_of(lambda m, d: d["players_slots"].__setitem__("H2", d["ball_xy"].copy())) != base   # a new slot is new content


def test_adding_a_key_or_changing_an_unread_one_keeps_the_hash():
    base = hash_of()
    assert hash_of(lambda m, d: d.update(restart_window=np.arange(4), restart_windows=[(1, 2)], report={"restart": 1})) == base
    assert hash_of(lambda m, d: d.__setitem__("uncertain", ~d["uncertain"])) == base
    assert hash_of(lambda m, d: d.__setitem__("event_f", np.arange(7))) == base


def test_match_order_is_fixed_and_ids_count():
    assert hash_of(mids=MIDS) == hash_of(mids=MIDS)
    assert hash_of(mids=MIDS) != hash_of(mids=MIDS[::-1])
    seen = []
    mp_.content_hash(MIDS, load=lambda m: seen.append(m) or fake_match(m))
    assert seen == MIDS
    # default list: the 40 development matches in the order of pff_split
    seen.clear()
    mp_.content_hash(load=lambda m: seen.append(m) or fake_match(m))
    assert seen == list(pff_split.DESARROLLO) and len(seen) == 40


def test_held_out_match_is_never_loaded():
    def boom(mid):
        raise AssertionError("loaded")
    with pytest.raises(AssertionError):
        mp_.content_hash([pff_split.RESERVADOS[0]], load=boom)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A fake loader directory with a pff.py, a private memo, and a counting content_hash."""
    (tmp_path / "pff.py").write_text("# v1")
    monkeypatch.setattr(pff_split, "LOADER_DIR", str(tmp_path))
    monkeypatch.setattr(mp_, "LOADER_MEMO", str(tmp_path / "memo.json"))
    calls = []
    monkeypatch.setattr(mp_, "content_hash", lambda *a, **k: calls.append(1) or "content-" + (tmp_path / "pff.py").read_text()[-2:].strip())
    return tmp_path, calls


def test_memo_is_used_and_invalidated_by_the_file_sha(lab):
    path, calls = lab
    assert mp_.loader_hash() == "content-v1" and len(calls) == 1
    assert mp_.loader_hash() == "content-v1" and len(calls) == 1             # same pff.py: read from the memo
    (path / "pff.py").write_text("# v2")                                     # other SHA: recomputed once
    assert mp_.loader_hash() == "content-v2" and len(calls) == 2
    assert mp_.loader_hash() == "content-v2" and len(calls) == 2
    (path / "pff.py").write_text("# v1")                                     # back to the first one: still memoised
    assert mp_.loader_hash() == "content-v1" and len(calls) == 2


def test_memo_is_invalidated_when_the_consumed_keys_change(lab, monkeypatch):
    _, calls = lab
    mp_.loader_hash()
    monkeypatch.setattr(mp_, "LOADER_KEYS", mp_.LOADER_KEYS + ("uncertain",))
    mp_.loader_hash()
    assert len(calls) == 2


def test_missing_loader(lab):
    path, _ = lab
    (path / "pff.py").unlink()
    assert mp_.loader_hash() == "pff.py missing"


def test_consumed_keys_are_the_ones_the_builders_read():
    """LOADER_KEYS must cover every d[...] that build_one / build_one_h read (the keys left out are loaded but never cached)."""
    import inspect
    import re
    import modelo_transfer as mt
    read = set()
    for f in (mp_.build_one, mt.build_one_h, mp_.rp.ball_variant):
        read |= set(re.findall(r'd\["(\w+)"\]', inspect.getsource(f)))
    read |= {"label_poss"}                                                    # through rp.ev.make_gt(d, "label_poss")
    read -= {"mid"}                                                           # the id, not loader content
    assert read <= set(mp_.LOADER_KEYS), read - set(mp_.LOADER_KEYS)
