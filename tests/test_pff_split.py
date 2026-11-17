"""Blind PFF split (bancos/pff_split.py): disjoint, complete, regenerable from the metadata, guard on the held-out set."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bancos"))
import pff_split as sp  # noqa: E402

needs_meta = pytest.mark.skipif(not os.path.isdir(sp.META_DIR), reason="PFF metadata not available (set ONSIDE_PFF_DIR)")


def test_disjoint_and_complete():
    assert len(sp.DESARROLLO) == 40 and len(sp.RESERVADOS) == 24
    assert not set(sp.DESARROLLO) & set(sp.RESERVADOS)
    assert len(set(sp.DESARROLLO) | set(sp.RESERVADOS)) == 64
    assert len(set(sp.DESARROLLO)) == 40 and len(set(sp.RESERVADOS)) == 24


@needs_meta
def test_regenerates_to_the_literals():
    phases = sp.read_phases()
    assert len(phases) == 64
    dev, held = sp.split(phases)
    assert tuple(dev) == sp.DESARROLLO and tuple(held) == sp.RESERVADOS
    assert sp.split(phases) == (dev, held)                        # deterministic


@needs_meta
def test_stratified_by_phase():
    phases = sp.read_phases()
    for ph, n_dev in sp.N_DEV_BY_PHASE.items():
        assert sum(phases[i] == ph for i in sp.DESARROLLO) == n_dev
    assert sum(phases[i] == "eliminatorias" for i in sp.RESERVADOS) == 6
    assert sum(phases[i] == "grupos" for i in sp.RESERVADOS) == 18


def test_depends_only_on_ids_and_phase():
    phases = {str(i): ("grupos" if i % 4 else "eliminatorias") for i in range(100, 164)}
    a = sp.split(phases, n_dev={"grupos": 30, "eliminatorias": 10})
    assert a == sp.split(dict(reversed(list(phases.items()))), n_dev={"grupos": 30, "eliminatorias": 10})
    assert a != sp.split(phases, salt="other", n_dev={"grupos": 30, "eliminatorias": 10})


def test_guard_blocks_held_out():
    for mid in sp.RESERVADOS:
        with pytest.raises(PermissionError):
            sp.load_dev(mid)
    with pytest.raises(KeyError):
        sp.load_dev("nope")
