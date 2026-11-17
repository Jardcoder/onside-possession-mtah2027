"""Blind development / held-out split of the 64 PFF World Cup 2022 matches, fixed in code (2026-10-28).

    challenge/.venv/bin/python bancos/pff_split.py            # prints the split regenerated from the metadata
    from pff_split import DESARROLLO, RESERVADOS, load_dev

Blind: the split reads ONLY the match id and the tournament phase (metadata `week`: 1-3 group stage, 4-8 knockout).
No performance figure, label or tracking is opened to build it. Within each phase the ids are ordered by
SHA-256(SALT + id) and the first N_DEV_BY_PHASE of them go to development, the rest are held out, so the held-out set has the
knockout matches in the same proportion as the whole tournament. Salt and counts are fixed here; the two lists below are
literals, and tests/test_pff_split.py checks that regenerating them gives the same lists.

Guard: every evaluation script of the PFF bank loads matches through `load_dev`, which raises for a held-out match. The
held-out matches are not loaded once before the final confirmation. PFF data are not redistributed: nothing PFF-derived
row by row leaves src/out/ (aggregates per match only).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import onside_paths  # noqa: E402

LOADER_DIR = HERE                                              # the PFF loader (pff.py) lives next to this file
META_DIR = os.path.join(onside_paths.PFF_DIR, "Metadata")

SALT = "onside-pff-split-v1"
N_DEV_BY_PHASE = {"grupos": 30, "eliminatorias": 10}            # 40 development of 64 (48 group + 16 knockout matches)
GROUP_WEEKS = (1, 2, 3)

DESARROLLO = (
    "3812", "3813", "3816", "3817", "3818", "3819", "3820", "3823", "3826", "3827", "3829", "3832", "3833",
    "3836", "3837", "3838", "3840", "3842", "3843", "3844", "3846", "3847", "3848", "3850", "3851", "3852",
    "3853", "3855", "3856", "3859", "10503", "10504", "10506", "10507", "10508", "10509", "10510", "10512",
    "10515", "10516",
)
RESERVADOS = (
    "3814", "3815", "3821", "3822", "3824", "3825", "3828", "3830", "3831", "3834", "3835", "3839", "3841",
    "3845", "3849", "3854", "3857", "3858", "10502", "10505", "10511", "10513", "10514", "10517",
)


def phase_of_week(week):
    return "grupos" if int(week) in GROUP_WEEKS else "eliminatorias"


def read_phases(meta_dir=META_DIR):
    """{match id: phase} from the metadata files (only `id` and `week` are read)."""
    out = {}
    for name in os.listdir(meta_dir):
        if name.endswith(".json"):
            m = json.load(open(os.path.join(meta_dir, name)))
            m = m[0] if isinstance(m, list) else m
            out[str(m["id"])] = phase_of_week(m["week"])
    return out


def split(phases, salt=SALT, n_dev=N_DEV_BY_PHASE):
    """(development, held out), both sorted by integer id. `phases` is {match id: phase}; nothing else is read."""
    dev, held = [], []
    for ph, n in n_dev.items():
        ids = sorted((i for i, p in phases.items() if p == ph),
                     key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())
        if len(ids) < n:
            raise ValueError(f"{ph}: {len(ids)} matches for {n} development slots")
        dev += ids[:n]
        held += ids[n:]
    if set(phases) - set(dev) - set(held):
        raise ValueError("matches of an unknown phase")
    key = lambda i: int(i)                                         # noqa: E731
    return sorted(dev, key=key), sorted(held, key=key)


def load_dev(mid):
    """pff.load(mid) (raw tracking, smoothed=False) for a development match; raises for any other id."""
    if mid in RESERVADOS:
        raise PermissionError(f"{mid} is a held-out match: it is not loaded before the final confirmation")
    if mid not in DESARROLLO:
        raise KeyError(f"{mid} is not a development match")
    if LOADER_DIR not in sys.path:
        sys.path.insert(0, LOADER_DIR)
    import pff
    return pff.load(mid)


if __name__ == "__main__":
    phases = read_phases()
    dev, held = split(phases)
    print(f"development {len(dev)}: {dev}")
    print(f"held out {len(held)}: {held}")
    for ph in N_DEV_BY_PHASE:
        print(f"{ph}: development {sum(phases[i] == ph for i in dev)}, held out {sum(phases[i] == ph for i in held)}")
