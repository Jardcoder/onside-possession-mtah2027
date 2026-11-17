"""Synthetic tests for the touch matching of bancos/evaluar_toques.py (no real data)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "bancos"), os.path.join(ROOT, "src")):
    sys.path.insert(0, p)
import evaluar_toques as ev  # noqa: E402


def match(tf, ef, tt=None, et=None, tol=12, same_team=False):
    tf, ef = np.array(tf), np.array(ef)
    tt = np.zeros(len(tf), int) if tt is None else np.array(tt)
    et = np.zeros(len(ef), int) if et is None else np.array(et)
    return ev.match_one_to_one(tf, tt, ef, et, tol=tol, same_team=same_team)


def test_greedy_losing_case_gives_two_pairs():
    # a greedy match takes (10, 9) first and leaves 0 and 20 unpaired: 1 pair; the maximum is 2
    assert match([0, 10], [9, 20]) == [(0, 0), (1, 1)]


def test_max_cardinality_then_min_distance():
    # both matchings have 2 pairs; (0-4, 5-6) costs 5, the crossing one (0-6, 5-4) costs 7
    assert match([0, 5], [4, 6]) == [(0, 0), (1, 1)]


def test_one_to_one_and_tolerance():
    assert match([0, 1, 2], [1]) == [(1, 0)]                  # three detections, one event: one pair, the closest
    assert match([0], [13]) == []                             # outside tolerance
    assert match([0], [12]) == [(0, 0)]                       # tolerance is inclusive
    assert match([], [3]) == [] and match([3], []) == []


def test_same_team_requirement():
    tf, tt, ef, et = [0, 10], [0, 1], [9, 20], [1, 1]
    assert match(tf, ef, tt, et) == [(0, 0), (1, 1)]
    # with the team required only det 1 (team 1) can pair: maximum cardinality is 1, the closest event (9)
    assert match(tf, ef, tt, et, same_team=True) == [(1, 0)]


def test_clusters_do_not_interact():
    pairs = match([0, 100, 200], [3, 98, 500])
    assert pairs == [(0, 0), (1, 1)]


def test_dp_matches_brute_force_on_random_cases():
    import itertools
    rng = np.random.default_rng(0)
    for _ in range(200):
        tf = np.sort(rng.integers(0, 60, rng.integers(0, 6)))
        ef = np.sort(rng.integers(0, 60, rng.integers(0, 6)))
        got = match(tf, ef)
        best = (0, 0)
        for r in range(min(len(tf), len(ef)) + 1):
            for di in itertools.combinations(range(len(tf)), r):
                for ei in itertools.permutations(range(len(ef)), r):
                    d = [abs(int(tf[a]) - int(ef[b])) for a, b in zip(di, ei)]
                    if all(x <= 12 for x in d):
                        best = max(best, (r, -sum(d)))
        assert (len(got), -sum(abs(int(tf[a]) - int(ef[b])) for a, b in got)) == best
        assert len({a for a, _ in got}) == len(got) and len({b for _, b in got}) == len(got)
