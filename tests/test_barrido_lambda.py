"""bancos/barrido_lambda.py: nested selection of lambda on synthetic spellF1 grids (no model, no cache)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import barrido_lambda as bw  # noqa: E402
import modelo_pff as mp_  # noqa: E402

K = len(bw.GRID)
I_A, I_B = bw.GRID.index((8, 16)), bw.GRID.index((16, 32))


def synthetic(n_per_fold=4, n_folds=5, best=I_A, rng=None):
    """f1 (n, 2 balls, K): a smooth bump around `best` plus noise; fold_of (n,)."""
    rng = rng or np.random.default_rng(0)
    n = n_per_fold * n_folds
    base = 0.5 - 0.01 * np.abs(np.arange(K) - best)
    f1 = base[None, None, :] + 0.001 * rng.normal(size=(n, 2, K))
    return f1, np.repeat(np.arange(n_folds), n_per_fold)


def test_grid_is_the_requested_one_in_lam_ha_major_order():
    assert bw.GRID[0] == (4, 8) and bw.GRID[-1] == (16, 192) and len(bw.GRID) == 27
    assert bw.GRID[:9] == [(4, d) for d in (8, 16, 24, 32, 48, 64, 96, 128, 192)]
    assert tuple(bw.FIXED) in bw.GRID


def test_nested_selection_picks_the_common_optimum_in_every_fold_and_reads_the_held_out_matches_there():
    f1, fold_of = synthetic()
    chosen, nested = bw.nested_select(f1, fold_of)
    assert chosen == {k: I_A for k in range(5)}
    assert np.array_equal(nested, f1[:, :, I_A])


def test_the_held_out_fold_never_decides_its_own_lambda():
    f1, fold_of = synthetic()
    spoiled = f1.copy()
    spoiled[fold_of == 2, :, I_B] = 5.0                     # fold 2 would love lambda B, the others must not see it
    chosen, nested = bw.nested_select(spoiled, fold_of)
    assert chosen[2] == I_A                                 # chosen with folds 0, 1, 3, 4 only
    assert all(chosen[k] == I_B for k in (0, 1, 3, 4))      # the other folds do see the spoiled fold 2
    held = fold_of == 2
    assert np.array_equal(nested[held], spoiled[held][:, :, I_A])   # fold 2 is scored at the lambda it did not choose itself


def test_the_criterion_is_the_min_of_the_two_balls_and_ties_go_to_the_first_grid_point():
    f1 = np.full((10, 2, K), 0.3)
    f1[:, 0, 4] = 0.9
    f1[:, 1, 4] = 0.2                                       # best g6h, but min = 0.2 < 0.3
    f1[:, :, 9] = 0.4                                       # min 0.4
    f1[:, :, 12] = 0.4                                      # same min, later in the grid
    chosen, _ = bw.nested_select(f1, np.repeat(np.arange(5), 2))
    assert set(chosen.values()) == {9}


def test_final_choice_is_the_most_chosen_and_a_tie_goes_to_the_best_mean():
    f1, _ = synthetic()
    f1[:, :, I_B] = f1[:, :, I_A] + 0.003                   # B is better over the 40 matches
    chosen = {0: I_A, 1: I_A, 2: I_A, 3: I_B, 4: I_B}
    assert bw.final_choice(chosen, f1)[0] == I_A            # 3 votes against 2, the mean does not decide
    chosen = {0: I_A, 1: I_A, 2: I_B, 3: I_B, 4: 0}
    idx, count = bw.final_choice(chosen, f1)
    assert idx == I_B and count == {I_A: 2, I_B: 2, 0: 1}   # 2 against 2: B has the better mean over the 40 matches


def test_on_edge():
    assert bw.on_edge(192, bw.LAM_D) == "high" and bw.on_edge(8, bw.LAM_D) == "low" and bw.on_edge(16, bw.LAM_D) is None


def test_uses_the_pick_lambda_of_m1():
    assert bw.nested_select.__defaults__[1] is mp_.pick_lambda
