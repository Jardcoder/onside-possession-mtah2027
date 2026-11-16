"""Challenge side of T1: the features of the challenge match, the T1 ensemble and its decoding, shared by bancos/dominio.py
(domain shift) and src/predecir_reto_m1.py (the CSV), so both read the challenge in exactly the same way.

    X, names, per, info = build_reto()           features of the challenge cut to the PFF layout (src/reto_periodos.py), no ground truth
    boosters = t1_boosters()                     the 5 T1 boosters of the outer folds (src/out/modelo_cache/transfer/modelos)
    P = ensemble_proba(boosters, X, names)       mean of the 5 symmetric probabilities (N, 3) (H, A, D)
    codes = decode(P, per, lam)                  Viterbi per period -> (N,) int8 0 = H, 1 = A, 2 = D
    labels = to_video(letters, info)             back to the video index: "X" outside the playing periods, "D" in the graded frames before the 2nd kick-off

The constructor is the one of T1 for a ball without added holes (limpieza.clean_atd without smoothing -> rp.build_method_ctx -> rp.pipeline ->
absorption tau 0.8 -> modelo.build_features), the same as modelo_transfer.build_one_h but on the cleaned challenge arrays.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import absorcion as ab  # noqa: E402
import atd  # noqa: E402
import limpieza  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import reto_periodos as rpd  # noqa: E402
import ronda_pff as rp  # noqa: E402

N_FOLDS = len(rp.FOLDS)
BASE_TAU = 1.6                                  # absorption of the current base (B3 + tau 1.6 s)


def build_reto():
    """(X (N, F) float32 full rate, names, per, info) of the challenge match with the constructor of T1 (no holes added, no ground truth).
    The cleaned arrays are cut to the PFF layout (src/reto_periodos.py: period 1 = kick-off .. 90,651, period 2 = kick-off 113,742 .. 190,197), so the
    clock and the windows of the features mean what they mean in PFF. `info`: `frames` (video index of every row), `n_video`, the graded frames left
    outside (before the 2nd kick-off: `pre2_n`, `pre2_no_ball`), and `b3` (N,) the "H" / "A" / "D" labels of B3 on the cut arrays."""
    d = limpieza.clean_atd()
    n = len(d["ball_xy"])
    frames = rpd.play_frames(n)
    per = rpd.periods(n)[frames]
    ball = d["ball_xy"][frames]
    slots = {k: v[frames] for k, v in d["players_slots"].items()}
    m = rp.build_method_ctx(ball, slots, per, atd.PITCH)
    b3, tf, _ = rp.pipeline(m)
    b3a = ab.absorb(b3, mp_.TAUS[0], per)
    X, names = mo.build_features(ball, slots, per, atd.PITCH, b3, b3a, tf)
    lo, hi = rpd.pre_kickoff_2nd()
    info = {"frames": frames, "n_video": n, "pre2_n": hi - lo + 1, "pre2_no_ball": float(np.isnan(d["ball_xy"][lo:hi + 1, 0]).mean()), "b3": b3}
    return X, list(names), per, info


def t1_boosters():
    import lightgbm as lgb
    return [lgb.Booster(model_file=os.path.join(mt.MODELOS, f"T1_k{k}.txt")) for k in range(N_FOLDS)]


def ensemble_proba(boosters, X, names):
    """Mean over the boosters of the symmetric probabilities (N, 3) float64 (H, A, D)."""
    return np.mean([mo.predict_proba(b, X, names) for b in boosters], axis=0)


def decode(P, per, lam):
    """Viterbi with lam = (lam_ha, lam_d) inside each period -> (N,) int8 0 = H, 1 = A, 2 = D."""
    return mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), per, [lam[0]], [lam[1]])[0]


def to_video(letters, info, pre_kickoff="D"):
    """(n_video,) array of "H" / "A" / "D" / "X": `letters` (N,) of the cut rows at their video frames, `pre_kickoff` in the graded frames before the 2nd
    kick-off (all D in the ground truth, FAQ.md:38-39), "X" everywhere else (not graded)."""
    full = rpd.to_full(np.asarray(letters, dtype="<U1"), info["frames"], info["n_video"])
    lo, hi = rpd.pre_kickoff_2nd()
    full[lo:hi + 1] = pre_kickoff
    return full
