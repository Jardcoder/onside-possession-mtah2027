"""Apply the B3 constants named by env B3_COMBO (evaluar_toques tag) to ronda_pff / evaluar_pff at import time,
so spawned workers inherit them. Import after promoted5_jumps / posesion_rnn and before building features."""
import os
import re


def parse(tag: str) -> dict:
    m = re.fullmatch(r"r([\d.]+)_dv([\d.]+)_w(\d+)_c([\d.]+)x(\d+)_g(\d+)", tag)
    if not m:
        raise ValueError(f"bad B3 tag: {tag}")
    r, dv, win, rc, k, cap = m.groups()
    return {"r": float(r), "dv": float(dv), "win": int(win), "r_ctrl": float(rc), "k": int(k), "cap": int(cap)}


def apply(rp, tag: str) -> dict:
    p = parse(tag)
    rp.V2.update({k: p[k] for k in ("r", "dv", "win", "r_ctrl", "k")})
    rp.B3_C1 = (p["dv"], p["win"], p["r"])
    rp.ep.CAP = p["cap"]
    return p


def apply_from_env(rp) -> dict | None:
    tag = os.environ.get("B3_COMBO")
    return apply(rp, tag) if tag else None
