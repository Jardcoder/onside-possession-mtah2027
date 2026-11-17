"""Recurrent possession model: per-frame H / A / D from tracking with a bidirectional GRU (approach A30).

    feats = build_features(ball, slots, per, pitch)       (N, 38) float32, tracking only
    net = Net(n_in, cfg)                                   multitask network: possession head (H, A, D) + touch head (none, H, A)
    P = predict_tta(net, X, Xs, per, cfg, device)          (N, 6) probabilities, normal and team-swapped views averaged
    codes = decode(P[:, :3], per, lam)                     Viterbi (modelo.viterbi) -> 0 = H, 1 = A, 2 = D

Training data: PFF development matches with explicit external gap aggregates and
unanchored injection. The possession target and short-spell weights are selected
on PFF; the touch target is the team at the PFF event touch. Labels are training
targets and scoring references only. Prediction inputs contain tracking features.
GPU allocation and duty cycle are bounded; sampled device watts are recorded.
"""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
from torch_reproducibility import enforce_deterministic_operations, seed_deterministically  # noqa: E402
from gpu_budget import GpuBudget  # noqa: E402
from training_data_guard import restrict_training_reads  # noqa: E402
onside_paths.add_challenge_to_path()
import atd  # noqa: E402
import dfl_degradado as dd  # noqa: E402
import inyeccion_huecos as ih  # noqa: E402
import limpieza  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_pff  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reto_periodos as rpd  # noqa: E402
import ronda_pff as rp  # noqa: E402

FPS = 25
VIEWS = {"g6h": "g6", "g8h": "g8"}           # training views: ball smoothing of each
CHALLENGE_SMOOTH = ("g6", "g8")
YOLO_TRACK = os.path.join(onside_paths.OUT_DIR, "balon_yolo", "track.npz")
CAP_SINCE = 10 * FPS
DIST_CAP_M = 40.0
SPEED_CAP = 12.0
BALL_SPEED_CAP = 35.0
VEL_HALF = 2
TOUCH_HALF = 2                               # frames either side of a PFF touch that carry the touch target
N_OUT = 6                                    # possession (H, A, D) + touch (none, H, A)
SWAP_COLS = [1, 0, 2, 3, 5, 4]               # columns of the team-swapped view mapped back
LAM_HA = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0)
LAM_D = (0.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0)
GPU_SHARE = 0.7
GPU_BUDGET: GpuBudget | None = None
BUILD_TIMEOUT_S = 600


# ------------------------------------------------------------------------------------------------ features

def orientation(slots: dict, per: np.ndarray) -> np.ndarray:
    """(N,) +1 where the home team defends the -x side in that period (its mean x below the away mean), else -1. Label free."""
    sign = np.ones(len(per))
    for p in np.unique(per):
        m = per == p
        hx, ax = np.nanmean(slots["H"][m, :, 0]), np.nanmean(slots["A"][m, :, 0])
        sign[m] = 1.0 if hx < ax else -1.0
    return sign


def period_starts(per: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.flatnonzero(np.diff(per) != 0) + 1, [len(per)]])


def last_seen(present: np.ndarray, per: np.ndarray) -> np.ndarray:
    """(N,) index of the last frame with ball at or before t inside the same period, -1 if none."""
    out = np.full(len(present), -1, np.int64)
    b = period_starts(per)
    for a, z in zip(b[:-1], b[1:]):
        idx = np.where(present[a:z], np.arange(a, z), -1)
        out[a:z] = np.maximum.accumulate(idx)
    return out


def central_diff(x: np.ndarray, per: np.ndarray, h: int) -> np.ndarray:
    """(x[t+h] - x[t-h]) / (2h / FPS) along axis 0 where both ends exist and lie in the period of t, else NaN."""
    n = len(x)
    out = np.full(x.shape, np.nan)
    if n <= 2 * h:
        return out
    ok = per[2 * h:] == per[:-2 * h]
    d = (x[2 * h:] - x[:-2 * h]) * (FPS / (2 * h))
    d[~ok] = np.nan
    out[h:n - h] = d
    return out


def team_block(p: np.ndarray, ref: np.ndarray, per: np.ndarray) -> list:
    """Columns of one team: players p (N, S, 2) and reference point ref (N, 2), metres in the canonical frame."""
    ok = np.isfinite(p[:, :, 0])
    n = ok.sum(1)
    p0 = np.where(ok[:, :, None], p, 0.0)
    cnt = np.maximum(n, 1)[:, None]
    cen = p0.sum(1) / cnt
    spread = np.sqrt(np.where(ok[:, :, None], (p - cen[:, None, :]) ** 2, 0.0).sum(1) / cnt)
    dist = np.where(ok, np.hypot(p[:, :, 0] - ref[:, None, 0], p[:, :, 1] - ref[:, None, 1]), np.inf)
    near3 = np.minimum(np.sort(np.partition(dist, 2, axis=1)[:, :3], axis=1), DIST_CAP_M)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(p))
    has = np.isfinite(dist[rows, j])
    rel = np.where(has[:, None], np.clip(p[rows, j] - ref, -DIST_CAP_M, DIST_CAP_M), 0.0)
    v = central_diff(p, per, 1)
    spd = np.nan_to_num(np.clip(np.hypot(v[:, :, 0], v[:, :, 1]), 0, SPEED_CAP), nan=0.0)
    nspd = np.where(has, spd[rows, j], 0.0)
    dmin = near3[:, 0]
    ddmin = np.nan_to_num(np.clip(central_diff(dmin, per, VEL_HALF), -20, 20), nan=0.0)
    return [n / 11.0, cen[:, 0] / 52.5, cen[:, 1] / 34.0, spread[:, 0] / 20.0, spread[:, 1] / 20.0,
            near3[:, 0] / DIST_CAP_M, near3[:, 1] / DIST_CAP_M, near3[:, 2] / DIST_CAP_M,
            rel[:, 0] / DIST_CAP_M, rel[:, 1] / DIST_CAP_M, nspd / SPEED_CAP, ddmin / 20.0,
            (dist < 2.0).sum(1) / 3.0, (dist < 5.0).sum(1) / 5.0]


def build_features(ball: np.ndarray, slots: dict, per: np.ndarray, pitch: tuple) -> np.ndarray:
    """(N, 38) float32 from tracking only: ball 0-1 (NaN = missing), slots {"H", "A"} 0-1, periods, pitch (m).
    Canonical frame: centre origin, metres, x flipped per period so that home always attacks +x."""
    per = np.asarray(per)
    sc = np.array(pitch, float)
    sign = orientation(slots, per)
    flip = np.stack([sign, np.ones_like(sign)], -1)
    b = (np.asarray(ball, float) - 0.5) * sc * flip
    P = {s: (np.asarray(slots[s], float) - 0.5) * sc * flip[:, None, :] for s in "HA"}
    present = np.isfinite(b[:, 0])
    ls = last_seen(present, per)
    seen = ls >= 0
    bf = np.where(seen[:, None], b[np.maximum(ls, 0)], 0.0)
    since = np.where(seen, np.minimum(np.arange(len(b)) - ls, CAP_SINCE), CAP_SINCE) / CAP_SINCE
    vel = np.nan_to_num(np.clip(central_diff(b, per, VEL_HALF), -BALL_SPEED_CAP, BALL_SPEED_CAP), nan=0.0)
    b0 = np.where(present[:, None], b, 0.0)
    cols = [present.astype(float), seen.astype(float), b0[:, 0] / 52.5, b0[:, 1] / 34.0, bf[:, 0] / 52.5, bf[:, 1] / 34.0,
            since, vel[:, 0] / BALL_SPEED_CAP, vel[:, 1] / BALL_SPEED_CAP, np.hypot(vel[:, 0], vel[:, 1]) / BALL_SPEED_CAP]
    for s in "HA":
        cols += team_block(P[s], bf, per)
    return np.stack(cols, -1).astype(np.float32)


def swapped(slots: dict) -> dict:
    return {"H": slots["A"], "A": slots["H"]}


# ------------------------------------------------------------------------------------------------ targets (training side)

def touch_teams(label: np.ndarray, frames: np.ndarray) -> tuple:
    """(frames, team 0 = H / 1 = A) of the PFF touches: the label of the touch frame or of the next 2; touches with no team dropped."""
    lab = np.asarray(label)
    keep_f, keep_t = [], []
    for f in frames:
        for g in range(f, min(f + 3, len(lab))):
            if lab[g] in ("H", "A"):
                keep_f.append(int(f))
                keep_t.append(0 if lab[g] == "H" else 1)
                break
    return np.array(keep_f, np.int64), np.array(keep_t, np.int8)


def touch_targets(n: int, frames: np.ndarray, teams: np.ndarray) -> np.ndarray:
    """(n,) int8: 0 none, 1 touch H, 2 touch A on the frames within TOUCH_HALF of a touch (a later touch overwrites)."""
    y = np.zeros(n, np.int8)
    for f, t in zip(frames, teams):
        y[max(f - TOUCH_HALF, 0):min(f + TOUCH_HALF + 1, n)] = 1 + t
    return y


def swap_codes(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[y == 0], out[y == 1] = 1, 0
    return out


def swap_touch(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[y == 1], out[y == 2] = 2, 1
    return out


# ------------------------------------------------------------------------------------------------ training data

def dev_split() -> tuple:
    """(36 fit matches, 4 early-stopping matches): the development matches ordered by SHA-256(modelo_pff.ES_SALT + id)."""
    dev = sorted((m for f in rp.FOLDS for m in f), key=lambda m: hashlib.sha256((mp_pff.ES_SALT + m).encode()).hexdigest())
    return dev[mp_pff.ES_MATCHES:], dev[:mp_pff.ES_MATCHES]


def external_gap_model(path: str) -> dict:
    """Read the explicitly selected PFF-only aggregate; never open challenge assets."""
    calibration = json.loads(Path(path).read_text())
    if calibration.get("source") != "PFF development videos only":
        raise ValueError("A30 training requires PFF-only external gap calibration")
    if calibration["anchor_probability"] != 0.0:
        raise ValueError("A30 external protocol requires unanchored gap injection")
    if not 0 < calibration["share"] < 1 or not sum(c["n_runs"] for c in calibration["classes"]):
        raise ValueError("External calibration must contain a valid absence rate and gap histogram")
    return calibration


def training_ball(d: dict, mid: str, view: str, calibration: dict) -> np.ndarray:
    """PFF ball with the external (PFF-only) gap statistics of `calibration`."""
    ball = dd.smooth_ball(np.asarray(d["ball_xy"], float), d["period"], VIEWS[view])
    return ih.inject_holes(ball, d["label"], d["period"], mid, target=calibration["share"],
                           draw=ih.length_model(calibration), anchor_p=calibration["anchor_probability"])[0]


def build_match(mid: str, calibration_path: str) -> dict:
    """Features (both views, normal and team-swapped) and targets of one development match."""
    restrict_training_reads((Path(onside_paths.CHALLENGE_DIR) / "data",
                             Path(onside_paths.EVALUATION_DIR) / "gt",
                             Path(onside_paths.OUT_DIR) / "balon_yolo"))
    d = rp.load_match(mid)
    per = np.asarray(d["period"])
    calibration = external_gap_model(calibration_path)
    out = {"mid": mid, "per": per,
           "targets": {key: mp_pff.label_codes(d[key]).astype(np.int64) for key in ("label", "label_poss")},
           "unc": np.asarray(d["uncertain"], bool), "gt": rp.ev.make_gt(d, "label")}
    ev_f, ev_t = touch_teams(d["label"], d["event_f"])
    out["touch"] = touch_targets(len(per), ev_f, ev_t).astype(np.int64)
    for view in VIEWS:
        ball = training_ball(d, mid, view, calibration)
        out[view] = {"X": build_features(ball, d["players_slots"], per, d["pitch"]).astype(np.float16),
                     "Xs": build_features(ball, swapped(d["players_slots"]), per, d["pitch"]).astype(np.float16)}
    return out


def build_matches(mids: list, procesos: int, calibration_path: str) -> dict:
    """build_match in worker processes; a dead worker raises BrokenProcessPool, a stuck one TimeoutError."""
    with ProcessPoolExecutor(procesos, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(build_match, m, calibration_path) for m in mids]
        out = {}
        for fut in as_completed(futs, timeout=BUILD_TIMEOUT_S * len(mids)):
            r = fut.result(timeout=BUILD_TIMEOUT_S)
            out[r["mid"]] = r
        return out


def short_spell_weights(y: np.ndarray, per: np.ndarray, cfg: dict) -> np.ndarray:
    """(N,) float32: min(cfg["short_max"], max(1, cfg["short_ref"] / d)) on every frame of a run of constant label of duration d (s);
    runs are cut at label and period changes."""
    w = np.ones(len(y), np.float32)
    edges = np.flatnonzero((y[1:] != y[:-1]) | (per[1:] != per[:-1])) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(y)]):
        w[a:b] = min(cfg["short_max"], max(1.0, cfg["short_ref"] / ((b - a) / FPS)))
    return w


def entries(mids: list, built: dict, cfg: dict) -> list:
    """Training entries (one per match and view): features, targets of both orientations, loss weights (uncertain PFF frames out,
    short possession runs weighted up), chunkable spans."""
    out = []
    chunk = cfg["chunk"]
    for mid in mids:
        b = built[mid]
        target = b["targets"][cfg["target"]]
        y = np.stack([target, b["touch"]], -1)
        ys = np.stack([swap_codes(target), swap_touch(b["touch"])], -1)
        w = (~b["unc"]).astype(np.float32) * short_spell_weights(target, b["per"], cfg)
        spans = [(int(a), int(z)) for a, z in zip(period_starts(b["per"])[:-1], period_starts(b["per"])[1:]) if z - a >= chunk]
        for view in VIEWS:
            out.append({"mid": mid, "view": view, "X": b[view]["X"], "Xs": b[view]["Xs"], "y": y, "ys": ys, "w": w, "spans": spans,
                        "per": b["per"], "gt": b["gt"]})
    return out


# ------------------------------------------------------------------------------------------------ network

def limit_gpu(device: torch.device) -> None:
    global GPU_BUDGET
    GPU_BUDGET = GpuBudget(device, GPU_SHARE, Path(onside_paths.OUT_DIR) / "gpu" / f"a30-{os.getpid()}.json")


def throttle(device: torch.device, t0: float) -> None:
    """Sleep so that the GPU work since t0 (time.monotonic) is at most GPU_SHARE of the elapsed time; capped pause."""
    if device.type == "cuda":
        if GPU_BUDGET is None:
            raise RuntimeError("Call limit_gpu before GPU work")
        GPU_BUDGET.finish_batch(t0)


class Net(nn.Module):
    """Per-frame encoder, strided convolution to FPS / stride, stacked single-layer bidirectional GRUs (torch dropout between them:
    the dropout of a multi-layer nn.GRU fails to compile on this ROCm stack), transposed convolution back to FPS and a per-frame head
    over [frame encoding, upsampled context]."""

    def __init__(self, n_in: int, cfg: dict):
        super().__init__()
        hidden, st = cfg["hidden"], cfg["stride"]
        self.stride = st
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.down = nn.Conv1d(hidden, hidden, st, stride=st)
        self.grus = nn.ModuleList([nn.GRU(hidden if i == 0 else 2 * hidden, hidden, 1, batch_first=True, bidirectional=True)
                                   for i in range(cfg["layers"])])
        self.drop = nn.Dropout(cfg["dropout"])
        self.up = nn.ConvTranspose1d(2 * hidden, hidden, st, stride=st)
        self.local = nn.Conv1d(2 * hidden, hidden, 2 * st + 1, padding=st)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, N_OUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        f = self.inp(nn.functional.pad(x, (0, 0, 0, (-t) % self.stride)))
        h = self.down(f.transpose(1, 2)).transpose(1, 2)
        for i, g in enumerate(self.grus):
            h = g(self.drop(h) if i else h)[0]
        u = self.up(h.transpose(1, 2))
        z = self.local(torch.cat([f.transpose(1, 2), u], 1)).transpose(1, 2)
        return self.out(z)[:, :t]


def masked_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), weight=cw, reduction="none")
    m = w.reshape(-1).float()
    return (ce * m).sum() / m.sum().clamp(min=1.0)


def loss_of(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """Possession head (weights w: uncertain PFF frames out) + touch head (every frame, class weights cw)."""
    return masked_loss(logits[..., :3], y[..., 0], w, torch.ones_like(cw)) + masked_loss(logits[..., 3:], y[..., 1], torch.ones_like(w), cw)


def sample_batch(data: list, rng: np.random.Generator, batch: int, length: int) -> tuple:
    """Random chunks inside one period on the device of `data`; each chunk picks the normal or the team-swapped view."""
    xs, ys, ws = [], [], []
    for _ in range(batch):
        e = data[int(rng.integers(len(data)))]
        spans = e["spans"]
        lens = np.array([z - a for a, z in spans], float)
        a, z = spans[int(rng.choice(len(spans), p=lens / lens.sum()))]
        s = int(rng.integers(a, z - length + 1))
        sw = bool(rng.integers(2))
        xs.append(e["Xs" if sw else "X"][s:s + length])
        ys.append(e["ys" if sw else "y"][s:s + length])
        ws.append(e["w"][s:s + length])
    return torch.stack(xs).float(), torch.stack(ys), torch.stack(ws)


def to_device(data: list, device: torch.device) -> list:
    keys = ("X", "Xs", "y", "ys", "w")
    return [{**e, **{k: torch.from_numpy(np.ascontiguousarray(e[k])).to(device) for k in keys}} for e in data]


def val_loss(net: Net, val: list, cfg: dict, device: torch.device) -> float:
    """Mean per-chunk loss over fixed non-overlapping chunks of the early-stopping matches (normal orientation)."""
    cw = torch.tensor([1.0, cfg["touch_weight"], cfg["touch_weight"]], device=device)
    chunks = [(e, s) for e in val for a, z in e["spans"] for s in range(a, z - cfg["chunk"] + 1, cfg["chunk"])]
    losses = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(chunks), cfg["batch"]):
            tb = time.monotonic()
            part = chunks[i:i + cfg["batch"]]
            x = torch.from_numpy(np.stack([e["X"][s:s + cfg["chunk"]] for e, s in part]).astype(np.float32)).to(device)
            y = torch.from_numpy(np.stack([e["y"][s:s + cfg["chunk"]] for e, s in part])).to(device)
            w = torch.from_numpy(np.stack([e["w"][s:s + cfg["chunk"]] for e, s in part])).to(device)
            logits = net(x)
            losses += [float(loss_of(logits[j:j + 1], y[j:j + 1], w[j:j + 1], cw)) for j in range(len(part))]
            throttle(device, tb)
    return float(np.mean(losses))


def train(fit: list, val: list, cfg: dict, device: torch.device, log: Callable) -> tuple:
    """(net, history): AdamW, early stopping on the validation loss, the best epoch's weights kept; seeded with cfg["seed"]."""
    seed_deterministically(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    net = Net(fit[0]["X"].shape[1], cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    cw = torch.tensor([1.0, cfg["touch_weight"], cfg["touch_weight"]], device=device)
    best, best_state, bad, hist = np.inf, None, 0, []
    deadline = time.monotonic() + cfg["maximum_training_seconds"]
    fit_dev = to_device(fit, device)
    for epoch in range(cfg["epochs"]):
        net.train()
        t0, tl = time.monotonic(), 0.0
        for _ in range(cfg["steps"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("A30 training exceeded its declared per-model deadline")
            tb = time.monotonic()
            x, y, w = sample_batch(fit_dev, rng, cfg["batch"], cfg["chunk"])
            loss = loss_of(net(x), y, w, cw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += float(loss.detach())
            throttle(device, tb)
        vl = val_loss(net, val, cfg, device)
        if not np.isfinite(tl) or not np.isfinite(vl):
            raise FloatingPointError(f"Non-finite A30 loss at epoch {epoch}, seed {cfg['seed']}")
        hist.append({"epoch": epoch, "train": tl / cfg["steps"], "val": vl})
        log(f"epoch {epoch:2d} train {tl / cfg['steps']:.4f} val {vl:.4f} ({time.monotonic() - t0:.0f} s)")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    return net, hist


# ------------------------------------------------------------------------------------------------ prediction and decoding

def predict_sequence(net: Net, X: np.ndarray, per: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    """(N, 6) softmax probabilities per head; each period in windows of cfg["win"] frames with cfg["margin"] context, batched."""
    enforce_deterministic_operations()
    net.eval()
    out = np.zeros((len(X), N_OUT), np.float32)
    b = period_starts(per)
    win, mg = cfg["win"], cfg["margin"]
    jobs = [(max(a, s - mg), min(z, s + win + mg), s, min(s + win, z)) for a, z in zip(b[:-1], b[1:]) for s in range(a, z, win)]
    with torch.no_grad():
        for i in range(0, len(jobs), cfg["infer_batch"]):
            tb = time.monotonic()
            part = jobs[i:i + cfg["infer_batch"]]
            L = max(hi - lo for lo, hi, _, _ in part)
            x = np.zeros((len(part), L, X.shape[1]), np.float32)
            for j, (lo, hi, _, _) in enumerate(part):
                x[j, :hi - lo] = X[lo:hi]
            logits = net(torch.from_numpy(x).to(device))
            p = torch.softmax(logits.reshape(*logits.shape[:-1], -1, 3), -1).reshape(logits.shape).cpu().numpy()
            for j, (lo, _, s, e) in enumerate(part):
                out[s:e] = p[j, s - lo:e - lo]
            throttle(device, tb)
    return out


def predict_tta(net: Net, X: np.ndarray, Xs: np.ndarray, per: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    """Mean of the normal view and the team-swapped view mapped back."""
    p = predict_sequence(net, X, per, cfg, device)
    q = predict_sequence(net, Xs, per, cfg, device)
    return (p + q[:, SWAP_COLS]) / 2


def decode(P: np.ndarray, per: np.ndarray, lam: tuple) -> np.ndarray:
    """Viterbi over H / A / D with penalty lam[0] per H <-> A change and lam[1] per entry / exit of D, per period."""
    return mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), per, [lam[0]], [lam[1]])[0]


def spell_f1(gt: np.ndarray, codes: np.ndarray) -> float:
    return float(pm.spell_scores(gt, codes, pm.valid_runs(gt))["mean_f1"])


def choose_lambda(probs: list, val: list) -> tuple:
    """Choose on PFF label: maximise the smaller mean over the two ball views."""
    grid_ha = np.repeat(LAM_HA, len(LAM_D))
    grid_d = np.tile(LAM_D, len(LAM_HA))
    f1 = {view: [] for view in VIEWS}
    for P, e in zip(probs, val):
        paths = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), e["per"], grid_ha, grid_d)
        f1[e["view"]].append([spell_f1(e["gt"], p) for p in paths])
    if not all(f1.values()):
        raise ValueError("Lambda selection requires both PFF ball views")
    objective = np.minimum.reduce([np.mean(f1[view], axis=0) for view in VIEWS])
    k = int(np.flatnonzero(objective >= objective.max() - rp.ep.TIE)[0])
    return (float(grid_ha[k]), float(grid_d[k])), float(objective[k])


# ------------------------------------------------------------------------------------------------ challenge inputs

def challenge_inputs(ball_video: np.ndarray) -> dict:
    """Challenge features in the PFF layout (src/reto_periodos.py): cleaned ATD players, the given ball, views g6 / g8."""
    d = limpieza.clean_atd()
    n = len(d["ball_xy"])
    frames = rpd.play_frames(n)
    per = rpd.periods(n)[frames].astype(np.int64)
    slots = {k: v[frames] for k, v in d["players_slots"].items()}
    views = []
    for s in CHALLENGE_SMOOTH:
        b = dd.smooth_ball(ball_video[frames], per, s)
        views.append({"X": build_features(b, slots, per, atd.PITCH), "Xs": build_features(b, swapped(slots), per, atd.PITCH)})
    return {"views": views, "per": per, "frames": frames, "n_video": n}
