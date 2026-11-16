"""Carga del ATD tactico a arrays por frame (solo numpy, sin kloppy).

Formato de la linea cruda (segun starter/baseline.py): "<frame_id>:<jugador>;<jugador>;...:<balon>",
cada grupo "x,y,speed", con "NaN" donde falta la observacion. Coordenadas normalizadas 0-1.

Alineacion: frame_id cuenta desde 1, el submission desde 0 -> video_frame = frame_id - 1.
El array resultante esta indexado por video_frame: arr[i] == fila i del submission.

Jugadores: el ATD tiene 46 "track slots" (tracklets), NO 11 por equipo. Cada slot pertenece a un
equipo (metadata XML). Por eso se devuelven dos vistas:
  - players_slots[team]: (N, n_slots_team, 2) alineado por slot, NaN si el slot no esta.
  - compact(slots, k=11): empaqueta por frame los slots presentes a la izquierda (orden de slot),
    (N, k, 2) con NaN de relleno. Si hay mas de k presentes (el tracker duplica a veces) se
    truncan los de mayor indice de slot; `n_present` informa de cuantos habia.
El orden dentro de compact NO es una identidad de jugador.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
from games import GAME, GAMES, asset_paths  # noqa: E402
from baseline import team_of_each_track, HOME, AWAY  # noqa: E402  (reutiliza el mapeo slot->equipo)

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH = GAMES[GAME]["pitch"]  # metros (largo, ancho)


def video_frame_of(frame_id):
    return int(frame_id) - 1


def frame_id_of(video_frame):
    return int(video_frame) + 1


def _group(text, n):
    """Primeros n valores float de "a,b,c"; NaN si falta o si el grupo es NaN."""
    out = np.full(n, np.nan)
    if not text or text[0] == "N":
        return out
    vals = text.split(",")
    for k in range(min(n, len(vals))):
        out[k] = float(vals[k])
    return out


def load_tactical(data_dir=None):
    """Devuelve dict con arrays indexados por video_frame (len N_FRAMES).

    ball_xy (N,2) | ball_extra (N,) tercer valor del grupo balon (en el formato EPTS es speed;
    NaN si el grupo solo trae 2 valores) | ball_present (N,) bool
    players_slots {"H": (N,Sh,2), "A": (N,Sa,2)} | n_present {"H": (N,), "A": (N,)}
    ball_group_len: histograma {n_valores: n_frames} del grupo balon, para auditar si hay z.
    """
    paths = asset_paths(GAME, data_dir)
    team = team_of_each_track(paths["atd_meta"])
    slots_h = np.flatnonzero(team == HOME)
    slots_a = np.flatnonzero(team == AWAY)
    n_slots = len(team)

    ball_xy = np.full((N_FRAMES, 2), np.nan)
    ball_extra = np.full(N_FRAMES, np.nan)
    ph = np.full((N_FRAMES, len(slots_h), 2), np.nan)
    pa = np.full((N_FRAMES, len(slots_a), 2), np.nan)
    seen = np.zeros(N_FRAMES, dtype=bool)
    glen = {}

    with open(paths["atd_raw"]) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = video_frame_of(parts[0])
            seen[i] = True
            b = parts[2].strip().rstrip(";")
            vals = _group(b, 3)
            ball_xy[i] = vals[:2]
            ball_extra[i] = vals[2]
            glen[len(b.split(","))] = glen.get(len(b.split(",")), 0) + 1
            groups = parts[1].split(";")
            xy = np.full((n_slots, 2), np.nan)
            for s, g in enumerate(groups[:n_slots]):
                xy[s] = _group(g, 2)
            ph[i] = xy[slots_h]
            pa[i] = xy[slots_a]

    return {
        "ball_xy": ball_xy,
        "ball_extra": ball_extra,
        "ball_present": ~np.isnan(ball_xy[:, 0]),
        "players_slots": {"H": ph, "A": pa},
        "n_present": {"H": (~np.isnan(ph[:, :, 0])).sum(1), "A": (~np.isnan(pa[:, :, 0])).sum(1)},
        "frame_seen": seen,
        "ball_group_len": glen,
    }


def compact(slots, k=11):
    """(N,S,2) alineado por slot -> (N,k,2) con los presentes a la izquierda, NaN de relleno."""
    present = ~np.isnan(slots[:, :, 0])
    order = np.argsort(~present, axis=1, kind="stable")  # presentes primero, orden de slot
    packed = np.take_along_axis(slots, order[:, :, None], axis=1)
    out = packed[:, :k]
    if out.shape[1] < k:
        pad = np.full((len(slots), k - out.shape[1], 2), np.nan)
        out = np.concatenate([out, pad], axis=1)
    return out


def to_metres(xy):
    return xy * np.array(PITCH)
