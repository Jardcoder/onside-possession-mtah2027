"""DFL / Sportec open data (IDSSE, CC BY 4.0) loader.

Per-frame DataFrame (25 Hz) with the ball, BallStatus, BallPossession and player
positions per team, plus H/A/D labels. Coordinates are the raw ones: meters, origin
at pitch center (x in [-52.5, 52.5], y in [-34, 34]).

Usage:
    from dfl import load_match, load_events, MATCHES
    df = load_match("J03WPY")        # cached as parquet in data/cache/
    ev = load_events("J03WPY")       # events with frame aligned by timestamp

Columns:
    frame        DFL frame number N (first half starts ~10000, second ~100000)
    period       1 / 2
    t_utc        timestamp (UTC, tz-naive datetime64)
    ball_x/y/z, ball_status (1 alive / 0 dead), ball_possession (1 home, 2 guest, NaN)
    label        'H' (home possession), 'A' (guest), 'D' (BallStatus == 0)
    h<shirt>_x/_y, a<shirt>_x/_y   player positions (NaN when not on the pitch)
Labelling note: label comes from the dataset's own BallPossession, see
dfl_ultimo_toque.py for the last-touch re-derivation.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import etree

ROOT = Path(__file__).parent
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import onside_paths  # noqa: E402

DATA = Path(onside_paths.BANKS_DIR) / "dfl"
CACHE = Path(onside_paths.BANKS_DIR) / "cache"
MATCHES = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
# Held out for the final confirmation (chosen blind, last two by list order, 2026-10-28).
# Never used to pick rules or parameters; only `--con-reservados` scores them.
RESERVADOS = ("J03WQQ", "J03WR9")
DESARROLLO = [m for m in MATCHES if m not in RESERVADOS]
HZ = 25


def _path(kind, mid):
    pre = {"meta": "DFL_02_01_matchinformation", "event": "DFL_03_02_events_raw",
           "tracking": "DFL_04_03_positions_raw_observed"}[kind]
    return DATA / f"{pre}_{MATCHES[mid]}_DFL-MAT-{mid}.xml"


def load_meta(mid):
    """Returns dict: home/guest team ids and names, shirt per PersonId, team per PersonId."""
    root = etree.parse(str(_path("meta", mid))).getroot()
    g = root.find(".//General")
    out = {"home_id": g.get("HomeTeamId"), "guest_id": g.get("GuestTeamId"),
           "home": g.get("HomeTeamName"), "guest": g.get("GuestTeamName"),
           "kickoff": g.get("KickoffTime"), "person": {}}
    for t in root.iter("Team"):
        side = "h" if t.get("TeamId") == out["home_id"] else "a"
        for p in t.iter("Player"):
            out["person"][p.get("PersonId")] = (side, int(p.get("ShirtNumber")), t.get("TeamId"))
    return out


def _parse_tracking(mid):
    meta = load_meta(mid)
    ball = {}
    players = {}
    for _, fs in etree.iterparse(str(_path("tracking", mid)), tag="FrameSet"):
        half = 1 if fs.get("GameSection") == "firstHalf" else 2
        team = fs.get("TeamId")
        frames = fs.findall("Frame")
        n = np.fromiter((int(f.get("N")) for f in frames), np.int64, len(frames))
        x = np.fromiter((float(f.get("X")) for f in frames), np.float32, len(frames))
        y = np.fromiter((float(f.get("Y")) for f in frames), np.float32, len(frames))
        if team == "BALL":
            z = np.fromiter((float(f.get("Z")) for f in frames), np.float32, len(frames))
            poss = np.fromiter((float(f.get("BallPossession", "nan")) for f in frames), np.float32, len(frames))
            st = np.fromiter((float(f.get("BallStatus", "nan")) for f in frames), np.float32, len(frames))
            t = pd.to_datetime([f.get("T")[:-6] for f in frames])  # strip +00:00 -> UTC naive
            ball[half] = pd.DataFrame({"frame": n, "period": half, "t_utc": t, "ball_x": x, "ball_y": y,
                                       "ball_z": z, "ball_status": st, "ball_possession": poss})
        else:
            if fs.get("PersonId") not in meta["person"]:  # referees are not in the squad lists
                fs.clear()
                continue
            side, shirt, _ = meta["person"][fs.get("PersonId")]
            players.setdefault(half, {})[f"{side}{shirt}"] = pd.DataFrame(
                {"frame": n, f"{side}{shirt}_x": x, f"{side}{shirt}_y": y}).set_index("frame")
        fs.clear()
    parts = []
    for half in (1, 2):
        b = ball[half].set_index("frame")
        pl = pd.concat(players[half].values(), axis=1) if half in players else None
        parts.append(b.join(pl, how="left") if pl is not None else b)
    df = pd.concat(parts).reset_index()
    pcols = sorted(c for c in df.columns if c[0] in "ha" and c[1:2].isdigit())
    return df[["frame", "period", "t_utc", "ball_x", "ball_y", "ball_z", "ball_status",
               "ball_possession"] + pcols]


def add_labels(df):
    lab = np.where(df["ball_status"] == 0, "D",
                   np.where(df["ball_possession"] == 1, "H",
                            np.where(df["ball_possession"] == 2, "A", "?")))
    df["label"] = lab
    return df


def load_match(mid, use_cache=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{mid}_frames.parquet"
    if use_cache and cp.exists():
        return pd.read_parquet(cp)
    df = add_labels(_parse_tracking(mid))
    df.to_parquet(cp)
    return df


def load_events(mid):
    """Flat events table: event_id, t_utc, kind (XML child tag), team (home/guest side 'H'/'A'),
    player, shirt, x, y, clearance (OtherBallAction DefensiveClearance), blocker (BlockedShot), plus tackling/play attributes useful for touch derivation.
    Events with <Delete/> are marked deleted=True. Frame alignment: nearest tracking frame by t_utc."""
    meta = load_meta(mid)
    side_of_team = {meta["home_id"]: "H", meta["guest_id"]: "A"}
    rows = []
    for _, ev in etree.iterparse(str(_path("event", mid)), tag="Event"):
        a = ev.attrib
        t = pd.Timestamp(a["EventTime"]).tz_convert("UTC").tz_localize(None)
        kids = list(ev)
        kind = kids[0].tag if kids else None
        rec = {"event_id": a["EventId"], "t_utc": t, "x": _f(a.get("X-Position")), "y": _f(a.get("Y-Position")),
               "calc_frame": _i(a.get("CalculatedFrame")), "kind": kind, "deleted": kind == "Delete"}
        node = kids[0] if kids else None
        if node is not None:
            sub = [c.tag for c in node.iter() if c is not node]
            rec["sub"] = ",".join(sub)
            rec["team"] = side_of_team.get(node.get("Team") or node.get("WinnerTeam"))
            rec["player"] = node.get("Player") or node.get("Winner")
            rec["loser_team"] = side_of_team.get(node.get("LoserTeam"))
            rec["loser"] = node.get("Loser")
            rec["possession_change"] = node.get("PossessionChange")
            rec["tackle_type"] = node.get("Type")
            rec["winner_result"] = node.get("WinnerResult")
            play = node if node.tag == "Play" else node.find(".//Play")
            if play is not None:
                rec["play_team"] = side_of_team.get(play.get("Team"))
                rec["play_player"] = play.get("Player")
                rec["recipient"] = play.get("Recipient")
                rec["evaluation"] = play.get("Evaluation")
                rec["height"] = play.get("Height")
            rec["clearance"] = node.get("DefensiveClearance") == "true"
            if node.tag == "ShotAtGoal":
                blk = node.find("BlockedShot")
                if blk is not None:  # the blocker belongs to the team opposite to the shooter
                    rec["blocker"] = blk.get("Player")
                    rec["blocker_team"] = "A" if rec["team"] == "H" else "H"
                rec["shot_type"] = node.get("TypeOfShot")
                rec["shot_ext"] = node.get("ExtendedTypeOfShot")
                rec["shot_taker_control"] = node.get("TakerBallControl")
        rows.append(rec)
        ev.clear()
    e = pd.DataFrame(rows)
    for c in ("player", "play_player", "recipient", "loser", "blocker"):
        if c in e:
            e[f"{c}_shirt"] = e[c].map(lambda p: meta["person"].get(p, (None, None))[1])
    return e.sort_values("t_utc", kind="stable").reset_index(drop=True)


def align_frames(ev, df, shift_s=0.0):
    """Adds `frame` to ev: nearest tracking frame to t_utc + shift_s (only inside the frame range)."""
    t = df["t_utc"].values
    q = (ev["t_utc"] + pd.to_timedelta(shift_s, unit="s")).values
    idx = np.searchsorted(t, q).clip(1, len(t) - 1)
    idx = np.where(np.abs(q - t[idx - 1]) < np.abs(q - t[idx]), idx - 1, idx)
    ev = ev.copy()
    ev["row"] = idx
    ev["frame"] = df["frame"].values[idx]
    return ev


def _f(v):
    return float(v) if v is not None else np.nan


def _i(v):
    return int(v) if v is not None else None


if __name__ == "__main__":
    import sys
    for mid in sys.argv[1:] or MATCHES:
        d = load_match(mid)
        print(mid, d.shape, d["label"].value_counts(normalize=True).round(3).to_dict())
