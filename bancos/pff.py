"""PFF FC Enhanced 2022 World Cup dataset as an external last-touch bank.

`label` / `label_ut` is the LAST-TOUCH team derived from PFF's game events: the team of the player who has control
of the ball (`game_event.home_team`), held until the next touch, with D while the ball is dead. It is event-derived
(annotated from video by PFF analysts), not physical ground truth. `label_poss` is the same timeline built from
PFF's possession-sequence team (`home_ball`) instead; it differs from `label_ut` on clearances, rebounds and
challenges, where the player touching the ball is not the team that keeps the sequence. The boolean `uncertain`
(N,) marks the stretches where the last-touch label is least reliable (see below); `label` is never changed because
of it.

Source: PFF FC (now Gradient Sports), "Enhanced 2022 World Cup dataset",
https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset (originally blog.fc.pff.com), accessed
2026-10-28. 64 matches. No licence file; used with credit and without redistributing the raw files.
Specs: <ONSIDE_PFF_DIR>/Tracking Data/PFF FC Tracking Data Specification v2.2.pdf, Event Data/... v2.5.pdf.

    from pff import MATCHES, PFF_VERSION, load
    d = load("3812")                   # same keys and conventions as dfl_degradado.load_degraded, plus extras
    d = load("3812", smoothed=True)    # PFF's smoothed series instead of the raw ones (see below)

`PFF_VERSION` identifies the content of what `load` returns (bumped when a key or a value changes); the SHA-256 of
this file is the exact fingerprint. 3.1 only ADDED keys over 3 (`restart_window`, `restart_windows`, and `restart` /
`restart_codes` in `report`): every key that 3 returned has identical values.

Data lives in <ONSIDE_PFF_DIR>/{Metadata,Tracking Data,Event Data} (read only; default ./data/pff, see src/onside_paths.py); the parsed tracking is cached in
<ONSIDE_PFF_DIR>/cache/<id>_tracking_v3.npz (native 29.97 fps, version tag in the file name: bump CACHE_VERSION when
the parsing changes). Parsing a match takes 26-40 s on one process (stdlib json, orjson is not installed in the venv), loading from
the cache 0.4-3 s. `python pff.py --build [id ...]` builds the caches with 8 processes (all 64 in 17 min, 2.6 GB).

Held-out matches (24 of 64, `RESERVADOS` of pff_split.py (next to this file), imported from there, never copied): the
command line never opens them. `--build` with no ids builds the 40 of `DESARROLLO`; an explicit held-out id fails unless
`--include-reserved` is given; `--selftest` and the default report (no ids) work on development matches only and raise
if a held-out one is requested (`_BLOCKED`). `load(mid)` itself serves any match: the caller decides what to load.

Conventions (same as load_degraded unless noted)
  ball_xy (N,2), players_slots {"H": (N,24,2), "A": (N,22,2)}: normalised 0-1 on the pitch of the match
      ((x + L/2) / L, (y + W/2) / W with L x W from the metadata, 105 x 68 in all 64), NaN when missing, float64.
      The PFF axes are NOT flipped: y grows from the bottom of the pitch to the top as seen by the main camera,
      the same sign as the DFL raw axes, which dfl_degradado normalises the same way (dfl_arrays: no flip). So the
      y of this bank has the same sign as DFL and the opposite of Metrica (top-left origin, y downwards). x is
      the raw one: left to right as seen by the main camera. The attack direction is NOT normalised: it is the raw
      one, as in DFL, and it flips every period. `report["attack_right_meta"]` gives, from the metadata
      (homeTeamStartLeft), whether each team attacks towards x = 1 in periods 1 and 2 (it agrees with the
      goalkeeper-based direction in the 128 periods of the 64 matches); it is known before kick off, so it is a
      legitimate feature. Extra time is not given: homeTeamStartLeftExtraTime contradicts the goalkeepers in 2 of
      the 3 matches with extra time in the tracking (in all three, period 3 has the direction of period 2 and
      period 4 that of period 1). `report["attack_right"]` is the whole period goalkeeper-based measurement (non
      causal, diagnostic only).
  H = home (metadata homeTeam), A = away.
  Slots: one column per (team, jerseyNum) of the whole match (starters and subs, stable identity, NaN while the
      player is not in the tracking), jerseys in ascending order, padded with NaN up to 24 (H) and 22 (A) so the
      shape is the one of DFL. `report["ids_H"/"ids_A"]` lists the jersey of each used column. Per frame there
      are 11 players per team in the source (12-13 for a few frames around substitutions, fewer after a red card).
      Players that the camera does not see are included with an ESTIMATED position (see `players_visible`).
  period (N,) 1-4 (3 and 4 are extra time: 3 matches, 10506 10508 10517; 10510 and 10511 are flagged as extra time
      in the metadata but have no tracking for it), frame (N,) the PFF frameNum of the native frame nearest in
      time (monotonic, not contiguous because of the resampling), label / label_ut / label_poss (N,) '<U1'
      "H"/"A"/"D", ball_status (N,) float32 (1 alive, 0 dead = label D, derived from the label, not from the
      source), pitch = (L, W) in m, fps = 25 (resampled).
  Extras: restart_window (N,) int8 and restart_windows (dict of arrays) (see "Restart windows"), n_present {"H","A"},
      uncertain (N,) bool, is_touch (N,) bool (frame nearest to the start of a touch
      event), period_time (N,) s since the first frame of the period (kick off), ball_visible (N,) int8 (1 VISIBLE,
      0 ESTIMATED, -1 no ball), ball_z (N,) float32 ball height in m (not in DFL), players_visible
      {"H": (N,24), "A": (N,22)} int8 (1 VISIBLE, 0 ESTIMATED, -1 absent), video_missing (N,) bool (the broadcast
      cut part of the event that fixes the frame; the positions of that stretch are estimated; NOT part of
      `uncertain`), report (dict: ids, decisions, % of `uncertain` per criterion, attack direction).
  smoothed=True: players from homePlayersSmoothed / awayPlayersSmoothed (Kalman filter) and ball from
      ballsSmoothed. WARNING: PFF builds that ball by placing it at the position of the player on the ball and
      interpolating linearly between events, i.e. it is made from the same events that build the label. Never
      use it to measure how well a method finds the last touch. The default is the raw series (homePlayers,
      awayPlayers, balls), which come from the broadcast tracking.

Resampling 29.97 -> 25 fps (per period). Native frames are contiguous inside a period (the only jumps in frameNum
are the breaks between periods) and 29.97 fps exactly (videoTimeMs and frameNum agree to < 1e-6 s), so the time of
a native frame is (frameNum - first frame of the period) / fps. Frames repeated in the source (frames with several
possession events are duplicated, 849 of 176 818 rows in match 10502) carry identical raw positions (checked: `report["decisions"]["dup_mismatch_raw"]` is 0 in the 64 matches) and
smoothed positions that differ by a few centimetres (`dup_mismatch_smooth`); the first one is kept. Output frame j is at j / 25 s and takes the
two native frames around it: both valid -> linear interpolation; only one valid -> that one if it is the nearest
(< half a native frame), otherwise NaN; so a hole is never bridged and grows by less than one native frame.
Labels, uncertain and the other per-frame flags take the value of the nearest native frame.

Last-touch rule (all from `game_event` of the tracking file, flags from the Event Data file by game event id)
  Game events are intervals [start_frame, end_frame] in frameNum. The tracking rows carry the event only until
  the next event overrides it, so the intervals of the fields (not the rows) are used.
  OTB with home_team 0/1, kickoffs (FIRSTKICKOFF, SECONDKICKOFF, THIRDKICKOFF, FOURTHKICKOFF)
      touch: the team of the player on the ball from start_frame; held through the flight of the ball until the
      next touch (ffill). The end of the interval does not change the label.
  OUT (ball out of play: touch line, goal line, goal H/A, whistle W) and END (end of period)
      stop: D from frame + 1 (the frame of the event is still live, as in Metrica) until the start of the next
      touch, which is the restart (throw in, free kick, goal kick, corner, penalty, drop ball, kick off).
  G (ball hits post or flag and stays in play), SUB, ON, OFF
      no effect.
  OTB with home_team null (a CHALLENGE with no clear player): not a touch for `label_ut`, the previous team stays
      (it still sets `label_poss` through home_ball). It is `uncertain`.
  Before the first touch of a period D (never happens, the tracking starts at the kick off).

`uncertain` (union of): (a) poss_vs_touch: the player's team (home_team) and the sequence team (home_ball)
differ, mostly clearances, rebounds and challenges: from the start of the event to the next touch, at most
UNC_SPAN after its end; (b) no_owner: OTB without player; same span; (c) non_event: events PFF marks as
`initialNonEvent` (disallowed after the fact); same span; (d) overlap: the next touch is of the other team and
starts before the previous event ends, from the start of the second to the end of the first; (e) contact: a
touch of the ball by a player of the other team than the one on the ball that the game events do not carry, read
from the possession events of the Event Data (deflectorPlayerId of a deflection, keeperTouchType of a shot,
CHALLENGE with challengeOutcomeType K, where the challenger touched the ball and the carrier kept it). The
player on the ball stays the last touch (label unchanged; the challenge winner never replaces it). Deflections
and keeper touches have no time of their own (they are attributes of the pass or shot), so the whole stretch
from the start of its game event to the next touch or stop is uncertain; a K challenge, a window of CONTACT_WIN_S on
either side of it; No contact in the data has a reliable time of its own, so none moves the label. (f) non_event_action: a possession event with possessionEvents.nonEvent (the action was
disallowed after the fact, 332 cases in 63 matches) from that action to the next touch or stop, not the previous
possession; (g) stop_inferred:
a restart (setpieceType T F G C P D K of the event data) with no OUT or END since the last touch: D opens at the
end of the last event before it and the stretch from that touch to the restart is flagged (1 case in the 64
matches, 3830).
There is NO "actor far from the ball" criterion as in Metrica: the tracking is broadcast based (49-75 % of the
player positions and 17-34 % of the ball frames are ESTIMATED, per match) and the player on the ball is within 3 m
of the raw ball at the start of 35-88 % of the touches (median 74 %), so it would flag a quarter of the touches or
more. `main()` reports that
distance as an independent check of the alignment of the events with the tracking: its median over a lag of
+-45 frames is lowest at lag 0 (event start frame = tracking frame, no offset).
Not detected: stoppages that PFF does not mark with an OUT event (57 SUB / ON / OFF events follow a touch without an
OUT in the 64 matches; no label change is made for them), and the ball absent from the tracking: 11 % of the frames
with no ball are live (the broadcast cuts away, 616 runs of 5 s or more, 1.8 % of the frames, the events keep
coming), so the ball absence is not a dead-ball signal here.

Restart windows (`restart_window`, `restart_windows`; independent of `label`, `label_poss` and `uncertain`, which they
never touch). Every restart in play after a stop is a window: a game event (OTB or kick off) with setpieceType T F G C
P D K of the Event Data (spec v2.5: T throw in, F free kick, G goal kick, C corner, P penalty, D drop ball, K kick off,
both the one that starts a period and the one after a goal). Codes, also in `report["restart_codes"]`:
    0 outside any window, 1 F free kick (foul), 2 G goal kick, 3 K kick off, 4 T throw in, 5 C corner, 6 P penalty,
    7 D drop ball
  restart_window (N,) int8: per output frame (25 fps), the code of the window the frame is in. A window runs from the
      output frame of the restart event (inclusive) up to the output frame of the next touch (exclusive), the next touch
      being the next OTB / kick off event of the period with a later start frame (the end of the period if none). It
      always holds at least the event frame. PFF dates the restart event 1 to 7 s before the ball rolls in a part of
      the cases (the event has duration 0), so the window covers the dead stretch and the flight of the ball.
  restart_windows: dict of equal length arrays, one row per restart in the order of the period / time (period, kind,
      event_frame, next_touch_frame, ball_move_frame, ball_move_ok); all frames are row indices of the arrays of
      this dict (25 fps, the same indices as `frame`, `label`...), not PFF frameNum:
      period int8, kind int8 (code above), event_frame int64, next_touch_frame int64, ball_move_frame int64 (-1 if it
      could not be fixed; it can be BEFORE event_frame, down to 1 s before), ball_move_ok bool.
  ball_move_frame: the frame where the ball starts to move according to the RAW tracking (`balls`, never the smoothed
      ball, which is built from the events), searched in the window [event - 1 s, next touch] of native frames. Only
      VISIBLE frames with a position count (ESTIMATED ones are interpolations and say nothing about the kick; a ball
      missing or ESTIMATED over the whole stretch gives ball_move_ok False, ball_move_frame -1). The criterion:
        rest: the first run of BALL_REF_N (5) consecutive VISIBLE samples spanning at most BALL_MAX_GAP_S (0.5 s) whose
            positions are all within BALL_REST_TOL_M (0.10 m) of their median R (the position of the quiet ball);
        departure: the first later sample from which BALL_SUSTAIN_N (4) consecutive VISIBLE samples (spanning at most
            1 s) are all farther than BALL_MOVE_M (0.5 m) from R;
        ball_move_frame: the last VISIBLE sample before the departure that is within BALL_REST_TOL_M of R, i.e. the
            last frame at which the ball was seen at rest; the departure sample must follow it by at most
            BALL_MAX_GAP_S, otherwise the onset is not located (ok False). It dates the onset with an error of at
            most 0.5 s in the worst case and about 1 native frame (0.04 s) when the ball is visible all along.
      The thresholds come from the jitter of the raw ball: the largest deviation from its median over 5 VISIBLE samples
      at the start of a window is < 0.01 m in 10 % of the restarts, < 0.02 m in 25 %, median 0.13 m (two modes: quiet
      below ~0.05 m, moving above 0.1 m), so a quiet ball is within 0.10 m; 0.5 m is far above that jitter and still
      reached within a few frames of a kick. Over the 40 development matches (4 034 restarts) the median delay
      barely moves with the thresholds (MOVE 0.3 / 0.5 / 1.0 m: -0.32 / -0.32 / -0.28 s; REST 0.05 / 0.10 / 0.20 m:
      -0.36 / -0.32 / -0.24 s; SUSTAIN 2 / 4 / 8: -0.32 / -0.32 / -0.28 s; the share fixed does: 17 / 15 / 10 %
      with MOVE, 11 / 15 / 19 % with REST). LIMITS: (1) around a restart the raw ball is mostly ESTIMATED (VISIBLE in
      4-8 % of the frames of a throw in up to the event, 30-38 % of a free kick, 10-15 % of a goal kick, 7-25 % of a
      corner; it recovers 1-2 s after the event), so the criterion fixes 15 % of the restarts (free kicks 33 %, throw
      ins 3 %). (2) Those that are fixed are the ones with the ball on the broadcast at the kick, i.e. mostly restarts
      taken quickly, where PFF is NOT early: the delay measured there (median -0.3 s, the ball starts to move ~0.3 s
      BEFORE the event time) is not representative of the early-dated restarts, which are the ones the ball does not
      let us measure. (3) Only one kick off of a period has a fixed ball move (3836, +0.56 s), so the offset of the
      measure itself could not be calibrated against restarts known to be on time.
      `report["restart"]` gives, per kind, the number of restarts, of ball_move_ok and the failure reasons.
"""
import bz2
import importlib.util
import json
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import onside_paths  # noqa: E402

RAW = Path(onside_paths.PFF_DIR)
CACHE = RAW / "cache"
CACHE_VERSION = "v3"            # v2: full-width text columns, contacts and non events; v3: start frame of the own game event
PFF_VERSION = "3.1-restart-window"      # content of load(): 3.1 = 3 + restart_window / restart_windows (keys only added)
FPS = 25
SLOTS = {"H": 24, "A": 22}
MATCHES = sorted((p.name.split(".")[0] for p in (RAW / "Metadata").glob("*.json")), key=int) \
    if (RAW / "Metadata").exists() else []

KICKOFFS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}
STOPS = {"OUT", "END"}
RESTART_SP = {"T", "F", "G", "C", "P", "D", "K"}   # setpieceType values that follow a stop (O = open play)
UNC_SPAN_S = 2.0                # seconds, cap of an `uncertain` span that runs to the next touch after an event ends
CONTACT_WIN_S = 1.0             # seconds either side of a challenge the carrier kept (K): `uncertain` window
RESTART_CODES = {"F": 1, "G": 2, "K": 3, "T": 4, "C": 5, "P": 6, "D": 7}    # setpieceType -> value of restart_window
BALL_BEFORE_S = 1.0             # the ball-move search starts this long before the restart event
BALL_REF_N = 5                  # consecutive VISIBLE samples that fix the quiet position of the ball
BALL_REST_TOL_M = 0.10          # m, spread of the quiet ball around its median (and "back at rest" tolerance)
BALL_MOVE_M = 0.5               # m, displacement from the quiet position that counts as moving
BALL_SUSTAIN_N = 4              # consecutive VISIBLE samples that must all be displaced
BALL_MAX_GAP_S = 0.5            # s, largest stretch between the last sample at rest and the first displaced one
_BLOCKED = frozenset()          # match ids that _tracking refuses to open (set by the command line, see _split)


# ---------------------------------------------------------------- development / held-out split
_SPLIT = None


def _split():
    """(DESARROLLO, RESERVADOS) of pff_split.py (next to this file), the single source of the split (never copied here).
    Raises RuntimeError if it cannot be imported: there is no fallback list."""
    global _SPLIT
    if _SPLIT is None:
        path = HERE / "pff_split.py"
        try:
            spec = importlib.util.spec_from_file_location("pff_split", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _SPLIT = (tuple(mod.DESARROLLO), tuple(mod.RESERVADOS))
        except Exception as e:
            raise RuntimeError(f"cannot import DESARROLLO / RESERVADOS from {path} ({e!r}): the held-out matches cannot "
                               f"be told apart, so nothing is built or tested") from e
        if set(_SPLIT[0]) & set(_SPLIT[1]) or not _SPLIT[0] or not _SPLIT[1]:
            raise RuntimeError(f"{path}: DESARROLLO and RESERVADOS must be non empty and disjoint")
    return _SPLIT


# ---------------------------------------------------------------- metadata
def _meta(mid):
    m = json.load(open(RAW / "Metadata" / f"{mid}.json"))
    m = m[0] if isinstance(m, list) else m
    day = m["date"][:10]
    pit = [p for p in m["stadium"]["pitches"]
           if (p["startDate"] is None or p["startDate"] <= day) and (p["endDate"] is None or p["endDate"] >= day)]
    if len(pit) != 1:
        raise ValueError(f"{mid}: {len(pit)} pitches valid on {day}")
    return {"fps": float(m["fps"]), "pitch": (float(pit[0]["length"]), float(pit[0]["width"])),
            "start_left": m["homeTeamStartLeft"], "start_left_et": m["homeTeamStartLeftExtraTime"],
            "date": day}


# ---------------------------------------------------------------- parsing (cached)
def _cache_path(mid):
    return CACHE / f"{mid}_tracking_{CACHE_VERSION}.npz"


def _sides(mid):
    """{playerId: "H"/"A"} from the roster and the home team of the metadata, {} if either file is missing."""
    try:
        m = json.load(open(RAW / "Metadata" / f"{mid}.json"))
        m = m[0] if isinstance(m, list) else m
        home = str(m["homeTeam"]["id"])
        return {int(p["player"]["id"]): "H" if str(p["team"]["id"]) == home else "A"
                for p in json.load(open(RAW / "Rosters" / f"{mid}.json"))}
    except FileNotFoundError:
        return {}


def _event_data(mid):
    """(flags, poss) from the Event Data file, ({}, []) if it is missing.
    flags {gameEventId: (setpieceType, outType, videoMissing, initialNonEvent)}, first row of each game event.
    poss [(time s of the video, kind, game event id)] one entry per possession event that matters for the last touch:
      "nonevent"      possessionEvents.nonEvent (the action was disallowed after the fact)
      "deflection"    deflectorPlayerId of the other team than the player on the ball (or of unknown team)
      "keeper"        keeperTouchType given and keeper of the other team than the shooter (or unknown)
      "challenge_k"   CHALLENGE with challengeOutcomeType K (the challenger touched the ball, the carrier kept it)
    None of the contacts has a time of its own: they are attributes of the pass / shot / challenge, so the time is
    the one of that possession event (the stretch they flag starts at the game event, see _parse)."""
    p = RAW / "Event Data" / f"{mid}.json"
    if not p.exists():
        return {}, []
    side = _sides(mid)
    flags, poss = {}, []
    for e in json.load(open(p)):
        g, pe = e["gameEvents"], e["possessionEvents"]
        flags.setdefault(int(e["gameEventId"]), (g["setpieceType"] or "", g["outType"] or "",
                                                int(bool(g["videoMissing"])), int(bool(g["initialNonEvent"]))))
        t = e["eventTime"] if e["eventTime"] is not None else e["startTime"]
        if t is None or e["possessionEventId"] is None:
            continue
        holder = None if g["homeTeam"] is None else ("H" if g["homeTeam"] else "A")
        rival = lambda pid: holder is None or side.get(int(pid)) in (None, "") or side[int(pid)] != holder   # noqa: E731
        kinds = []
        if pe["nonEvent"]:
            kinds.append("nonevent")
        if pe.get("deflectorPlayerId") is not None and rival(pe["deflectorPlayerId"]):
            kinds.append("deflection")
        if pe.get("keeperTouchType") and (pe.get("keeperPlayerId") is None or rival(pe["keeperPlayerId"])):
            kinds.append("keeper")
        if pe["possessionEventType"] == "CH" and pe.get("challengeOutcomeType") == "K":
            kinds.append("challenge_k")
        poss += [(float(t), k, int(e["gameEventId"])) for k in kinds]
    return flags, poss


def _dense(flat, u, ids, n):
    """Flat (row, jersey, x, y, vis) entries -> dense (n, len(ids), 2) float32 with NaN, vis (n, len(ids)) int8 with -1,
    and the number of entries whose x differs from the one in the cell (rows of a duplicated frame; the first row
    wins, so these are the later duplicates that disagree with it)."""
    jx = np.searchsorted(ids, flat["j"])
    xy = np.full((n, len(ids), 2), np.nan, np.float32)
    vis = np.full((n, len(ids)), -1, np.int8)
    r = slice(None, None, -1)                              # assign backwards so that the first duplicate stays
    xy[u[r], jx[r], 0], xy[u[r], jx[r], 1], vis[u[r], jx[r]] = flat["x"][r], flat["y"][r], flat["v"][r]
    bad = int(((xy[u, jx, 0] != flat["x"]) & ~(np.isnan(xy[u, jx, 0]) & np.isnan(flat["x"]))).sum())
    return xy, vis, bad


def _parse(mid):
    """Reads Tracking Data/<id>.jsonl.bz2 into the arrays stored in the cache (see _tracking)."""
    A = lambda c: array(c)                                  # noqa: E731
    nan = float("nan")
    F = lambda v: nan if v is None else v                   # noqa: E731
    fr, per, vt = A("q"), A("b"), A("d")
    ball = {k: A(c) for k, c in (("x", "f"), ("y", "f"), ("z", "f"), ("v", "b"), ("sx", "f"), ("sy", "f"), ("sv", "b"))}
    cols = (("row", "i"), ("j", "h"), ("x", "f"), ("y", "f"), ("v", "b"))
    pl = {(s, k): {c: A(t) for c, t in cols} for s in "HA" for k in ("raw", "smooth")}
    ev = {}                                                 # game_event_id -> dict, from its first row
    ev_pos = {}
    n_bad_jersey = 0
    with bz2.open(RAW / "Tracking Data" / f"{mid}.jsonl.bz2", "rt") as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            fr.append(d["frameNum"])
            per.append(d["period"])
            vt.append(d["videoTimeMs"])
            b = d["balls"]
            if b:
                ball["x"].append(F(b[0]["x"]))
                ball["y"].append(F(b[0]["y"]))
                ball["z"].append(F(b[0]["z"]))
                ball["v"].append(1 if b[0]["visibility"] == "VISIBLE" else 0)
            else:
                for k in "xyz":
                    ball[k].append(nan)
                ball["v"].append(-1)
            bs = d["ballsSmoothed"]
            if bs:
                ball["sx"].append(F(bs["x"]))
                ball["sy"].append(F(bs["y"]))
                ball["sv"].append(1 if bs["visibility"] == "VISIBLE" else 0)
            else:
                ball["sx"].append(nan)
                ball["sy"].append(nan)
                ball["sv"].append(-1)
            for s, key, kind in (("H", "homePlayers", "raw"), ("A", "awayPlayers", "raw"),
                                 ("H", "homePlayersSmoothed", "smooth"), ("A", "awayPlayersSmoothed", "smooth")):
                q = pl[(s, kind)]
                for p in d[key] or ():
                    try:
                        jersey = int(p["jerseyNum"])
                    except (TypeError, ValueError):
                        n_bad_jersey += 1
                        continue
                    q["row"].append(i)
                    q["j"].append(jersey)
                    q["x"].append(F(p["x"]))
                    q["y"].append(F(p["y"]))
                    q["v"].append(1 if p["visibility"] == "VISIBLE" else 0)
            g, gid = d["game_event"], d["game_event_id"]
            if g and gid is not None:
                gid = int(gid)
                if gid not in ev:
                    hb, ht = g.get("home_ball"), g.get("home_team")
                    try:
                        shirt = int(g.get("shirt_number"))
                    except (TypeError, ValueError):
                        shirt = -1
                    ev[gid] = dict(id=gid, type=g["game_event_type"], period=d["period"], frame0=d["frameNum"],
                                   ht=-1 if ht is None else int(ht), hb=-1 if hb is None else int(hb),
                                   sf=g.get("start_frame") if g.get("start_frame") is not None else -1,
                                   ef=g.get("end_frame") if g.get("end_frame") is not None else -1, shirt=shirt)
                    ev_pos[gid] = set()
                if d["possession_event"]:
                    ev_pos[gid].add(d["possession_event"]["possession_event_type"])
    fr, per, vt = (np.frombuffer(x, dtype=t) for x, t in ((fr, np.int64), (per, np.int8), (vt, np.float64)))
    uf, first = np.unique(fr, return_index=True)
    u_of_row = np.searchsorted(uf, fr)
    n = len(uf)
    out = {"frame": uf, "period": per[first], "vt": vt[first] / 1000.0}
    for k, dt in (("x", np.float32), ("y", np.float32), ("z", np.float32), ("v", np.int8),
                  ("sx", np.float32), ("sy", np.float32), ("sv", np.int8)):
        out["ball_" + k] = np.frombuffer(ball[k], dtype=dt)[first]
    dup_bad = {"raw": 0, "smooth": 0}
    ids = {}
    for s in "HA":
        flat = {}
        for kind in ("raw", "smooth"):
            q = pl[(s, kind)]
            flat[kind] = {"row": np.frombuffer(q["row"], np.int32), "j": np.frombuffer(q["j"], np.int16),
                          "x": np.frombuffer(q["x"], np.float32), "y": np.frombuffer(q["y"], np.float32),
                          "v": np.frombuffer(q["v"], np.int8)}
        ids[s] = np.unique(np.concatenate([flat["raw"]["j"], flat["smooth"]["j"]])).astype(np.int16)
        for kind in ("raw", "smooth"):
            f = flat[kind]
            xy, vis, bad = _dense(f, u_of_row[f["row"]], ids[s], n)
            dup_bad[kind] += bad
            out[f"{s}_{kind}"], out[f"{s}_{kind}_vis"] = xy, vis
        out["ids_" + s] = ids[s]
    # events: one row per game_event_id, flags from the Event Data file
    flags, poss = _event_data(mid)
    rows = sorted(ev.values(), key=lambda e: (e["frame0"], e["id"]))
    fl = [flags.get(e["id"], ("", "", -1, -1)) for e in rows]
    for k, dt in (("id", np.int64), ("period", np.int8), ("ht", np.int8), ("hb", np.int8), ("sf", np.int64),
                  ("ef", np.int64), ("shirt", np.int16), ("frame0", np.int64)):
        out["ev_" + k] = np.array([e[k] for e in rows], dt)
    # text columns take the width of their longest value (a fixed width cut SECONDKICKOFF and FOURTHKICKOFF in v1)
    out["ev_type"] = np.array([e["type"] for e in rows], dtype=str)
    out["ev_pos"] = np.array(["+".join(sorted(ev_pos[e["id"]])) for e in rows], dtype=str)
    out["ev_sp"] = np.array([f[0] for f in fl], dtype=str)
    out["ev_out"] = np.array([f[1] for f in fl], dtype=str)
    # possession-event contacts and non events: video time -> nearest native frame (within 0.5 s of a tracked frame)
    pt = np.array([x[0] for x in poss], np.float64)
    j = np.clip(np.searchsorted(out["vt"], pt), 1, max(n - 1, 1))
    j = np.where(np.abs(out["vt"][j - 1] - pt) < np.abs(out["vt"][j] - pt), j - 1, j) if n > 1 else np.zeros(len(pt), int)
    ok = np.abs(out["vt"][j] - pt) <= 0.5 if len(pt) else np.zeros(0, bool)
    pf = uf[j] if len(pt) else uf[:0]
    # a contact without time of its own flags from the start of its game event (the carrier may have lost the ball
    # at any point after it took it); a disallowed action and a K challenge keep their own possession event frame
    kind = np.array([x[1] for x in poss], dtype=str)
    start = {e["id"]: e["sf"] for e in rows}
    for i, x in enumerate(poss):
        if x[1] in ("deflection", "keeper") and start.get(x[2], -1) >= 0:
            pf[i] = min(pf[i], start[x[2]])
    own = np.array([start.get(x[2], -1) for x in poss], np.int64)       # start frame of the game event of each one
    out["pe_start"] = own[ok]
    out["pe_frame"] = pf[ok]
    out["pe_period"] = out["period"][np.searchsorted(uf, pf)][ok] if len(pt) else out["period"][:0]
    out["pe_kind"] = kind[ok]
    out["ev_vidmiss"] = np.array([f[2] for f in fl], np.int8)
    out["ev_nonevent"] = np.array([f[3] for f in fl], np.int8)
    out["stats"] = np.array([len(fr), len(fr) - n, dup_bad["raw"], dup_bad["smooth"], n_bad_jersey, int(bool(flags)),
                             len(poss) - int(ok.sum())], np.int64)
    return out


def _tracking(mid):
    if mid in _BLOCKED:
        raise PermissionError(f"{mid} is a held-out match (RESERVADOS): not opened by --build / --selftest / the report")
    p = _cache_path(mid)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    out = _parse(mid)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.rename(p)
    return out


# ---------------------------------------------------------------- labels
def _timeline(K, touches, stops):
    """(K,) '<U1': D until the first action; a touch (index, team) sets its team from its index on, a stop (index) sets
    D from its index on. Both lists are in native frame indices relative to the period; at the same index the stop
    goes first (the touch is the restart)."""
    acts = sorted([(s, 1, t) for s, t in touches] + [(f, 0, "D") for f in stops])
    lab = np.full(K, "D", dtype="<U1")
    cur, prev = "D", 0
    for eff, _, v in acts:
        eff = min(max(eff, 0), K)
        lab[prev:eff] = cur
        cur, prev = v, max(prev, eff)
    lab[prev:] = cur
    return lab


def _period_labels(K, f0, ev, fps, contacts=()):
    """Per native frame of one period: label_ut, label_poss (K,) '<U1', uncertain parts {name: (K,) bool},
    touch start indices (list), stats. `ev` is the list of event dicts of the period (keys type, ht, hb, sf, ef, sp,
    nonevent), `f0` the first frame of the period, `contacts` the (frame, kind, start frame of its own
    game event or -1) of its possession events that touch
    the ball of the player on the ball or disallow an action (see _event_data)."""
    span = int(round(UNC_SPAN_S * fps))
    st = dict(touches=0, touches_no_owner=0, poss_vs_touch=0, overlaps=0, stops=0, restarts=0, stops_inferred=0,
              events_outside=0, contacts_deflection=0, contacts_keeper=0, contacts_challenge_k=0, non_event_actions=0)
    ev = sorted(ev, key=lambda e: (e["sf"], e["id"]))
    rel = lambda f: int(f) - f0                                   # noqa: E731
    tch = [e for e in ev if (e["type"] == "OTB" or e["type"] in KICKOFFS) and e["sf"] >= 0]
    for e in tch:
        e["s"] = rel(e["sf"])
        e["e"] = rel(e["ef"]) if e["ef"] >= e["sf"] else e["s"]
    inside = [e for e in tch if 0 <= e["s"] < K]
    st["events_outside"] = len(tch) - len(inside)
    tch = inside
    ut = [e for e in tch if e["ht"] in (0, 1)]
    stops = [rel(e["sf"]) + 1 for e in ev if e["type"] in STOPS and e["sf"] >= 0 and rel(e["sf"]) + 1 < K]
    st["stops"] = len(stops)
    team = lambda v: "H" if v == 1 else "A"                       # noqa: E731
    parts = {k: np.zeros(K, bool) for k in ("poss_vs_touch", "no_owner", "non_event", "overlap", "stop_inferred",
                                            "contact", "non_event_action")}
    # --- restart reconciliation: every restart must follow a stop after the last touch (UT touches)
    ut_s = np.array([e["s"] for e in ut], dtype=np.int64)
    stops_arr = np.array(sorted(stops), dtype=np.int64)
    for e in tch:
        if e["sp"] in RESTART_SP and e["type"] == "OTB":
            st["restarts"] += 1
            prior = ut_s[ut_s < e["s"]]
            if not len(prior):
                continue
            lt = prior.max()
            if ((stops_arr >= lt) & (stops_arr <= e["s"])).any():
                continue
            last_end = max(x["e"] for x in tch if x["s"] < e["s"])
            stops.append(min(max(last_end, lt) + 1, e["s"]))
            parts["stop_inferred"][lt:e["s"]] = True
            st["stops_inferred"] += 1
    lab_ut = _timeline(K, [(e["s"], team(e["ht"])) for e in ut], stops)
    lab_poss = _timeline(K, [(e["s"], team(e["hb"])) for e in tch if e["hb"] in (0, 1)], stops)
    # --- uncertain parts
    order = sorted(tch, key=lambda e: (e["s"], e["id"]))
    nxt = {e["id"]: (order[k + 1]["s"] if k + 1 < len(order) else K) for k, e in enumerate(order)}
    for k, e in enumerate(order):
        hi = min(nxt[e["id"]], e["e"] + 1 + span, K)
        if e["ht"] not in (0, 1):
            parts["no_owner"][e["s"]:hi] = True
            st["touches_no_owner"] += 1
        elif e["hb"] in (0, 1) and e["ht"] != e["hb"]:
            parts["poss_vs_touch"][e["s"]:hi] = True
            st["poss_vs_touch"] += 1
        if e["nonevent"] == 1:
            parts["non_event"][e["s"]:hi] = True
        if k + 1 < len(order):
            b = order[k + 1]
            if b["ht"] != e["ht"] and e["ht"] in (0, 1) and b["ht"] in (0, 1) and b["s"] <= e["e"]:
                parts["overlap"][b["s"]:min(e["e"] + 1, K)] = True
                st["overlaps"] += 1
    # --- rival contacts and disallowed actions of the possession events. Contacts have no time of their own (they are
    # attributes of the pass / shot), so the label is not changed: the stretch from the possession event to the next
    # touch or stop is uncertain. A challenge that the carrier keeps (K) is uncertain in a short window.
    ut_arr = np.sort(ut_s)
    stop_arr = np.array(sorted(stops), dtype=np.int64)

    def horizon(a, own):                  # end of the stretch the event can affect: the next touch AFTER its own game
        a = max(a, own)                   # event (the converted time can fall 1-2 frames before the event starts)
        nt, ns = ut_arr[ut_arr > a], stop_arr[stop_arr > a]
        return int(min(nt[0] if len(nt) else K, ns[0] if len(ns) else K, K))
    win = int(round(CONTACT_WIN_S * fps))
    for f, kind, own in contacts:
        a, own = rel(f), (rel(own) if own >= 0 else -1)
        if not 0 <= a < K:
            continue
        if kind == "nonevent":
            parts["non_event_action"][a:horizon(a, own)] = True
            st["non_event_actions"] += 1
        elif kind == "challenge_k":
            parts["contact"][max(a - win, 0):min(a + win + 1, K)] = True
            st["contacts_challenge_k"] += 1
        else:
            parts["contact"][a:horizon(a, own)] = True
            st["contacts_" + kind] += 1
    st["touches"] = len(ut)
    return lab_ut, lab_poss, parts, [e["s"] for e in tch], st


# ---------------------------------------------------------------- resampling
def _resample_xy(X, k0, k1, w):
    """X (K, ..., 2) float32 -> (len(k0), ..., 2) float64: linear interpolation between native frames k0 and k1 (weight
    w of k1) when both are valid, else the valid one if it is the nearest, else NaN. Holes are never bridged."""
    a, b = X[k0].astype(np.float64), X[k1].astype(np.float64)
    wa = (1 - w).reshape((-1,) + (1,) * (a.ndim - 1))
    wb = w.reshape(wa.shape)
    va, vb = ~np.isnan(a[..., 0:1]), ~np.isnan(b[..., 0:1])
    near_a, near_b = wa >= 0.5, wb > 0.5
    out = np.where(va & vb, a * wa + b * wb, np.where(va & near_a, a, np.where(vb & near_b, b, np.nan)))
    return out


# ---------------------------------------------------------------- restart windows
def _ball_move(x, y, vis, fps_n):
    """(index, reason) of the frame where the ball starts to move in one window of native frames, from the RAW ball
    x, y (m) and its visibility (1 VISIBLE); index -1 and the reason when it cannot be fixed (criterion: module
    docstring, "ball_move_frame"). reason is "ok", "few_visible", "no_rest", "no_move" or "gap"."""
    gap = int(round(BALL_MAX_GAP_S * fps_n))
    vi = np.flatnonzero((vis == 1) & np.isfinite(x) & np.isfinite(y))
    n = len(vi)
    if n < BALL_REF_N + BALL_SUSTAIN_N:
        return -1, "few_visible"
    xy = np.stack([x[vi], y[vi]], -1).astype(np.float64)
    why = "no_rest"
    for k in range(n - BALL_REF_N - BALL_SUSTAIN_N + 1):
        ref = xy[k:k + BALL_REF_N]
        R = np.median(ref, 0)
        if np.linalg.norm(ref - R, axis=1).max() > BALL_REST_TOL_M or vi[k + BALL_REF_N - 1] - vi[k] > gap:
            continue                                        # not a quiet run: the next one decides
        k1 = k + BALL_REF_N
        moved = np.linalg.norm(xy[k1:] - R, axis=1) > BALL_MOVE_M
        why = "no_move"
        for j in range(len(moved) - BALL_SUSTAIN_N + 1):
            if moved[j:j + BALL_SUSTAIN_N].all() and vi[k1 + j + BALL_SUSTAIN_N - 1] - vi[k1 + j] <= 2 * gap:
                rest = np.flatnonzero(np.linalg.norm(xy[:k1 + j] - R, axis=1) <= BALL_REST_TOL_M)
                last = rest[-1]                             # the last frame at which the ball was seen at rest
                return (int(vi[last]), "ok") if vi[k1 + j] - vi[last] <= gap else (-1, "gap")
        return -1, why                                      # the first quiet run decides
    return -1, why


def _restarts(ev, K, N, f0, fps_n, bx, by, bv):
    """Restart windows of one period. `ev` the event dicts of the period, K native frames, N output frames, `f0` first
    native frame, bx / by / bv the dense native raw ball (m, m, visibility). Returns (restart_window (N,) int8, rows,
    reasons) with rows [(kind, event, next touch, ball move)] in output frames relative to the period (move -1 if
    not fixed) and reasons the list of the reason of each row."""
    out = lambda s: int(min(max(np.rint(s / fps_n * FPS), 0), N - 1))                       # noqa: E731
    tch = sorted((e["sf"] - f0, e["id"]) for e in ev
                 if (e["type"] == "OTB" or e["type"] in KICKOFFS) and e["sf"] >= 0 and 0 <= e["sf"] - f0 < K)
    starts = np.array([t[0] for t in tch], np.int64)
    rw = np.zeros(N, np.int8)
    rows, reasons = [], []
    back = int(round(BALL_BEFORE_S * fps_n))
    for e in sorted(ev, key=lambda e: (e["sf"], e["id"])):
        s = e["sf"] - f0
        if e["sp"] not in RESTART_CODES or not (e["type"] == "OTB" or e["type"] in KICKOFFS) or not 0 <= s < K:
            continue
        later = starts[starts > s]
        nt = int(later[0]) if len(later) else K - 1
        a, b = out(s), out(nt)
        rw[a:max(b, a + 1)] = RESTART_CODES[e["sp"]]
        lo = max(s - back, 0)
        r, why = _ball_move(bx[lo:nt + 1], by[lo:nt + 1], bv[lo:nt + 1], fps_n)
        rows.append((RESTART_CODES[e["sp"]], a, b, out(lo + r) if r >= 0 else -1))
        reasons.append(why)
    return rw, rows, reasons


# ---------------------------------------------------------------- public API
def _events_of(z, p):
    keys = ("id", "type", "ht", "hb", "sf", "ef", "shirt", "sp", "out", "vidmiss", "nonevent", "pos")
    idx = np.flatnonzero(z["ev_period"] == p)
    return [{k: z["ev_" + k][i].item() if hasattr(z["ev_" + k][i], "item") else z["ev_" + k][i] for k in keys}
            for i in idx]


def _attack_right(P, period):
    """{period: True if the team attacks towards x = 1}: its goalkeeper (the player whose mean x is farthest from the
    centre, among those on the pitch for at least half of the period) defends x = 0. Whole-period mean: diagnostic
    only (non causal)."""
    out = {}
    for p in np.unique(period):
        x = P[period == p][:, :, 0]
        ok = (~np.isnan(x)).mean(0) >= 0.5
        mx = np.full(x.shape[1], np.nan)
        mx[ok] = np.nanmean(x[:, ok], 0)
        out[int(p)] = bool(mx[np.nanargmax(np.abs(mx - 0.5))] < 0.5)
    return out


def _attack_meta(meta, periods):
    """{"H": {period: bool}, "A": ...} from the metadata: home plays left to right (towards x = 1) in period 1 if
    homeTeamStartLeft, and the sides swap in period 2. Only periods 1 and 2: homeTeamStartLeftExtraTime does not
    predict the goalkeeper-based direction of extra time (it fails in 2 of the 3 matches with extra time in the
    tracking, 10506 and 10517), so periods 3 and 4 are not given."""
    out = {"H": {}, "A": {}}
    for p in periods:
        if p in (1, 2):
            h = bool(meta["start_left"]) if p == 1 else not meta["start_left"]
            out["H"][int(p)], out["A"][int(p)] = h, not h
    return out


def load(mid, smoothed=False):
    """Match dict (see module docstring). smoothed=True takes PFF's smoothed players and ball."""
    if mid not in MATCHES:
        raise KeyError(mid)
    z, meta = _tracking(mid), _meta(mid)
    fps_n = meta["fps"]
    L, W = meta["pitch"]
    scale = np.array([L, W])
    kind = "smooth" if smoothed else "raw"
    uf, up = z["frame"], z["period"]
    ids = {s: z["ids_" + s].tolist() for s in "HA"}
    for s in "HA":
        if len(ids[s]) > SLOTS[s]:
            raise ValueError(f"{mid}: {len(ids[s])} jerseys for {SLOTS[s]} slots in team {s}")
    acc = {k: [] for k in ("frame", "period", "t", "ball", "bvis", "bz", "H", "A", "Hv", "Av", "ut", "poss", "touch",
                           "vid", "unc", "rw")}
    rtab = {k: [] for k in ("period", "kind", "event_frame", "next_touch_frame", "ball_move_frame")}
    rwhy, offset = [], 0
    st_tot, parts_tot = {}, {}
    ball_xy_n = np.stack([z["ball_sx" if smoothed else "ball_x"], z["ball_sy" if smoothed else "ball_y"]], -1)
    ball_vis_n = z["ball_sv" if smoothed else "ball_v"]
    for p in np.unique(up):
        sel = np.flatnonzero(up == p)
        f0, f1 = int(uf[sel[0]]), int(uf[sel[-1]])
        K = f1 - f0 + 1
        k_of = (uf[sel] - f0).astype(np.int64)
        # the clock of the video must agree with the frame number (fps of the metadata)
        if len(sel) > 1 and abs((z["vt"][sel[-1]] - z["vt"][sel[0]]) * fps_n / (f1 - f0) - 1) > 1e-4:
            raise ValueError(f"{mid} period {p}: videoTimeMs does not match frameNum at {fps_n} fps")

        def dense(X, fill):
            out = np.full((K,) + X.shape[1:], fill, X.dtype)
            out[k_of] = X[sel]
            return out
        N = int((K - 1) / fps_n * FPS) + 1
        pos = np.arange(N) / FPS * fps_n
        k0 = np.minimum(np.floor(pos).astype(np.int64), K - 1)
        k1 = np.minimum(k0 + 1, K - 1)
        w = pos - k0
        kn = np.minimum(np.rint(pos).astype(np.int64), K - 1)
        ev = _events_of(z, int(p))
        cm = z["pe_period"] == p
        contacts = [(int(f), str(k), int(o)) for f, k, o in zip(z["pe_frame"][cm], z["pe_kind"][cm], z["pe_start"][cm])]
        lut, lps, parts, tstart, st = _period_labels(K, f0, ev, fps_n, contacts)
        for k, v in st.items():
            st_tot[k] = st_tot.get(k, 0) + v
        unc = np.logical_or.reduce(list(parts.values()))
        for k, v in parts.items():
            parts_tot.setdefault(k, []).append(v[kn])
        vid = np.zeros(K, bool)
        for e in ev:
            if e["vidmiss"] == 1 and e["type"] == "OTB" and e["sf"] >= f0:
                vid[max(e["sf"] - f0, 0):max(min(e["ef"], f1) - f0 + 1, 0)] = True
        is_t = np.zeros(N, bool)                        # nearest output frame of each touch start, none is lost
        is_t[np.clip(np.rint(np.array(tstart) / fps_n * FPS).astype(np.int64), 0, N - 1)] = True
        acc["frame"].append((f0 + kn).astype(np.int64))
        acc["period"].append(np.full(N, p, np.int64))
        acc["t"].append(np.arange(N) / FPS)
        acc["ball"].append(_resample_xy(dense(ball_xy_n, np.nan), k0, k1, w))
        acc["bvis"].append(dense(ball_vis_n, -1)[kn])
        acc["bz"].append(dense(z["ball_z"], np.nan)[kn])
        for s in "HA":
            acc[s].append(_resample_xy(dense(z[f"{s}_{kind}"], np.nan), k0, k1, w))
            acc[s + "v"].append(dense(z[f"{s}_{kind}_vis"], -1)[kn])
        acc["ut"].append(lut[kn])
        acc["poss"].append(lps[kn])
        acc["touch"].append(is_t)
        acc["vid"].append(vid[kn])
        acc["unc"].append(unc[kn])
        # restart windows: the RAW ball also when smoothed=True (the smoothed one is built from the events)
        rw, rows, why = _restarts(ev, K, N, f0, fps_n, dense(z["ball_x"], np.nan), dense(z["ball_y"], np.nan),
                                  dense(z["ball_v"], -1))
        acc["rw"].append(rw)
        rwhy += why
        rtab["period"] += [int(p)] * len(rows)
        for code, a, b, mv in rows:                    # frames of the whole match: add the output frames before this period
            rtab["kind"].append(code)
            rtab["event_frame"].append(a + offset)
            rtab["next_touch_frame"].append(b + offset)
            rtab["ball_move_frame"].append(mv + offset if mv >= 0 else -1)
        offset += N
    cat = {k: np.concatenate(v) for k, v in acc.items() if v}
    ball = (cat["ball"] + scale / 2) / scale                # metres, centre origin -> 0-1

    def slots(s):
        n, c = cat[s].shape[:2]
        out = np.full((n, SLOTS[s], 2), np.nan)
        out[:, :c] = (cat[s] + scale / 2) / scale
        return out

    def vis(s):
        n, c = cat[s + "v"].shape
        out = np.full((n, SLOTS[s]), -1, np.int8)
        out[:, :c] = cat[s + "v"]
        return out
    H, A = slots("H"), slots("A")
    lab = cat["ut"]
    unc = cat["unc"]
    period = cat["period"]
    stats = z["stats"]
    uncertain_pct = {"total": round(100 * float(unc.mean()), 2)}
    for k, v in parts_tot.items():
        uncertain_pct[k] = round(100 * float(np.concatenate(v).mean()), 2)
    rwin = {k: np.array(v, dt) for (k, v), dt in zip(rtab.items(), (np.int8, np.int8, np.int64, np.int64, np.int64))}
    rwin["ball_move_ok"] = rwin["ball_move_frame"] >= 0
    rwhy = np.array(rwhy, dtype=str)
    restart = {}
    for name, code in RESTART_CODES.items():
        m = rwin["kind"] == code
        ok = m & rwin["ball_move_ok"]
        dl = (rwin["ball_move_frame"][ok] - rwin["event_frame"][ok]) / FPS          # s, ball move - event
        restart[name] = {"n": int(m.sum()), "ball_move_ok": int(ok.sum()),
                         "delay_median_s": round(float(np.median(dl)), 3) if len(dl) else None,
                         "delay_p90_s": round(float(np.percentile(dl, 90)), 3) if len(dl) else None,
                         "failed": {w: int((m & (rwhy == w)).sum()) for w in ("few_visible", "no_rest", "no_move", "gap")
                                    if (m & (rwhy == w)).any()}}
    restart["pct_frames_in_window"] = round(100 * float((cat["rw"] > 0).mean()), 2)
    return {
        "ball_xy": ball, "ball_present": ~np.isnan(ball[:, 0]),
        "players_slots": {"H": H, "A": A},
        "n_present": {"H": (~np.isnan(H[:, :, 0])).sum(1), "A": (~np.isnan(A[:, :, 0])).sum(1)},
        "frame": cat["frame"], "period": period, "ball_status": (lab != "D").astype(np.float32),
        "label": lab, "label_ut": lab.copy(), "label_poss": cat["poss"], "is_touch": cat["touch"],
        "uncertain": unc, "period_time": cat["t"], "ball_visible": cat["bvis"], "ball_z": cat["bz"].astype(np.float32),
        "players_visible": {"H": vis("H"), "A": vis("A")}, "video_missing": cat["vid"],
        "pitch": (L, W), "fps": FPS,
        "restart_window": cat["rw"], "restart_windows": rwin,
        "report": {"match": mid, "restart_codes": dict(RESTART_CODES), "restart": restart, "smoothed": smoothed, "ids_H": ids["H"], "ids_A": ids["A"],
                   "fps_native": fps_n, "attack_right": {"H": _attack_right(H, period), "A": _attack_right(A, period)},
                   "attack_right_meta": _attack_meta(meta, np.unique(period)),
                   "decisions": {**st_tot, "rows": int(stats[0]), "dup_rows": int(stats[1]),
                                 "dup_mismatch_raw": int(stats[2]), "dup_mismatch_smooth": int(stats[3]),
                                 "bad_jersey": int(stats[4]), "event_data": bool(stats[5]),
                                 "poss_events_outside": int(stats[6])},
                   "uncertain_pct": uncertain_pct},
    }


# ---------------------------------------------------------------- verification
def _ev(i, type_, ht, hb, sf, ef, sp="O", nonevent=0):
    return dict(id=i, type=type_, ht=ht, hb=hb, sf=sf, ef=ef, shirt=-1, sp=sp, out="", vidmiss=0, nonevent=nonevent,
                pos="")


def _actor_distance(z, lag=0):
    """Distance (m) between the player on the ball of every touch event and the raw ball: at the start frame (+ `lag`
    native frames) and the minimum over the first 0.5 s of the event. NaN when the ball or the player is not tracked
    there. Independent of the label construction (tracking against the event annotation)."""
    ev = [i for i in range(len(z["ev_id"])) if z["ev_type"][i] == "OTB" and z["ev_ht"][i] in (0, 1) and z["ev_shirt"][i] >= 0]
    uf = z["frame"]
    out = []
    for i in ev:
        s = "H" if z["ev_ht"][i] == 1 else "A"
        ids = z["ids_" + s].tolist()
        if int(z["ev_shirt"][i]) not in ids:
            out.append((np.nan, np.nan))
            continue
        col = ids.index(int(z["ev_shirt"][i]))
        k0 = int(np.searchsorted(uf, z["ev_sf"][i])) + lag
        k1 = int(np.searchsorted(uf, min(z["ev_ef"][i], z["ev_sf"][i] + 15), side="right"))
        k1 += lag
        if k0 < 0 or k1 > len(uf):
            out.append((np.nan, np.nan))
            continue
        sl = slice(k0, max(k1, k0 + 1))
        d = np.linalg.norm(z[f"{s}_raw"][sl, col] - np.stack([z["ball_x"][sl], z["ball_y"][sl]], -1), axis=1)
        out.append((d[0] if len(d) else np.nan, np.nanmin(d) if np.isfinite(d).any() else np.nan))
    return np.array(out)


def _actor_lag(z, lags=range(-45, 46, 5)):
    """(lag, median distance at the touch start) over native frame lags: the lag with the smallest distance is the
    time offset between the event annotation and the tracking (0 when they are aligned)."""
    return [(lag, float(np.nanmedian(_actor_distance(z, lag)[:, 0]))) for lag in lags]


def _teleports(z):
    """Per team: number of native steps where a slot moves more than 1.5 m in one frame (45 m/s), which would be an
    identity swap or a track restart, over the frames where the slot is valid in two consecutive frames."""
    out = {}
    for s in "HA":
        p = z[f"{s}_raw"].astype(np.float64)
        same_period = (np.diff(z["period"]) == 0)[:, None] & (np.diff(z["frame"]) == 1)[:, None]
        step = np.where(same_period, np.linalg.norm(np.diff(p, axis=0), axis=2), np.nan)
        out[s] = (int((step > 1.5).sum()), int(np.isfinite(step).sum()))
    return out


def _selftest():
    """Runs `_selftest_cases` with the held-out matches (RESERVADOS) blocked: any attempt to open one raises."""
    global _BLOCKED
    held = frozenset(_split()[1])
    old, _BLOCKED = _BLOCKED, held
    try:
        try:
            _tracking(sorted(held)[0])
        except PermissionError:
            pass
        else:
            raise AssertionError("the guard does not block a held-out match")
        _selftest_cases()
    finally:
        _BLOCKED = old


def _selftest_cases():
    """Synthetic checks of the label rules, the resampling and the restart windows, then consistency checks on three
    development matches (3813, 10506 with extra time, 3812) and reproductions on development matches. Raises
    AssertionError on failure."""
    # 1. label rules, native fps 30, one period of 1000 frames starting at frame 100
    ev = [_ev(1, "FIRSTKICKOFF", 1, 1, 100, 100),
          _ev(2, "OTB", 0, 1, 140, 150),                       # A clears while the sequence is H
          _ev(3, "OTB", 0, 0, 300, 310),
          _ev(4, "OUT", -1, 0, 400, 400),
          _ev(5, "OTB", 1, 1, 600, 610, sp="T"),               # throw in after the OUT
          _ev(6, "OTB", 1, 1, 700, 710),
          _ev(7, "OTB", 0, 0, 900, 905, sp="F"),               # free kick with no stop before: inferred
          _ev(8, "OTB", -1, 1, 950, 950),                      # challenge with no player
          _ev(9, "OTB", 1, 1, 960, 970, nonevent=1),
          _ev(10, "OTB", 0, 0, 965, 980),                      # starts before 9 ends: overlap of the other team
          _ev(11, "G", -1, -1, 990, 990)]
    ut, ps, parts, _, st = _period_labels(1000, 100, ev, 30.0)
    assert ut[0] == "H" and ut[39] == "H" and ut[40] == "A" and ps[40] == "H", "touch of A vs sequence of H"
    assert ut[200] == "A" and ut[300] == "A" and ut[301] == "D" and ut[499] == "D" and ut[500] == "H", "OUT -> D -> restart"
    assert ut[610] == "H" and ut[611] == "D" and ut[799] == "D" and ut[800] == "A", "stop inferred before a restart"
    assert st["stops_inferred"] == 1 and parts["stop_inferred"][600:800].all() and not parts["stop_inferred"][599], st
    assert ut[850] == "A" and ps[850] == "H", "no-owner challenge: ut keeps A, poss takes home_ball"
    assert parts["no_owner"][850] and parts["poss_vs_touch"][40] and parts["non_event"][860], parts
    assert ut[862] == "H" and ut[865] == "A" and parts["overlap"][865] and parts["overlap"][870] \
        and not parts["overlap"][871] and not parts["overlap"][862], "overlap of the other team flagged"
    # 1b. contacts and disallowed actions (frames relative to the period: 100 + index)
    ev2 = [_ev(1, "FIRSTKICKOFF", 1, 1, 100, 100), _ev(2, "OTB", 1, 1, 150, 160), _ev(3, "OTB", 0, 0, 400, 410),
           _ev(4, "OUT", -1, 0, 600, 600), _ev(5, "OTB", 1, 1, 800, 810, sp="T")]
    contacts = [(300, "deflection", -1), (350, "keeper", -1), (500, "challenge_k", -1), (560, "nonevent", -1),
                (650, "nonevent", -1)]
    ut, ps, parts, _, st = _period_labels(1000, 100, ev2, 30.0, contacts)
    assert ut[60] == "H" and ut[250] == "H" and ut[300] == "A", "contacts do not change the label"
    assert parts["contact"][200:300].all() and not parts["contact"][199] and not parts["contact"][300], \
        "a deflection / keeper touch flags from its possession event to the next touch"
    assert parts["contact"][400 - 30:400 + 31].all() and not parts["contact"][400 - 31] and not parts["contact"][400 + 31], \
        "a K challenge flags a short window"
    assert parts["non_event_action"][460:501].all() and not parts["non_event_action"][459] \
        and not parts["non_event_action"][502], "a disallowed action flags up to the stop"
    assert st["contacts_deflection"] == 1 and st["contacts_keeper"] == 1 and st["contacts_challenge_k"] == 1 \
        and st["non_event_actions"] == 2, st
    # 1c. a flag converted 1 frame before the start of its own event must not take that event as the next touch
    ut, ps, parts, _, st = _period_labels(1000, 100, ev2, 30.0, [(399, "deflection", 400), (399, "nonevent", 400)])
    assert ut[299] == "H" and ut[300] == "A" and parts["contact"][299:501].all() and not parts["contact"][298] \
        and not parts["contact"][501] and parts["non_event_action"][299:501].all(), "own event is not the next touch"
    # 2. resampling: ramp with a hole, native 29.97 -> 25
    K = 100
    X = np.stack([np.arange(K) * 0.1, np.zeros(K)], -1).astype(np.float32)
    X[50:60] = np.nan
    pos = np.array([10.4, 49.5, 49.6, 59.4, 59.5, 59.6, 70.25])
    k0 = np.floor(pos).astype(int)
    r = _resample_xy(X, k0, np.minimum(k0 + 1, K - 1), pos - k0)
    assert abs(r[0, 0] - 1.04) < 1e-5 and abs(r[6, 0] - 7.025) < 1e-5, r
    assert abs(r[1, 0] - 4.9) < 1e-5 and np.isnan(r[2, 0]) and np.isnan(r[3, 0]), "holes are not bridged"
    assert abs(r[5, 0] - 6.0) < 1e-5, "nearest valid frame after the hole"
    # 3. real matches
    for mid in ("3813", "10506", "3812"):
        d = load(mid)
        z = _tracking(mid)
        N = len(d["frame"])
        lab, per = d["label"], d["period"]
        assert set(np.unique(lab)) <= {"H", "A", "D"} and all(len(np.unique(lab[per == p])) == 3 for p in np.unique(per))
        for p in np.unique(per):
            m = per == p
            assert (np.diff(d["frame"][m]) > 0).all() and np.allclose(d["period_time"][m], np.arange(m.sum()) / FPS)
            span = (z["frame"][z["period"] == p][-1] - z["frame"][z["period"] == p][0]) / d["report"]["fps_native"]
            assert abs(m.sum() / FPS - span) < 1 / FPS + 1e-9, "duration of the period is kept"
        assert abs(d["ball_present"].mean() - np.mean(~np.isnan(z["ball_x"]))) < 0.01, "ball presence is kept"
        for s in "HA":
            P = d["players_slots"][s]
            assert P.shape == (N, SLOTS[s], 2) and np.nanmin(P) > -0.3 and np.nanmax(P) < 1.3
            assert 10 <= np.median(d["n_present"][s]) <= 12
        # every OUT / END: D three native frames after it; every touch with owner: its team at its own frame
        ok_out = n_out = ok_t = n_t = 0
        f2j = {int(f): j for j, f in enumerate(d["frame"])}
        for i in range(len(z["ev_id"])):
            sf = int(z["ev_sf"][i])
            if z["ev_type"][i] in STOPS and sf >= 0:
                j = int(np.searchsorted(d["frame"], sf + 4))
                if j < N and d["frame"][j] - sf <= 6 and d["period"][j] == z["ev_period"][i]:
                    n_out += 1
                    ok_out += lab[j] == "D"
            elif z["ev_type"][i] == "OTB" and z["ev_ht"][i] in (0, 1) and sf in f2j:
                n_t += 1
                ok_t += lab[f2j[sf]] == ("H" if z["ev_ht"][i] == 1 else "A")
        assert n_out > 50 and ok_out == n_out, (mid, ok_out, n_out)
        assert n_t > 500 and ok_t / n_t > 0.98, (mid, ok_t, n_t)
        meta = d["report"]["attack_right_meta"]
        for p, v in d["report"]["attack_right"]["H"].items():
            if p <= 2:
                assert meta["H"][p] == v, (mid, p, meta, d["report"]["attack_right"])
        assert set(np.unique(z["ev_type"])) <= {"OTB", "OUT", "END", "SUB", "ON", "OFF", "G"} | KICKOFFS, \
            "event types are not truncated"
        dd = load(mid, smoothed=True)
        assert dd["ball_xy"].shape == d["ball_xy"].shape and (dd["label"] == lab).all()
    # 4. reproductions of the review of 2026-10-28 on development matches (frames are PFF frameNum): the phenomena of
    # the reviewed cases (the review's own matches are held out), found in the 40 development matches
    d = load("3812")

    def seg(a, b, d=d):
        m = (d["frame"] >= a) & (d["frame"] < b)
        assert m.sum() > 0
        return d["label"][m], d["uncertain"][m]
    d13 = load("3813")
    lab, unc = seg(114026, 114069, d13)  # second half kick off of H (SECONDKICKOFF cut to 12 characters in v1)
    assert (lab == "H").all() and not unc.any(), "second half kick off is a touch of H"
    d6 = load("10506")
    for kick, nxt in (("THIRDKICKOFF", None), ("FOURTHKICKOFF", None)):
        z6 = _tracking("10506")
        i = int(np.flatnonzero(z6["ev_type"] == kick)[0])
        j = int(np.searchsorted(d6["frame"], z6["ev_sf"][i] + 3))
        assert d6["label"][j] == ("H" if z6["ev_ht"][i] == 1 else "A"), kick
    lab, unc = seg(93433, 93597)         # pass of A deflected by a player of H before the reception of A
    assert (lab == "A").all() and unc.all(), "rival deflection: label kept, whole stretch uncertain"
    lab, unc = seg(186913, 186943)       # shot of A, touch of the keeper of H before the next touch
    assert (lab == "A").all() and unc.all(), "keeper touch: label kept, whole stretch uncertain"
    lab, unc = seg(99250, 99257)         # challenge K against A, who keeps the ball
    assert (lab == "A").all() and unc.all(), "challenge K: label of the carrier, window uncertain"
    lab, unc = seg(27440, 27497)         # possession event with nonEvent=True up to the next touch
    assert (lab == "A").all() and unc.all(), "disallowed action: uncertain up to the next touch"
    lab, unc = seg(180456, 180538)       # nonEvent flag converted 1 frame before the start of its own OTB (second review)
    assert (lab == "H").all() and unc.all(), "flag 1 frame before its own event: the event is not the next touch"
    assert d["report"]["uncertain_pct"]["contact"] > 0 and d["report"]["decisions"]["contacts_deflection"] > 0
    # 5. restart windows
    _selftest_restarts()
    print("selftest ok")


def _selftest_restarts():
    """Synthetic checks of the ball-move criterion and of the windows, then invariants on three development matches."""
    fps = 29.97

    def ball(rest, move, hidden=0):             # rest frames at (10, 5) with 2 cm jitter, hidden estimated, then a kick
        x = np.concatenate([10 + 0.02 * np.sin(np.arange(rest)), np.full(hidden, 10.0), 10 + np.arange(1, move + 1) * 0.6])
        y = np.concatenate([5 + 0.02 * np.cos(np.arange(rest)), np.full(hidden, 5.0), np.full(move, 5.0)])
        v = np.concatenate([np.ones(rest), np.zeros(hidden), np.ones(move)]).astype(int)
        return x, y, v
    assert _ball_move(*ball(30, 20), fps) == (29, "ok"), "last frame at rest before a kick"
    assert _ball_move(*ball(30, 20, hidden=10), fps) == (29, "ok"), "a short hidden stretch keeps the onset located"
    assert _ball_move(*ball(30, 20, hidden=20), fps) == (-1, "gap"), "onset hidden for more than 0.5 s is not located"
    x, y, v = ball(30, 20)
    assert _ball_move(x, y, np.zeros_like(v), fps) == (-1, "few_visible"), "ESTIMATED over the window"
    assert _ball_move(np.where(v == 1, x, np.nan), y, v, fps)[0] == 29 and _ball_move(np.full(50, np.nan), y, v, fps)[0] == -1
    assert _ball_move(np.arange(50) * 0.5, np.zeros(50), np.ones(50, int), fps) == (-1, "no_rest"), "ball moving all along"
    assert _ball_move(np.full(50, 3.0), np.full(50, 4.0), np.ones(50, int), fps) == (-1, "no_move"), "ball that never moves"
    x2 = np.full(50, 10.0)
    x2[30:32] += 3.0                                # two displaced samples only: not sustained
    assert _ball_move(x2, np.full(50, 5.0), np.ones(50, int), fps)[1] == "no_move", "a spike is not a departure"
    # windows: native fps 30 (output index = native index * 25 / 30), one period of 300 frames from frame 100
    ev = [_ev(1, "FIRSTKICKOFF", 1, 1, 100, 100, sp="K"), _ev(2, "OTB", 0, 0, 160, 165),
          _ev(3, "OTB", 1, 1, 210, 210, sp="T"), _ev(4, "OTB", 0, 0, 280, 285),
          _ev(5, "OUT", -1, 0, 330, 330), _ev(6, "OTB", 1, 1, 380, 380, sp="C")]
    bx = np.full(300, 20.0)
    bx[130:] += 0.6 * np.arange(1, 171)
    bv = np.ones(300, int)
    rw, rows, why = _restarts(ev, 300, 250, 100, 30.0, bx, np.zeros(300), bv)
    assert rows == [(3, 0, 50, -1), (4, 92, 150, 108), (5, 233, 249, -1)], rows        # K, T (kick at native 130), C
    assert why == ["no_move", "ok", "no_rest"], why
    assert rw[0] == 3 and rw[49] == 3 and rw[50] == 0 and rw[92] == 4 and rw[149] == 4 and rw[150] == 0 \
        and rw[233] == 5 and rw[248] == 5 and rw[249] == 0 and (rw > 0).sum() == 50 + 58 + 16, "window [event, next touch)"
    # 6. invariants on development matches
    codes = set(RESTART_CODES.values())
    for mid in ("3812", "3816", "10506"):
        d = load(mid)
        t, rwv = d["restart_windows"], d["restart_window"]
        N = len(d["frame"])
        assert rwv.dtype == np.int8 and rwv.shape == (N,) and set(np.unique(rwv)) <= codes | {0}
        n = len(t["kind"])
        assert n > 20 and all(len(v) == n for v in t.values()) and set(t["kind"].tolist()) <= codes
        assert (t["event_frame"] >= 0).all() and (t["next_touch_frame"] < N + 1).all() and (t["next_touch_frame"] >= t["event_frame"]).all()
        assert (np.diff(t["event_frame"]) > 0).all(), "restarts in time order"
        assert (rwv[t["event_frame"]] == t["kind"]).all() and (rwv > 0).sum() == (t["next_touch_frame"] - t["event_frame"]).clip(1).sum()
        ok = t["ball_move_ok"]
        assert (ok == (t["ball_move_frame"] >= 0)).all()
        lo = t["event_frame"] - int(BALL_BEFORE_S * FPS) - 1
        assert ((t["ball_move_frame"][ok] >= lo[ok]) & (t["ball_move_frame"][ok] <= t["next_touch_frame"][ok])).all()
        assert (d["period"][t["event_frame"]] == t["period"]).all() and (d["period"][t["next_touch_frame"].clip(0, N - 1)] == t["period"]).all()
        assert sum(v["n"] for k, v in d["report"]["restart"].items() if k in RESTART_CODES) == n
        dd = load(mid, smoothed=True)
        assert (dd["restart_window"] == rwv).all() and (dd["restart_windows"]["ball_move_frame"] == t["ball_move_frame"]).all(), \
            "the restart windows come from the raw ball also with smoothed=True"


def main(ids=None):
    """One block per match (the command line passes the development matches by default): size, labels, uncertain, ball, players, and the checks that do not depend on the label
    construction (tracking against the event annotation)."""
    for mid in ids or MATCHES:
        d = load(mid)
        z = _tracking(mid)
        N, per, lab, r = len(d["frame"]), d["period"], d["label"], d["report"]
        dist = _actor_distance(z)
        lag = min(_actor_lag(z), key=lambda t: t[1])
        tp = _teleports(z)
        print(f"== {mid}: N={N} ({N / FPS / 60:.1f} min at {d['fps']} fps, native {r['fps_native']}), pitch {d['pitch']}, "
              f"periods {np.unique(per).tolist()}")
        print("   label: " + ", ".join(f"{k}={100 * (lab == k).mean():.1f} %" for k in "HAD")
              + " | label_poss: " + ", ".join(f"{k}={100 * (d['label_poss'] == k).mean():.1f} %" for k in "HAD")
              + f" | ut != poss {100 * (lab != d['label_poss']).mean():.2f} %")
        print(f"   uncertain %: " + ", ".join(f"{k}={v}" for k, v in r["uncertain_pct"].items())
              + f" | video_missing {100 * d['video_missing'].mean():.1f} %")
        print(f"   ball present {100 * d['ball_present'].mean():.1f} % (VISIBLE {100 * (d['ball_visible'] == 1).mean():.1f} %, "
              f"ESTIMATED {100 * (d['ball_visible'] == 0).mean():.1f} %); D with ball {100 * d['ball_present'][lab == 'D'].mean():.1f} %, "
              f"ball absent labelled D {100 * (lab == 'D')[~d['ball_present']].mean():.1f} %")
        print("   players present mean H/A: " + f"{d['n_present']['H'].mean():.2f} / {d['n_present']['A'].mean():.2f}"
              + f", columns used {len(r['ids_H'])}/{len(r['ids_A'])}, visible share H "
              + f"{100 * (d['players_visible']['H'] == 1).sum() / (d['players_visible']['H'] >= 0).sum():.0f} %")
        print("   decisions: " + ", ".join(f"{k}={v}" for k, v in r["decisions"].items()))
        a0, a1 = dist[:, 0], dist[:, 1]
        ok0, ok1 = ~np.isnan(a0), ~np.isnan(a1)
        print(f"   actor to ball at touch start: {ok0.sum()} of {len(dist)} measurable, median {np.median(a0[ok0]):.2f} m, "
              f"< 3 m {100 * (a0[ok0] < 3).mean():.1f} %; min over first 0.5 s: median {np.median(a1[ok1]):.2f} m, "
              f"< 3 m {100 * (a1[ok1] < 3).mean():.1f} %")
        print(f"   event-to-tracking lag with the smallest actor-ball distance: {lag[0]} native frames ({lag[1]:.2f} m)")
        print(f"   native steps > 1.5 m per frame (identity swaps): H {tp['H'][0]} of {tp['H'][1]}, A {tp['A'][0]} of {tp['A'][1]}")
        print("   attack towards x=1 (period: H, A) meta vs goalkeeper: " + ", ".join(
            f"P{p}: {r['attack_right_meta']['H'].get(p, '-')}/{r['attack_right']['H'][p]}" for p in r["attack_right"]["H"]))


def _build_one(mid):
    try:
        _tracking(mid)
        return mid, None
    except Exception as e:                                  # one bad match must not stop the batch
        return mid, repr(e)


def _cli_ids(argv):
    """Ids of the command line, with the held-out matches blocked. No ids: the 40 of DESARROLLO. A held-out id given
    explicitly raises unless `--include-reserved` is present (then it is the caller's decision)."""
    dev, held = _split()
    ids = [a for a in argv if not a.startswith("-")]
    bad = sorted(set(ids) & set(held))
    if bad and "--include-reserved" not in argv:
        raise SystemExit(f"refusing to open held-out matches {bad} (RESERVADOS of pff_split.py): "
                         f"pass --include-reserved to do it on purpose")
    return ids or list(dev)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    if "--build" in sys.argv:
        ids = _cli_ids(sys.argv[1:])
        with ProcessPoolExecutor(max_workers=8) as ex:
            for m, err in ex.map(_build_one, ids):
                print("built" if err is None else "FAILED", m, err or "", flush=True)
    elif "--selftest" in sys.argv:
        _selftest()
    else:
        ids = _cli_ids(sys.argv[1:])
        if "--include-reserved" not in sys.argv:
            _BLOCKED = frozenset(_split()[1])
        main(ids)
