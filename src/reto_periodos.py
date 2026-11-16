"""Periods of the challenge match (BAR-ATH) built with the semantics of the PFF bank, so that features and the CSV agree on what a period is.

In PFF (bancos/pff.py) the arrays of a match hold only the two playing periods: period 1 starts at the kick-off of the first half and
ends at its last frame, period 2 starts at its kick-off and ends at its last frame; there is no pre-match, half-time or post-match. The features of
src/modelo.py take the clock from that layout (`t_in_period_s` = seconds since the first frame of the period, `t_to_period_end_s` = seconds to its
last frame, windows and "time since / until" never cross a period). The challenge video does not look like that, so the challenge arrays are cut
to the same layout before the features are built.

Frames of the video (0-based, the index of src/atd.py arrays), from challenge/DATA.md:256-263 and challenge/FAQ.md:35-41:
    period 1   11,185 .. 90,651   (graded from the kick-off itself)
    period 2  113,742 .. 190,197  (kick-off 113,742; graded from 110,808, so the 2,934 frames 110,808 .. 113,741 are graded, all D, and outside the period)
    everything else (before the match, half-time, after the final whistle): 40,462 frames, not graded.

    per = periods()                       # (N,) int8: 1, 2 or 0 (outside the playing periods)
    frames = play_frames()                # video index of every row of the cut arrays
    cut = per[frames]                     # period of each row, what the method side receives
    full = to_full(pred_cut, frames, n)   # back to the video index ("X" outside the playing periods)
"""

import numpy as np

import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
from games import GAME, GAMES, half_video_blocks  # noqa: E402

KICKOFF_2ND = 113_742                           # FAQ.md:38-39; the first half has no gap: its kick-off is the first graded frame


def blocks(game=GAME):
    """[(period, first video frame, last video frame)] of the playing periods: the graded halves with the second starting at its kick-off."""
    (h1, a1, b1), (h2, a2, b2) = half_video_blocks(game)
    return [(1, a1, b1), (2, KICKOFF_2ND, b2)]


def n_frames(game=GAME):
    return GAMES[game]["n_video_frames"]


def periods(n=None, game=GAME):
    """(N,) int8: 1 inside period 1, 2 inside period 2, 0 elsewhere."""
    per = np.zeros(n_frames(game) if n is None else n, np.int8)
    for p, a, b in blocks(game):
        per[a:b + 1] = p
    return per


def play_frames(n=None, game=GAME):
    """Video frames (sorted) of the two playing periods: the rows kept when an array of the video is cut to the PFF layout."""
    return np.flatnonzero(periods(n, game) > 0)


def pre_kickoff_2nd(game=GAME):
    """(first, last) video frames that are graded but before the second-half kick-off (all D in the ground truth, FAQ.md:38-39), outside both periods."""
    (_, _, _), (_, a2, _) = half_video_blocks(game)
    return a2, KICKOFF_2ND - 1


def cut(arr, frames):
    """Rows `frames` of an array indexed by video frame."""
    return arr[frames]


def to_full(values, frames, n, fill="X"):
    """Put the per-row `values` of the cut arrays back at their video frames; `fill` everywhere else."""
    values = np.asarray(values)
    out = np.full(n, fill, dtype=values.dtype)
    out[frames] = values
    return out
