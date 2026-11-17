"""Where the data, the caches and the outputs live. Every path of the repository goes through here.

Environment variables (a variable that is unset or empty falls back to its default, which is inside the repository and gitignored):

  ONSIDE_CHALLENGE_DIR  clone of the official challenge repository (data/, evaluation/, starter/, games.py)   default ./challenge
  ONSIDE_PFF_DIR        PFF World Cup 2022 files (Metadata/, Rosters/, Event Data/, Tracking Data/)            default ./data/pff
  ONSIDE_OUT            outputs and caches; the layout below this directory is fixed by the code               default ./src/out
  ONSIDE_BANKS_DIR      DFL bank (dfl/) and its calibration caches (cache/)                                    default ./bancos/data
  ONSIDE_MODELS_DIR     frozen model assets                                                                     default ./models

`add_challenge_to_path()` puts the challenge modules (games, baseline, possession_metrics, ...) on sys.path.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir(var, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(var) or os.path.join(ROOT, *default.split("/"))))


CHALLENGE_DIR = _dir("ONSIDE_CHALLENGE_DIR", "challenge")
PFF_DIR = _dir("ONSIDE_PFF_DIR", "data/pff")
OUT_DIR = _dir("ONSIDE_OUT", "src/out")
BANKS_DIR = _dir("ONSIDE_BANKS_DIR", "bancos/data")
MODELS_DIR = _dir("ONSIDE_MODELS_DIR", "models")

EVALUATION_DIR = os.path.join(CHALLENGE_DIR, "evaluation")
STARTER_DIR = os.path.join(CHALLENGE_DIR, "starter")


def add_challenge_to_path():
    """Idempotent: starter/ (baseline), evaluation/ (possession_metrics) and the challenge root (games.py) at the front of sys.path, in that order."""
    for p in (CHALLENGE_DIR, EVALUATION_DIR, STARTER_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
