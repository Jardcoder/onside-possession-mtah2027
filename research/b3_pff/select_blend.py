"""Blend selection of the submission on the inner 4 (CPU): alpha and lambda for short8 GRU + T1w_external refit with
the PFF B3 constants (selection part of the study's stage-2 script; the result is stage2_selection.json).

    python select_blend.py      # -> stage2/selection.json

pj.select on the inner 4 training-style balls: standalone lambda of the 5-seed GRU mean, then for alpha in
{0, .25, .5, .75, 1} the lambda pair maximising the worse of the two views; the best (alpha, lambda) is kept.
Inputs: short8 inner probabilities (5-seed mean) and the new T1w inner probabilities written by stage2_t1w.py.
Nothing from BAR-ATH is read.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT / "promoted5_jumps_20261111"))
import promoted5_jumps as pj  # noqa: E402

SHORT8 = OUT / "promoted5_short8_20261111"


def main() -> None:
    combo = os.environ["B3_COMBO"]
    protocol = json.loads((SHORT8 / "protocol.json").read_text())
    entries, nn, t1 = [], [], []
    for mid in protocol["inner"]:
        for view in ("g6h", "g8h"):
            z = pj.npz(SHORT8 / "inner" / f"{mid}__{view}.npz")
            t = pj.npz(HERE / f"t1w_{combo}" / "inner" / f"{mid}__{view}.npz")
            if not np.array_equal(z["gt"], t["gt"]):
                raise ValueError(f"inner ground truth differs: {mid} {view}")
            entries.append({"mid": mid, "view": view, "gt": z["gt"], "per": z["period"]})
            nn.append(z["probabilities"])
            t1.append(t["probabilities"])
    selection = pj.select(nn, entries, t1)
    (HERE / "stage2").mkdir(exist_ok=True)
    (HERE / "stage2/selection.json").write_text(json.dumps({"combo": combo, **selection}, indent=1))
    print(json.dumps(selection["blend"]))


if __name__ == "__main__":
    main()
