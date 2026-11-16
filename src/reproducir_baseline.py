"""Reproduce el baseline oficial (smooth50), lo valida y lo puntua contra el GT publico.
Uso: challenge/.venv/bin/python src/reproducir_baseline.py [--window 50]
Solo imprime agregados."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import numpy as np  # noqa: E402
import baseline as bl  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = onside_paths.OUT_DIR
GT = os.path.join(onside_paths.EVALUATION_DIR, "gt", "BAR-ATH_possession_gt_public.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=50)
args = ap.parse_args()

codes = bl.build()
labels = bl.smooth(codes, args.window)
path = bl.write_submission(os.path.join(OUT, "baseline_smooth{}.csv".format(args.window)), labels)

print("--- validate_submission.py ---")
r = subprocess.run([sys.executable, os.path.join(onside_paths.EVALUATION_DIR, "validate_submission.py"), path],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()[-300:])

gt = pm.label_codes(pm.read_label_csv(GT))
res, reason = None, None
table = pm.read_label_csv(path)
reason = pm.validate_submission(table, len(gt))
if reason:
    sys.exit("invalid: " + reason)
res = pm.evaluate(gt, pm.label_codes(table))
print("--- score vs GT publico (ventana 5 min) ---")
for k in ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells"):
    print("{}: {:.4f}".format(k, res[k]) if isinstance(res[k], float) else "{}: {}".format(k, res[k]))
print("referencia oficial full-match (privado) smooth50: spellF1 0.2433; publico segun EVALUATION.md: 0.3256")
