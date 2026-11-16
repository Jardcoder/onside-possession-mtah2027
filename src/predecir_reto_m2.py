"""CSV of the challenge with T1w plus the border detector.

    .venv/bin/python src/predecir_reto_m2.py [--salida RUTA] [--lam HA D] [--modelos DIR] [--forzar] [--cache]

Same input as `predecir_reto_m1.py --modelo T1w`: the features of the challenge (bancos/features_reto.py), the mean of the 5 T1w boosters and the lambda
of models/t1w/lambda_final_t1w.json (committed with the boosters; `--cache` reads src/out instead). The difference is the decoding: the price of changing state is multiplied by `f` within `k` frames of a peak of the
border detector (models/detector_bordes.txt, trained by `./run.sh bordes`) and by `g` elsewhere (`borde.decode`); with f = g = 1 the path would be the one
of `predecir_reto_m1.py`. `predecir_reto_m1.py` and its hash in scripts/expected_sha256.txt are untouched.

The detector reads the T1w feature matrix of the challenge, so its layout has to be the one it was trained on: `borde.column_index` fails loudly otherwise.

Output: the CSV (default src/out/reto_m2_t1w_bordes.csv; `frame,label`, same format and checks as the T1w CSV, `validate_submission.py` is run on it) and a report with
aggregates only (no per-frame values): how many graded frames differ from T1w, possessions per minute and share of D with and without the detector, and its peaks per
minute on the challenge next to the PFF reference stored with the weights. The public ground-truth window is not read: it is a sanity check of the T1w report, not an input here.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bancos"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import borde  # noqa: E402
import features_reto as fr  # noqa: E402
import modelo as mo  # noqa: E402
import predecir_reto_m1 as m1  # noqa: E402

FPS = 25
OUT = os.path.join(onside_paths.OUT_DIR, "reto_m2_t1w_bordes.csv")
REPORT = os.path.join(onside_paths.OUT_DIR, "predecir_reto_m2_informe.txt")


def border_probability(booster, meta, X, names):
    """(N,) probability of a possession edge at each row of the feature matrix X (names = its column names)."""
    cols = borde.column_index(names, meta["feature_names"])
    return booster.predict(np.asarray(X[:, cols], np.float32), num_threads=1)


def changed_fraction(a, b, mask):
    """Share of the frames of `mask` where the label arrays a and b differ."""
    a, b = np.asarray(a), np.asarray(b)
    return float((a[mask] != b[mask]).mean())


def provenance_warnings(meta, lam_given, cache=False):
    """Lines warning that the lambda file or the T1w boosters used are not the ones the detector was evaluated with."""
    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    files, lam_file = m1.booster_files("T1w", cache), m1.lambda_path("T1w", cache)
    out = []
    try:
        if not cache:    # committed boosters: the reference is models/t1w/SHA256SUMS (the detector JSON records the hashes of the machine that trained it)
            sums = dict(line.split()[::-1] for line in open(os.path.join(onside_paths.T1W_DIR, "SHA256SUMS")).read().splitlines() if line.strip())
            ok = [sha(f) == sums.get(os.path.basename(f)) for f in files] + [lam_given or sha(lam_file) == sums.get(os.path.basename(lam_file))]
            if not all(ok):
                out.append("WARNING: the T1w boosters or lambda in models/t1w do not match models/t1w/SHA256SUMS")
            return out
        if [sha(f) for f in files] != meta.get("t1w_boosters_sha256"):
            out.append("WARNING: the T1w boosters are not the ones the detector was evaluated with (SHA-256 differ)")
        if not lam_given and sha(lam_file) != meta.get("lambda_file_sha256"):
            out.append("WARNING: lambda_final_t1w.json is not the one the detector was evaluated with (SHA-256 differs)")
    except OSError as e:
        out.append(f"WARNING: could not check the provenance of the T1w files ({e.filename})")
    return out


def provenance_info(meta, cache=False):
    """Informational lines (not warnings): the committed boosters differ from the ones recorded when the detector was trained."""
    if cache:
        return []
    try:
        files = m1.booster_files("T1w")
        if [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files] != meta.get("t1w_boosters_sha256"):
            return ["INFO: the committed T1w boosters differ from those recorded in detector_bordes.json (the detector was trained on boosters built on another operating system); the final CSV is the same"]
    except OSError:
        pass
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=OUT, help="CSV path (default src/out/reto_m2_t1w_bordes.csv)")
    ap.add_argument("--lam", type=float, nargs=2, metavar=("HA", "D"), help="penalties of the Viterbi; default: models/t1w/lambda_final_t1w.json")
    ap.add_argument("--cache", action="store_true", help="read the T1w boosters and lambda from the cache under ONSIDE_OUT instead of models/t1w")
    ap.add_argument("--modelos", default=borde.MODELS_DIR, help="folder with detector_bordes.txt and detector_bordes.json (default ONSIDE_MODELS_DIR, ./models)")
    ap.add_argument("--forzar", action="store_true", help="use the detector even if its metadata say it did not pass the entry rule")
    args = ap.parse_args()
    t0 = time.time()
    booster, meta = borde.load_detector(args.modelos)
    if not meta.get("entry_rule_passes") and not args.forzar:
        sys.exit("the detector in this folder did not pass the entry rule (ronda_pff.verdict) when it was trained: pass --forzar to use it anyway")
    lam = tuple(args.lam) if args.lam else m1.lambda_final(m1.lambda_path("T1w", args.cache))
    print("model: T1w + border detector; lambda:", lam, "; setting:", meta["config"], flush=True)
    X, names, per, info = fr.build_reto()
    print(f"challenge features {X.shape} in {time.time() - t0:.0f} s", flush=True)
    print("feature columns checked against:", m1.check_feature_names(names, "T1w") or "nothing", flush=True)
    P = fr.ensemble_proba(m1.boosters_of("T1w", args.cache), X, names)
    p_border = border_probability(booster, meta, X, names)
    letters = np.array(mo.STATES)[borde.decode(P, per, lam, p_border, meta["config"])]
    base_letters = np.array(mo.STATES)[fr.decode(P, per, lam)]
    labels = fr.to_video(letters, info)
    base = fr.to_video(base_letters, info)
    m1.check_labels(labels)
    m1.check_labels(base)
    m1.write_csv(args.salida, labels)
    r = subprocess.run([sys.executable, os.path.join(onside_paths.EVALUATION_DIR, "validate_submission.py"), args.salida], capture_output=True, text=True)
    g = m1.graded_mask(len(labels))
    d_b, n_b, ppm_b, _ = m1.counters(np.where(g, base, "X"), g)
    d_m, n_m, ppm_m, minutes = m1.counters(np.where(g, labels, "X"), g)
    thr = meta["config"]["threshold"]
    peaks_min = len(borde.peaks(p_border, thr, borde.PEAK_WINDOW)) / (len(p_border) / FPS / 60)
    sha = hashlib.sha256(open(args.salida, "rb").read()).hexdigest()
    out = [f"CSV: {args.salida}", f"SHA-256 {sha}", f"model T1w + border detector; lambda {lam}; setting {meta['config']}; rows {len(labels)}",
           "validate_submission: " + (r.stdout.strip() or r.stderr.strip())[-400:],
           f"\nAggregates over the {int(g.sum())} graded frames ({minutes:.1f} min):",
           f"  graded frames that differ from T1w: {100 * changed_fraction(base, labels, g):.2f} %",
           f"  possessions: T1w {n_b} ({ppm_b:.2f} / min), with borders {n_m} ({ppm_m:.2f} / min)",
           f"  share of D: T1w {100 * d_b:.1f} %, with borders {100 * d_m:.1f} %",
           f"  detector peaks per minute at threshold {thr}: challenge {peaks_min:.2f}, PFF development matches {meta['reference_peaks_per_min']:.2f}",
           *provenance_warnings(meta, bool(args.lam), args.cache),
           *provenance_info(meta, args.cache),
           f"\nTime: {time.time() - t0:.0f} s."]
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)     # a fresh clone has no src/out
    with open(REPORT, "w") as f:
        f.write(text + "\n")
    if r.returncode != 0:
        sys.exit(f"validate_submission.py rejected {args.salida}")


if __name__ == "__main__":
    main()
