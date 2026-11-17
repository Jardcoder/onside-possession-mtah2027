# Possession labelling from tracking (H / A / D per frame), no video

Code for the hackathon task of labelling, frame by frame, which team has the ball (`H` home, `A` away, `D` dead ball) from
the tracking of one match. The submitted model blends two frozen models and decodes the result:

* **GRU short8**: a bidirectional GRU (2 layers, 128 hidden units, windows of 2,000 frames), 5 seeds averaged with a
  home / away swap test-time augmentation. Trained on the PFF FC World Cup 2022 bank with ball holes injected at the rate
  and gap lengths of a calibration file, with weights that favour short possessions ("short8").
* **T1w_external**: a LightGBM classifier per frame on ball, player and rule-base features (touch detector B3 with the
  constants in `models/submission_a2p/b3.json`, plus absorption of micro-possessions).
* **Blend and decoder**: class probabilities blended at alpha 0.75 (GRU weight), averaged over two ball smoothings
  (Gaussian sigma 6 and 8 frames), decoded with a Viterbi pass per period with switch penalties lambda_HA = 8 between teams
  and lambda_D = 32 to and from dead ball.

The ball is the ball of the challenge tracking, cleaned by rules only (`src/limpieza.py`); holes stay holes. No video and no
detector are used.

The submission file is produced by **inference only**, from the weights committed in `models/submission_a2p/`. **No match
data, cache or per frame output is committed**: the licences forbid redistributing the challenge data or anything that
reconstructs it, and the training data (PFF) is not redistributed either.

The technical document is in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Requirements

* Python 3.11 (verified with 3.11.15).
* Dependencies pinned in `requirements.lock` (includes `torch==2.9.1`, `lightgbm==4.6.0`, `numpy==2.4.6`). The lock was
  frozen on macOS arm64 and the hash below was verified there from a fresh environment. The hash was also verified on Linux x86-64 (see the table below). Other platforms are not verified.
* The data bundle of the challenge and a clone of the official challenge repository (it provides `evaluation/`, `starter/`
  and `games.py`, which the code imports).
* Disk: the repository is about 29 MB (the committed weights). CPU only; no GPU is needed to regenerate the file.

## Installation

```
git clone <this repository> onside-entrega-sinvideo && cd onside-entrega-sinvideo
python3.11 -m venv .venv                 # or: uv venv --seed --python 3.11 .venv
.venv/bin/pip install -r requirements.lock
```

Or `./run.sh setup`, which does the same, clones the challenge repository if it is missing, runs the tests and, when the
challenge data is in place, runs the hash check below. On Linux it adds `--extra-index-url https://download.pytorch.org/whl/cpu`
so that torch comes as the CPU wheel (by hand: `pip install -r requirements.lock --extra-index-url
https://download.pytorch.org/whl/cpu`). That path is not verified.

`run.sh` uses `$ONSIDE_PYTHON` if set, else `./.venv/bin/python`; with neither it stops with exit code 1 and asks for `./run.sh setup`.

## Challenge data

1. Enrol at https://landing.mobileworldcapital.com/more-than-a-hack-2027 and accept the terms of data use. The match assets
   arrive by email.
2. Clone the official repository anywhere:
   `git clone https://github.com/metrica-sports/more-than-a-hack-2027 challenge`
3. Unpack the match assets into `challenge/data/`, keeping the names they arrive with, as its `DATA.md` says
   (`tactical/BAR-ATH_tactical_atd.txt`, `..._homography.csv`, `..._smart_tagging.xml`, and so on). The code reads
   `challenge/data/`, `challenge/evaluation/`, `challenge/starter/` and `challenge/games.py`. The tactical video is not read.
4. Point the repository to that folder (default is `./challenge`, already gitignored):

```
export ONSIDE_CHALLENGE_DIR=/path/to/challenge
```

## Regenerating the submission file

```
export ONSIDE_CHALLENGE_DIR=/path/to/challenge
./run.sh submission --output <team_name>.csv
shasum -a 256 <team_name>.csv
```

Replace `<team_name>` with the team name (the form wants `<team_name>.csv`); the name does not change the content.
`--ball tracking` is accepted and is the default (and the only ball source). The command verifies the frozen assets against
`models/submission_a2p/SHA256SUMS`, writes the CSV at the given path, runs the organisers' `validate_submission.py` on it and
prints the SHA-256. Without `--output` it writes `$ONSIDE_OUT/submission_a2p/submission_a2p.csv` (`./src/out/...` by default).

Expected SHA-256 of the content:

```
ddac274ad809b10d12b6f8d68d252c87561515797ff202178405c9c374bfbd1c
```

It is also in `scripts/expected_sha256.txt`. One command that regenerates the file into a temporary folder and compares the
hash (exit code 0 on a match, only the two hashes are printed):

```
./run.sh verify
```

To compare a file that already exists: `./run.sh verify path/to/file.csv`. To check it again by hand:

```
.venv/bin/python "$ONSIDE_CHALLENGE_DIR/evaluation/validate_submission.py" <team_name>.csv
```

### Environments where the hash is verified

| Platform | Python | torch | lightgbm | numpy | Result |
|---|---|---|---|---|---|
| macOS 27.2.1 arm64 (Apple M4 Pro) | 3.11.15 | 2.9.1 | 4.6.0 | 2.4.6 | hash matches, from a clean clone and a new venv built from `requirements.lock` |
| Linux x86-64 (AMD Ryzen 5 9600X, CPU) | 3.11.15 | 2.9.1+cpu | 4.6.0 | 2.4.6 | hash identical |

Bit-identical output is only claimed for the verified rows. The torch settings that make runs deterministic are in
`src/torch_reproducibility.py` (deterministic algorithms, fixed thread counts, TF32 off) and LightGBM runs with
`deterministic` and `force_row_wise` (`src/modelo.py`); a different platform or library build may still change the
floating point results.

### Verifying the weights

```
cd models/submission_a2p && shasum -a 256 -c SHA256SUMS
```

It lists the 5 GRU checkpoints (`short8/<seed>/weights.pt`), their protocol and blend selection (`short8/protocol.json`,
`short8/selection.json`), the T1w_external booster (`t1w_external_fold1.txt`) and the B3 constants (`b3.json`).

## Tests

```
./run.sh test
```

Without the challenge clone or the PFF files, the tests that need them are skipped, not failed. The tests that run on
synthetic data always run, including the check of the frozen assets against their checksums and of the hash comparison.
To include the full regeneration in pytest: `ONSIDE_VERIFY_SUBMISSION=1 ./run.sh test tests/test_verify_submission.py`.

## Layout

```
src/           method modules (rules, features, model, decoder) and the CSV writer used by the submission (predecir_reto_m1.py)
bancos/        development banks, evaluation harnesses and training code (PFF loader: pff.py, split: pff_split.py, GRU: posesion_rnn.py)
models/        frozen assets of the submission (submission_a2p/)
configs/       protocol file of the GRU training harness
research/      scripts and decisions that selected the constants (see below)
tests/         pytest suite (synthetic data; the tests that need the real data are skipped when it is missing)
pff_download/  SHA-256 manifest of the PFF files the code reads and the script that checks a copy of them against it
scripts/       submission_a2p.py (the entry point of the delivery), verify_submission.py, expected hash, setup.sh
docs/          technical report
run.sh         one entry point for every command (list below)
```

Module and script names are partly in Spanish. A short glossary:

| Name | Meaning |
|---|---|
| `bancos` | banks (the development datasets and the scripts around them) |
| `predecir_reto` | predict the challenge |
| `toques` | touches |
| `huecos` | holes (ball absent from the tracking) |
| `muerto` | dead ball |
| `reinicios` | restarts |
| `limpieza` | cleaning |
| `ronda` | evaluation round |
| `barrido` | sweep |
| `salida` | output |
| `entrenar`, `predecir`, `comparar`, `empaquetar` | train, predict, compare, package |

## External training data

None of it comes from the provided game.

* **PFF FC World Cup 2022** (64 matches: tracking, event data, rosters, metadata). Downloaded from the public link that
  PFF FC (now Gradient Sports) publishes on the page cited below. This repository does not redistribute any of it. Citation: PFF FC (now Gradient Sports),
  Enhanced 2022 World Cup dataset, https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset (accessed 2026-10-28).
  It is needed only to retrain or to run the PFF harnesses, not to regenerate the CSV. The code reads `Metadata/<id>.json`,
  `Rosters/<id>.json`, `Event Data/<id>.json` and `Tracking Data/<id>.jsonl.bz2` from `ONSIDE_PFF_DIR` (default `./data/pff`).
  `pff_download/manifest.csv` lists the path, size and SHA-256 of the 256 files the code opens, so a copy can be checked with
  `./run.sh pff-check`; it contains no data. 40 matches are used for development and 24 are held out and never opened by the
  pipeline (`bancos/pff_split.py`).
* **DFL / Sportec open dataset** (CC BY 4.0), only to calibrate how often a ball hole is dead ball by hole length
  (`./run.sh calibrar-largo`) and the player-cleaning constants (`research/limpieza_dfl/`): Bassek, Rein, Weber and Memmert,
  Scientific Data 2025, https://doi.org/10.6084/m9.figshare.28196177. Its XML files go in `bancos/data/dfl/` (or
  `$ONSIDE_BANKS_DIR/dfl`). Not redistributed here either.
* The tracking of the challenge game is used as input (features) and, for the ball regime aggregates, as statistics of the
  tracking only. Its labels are never used for training or tuning, and the training code installs a read guard that refuses
  to open any challenge file (`src/training_data_guard.py`).

## Retraining and where the constants come from

Not needed to regenerate the submission. What this repository contains of it:

| Component | Code | Notes |
|---|---|---|
| GRU short8 (model, features, training loop, inference) | `bancos/posesion_rnn.py`, `bancos/a30_protocol.py`, `bancos/enfoque_a30.py` (`./run.sh a30`) | the configuration of the frozen checkpoints is `models/submission_a2p/short8/protocol.json` (hidden 128, stride 2, `short_ref` / `short_max` 8, seeds 20261101, 7, 11, 23, 42, fit 28 / inner 4 / outer 8 PFF matches) |
| GRU short8, driver of the study | `research/possession_short8/` | `promoted5_short8.py run` trains the 5 seeds; it reads intermediate outputs of earlier stages of the study (a reference T1w booster, calibrations, a protocol seal under `src/out/`) that are not in this repository |
| Hole calibration of the training balls | `research/tracker_calibration/C00_release_calibration.json`, `bancos/inyeccion_huecos.py` | aggregates only (absence share 0.295 and gap-length histogram); the video and detector runs that produced them are not part of this repository |
| T1w features and training recipe (LightGBM per frame, per-frame weights, lambda) | `src/modelo.py`, `bancos/modelo_pff.py`, `bancos/modelo_transfer.py`, `bancos/barrido_lambda.py`, `bancos/ronda5_t1.py` | `modelo`, `transfer`, `barrido-lambda`, `ronda5` below, in that order, produce the T1w family of boosters on PFF |
| B3 constants and blend selection | `research/b3_pff/` (`select_b3.py`, `select_blend.py`, `decision.json`, `stage2_selection.json`) | B3 constants chosen on the 32 PFF training matches (432 combinations); alpha 0.75 and lambda (8, 32) chosen on the inner 4 |
| Ball smoothing (g6, g8) | `research/smoothing_pff/` | selection on a ball with detector-like noise; decisions in the JSON files |
| Player-cleaning constants | `research/limpieza_dfl/` | `calibrate_dfl.py` on the DFL bank; decisions in `decision_dfl.json` |

The scripts under `research/` (including the refit of `t1w_external_fold1.txt` in `research/t1w_external/` and the short8
grid in `research/possession_short8/promoted5_jumps_protocol.json`) need the PFF data and intermediate outputs of earlier
stages of the study. Those outputs derive from PFF data and labels and are not distributed, so the retraining does not run
from this repository alone. With those inputs in place, the booster refit was reproduced byte for byte on CPU, and the five
short8 GRU seeds only on the GPU where they were trained (AMD Radeon RX 7800 XT, ROCm 6.3). The delivered file does not
depend on retraining: it is regenerated from the frozen weights, verified by checksum. The `calibrar` and `validar` actions of
`./run.sh a30` need the tactical video of the PFF clips and are not supported here.

Commands of the T1w family on PFF, in dependency order (they need the PFF files, the DFL files for `calibrar-largo`, and several
hours of compute; the scripts assert up to 9 cores):

```
./run.sh pff-check                      # PFF files in place, verified against the manifest
./run.sh regimen                        # src/out/regimen_reto.csv: aggregates of the challenge ball regime (tracking only)
./run.sh calibrar-largo                 # bancos/data/cache/huecos_largo_atd.json (needs the DFL files)
./run.sh modelo                         # M1 on PFF, nested 5 x 8 (feature cache of about 16 GB)
./run.sh transfer                       # T0 / T1: M1 with ball holes calibrated to the challenge
./run.sh barrido-lambda                 # nested lambda sweep of T1 (seconds)
./run.sh ronda5                         # T1w: weighted training of the 5 boosters, writes them under ONSIDE_OUT
```

Retraining the T1w boosters on another operating system can produce different trees; the byte-for-byte refit above was done
on Linux x86.

## Environment variables

Unset or empty means the default. All defaults are inside the repository and gitignored. A `.env` file at the root is sourced
by `run.sh`.

| Variable | Meaning | Default |
|---|---|---|
| `ONSIDE_CHALLENGE_DIR` | clone of the official challenge repo (`data/`, `evaluation/`, `starter/`, `games.py`) | `./challenge` |
| `ONSIDE_PFF_DIR` | PFF files (`Metadata/`, `Rosters/`, `Event Data/`, `Tracking Data/`, `cache/`) | `./data/pff` |
| `ONSIDE_OUT` | outputs and caches; the layout below it is fixed by the code | `./src/out` |
| `ONSIDE_BANKS_DIR` | DFL bank (`dfl/`) and calibration caches (`cache/`) | `./bancos/data` |
| `ONSIDE_PYTHON` | interpreter used by `run.sh` | `./.venv/bin/python` |

All paths go through `src/onside_paths.py`.

## Subcommands of `run.sh`

| Subcommand | Script | What it does |
|---|---|---|
| `setup` | `scripts/setup.sh` | environment from `requirements.lock`, challenge repository, tests, hash check |
| `submission` | `scripts/submission_a2p.py` | the submitted CSV: `--output PATH` |
| `verify` | `scripts/verify_submission.py` | regenerate the CSV (or take one) and compare its SHA-256 with `scripts/expected_sha256.txt` |
| `test` | `pytest tests/` | test suite |
| `pff-check` | `pff_download/pff_manifest.py --check` | verify the PFF files |
| `pff-build` | `bancos/pff.py --build` | parse the development matches into the PFF cache |
| `regimen` | `bancos/regimen_reto.py` | ball regime of the challenge against PFF (tracking only, aggregates) |
| `calibrar-largo` | `bancos/dfl_degradado.py --calibrar-largo` | P(dead ball \| ball absent) by hole length |
| `verificar-huecos` | `bancos/verificar_huecos_largo.py` | hole lengths: challenge against the degraded bank |
| `calibrar-balon` | `bancos/calibrar_balon.py` | ball noise: challenge against bank and PFF smoothing candidates |
| `pff` | `bancos/evaluar_pff.py` | rule ladder B0 to B3 on the 40 PFF development matches |
| `toques` | `bancos/evaluar_toques.py` | touch rules on the DFL development banks |
| `equipo` | `bancos/evaluar_equipo.py` | team of each deflection on the DFL banks |
| `ruido-gt` | `bancos/ruido_gt_pff.py` | label noise floor (possession edges shifted) |
| `sobrantes` | `bancos/diagnostico_pff_sobrantes.py` | extra detected touches against event touches |
| `oraculo-pff` | `bancos/oraculo_pff.py` | oracle ceilings on PFF (diagnostic, reads the ground truth on purpose) |
| `ronda` | `bancos/ronda_pff.py` | rules rounds 1 to 4 on PFF, nested validation, entry rule |
| `aerial` | `bancos/gt_aerial.py` | `label_aerial` stress ground truth against `label` |
| `modelo` | `bancos/modelo_pff.py` | M1: LightGBM per frame plus Viterbi |
| `transfer` | `bancos/modelo_transfer.py` | T0 and T1 on PFF with injected ball holes |
| `barrido-lambda` | `bancos/barrido_lambda.py` | nested lambda sweep |
| `ronda5` | `bancos/ronda5_t1.py` | T1w / T1wm training and entry rule against T1 |
| `predecir-reto-m1` | `src/predecir_reto_m1.py` | T1w family CSV from boosters left under `ONSIDE_OUT` by `ronda5`: `--modelo T1\|T1w\|T1wm`, `--salida PATH`, `--solo-csv` (not the submitted file) |
| `a30` | `bancos/enfoque_a30.py` | GRU harness: `entrenar`, `predecir`, `comparar`, `empaquetar` |
| `dominio` | `bancos/dominio.py` | domain shift of the T1 features, challenge against PFF |
| `diag-t1` | `bancos/diagnostico_t1.py` | where T1 loses spellF1 (diagnostic, reads the ground truth on purpose) |

## License

MIT, see `LICENSE`. Authors are listed in `AUTHORS`.
