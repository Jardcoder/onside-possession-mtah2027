# Possession labelling from tracking (H / A / D per frame)

Code for the hackathon task of labelling, frame by frame, which team has the ball (`H` home, `A` away, `D` dead ball) from
the tracking of one match. The submitted model is **T1w with a border detector**:

* **T1w**: a LightGBM classifier per frame (5 boosters, averaged), trained on the PFF FC World Cup 2022 bank with ball holes
  injected so that it looks like the challenge ball, with per frame training weights, decoded with a Viterbi pass per period.
* **Border detector**: a small LightGBM model that scores how likely a possession edge is at each frame. It does not change
  the T1w probabilities; it only makes the Viterbi cheaper to change state near the detector peaks (`src/borde.py`).

The submission file is produced by **inference only**, from weights committed in `models/`. Nothing is trained to regenerate
it. **No match data, cache or per frame output is committed**: the licences forbid redistributing the challenge data or
anything that reconstructs it, and the training data (PFF) is not redistributed either.

The technical document is in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Requirements

* Python 3.11 (tested with 3.11.15).
* Dependencies pinned in `requirements.lock`. The lock was frozen on macOS arm64 and the submission hashes below were
  verified there from a fresh environment. Other platforms are not verified.
* The data bundle of the challenge and a clone of the official challenge repository (it provides `evaluation/`, `starter/`
  and `games.py`, which the code imports).
* Disk: the repository is about 62 MB (the committed weights).

## Installation

```
git clone <this repository> onside-entrega && cd onside-entrega
python3.11 -m venv .venv                 # or: uv venv --seed --python 3.11 .venv
.venv/bin/pip install -r requirements.lock
```

`run.sh` uses `$ONSIDE_PYTHON` if set, else `./.venv/bin/python`.

## Challenge data

1. Enrol at https://landing.mobileworldcapital.com/more-than-a-hack-2027 and accept the terms of data use. The match assets
   arrive by email.
2. Clone the official repository anywhere:
   `git clone https://github.com/metrica-sports/more-than-a-hack-2027 challenge`
3. Unpack the match assets into `challenge/data/`, keeping the names they arrive with, as its `DATA.md` says
   (`tactical/BAR-ATH_tactical_atd.txt`, `..._homography.csv`, `..._smart_tagging.xml`, and so on). The code reads
   `challenge/data/`, `challenge/evaluation/`, `challenge/starter/` and `challenge/games.py`.
4. Point the repository to that folder (default is `./challenge`, already gitignored):

```
export ONSIDE_CHALLENGE_DIR=/path/to/challenge
```

## Regenerating the submission file

The file that is submitted is the second one: T1w decoded with the border detector. Replace `<team_name>` with the team name
(the form wants `<team_name>.csv`); the name does not change the content.

```
export ONSIDE_CHALLENGE_DIR=/path/to/challenge
./run.sh predecir-reto-m2 --salida <team_name>.csv
shasum -a 256 <team_name>.csv
```

Expected SHA-256 of the content (it takes under a minute):

```
cbd69066cdc2416fce77126aeca38ca559fae9a861e346c77b17fa0d2c904303
```

The command writes the CSV at the given path (relative to the repository root), runs the organisers' `validate_submission.py`
on it and prints an aggregate report (no per frame values). Without `--salida` it writes `src/out/reto_m2_t1w_bordes.csv`.
To check it again by hand:

```
.venv/bin/python "$ONSIDE_CHALLENGE_DIR/evaluation/validate_submission.py" <team_name>.csv
```

### The T1w file without the detector (optional)

The first stage on its own, kept as a reference. It is not the submitted file.

```
./run.sh predecir-reto-m1 --salida reto_m1_t1w.csv --solo-csv       # 5ee2982e549aaa0d445760f0705bda05d6a4b3d3b992412cdf097c419ff7018b
```

Both expected hashes are also in `scripts/expected_sha256.txt` (`shasum -a 256 -c scripts/expected_sha256.txt` works when
the two files, named as in that list, are in the current folder). `--solo-csv` skips the sanity report, which needs the PFF
caches (without them m1 prints a note and skips it).

### Verifying the weights

```
cd models/t1w && shasum -a 256 -c SHA256SUMS
```

It lists the 5 boosters, `lambda_final_t1w.json` and `feature_names.json`. The detector weights are `models/detector_bordes.txt`
with its metadata in `models/detector_bordes.json` (which records the SHA-256 of the text file; `tests/test_borde.py` checks it).
The detector metadata also records the hashes of the T1w boosters the detector was trained with. Those were built on another
operating system and differ from `models/t1w`; `predecir-reto-m2` prints an informational line about it and the CSV is the
same. The reference for the boosters used at inference is `models/t1w/SHA256SUMS`.

## Tests

```
./run.sh test
```

Without the challenge clone or the PFF files, the tests that need them are skipped, not failed. The tests that run on
synthetic data always run.

## Layout

```
src/           method modules (rules, features, model, decoder) and the entry points predecir_reto_m1.py and predecir_reto_m2.py
bancos/        development banks, evaluation harnesses and training scripts (PFF loader: pff.py, split: pff_split.py)
models/        committed weights: T1w boosters in t1w/, border detector as detector_bordes.{txt,json}
tests/         pytest suite (synthetic data; the tests that need the real data are skipped when it is missing)
pff_download/  SHA-256 manifest of the PFF files the code reads and the script that checks a copy of them against it
scripts/       expected SHA-256 of the two CSVs
docs/          technical report
run.sh         one entry point for every command (list below)
```

Module and script names are partly in Spanish (the working language of the team). A short glossary:

| Name | Meaning |
|---|---|
| `bancos` | banks (the development datasets and the scripts around them) |
| `predecir_reto` | predict the challenge |
| `bordes`, `borde` | borders (possession edges) |
| `toques` | touches |
| `huecos` | holes (ball absent from the tracking) |
| `muerto` | dead ball |
| `reinicios` | restarts |
| `limpieza` | cleaning |
| `ronda` | evaluation round |
| `barrido` | sweep |
| `salida` | output |
| `modelos` | models |
| `forzar`, `sobrescribir` | force, overwrite |

## External training data

None of it comes from the provided game.

* **PFF FC World Cup 2022** (64 matches: tracking, event data, rosters, metadata). Downloaded from the public link that
  PFF FC (now Gradient Sports) publishes on the page cited below. This repository does not redistribute any of it. Citation: PFF FC (now Gradient Sports),
  Enhanced 2022 World Cup dataset, https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset (accessed 2026-10-28).
  The code reads `Metadata/<id>.json`, `Rosters/<id>.json`, `Event Data/<id>.json` and `Tracking Data/<id>.jsonl.bz2` from
  `ONSIDE_PFF_DIR` (default `./data/pff`). `pff_download/manifest.csv` lists the path, size and SHA-256 of the 256 files the
  code opens, so a copy can be checked with `./run.sh pff-check`; it contains no data. 40 matches are used for development
  and 24 are held out and never opened by the pipeline (`bancos/pff_split.py`).
* **DFL / Sportec open dataset** (7 matches, CC BY 4.0), only to calibrate how often a ball hole is dead ball by hole length
  (`./run.sh calibrar-largo`): Bassek, Rein, Weber and Memmert, Scientific Data 2025, https://doi.org/10.6084/m9.figshare.28196177.
  Its XML files go in `bancos/data/dfl/` (or `$ONSIDE_BANKS_DIR/dfl`). Not redistributed here either.
* The tracking of the challenge game is used as input (features) and, for the ball regime aggregates, as statistics of the
  tracking only. Its labels are never used for training or tuning.

## Reproducing the training (not needed to regenerate the submission)

The weights in `models/` are committed so that the submission is regenerated by inference. The commands that produced them are
below, in dependency order. They need the PFF files (and the DFL files for `calibrar-largo`) and several hours of compute
(times are wall clock on a 9 core machine, and the scripts assert up to 9 cores). Retraining on another operating system
produces different trees, so it will not reproduce the committed weights byte by byte.

```
./run.sh pff-check                      # PFF files in place, verified against the manifest
./run.sh regimen                        # src/out/regimen_reto.csv: aggregates of the challenge ball regime (tracking only)
./run.sh calibrar-largo                 # bancos/data/cache/huecos_largo_atd.json (needs the DFL files)
./run.sh modelo                         # M1 on PFF, nested 5 x 8 (about 85 min, feature cache of about 16 GB)
./run.sh transfer                       # T0 / T1: M1 with ball holes calibrated to the challenge (about 45 min)
./run.sh barrido-lambda                 # nested lambda sweep of T1 (seconds)
./run.sh ronda5                         # T1w: weighted training of the 5 boosters, writes the T1w_k0..4.txt boosters under ONSIDE_OUT (about 1 h)
./run.sh ronda5 --lam-ha 4 8 16 24 32 48 --lam-d 8 16 24 32 48 64 96 128 192   # wider lambda grid, writes lambda_final_t1w.json (evaluation only)
./run.sh bordes                         # border detector: nested evaluation against T1w, final weights to models/detector_bordes.{txt,json} (about 36 min; --sobrescribir to replace the committed ones)
```

`ronda5` writes the boosters to `ONSIDE_OUT/modelo_cache/ronda5/modelos/`; the committed copies in `models/t1w/` are those files
(the CSV commands read `models/t1w` by default and `ONSIDE_OUT` with `--cache`). `bordes` reads the caches that `transfer` and
`ronda5` leave under `ONSIDE_OUT` and does not use the held-out matches or the challenge data. It refuses to overwrite the
committed weights unless `--sobrescribir` is given.

The other subcommands of `run.sh` are evaluation harnesses and diagnostics on the development banks; the full list is in the
usage message of `./run.sh`.

## Environment variables

Unset or empty means the default. All defaults are inside the repository and gitignored. A `.env` file at the root is sourced
by `run.sh`.

| Variable | Meaning | Default |
|---|---|---|
| `ONSIDE_CHALLENGE_DIR` | clone of the official challenge repo (`data/`, `evaluation/`, `starter/`, `games.py`) | `./challenge` |
| `ONSIDE_PFF_DIR` | PFF files (`Metadata/`, `Rosters/`, `Event Data/`, `Tracking Data/`, `cache/`) | `./data/pff` |
| `ONSIDE_OUT` | outputs and caches; the layout below it is fixed by the code | `./src/out` |
| `ONSIDE_BANKS_DIR` | DFL bank (`dfl/`) and calibration caches (`cache/`) | `./bancos/data` |
| `ONSIDE_MODELS_DIR` | committed weights of the border detector (`detector_bordes.txt`, `.json`) | `./models` |
| `ONSIDE_T1W_DIR` | committed T1w boosters, `lambda_final_t1w.json` and `SHA256SUMS` | `./models/t1w` |
| `ONSIDE_PYTHON` | interpreter used by `run.sh` | `./.venv/bin/python` |

All paths go through `src/onside_paths.py`.

## Subcommands of `run.sh`

| Subcommand | Script | What it does |
|---|---|---|
| `test` | `pytest tests/` | test suite |
| `baseline` | `src/reproducir_baseline.py` | official smooth50 baseline: CSV, validation, score on the public window |
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
| `oraculo-pff` | `bancos/oraculo_pff.py` | oracle ceilings on PFF (diagnostic) |
| `ronda` | `bancos/ronda_pff.py` | rules rounds 1 to 4 on PFF, nested validation, entry rule |
| `aerial` | `bancos/gt_aerial.py` | `label_aerial` stress ground truth against `label` |
| `modelo` | `bancos/modelo_pff.py` | M1: LightGBM per frame plus Viterbi |
| `transfer` | `bancos/modelo_transfer.py` | T0 and T1 on PFF with injected ball holes |
| `barrido-lambda` | `bancos/barrido_lambda.py` | nested lambda sweep |
| `ronda5` | `bancos/ronda5_t1.py` | T1w training and entry rule against T1 |
| `dominio` | `bancos/dominio.py` | domain shift of the T1 features, challenge against PFF |
| `diag-t1` | `bancos/diagnostico_t1.py` | where T1 loses spellF1 (diagnostic on PFF) |
| `predecir-reto-m1` | `src/predecir_reto_m1.py` | challenge CSV: `--modelo T1w\|T1\|T1wm` (default T1w), `--salida PATH`, `--solo-csv`, `--cache` |
| `bordes` | `bancos/ronda_bordes.py` | border detector on top of T1w: nested evaluation with the entry rule, final weights in `models/` |
| `predecir-reto-m2` | `src/predecir_reto_m2.py` | challenge CSV with T1w plus the border detector (submitted): `--salida PATH`, `--lam HA D`, `--modelos DIR`, `--cache` |
| `ablacion-presencia` | `bancos/ablacion_presencia.py` | border detector without / with the ball-presence columns (PFF only) |
| `all` | | `test` and `baseline` |

## License

MIT, see `LICENSE`. Authors are listed in `AUTHORS`.
