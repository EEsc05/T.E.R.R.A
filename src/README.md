# Stage 1 models: train from scratch

All model definitions, training, preparation and output code live in `src`.
The implementation follows the model families and structure of the original
stage 1 scripts. Nothing in `src` automatically loads their pretrained checkpoints,
configs, selection results or data. `hackathon_stage1` is not a runtime dependency.

## Query and prepare the NISAR data

The notebook-based pipeline now runs with `python -m src prepare-data`. It queries
ASF, downloads with locally configured Earthdata authentication, processes GUNW
pairs, and exports model inputs under `data/nisar`. See [NISAR_DATA.md](NISAR_DATA.md)
for commands, outputs, processing assumptions and authentication setup.

## Training data

Use **NumPy `.npz`**, with a separate file for training, validation and optionally
testing. These are supervised forecasting examples: historical maps paired with
an actually observed future map. CSV is not the training input format.

| Key | Shape | Meaning |
| --- | --- | --- |
| `maps` | `(N,T,H,W)` | Historical signed LOS displacement maps |
| `dates` | `(N,T)` or shared `(T,)` | Chronological dates of historical acquisitions |
| `targets` | `(N,H,W)` | Observed future maps in the same units/reference/grid |
| `target_dates` | `(N,)` | Actual acquisition date of each future map |
| `valid_masks` | `(N,T,H,W)` | Optional boolean historical validity masks |
| `target_masks` | `(N,H,W)` | Optional boolean target validity masks |
| `volcano_ids` | `(N,)` | Optional site IDs; omitted means a single site |
| `metadata_json` | scalar or `(N,)` | JSON string(s): shared or per-example metadata |

`N` is the number of examples. `T` must be at least six for all models except
persistence, which accepts one. Models use the final six historical acquisitions.
Targets must be later than their history. The requested horizon is calculated
from each last historical date to its observed target date. Do not put target
maps into `maps`.

Dates can be ISO strings, NumPy datetime arrays, or numeric days since 1970-01-01.
Histories and targets must already share the same LOS sign, grid and displacement
reference. No registration, reference alignment or resampling is performed here.
Missing masks default to finite pixels; zero displacement remains valid. Optional
`--nodata` excludes a sentinel in source units. `--units mm|cm|m` converts both
histories and targets to millimetres, overriding metadata units.

Four-branch and CVAE require 32×32 maps and horizons of 1–180 days. Declare
`native_center32` for native patches or `aoi_block_mean32` for the notebook's
block-averaged AOI grid. The model does not resample either grid. Each
example needs metadata with actual spacing and geometry:

```python
metadata = {
    "units": "mm",
    "grid_convention": "native_center32",
    "spacing_km": [actual_row_spacing_km, actual_column_spacing_km],
    "geometry_available": False,  # explicitly unknown; use True when known
}
# For known geometry also include los_vector=[east, north, up].
```

Use your real values. `--center-crop` can extract the central native 32×32 pixels
from larger registered maps: starting indices `height//2-16, width//2-16`. It crops
histories, masks, targets and target masks identically. Supply geometry appropriate
to that patch. It does not resize or infer spatial metadata.

Create an archive from arrays you have prepared:

```python
import json
import numpy as np

np.savez_compressed(
    "data/train.npz",
    maps=train_histories,  # (N,6,32,32)
    dates=train_history_dates,  # (N,6)
    targets=train_future_maps,  # (N,32,32)
    target_dates=train_future_dates,  # (N,)
    valid_masks=train_history_masks,  # optional bool
    target_masks=train_future_masks,  # optional bool
    volcano_ids=train_site_ids,  # (N,) string array
    metadata_json=json.dumps(metadata),
)
# Write validation.npz using the same keys and held-out observations.
```

Use distinct files. For each site present in both partitions, all training target
dates must precede the **first historical date** of every validation example.
This purged temporal split avoids overlap across training and validation windows.
An optional final test partition must similarly follow training and validation.
Use actual site IDs when combining volcanoes; disjoint sites are allowed as a
geographic holdout.

## Train from the terminal

```bash
conda activate terra
python -m src train --help

python -m src train \
  --model cvae \
  --input data/train.npz \
  --validation data/validation.npz \
  --output outputs/cvae.joblib \
  --report outputs/cvae_training.json \
  --epochs 80 --batch-size 64 --learning-rate 0.001 \
  --patience 15 --basis-rank 8 --seed 42 --device cpu

python -m src train \
  --model four_branch \
  --input data/train.npz \
  --validation data/validation.npz \
  --output outputs/four_branch.joblib \
  --report outputs/four_branch_training.json \
  --max-context-pixels 10000 --seed 42
```

Four-branch and CVAE require separate validation data in the CLI and at least two
training examples. That is a numerical minimum, not a sufficient dataset for
useful generalization. `--test data/test.npz` adds final held-out MAE, RMSE and bias
without fitting on test data. Existing output files require `--overwrite`.
No training has been run on your real data yet.

| Model / existing file | Training behavior |
| --- | --- |
| `persistence` / `models/persistence.py` | Last observed value; no global weights to train |
| `ema30` / `models/ema30.py` | Local EMA with 30-day half-life; no global weights |
| `smoothing` / `models/smoothing.py` | Local robust quality smoothing; no global weights |
| `robust` / `models/robust.py` | Robust local low-rank and coefficient regression fitted to each history |
| `four_branch` / `models/four_branch.py` | Training-only block spatial basis and scaled SVR residual head; validation fits horizon-bin fusion weights |
| `cvae` / `models/cvae.py` | Training-only block PCA/scalers/residual noise and freshly initialized conditional neural network |

All families expose `fit()`/`train()` and `predict()`. Baseline fitting validates
data and reports metrics; it does not pretend to learn global parameters.

The four-branch structure retains temporal EMA, spatial coefficient forecasting,
context residual regression and quality smoothing. Its fusion weights are fitted
on validation examples, so the report calls those **calibration metrics**, not an
independent test score. Missing horizon bins retain equal branch weights. Context
training samples at most `--max-context-pixels` valid training pixels for practical
SVR memory/runtime use.

The CVAE preserves the conditional prior/posterior/decoder structure, 75-coordinate
conditioning, EMA parent, block spatial representation and physical-map decoder.
Its network uses AdamW, clipping and validation-based early stopping. All PCA and
scalers fit training data only. Missing target pixels are imputed with a
training-only per-pixel residual mean for the spatial representation. Block PCA
rank is capped by available training examples. Sample spread is **uncalibrated**;
validation early stopping is not uncertainty calibration. These are fresh training
recipes based on the source structure, not the original fitted model selection or
its reported performance.

## Train all six and compare against persistence

```bash
python -m src.train_all --data-root data/nisar \
  --sites sabancaya ubinas misti \
  --output-dir outputs/my_training_run
```

This pools the existing training archives, uses leave-one-volcano-out folds within
those archives to select CVAE duration and ensemble weights, then fits final models
on all pooled training examples. The original `validation.npz` archives are kept
untouched for a single final comparison. Every model uses identical observed-pixel
masks. Scores weight volcanoes equally and windows equally within each volcano;
RMSE is the square root of that balanced mean squared error.

The output directory contains six saved models, pooled datasets, held-out predictions,
per-case and per-site CSV scores, internal selection reports, package versions,
source-data checksums and `REPORT.md`. Positive MAE skill means improvement over
persistence. CVAE additionally reports empirical CRPS and interval coverage.
These are temporal holdout scores on the prepared sites, not independent geographic
validation. Baselines fit locally; they do not acquire neural/global weights.

## Combined Peru and Japan model set

```bash
python -m src.train_all --data-root data/nisar \
  --sites sabancaya ubinas misti \
  --extra-training data/nisar_japan/mount_asama_ascending/examples.npz \
                   data/nisar_japan/mount_asama_descending/examples.npz \
  --compare-to outputs/nisar_training_20261115_041618 \
  --output-dir outputs/my_peru_japan_run
```

This fits a separate set of all six models on 33 examples (30 Peru, 3 Japan).
Asama's ascending and descending histories share one physical-volcano group in
internal validation. The same six Peru reporting windows are reused for a direct
comparison; `comparison.csv` records changes from the first model set. Original
artifacts are preserved. All Japan examples enter training, so there is no held-out
Japan score. Some Japan target dates are later than the Peru forecast origins:
these results are retrospective diagnostics, not a strict forward-time test.
The protocol records this explicitly in `temporal_audit`.

## Expanded international model set

```bash
python -m src.train_all --data-root data/nisar_training_global \
  --sites sabancaya ubinas misti popocatepetl dabbahu \
  --extra-training data/nisar_japan/mount_asama_ascending/examples.npz \
                   data/nisar_japan/mount_asama_descending/examples.npz \
  --compare-to outputs/nisar_peru_japan_combined --rescore-comparison \
  --output-dir outputs/my_expanded_run
```

The prepared pool contains 53 training examples across six volcanoes in Peru,
Japan, Mexico and Ethiopia, with 12 reporting windows across five volcanoes.
Ubinas was rebuilt excluding two BETA pairs; previous data and model sets remain
preserved. `data/nisar_training_global/input_sources.json` records the sources
and hashes. Japan contributes three training examples and no reporting windows.

`--rescore-comparison` evaluates the previous saved models on the current reporting
windows without fitting them. Use these paired scores instead of comparing the
new 12-window results with the old six-window summary. Training-only physical
volcano folds select CVAE duration and ensemble weights. Dates overlap globally,
so these are retrospective diagnostics, not a prospective forecast test. The
expanded data and Ubinas cleanup both changed; score differences cannot isolate
the effect of adding data alone.

## Predict with a model you trained

Inference data can be one `.npz` containing `maps (T,H,W)`, `dates (T,)`, optional
`valid_masks (T,H,W)` and scalar `metadata_json`. Batched archives need an explicit
`--case-index`. Separate `--maps maps.npy --dates dates.npy --masks masks.npy` also
work. Supply `--volcano-id` and `--horizon-days` for the requested forecast.

```bash
python -m src predict \
  --model cvae --checkpoint outputs/cvae.joblib \
  --input data/history.npz \
  --volcano-id corbetti --horizon-days 30 \
  --output outputs/forecast.npz --summary outputs/forecast.json \
  --n-samples 32 --sampling-seed 42
```

The learned models reject prediction until fitted, and the terminal requires an
explicit `--checkpoint` pointing to your newly saved artifact. Joblib artifacts
use pickle: load only files you trust. No pretrained model is bundled in `src`.
Baselines can predict directly without `--checkpoint`.

Metadata can come from archive `metadata_json`, a `--metadata metadata.json` file,
and explicit flags (in that precedence order). Available flags include
`--spacing-km ROW COLUMN`, `--geometry-available`/`--no-geometry-available`,
`--los-vector EAST NORTH UP`, and `--grid-convention native_center32`.

## Python API and output class

```python
from src import TrainingDataset, TrainingModel, InputPreparation, ForecastOutput

training = TrainingDataset.from_npz("data/train.npz")
validation = TrainingDataset.from_npz("data/validation.npz")
manager = TrainingModel("cvae", {"epochs": 80, "seed": 42})
manager.fit(training, validation)  # always a fresh model, never a warm start
manager.save("outputs/cvae.joblib")

history = InputPreparation().from_files(
    input_path="data/history.npz",
    volcano_id="corbetti",
    horizon_days=30,
    model_name="cvae",
)
forecast = manager.predict(history)
output = ForecastOutput(forecast, history, model_key="cvae")
output.save("outputs/forecast.npz")
output.save_summary("outputs/forecast.json")
```

`TrainingModel` is exported through `src.models`, with its coordinator in
`src/training.py` to avoid the `models.py`/`models/` import collision.
`ForecastOutput` writes predictions, change, target date and model/input metadata.
CVAE archives also contain samples, standard deviation and interval arrays such
as `lower90_mm`/`upper90_mm`. Missing pixels remain NaN; JSON represents nonfinite
numbers as null. Read outputs with `np.load(path, allow_pickle=False)` and decode
`metadata_json` with `json.loads`. Existing files require explicit overwrite.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s src/tests -t . -v
```

Tests build synthetic datasets, train fresh models, check neural weight updates,
training-only preprocessing, split rejection, masks, save/load round trips and
terminal training/prediction. They require neither the old source folder nor its
saved models/data. They establish workflow behavior, not scientific performance.

## Prepare historical Sentinel-1 ARIA GUNW

```bash
python -m src.prepare_aria --token-file src/token.txt --workers 4
```

This reads the existing `data/gunw_discovery/aria_survey.json` and its catalogues,
then prepares `data/aria/SITE/{series,history,examples,train,validation}.npz`.
Only sites with a valid purged temporal split receive train/validation archives.
`summary.json` records failures and sites without archive coverage;
`ready_sites.json` lists sites that can enter training. No new models are trained.

The default selects up to 60 connected acquisitions per site from one track,
direction and nominal footprint, trying up to three candidate stacks. It uses
latest dates within the inventoried component, not a complete archive download.
Use `--max-epochs 0` for the complete inventoried component. The saved Reykjanes
search hit its 5,000-result cap; this does not recover products absent from that
inventory. Optional `--sites coropuna cotopaxi` limits the batch.

The adapter reads NetCDF4 through h5py and authenticated HTTP (files up to 128 MiB
are fetched into bounded memory; larger files use byte ranges), converts
phase using each product's wavelength, bins native pixel centers onto a 32x32
UTM grid covering the configured physical AOI, and retains coherence >=0.4,
>=50% valid native coverage per cell and >=20% valid output cells per pair.
Only the dominant valid unwrapping component is retained. Reduced pair caches
are keyed by configuration and adapter version; source URLs and dates are saved.
Tokens are never included in these files.

ARIA native positive phase means motion toward the sensor, with the newer date
as reference. Native NISAR uses the opposite order/sign; its existing -1 phase
multiplier is consistent with the same toward-sensor displacement convention.
See the [ASF ARIA guide](https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/)
and [ARIA-tools convention documentation](https://github.com/aria-tools/ARIA-tools#readme).
Previous NISAR exports retain their original metadata and values.

Inputs and actual future acquisition targets use the existing causal inversion
and purged split. Product publication latency is not modeled. Displacement is
relative to each pair's retained-pixel median, so changing reference coverage
can introduce offsets and broad uniform motion is removed. Atmospheric,
ionospheric and tide corrections are not applied. Sensor, polarization, source
and stack remain explicit metadata; Sentinel-1 and NISAR histories are never
spliced together. Matching units and grids do not remove sensor differences.

No additional dependencies are required beyond `environment.yml`.

## Combine all prepared data and export Volcano Watch

After the ARIA batch completes:

```bash
python -m src.run_combined_workflow
```

This creates a separate `data/nisar_aria_training` staging pool and trains all six
families from scratch in `outputs/nisar_aria_all_data`. It combines the cleaned
Peru, Mexico and Ethiopia NISAR partitions, Mount Asama training-only examples,
quality-screened ARIA partitions, and any generated ARIA examples too short for
a purged split (training only). Original data and previous models are preserved.
Training configuration is selected using physical-volcano folds; reporting
windows are never passed to fitting. Previous expanded models are rescored on
those same reporting windows for comparison. This is a retrospective benchmark.

The workflow exports `outputs/volcano_watch/index.html` and `details.html`, based
on the original Peru Volcano Watch concept. Open index.html directly: the globe,
country geometry, maps and model outputs are bundled locally and need no server
or internet connection. Country/region, sensor history, model and date controls
select actual data. Forecast errors use paired reporting targets; observation
maps are full retrospective reconstructions. CVAE spread is labeled uncalibrated.
No official alerts or eruption probabilities are inferred.

Regenerate only the dashboard after editing files in `src/dashboard`:

```bash
python -m src.export_dashboard --manifest data/nisar_aria_training/manifest.json \
  --run outputs/nisar_aria_all_data --output-dir outputs/volcano_watch
```

`src/assemble_training.py` builds a provenance-hashed pool without training.
`src/export_dashboard.py` exports real maps, predictions and scores; it never
creates demonstration measurements. `dataset.json` describes the exported data,
and `data_sha256.json` records exported data scripts. Raw products, tokens and
model pickle files are not bundled into the dashboard. D3 and Natural Earth
source/license information is under `src/dashboard/assets`.

The island-specific Ambrym run uses a 20 km square (same center; thresholds
unchanged):

```bash
python -m src.prepare_aria --sites ambrym --token-file src/token.txt \
  --sites-config data/aria/island_sites_config.json --output-dir data/aria_island
```
