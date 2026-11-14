# Stage 1 models: train from scratch

All model definitions, training, preparation and output code live in `src`.
The implementation follows the model families and structure of the original
stage 1 scripts. Nothing in `src` automatically loads their pretrained checkpoints,
configs, selection results or data. `hackathon_stage1` is not a runtime dependency.

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

Four-branch and CVAE require native 32×32 maps and horizons of 1–180 days. Each
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
