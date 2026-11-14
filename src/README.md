# Stage 1 forecasting

Run from the project root after `conda activate terra`:

```python
from src import ForecastInput, PersistenceModel, TrainingModel

data = ForecastInput(
    volcano_id="site_id",
    maps=historical_maps_mm,  # (T, H, W), signed LOS displacement in mm
    dates=acquisition_dates,  # UTC dates or numeric days since 1970-01-01
    valid_masks=historical_masks,  # boolean, same shape as maps
    horizon_days=30,
    metadata={"units": "mm"},
)
result = PersistenceModel().predict(data)
prediction = result.prediction_mm
change = result.change_mm

# The coordinator uses the same input and returns the same ForecastResult.
model = TrainingModel("ema30")
result = model.predict(data)
```

| Module | Class | Coordinator name |
| --- | --- | --- |
| `src.models.persistence` | `PersistenceModel` | `persistence` |
| `src.models.ema30` | `EMA30Model` | `ema30` |
| `src.models.smoothing` | `SmoothingModel` | `smoothing` |
| `src.models.robust` | `RobustModel` | `robust` |
| `src.models.four_branch` | `FourBranchModel` | `four_branch` |
| `src.models.cvae` | `CVAEModel` | `cvae` |

The implementations preserve the supplied frozen inference formulas and weights.
`TrainingModel` retains the existing coordinator name; its methods return forecasts
without training or writing pickle files. It caches model instances for reuse.
Its implementation is in `src/training.py`, exported through `src.models` to
resolve the existing `models.py` and `models/` naming collision. CVAE accepts
`parameters={"device": "cpu", "n_samples": 32}` through the coordinator.

Models use the final six acquisitions. Persistence also accepts a single
acquisition. Valid finite zeros are observations; masked or nonfinite pixels are
missing. Never-observed output pixels are NaN. All models return `ForecastResult`
with prediction, change, target date and metadata. CVAE additionally returns
samples, standard deviation and intervals.

Four-branch and CVAE require native, co-registered 32×32 grids, a horizon of
1–180 days, and actual metadata: `grid_convention="native_center32"`, positive
`spacing_km=[row_spacing, column_spacing]`, and explicit
`geometry_available=True/False`. Known geometry also requires
`los_vector=[east, north, up]`. The code does not infer geometry or resize maps.

All runtime Python code is under `src`. Checkpoints, configurations, manifest,
and regression fixtures remain in `hackathon_stage1`; keep that directory beside
`src`. The loader verifies artifact checksums before loading. Historical version
prefixes were removed from public model labels, while original artifact paths
and serialized class identifiers remain intact for compatibility. Archived
research runners and fixture-generation scripts are not needed for inference.

The environment records the supplied scientific package versions and CPU PyTorch:

```bash
mamba env update -n terra -f environment.yml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest src.tests.test_inference -v
```

The tests compare all six models against the existing reference fixtures. This
checks integration equivalence, not performance on new forecasting inputs.
