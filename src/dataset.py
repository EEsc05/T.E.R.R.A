"""Supervised map histories and observed future maps for training from scratch."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .preparation import InputPreparation
from .input import ForecastInput


@dataclass
class TrainingDataset:
    inputs: list[ForecastInput]
    targets: np.ndarray
    target_masks: np.ndarray
    target_dates: np.ndarray

    def __post_init__(self):
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.target_masks = np.asarray(self.target_masks)
        self.target_dates = np.asarray(self.target_dates, dtype="datetime64[ns]")
        if (
            not self.inputs
            or self.targets.ndim != 3
            or len(self.inputs) != len(self.targets)
        ):
            raise ValueError("Training requires N inputs and targets shaped (N,H,W)")
        if (
            self.target_masks.dtype != np.bool_
            or self.target_masks.shape != self.targets.shape
        ):
            raise ValueError("target_masks must be boolean and match targets")
        if (
            self.target_dates.shape != (len(self.inputs),)
            or np.isnat(self.target_dates).any()
        ):
            raise ValueError("target_dates must contain N valid observation dates")
        self.target_masks = self.target_masks & np.isfinite(self.targets)
        for i, inp in enumerate(self.inputs):
            if inp.shape != self.targets[i].shape:
                raise ValueError("History and target grids must match")
            if (
                self.target_dates[i] <= inp.forecast_origin
                or self.target_dates[i] != inp.target_date
            ):
                raise ValueError(
                    "Each target date must match the requested future horizon"
                )
            if not (
                self.target_masks[i]
                & np.any(inp.valid_masks & np.isfinite(inp.maps), axis=0)
            ).any():
                raise ValueError(
                    "Each target needs observed pixels on historical support"
                )

    def __len__(self):
        return len(self.inputs)

    @classmethod
    def from_npz(cls, path, *, units=None, nodata=None, center_crop=False):
        """Required keys: maps (N,T,H,W), dates (N,T), targets (N,H,W), target_dates (N,).

        Dates (T,) may be shared. Optional masks, target_masks, volcano_ids (N,),
        and metadata_json (scalar JSON or N JSON strings). Numeric dates are days
        since 1970. Targets and histories use the same explicitly supplied units.
        """
        with np.load(Path(path), allow_pickle=False) as a:
            required = {"maps", "dates", "targets", "target_dates"}
            if not required.issubset(a.files):
                raise ValueError(
                    f"Training archive missing keys: {sorted(required - set(a.files))}"
                )
            maps, dates, targets, future = (
                a[k] for k in ("maps", "dates", "targets", "target_dates")
            )
            if maps.ndim != 4 or targets.shape != (len(maps), *maps.shape[2:]):
                raise ValueError("maps must be (N,T,H,W), targets must be (N,H,W)")
            n = len(maps)
            if dates.shape not in (
                (maps.shape[1],),
                maps.shape[:2],
            ) or future.shape != (n,):
                raise ValueError(
                    "dates must be (T,) or (N,T); target_dates must be (N,)"
                )
            masks = a["valid_masks"] if "valid_masks" in a else np.isfinite(maps)
            target_masks = (
                a["target_masks"] if "target_masks" in a else np.isfinite(targets)
            )
            if masks.shape != maps.shape or masks.dtype != np.bool_:
                raise ValueError("valid_masks must be boolean and match maps")
            if target_masks.shape != targets.shape or target_masks.dtype != np.bool_:
                raise ValueError("target_masks must be boolean and match targets")
            ids = a["volcano_ids"] if "volcano_ids" in a else np.repeat("site", n)
            if ids.shape != (n,):
                raise ValueError("volcano_ids must have shape (N,)")
            metadata = a["metadata_json"] if "metadata_json" in a else np.asarray("{}")
            if metadata.shape not in ((), (n,)) or metadata.dtype.kind not in "US":
                raise ValueError(
                    "metadata_json must be one JSON string or N JSON strings"
                )
            inputs, converted, converted_masks, target_dates = [], [], [], []
            preparer = InputPreparation(units, nodata, center_crop)
            for i in range(n):
                md = json.loads(metadata.item() if metadata.ndim == 0 else metadata[i])
                if not isinstance(md, dict):
                    raise ValueError("metadata_json must describe objects")
                t = dates if dates.ndim == 1 else dates[i]
                # Use the shared date conversion for both history and observed target.
                origin = ForecastInput(
                    str(ids[i]), maps[i], t, masks[i], 1, md
                ).forecast_origin
                f = np.asarray([future[i]])
                if np.issubdtype(f.dtype, np.number):
                    f = np.rint(f.astype(float) * 86400e9).astype("int64")
                target_date = f.astype("datetime64[ns]")[0]
                horizon = float((target_date - origin) / np.timedelta64(1, "D"))
                inp = preparer.from_arrays(
                    maps[i],
                    t,
                    volcano_id=str(ids[i]),
                    horizon_days=horizon,
                    valid_masks=masks[i],
                    metadata=md,
                )
                # Apply exactly the same conversion and native crop to the future map.
                target = preparer.from_arrays(
                    targets[i : i + 1],
                    [target_date],
                    volcano_id=str(ids[i]),
                    horizon_days=1,
                    valid_masks=target_masks[i : i + 1],
                    metadata=md,
                )
                inputs.append(inp)
                converted.append(target.maps[0])
                converted_masks.append(target.valid_masks[0])
                target_dates.append(target_date)
        return cls(
            inputs,
            np.asarray(converted),
            np.asarray(converted_masks),
            np.asarray(target_dates),
        )


def validate_split(training, validation):
    """Require a forward temporal holdout for every site appearing in both sets."""
    for site in {i.volcano_id for i in training.inputs} & {
        i.volcano_id for i in validation.inputs
    }:
        latest = max(
            training.target_dates[k]
            for k, i in enumerate(training.inputs)
            if i.volcano_id == site
        )
        earliest = min(i.dates[0] for i in validation.inputs if i.volcano_id == site)
        if latest >= earliest:
            raise ValueError(
                f"Training targets must precede all validation history for site {site!r}; use a purged temporal split"
            )
