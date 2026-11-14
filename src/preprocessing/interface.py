"""New compatibility contracts. No future observations or training operations."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..input import ForecastInput


@dataclass
class ForecastResult:
    prediction_mm: np.ndarray
    change_mm: np.ndarray
    target_date: np.datetime64
    model_name: str
    model_version: str
    metadata: dict = field(default_factory=dict)
    samples_mm: Optional[np.ndarray] = None
    std_mm: Optional[np.ndarray] = None
    intervals_mm: Optional[dict] = None

    @property
    def mean_prediction_mm(self):
        """For CVAE this is the sample mean; for other models the point forecast."""
        return self.prediction_mm


@dataclass
class PreparedInput:
    raw: np.ndarray
    days: np.ndarray
    last: np.ndarray
    seen: np.ndarray
    origin_date: np.datetime64
    target_date: np.datetime64
    horizon: float
    volcano_id: str
    metadata: dict
    total_acquisitions: int


def prepare(inp, min_history=6, fixed_grid=False):
    from .numerics import fill_past

    maps = np.asarray(inp.maps)
    if maps.ndim != 3 or min(maps.shape) < 1:
        raise ValueError("maps must have nonempty shape(T,H,W).")
    if not np.issubdtype(maps.dtype, np.number) or np.iscomplexobj(maps):
        raise ValueError("maps must be real signed LOS displacement in mm.")
    masks = np.asarray(inp.valid_masks)
    if masks.shape != maps.shape or masks.dtype != np.bool_:
        raise ValueError(
            "valid_masks must be boolean with the same(T,H,W) shape as maps."
        )
    dates = np.asarray(inp.dates)
    if dates.ndim != 1 or len(dates) != len(maps):
        raise ValueError("dates must be one-dimensional and match T.")
    if np.issubdtype(dates.dtype, np.number):
        days = dates.astype(float)
        if not np.isfinite(days).all():
            raise ValueError("Dates must be finite.")
        ns = np.rint(days * 86400e9).astype("int64")
        datetimes = ns.astype("datetime64[ns]")
    else:
        try:
            datetimes = dates.astype("datetime64[ns]")
            days = datetimes.astype("int64") / 86400e9
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Use chronological UTC dates or numeric days since1970-01-01."
            ) from error
        if np.isnat(datetimes).any():
            raise ValueError("dates cannot contain NaT.")
    if not (np.diff(days) > 0).all():
        raise ValueError(
            "Acquisition dates must be strictly chronological; no sorting or deduplication is performed."
        )
    if len(maps) < min_history:
        raise ValueError(f"This model requires at least{min_history} acquisitions.")
    h = float(inp.horizon_days)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("horizon_days must be positive and finite.")
    if fixed_grid and (maps.shape[1:] != (32, 32) or not 1 <= h <= 180):
        raise ValueError(
            "Hybrid/CVAE require native32x32 maps and horizon1–180days; no resizing is performed."
        )
    if not isinstance(inp.metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    if inp.metadata.get("units", "mm") != "mm":
        raise ValueError("Convert maps to signed LOS millimetres before inference.")
    raw = np.where(masks[-6:] & np.isfinite(maps[-6:]), maps[-6:], np.nan).astype(
        "float32"
    )
    f, v = fill_past(raw)
    if not v.any():
        raise ValueError("No valid historical pixels are available.")
    last = np.where(v.any(0), f[-1], np.nan)
    target = datetimes[-1] + np.timedelta64(int(round(h * 86400e9)), "ns")
    return PreparedInput(
        raw,
        days[-6:],
        last,
        v.any(0),
        datetimes[-1],
        target,
        h,
        str(inp.volcano_id),
        dict(inp.metadata),
        len(maps),
    )


def result(prep, p, name, version, metadata=None, **kwargs):
    p = np.array(p, copy=True)
    p[~prep.seen] = np.nan
    if p.shape != prep.last.shape or not np.isfinite(p[prep.seen]).all():
        raise ValueError(
            "Model returned invalid values on historically observed support."
        )
    info = dict(
        units="signed LOS mm",
        history_acquisitions_used=len(prep.raw),
        older_acquisitions_ignored=prep.total_acquisitions - len(prep.raw),
        latest_observation="Per-pixel historical last observation, forward-filled across missing acquisitions",
        horizon_within_documented_1_to_180_days=bool(1 <= prep.horizon <= 180),
        operational_asof_verified=False,
    )
    info.update(metadata or {})
    return ForecastResult(
        p, p - prep.last, prep.target_date, name, version, info, **kwargs
    )
