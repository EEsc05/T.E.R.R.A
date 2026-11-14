from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ForecastInput:
    """
    Standardized input for Stage 1 deformation forecasting.

    All models receive the same historical observations,
    acquisition dates, validity masks, and forecast horizon.
    """

    volcano_id: str
    maps: np.ndarray
    dates: np.ndarray
    valid_masks: np.ndarray
    horizon_days: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        maps = np.asarray(self.maps)
        if not np.issubdtype(maps.dtype, np.number) or np.iscomplexobj(maps):
            raise ValueError("maps must contain real signed LOS displacement in mm")
        self.maps = maps.astype(np.float32)
        dates = np.asarray(self.dates)
        if np.issubdtype(dates.dtype, np.number):
            if not np.isfinite(dates).all():
                raise ValueError("dates must be finite")
            # Research fixtures encode dates as days since 1970, including fractions.
            dates = np.rint(dates.astype(float) * 86400e9).astype("int64")
        self.dates = dates.astype("datetime64[ns]")
        self.valid_masks = np.asarray(self.valid_masks)
        if self.valid_masks.dtype != np.bool_:
            raise ValueError("valid_masks must be boolean")

        self.validate()

    def validate(self):
        """Check input dimensions, dates, and values."""

        # Historical displacement maps: (time, height, width)
        if self.maps.ndim != 3 or min(self.maps.shape) < 1:
            raise ValueError("maps must have shape (T, H, W)")

        # One validity mask per displacement map
        if self.valid_masks.shape != self.maps.shape:
            raise ValueError("valid_masks must match maps shape")

        # One acquisition date per map
        if self.dates.ndim != 1 or len(self.dates) != len(self.maps):
            raise ValueError("dates must have shape (T,)")

        # Persistence accepts one acquisition; the other models enforce six.

        # Acquisition dates must be chronological
        if np.isnat(self.dates).any():
            raise ValueError("Acquisition dates cannot contain NaT")

        if np.any(np.diff(self.dates) <= np.timedelta64(0, "ns")):
            raise ValueError("Acquisition dates must be strictly increasing")

        if not np.isfinite(self.horizon_days) or self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive and finite")

        if not np.any(self.valid_masks & np.isfinite(self.maps)):
            raise ValueError("No valid displacement pixels found")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    @property
    def shape(self):
        """Spatial dimensions (height, width)."""
        return self.maps.shape[1:]

    @property
    def history_length(self):
        """Number of historical observations."""
        return self.maps.shape[0]

    @property
    def forecast_origin(self):
        """Date of the latest historical observation."""
        return self.dates[-1]

    @property
    def target_date(self):
        """Requested future prediction date."""
        nanoseconds = int(round(self.horizon_days * 86400e9))
        return self.forecast_origin + np.timedelta64(nanoseconds, "ns")

    def latest_map(self):
        """Return the most recent displacement map."""
        return self.maps[-1].copy()

    def time_offsets(self):
        """Days elapsed since the first observation."""
        return ((self.dates - self.dates[0]) / np.timedelta64(1, "D")).astype(
            np.float64
        )

    def model_arrays(self):
        """
        Return arrays with invalid pixels filled by zero.

        The mask must be passed alongside these arrays so
        models do not interpret filled pixels as observations.
        """
        masks = self.valid_masks & np.isfinite(self.maps)
        clean_maps = np.where(masks, self.maps, 0.0)

        return {
            "maps": clean_maps,
            "valid_masks": masks.copy(),
            "time_days": self.time_offsets(),
            "horizon_days": self.horizon_days,
        }
