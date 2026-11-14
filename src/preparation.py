"""Load NumPy observations and prepare the shared forecasting input contract."""

from pathlib import Path
import json
import numpy as np
from .input import ForecastInput
from .preprocessing.interface import prepare
from .preprocessing.shared import prepare_batch


class InputPreparation:
    """Convert units/masks and optionally extract a native central 32×32 grid.

    Maps must already share a grid, LOS direction and displacement reference.
    This class does not register images, interpolate dates or resample pixels.
    """

    UNIT_SCALES = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    MODEL_NAMES = ("persistence", "ema30", "smoothing", "robust", "four_branch", "cvae")

    def __init__(self, units=None, nodata=None, center_crop=False):
        if units is not None and units not in self.UNIT_SCALES:
            raise ValueError("units must be mm, cm or m")
        self.units = units
        self.nodata = nodata
        self.center_crop = center_crop

    def from_arrays(
        self,
        maps,
        dates,
        *,
        volcano_id,
        horizon_days,
        valid_masks=None,
        metadata=None,
        model_name=None,
    ):
        metadata = dict(metadata or {})
        units = self.units or metadata.get("units", "mm")
        if units not in self.UNIT_SCALES:
            raise ValueError("Input units must be mm, cm or m")
        maps = np.asarray(maps)
        if maps.ndim != 3 or min(maps.shape) < 1:
            raise ValueError("maps must have nonempty shape (T, H, W)")
        if not np.issubdtype(maps.dtype, np.number) or np.iscomplexobj(maps):
            raise ValueError("maps must contain real numeric displacements")
        if valid_masks is None:
            masks = np.ones(maps.shape, dtype=bool)
            mask_source = "finite pixels, excluding explicit nodata when provided"
        else:
            masks = np.asarray(valid_masks)
            if masks.dtype != np.bool_ or masks.shape != maps.shape:
                raise ValueError("valid_masks must be boolean and match maps shape")
            mask_source = "explicit boolean masks, excluding nonfinite/nodata pixels"
        masks = masks & np.isfinite(maps)
        if self.nodata is not None:
            masks &= maps != self.nodata
        # Apply nodata in source units, then convert displacements to millimetres.
        maps = maps.astype(np.float64) * self.UNIT_SCALES[units]
        source_shape = list(maps.shape)
        crop = None
        if self.center_crop:
            height, width = maps.shape[1:]
            if height < 32 or width < 32:
                raise ValueError(
                    "Center crop requires spatial dimensions at least 32×32"
                )
            row, col = height // 2 - 16, width // 2 - 16
            maps = maps[:, row : row + 32, col : col + 32]
            masks = masks[:, row : row + 32, col : col + 32]
            crop = {"row_start": row, "column_start": col, "height": 32, "width": 32}
            metadata["grid_convention"] = "native_center32"
        maps = np.where(masks, maps, np.nan)
        metadata["units"] = "mm"
        metadata["input_preparation"] = {
            "source_units": units,
            "unit_scale_to_mm": self.UNIT_SCALES[units],
            "source_shape": source_shape,
            "mask_source": mask_source,
            "nodata": self.nodata,
            "crop": crop,
        }
        data = ForecastInput(volcano_id, maps, dates, masks, horizon_days, metadata)
        self.validate_for_model(data, model_name)
        return data

    @classmethod
    def validate_for_model(cls, data, model_name):
        if model_name is None:
            return
        if model_name not in cls.MODEL_NAMES:
            raise ValueError(f"Unknown model {model_name!r}")
        learned = model_name in ("four_branch", "cvae")
        prepared = prepare(
            data,
            min_history=1 if model_name == "persistence" else 6,
            fixed_grid=learned,
        )
        if learned:
            # Enforce the same geometry and spacing contract before loading weights.
            prepare_batch(prepared)

    @staticmethod
    def _array(path):
        path = Path(path)
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy array: {path}")
        return np.load(path, allow_pickle=False)

    @staticmethod
    def _metadata(path):
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("Metadata JSON must contain an object")
        return value

    def from_files(
        self,
        *,
        volcano_id,
        horizon_days,
        input_path=None,
        maps_path=None,
        dates_path=None,
        masks_path=None,
        metadata_path=None,
        metadata=None,
        case_index=None,
        model_name=None,
    ):
        """Load `maps`, `dates`, optional `valid_masks` and `metadata_json` from NPZ.

        Separate .npy files can be used instead. An explicit case_index selects
        one case from (N,T,H,W) archives; cases are never selected implicitly.
        Explicit dates/masks override matching NPZ fields. Metadata precedence:
        bundle, JSON file, explicit metadata dictionary.
        """
        if (input_path is None) == (maps_path is None):
            raise ValueError("Provide either input_path (.npz) or maps_path (.npy)")
        bundle_metadata = {}
        if input_path is not None:
            if Path(input_path).suffix.lower() != ".npz":
                raise ValueError("input_path must be a .npz archive")
            with np.load(input_path, allow_pickle=False) as archive:
                if "maps" not in archive:
                    raise ValueError("NPZ archive must contain a 'maps' array")
                maps = archive["maps"]
                dates = archive["dates"] if "dates" in archive else None
                masks = archive["valid_masks"] if "valid_masks" in archive else None
                if "metadata_json" in archive:
                    value = archive["metadata_json"]
                    if value.ndim != 0 or value.dtype.kind not in "US":
                        raise ValueError("metadata_json must be a scalar JSON string")
                    item = value.item()
                    bundle_metadata = json.loads(item)
                    if not isinstance(bundle_metadata, dict):
                        raise ValueError("metadata_json must describe an object")
        else:
            maps, dates, masks = self._array(maps_path), None, None
        if dates_path is not None:
            dates = self._array(dates_path)
        if masks_path is not None:
            masks = self._array(masks_path)
        if dates is None:
            raise ValueError("Supply dates in the archive or through --dates dates.npy")
        if case_index is not None:
            if maps.ndim != 4:
                raise ValueError("--case-index requires maps with shape (N, T, H, W)")
            if not 0 <= case_index < len(maps):
                raise ValueError("case_index is outside the input archive")
            if dates.ndim == 2:
                if dates.shape[0] != len(maps):
                    raise ValueError("Batched dates must match the number of map cases")
                dates = dates[case_index]
            if masks is not None and masks.ndim == 4:
                if masks.shape[0] != len(maps):
                    raise ValueError("Batched masks must match the number of map cases")
                masks = masks[case_index]
            maps = maps[case_index]
        elif maps.ndim == 4:
            raise ValueError("Batched input requires an explicit --case-index")
        if metadata_path is not None:
            bundle_metadata.update(self._metadata(metadata_path))
        bundle_metadata.update(metadata or {})
        bundle_metadata["input_source"] = {
            "maps": str(input_path if input_path is not None else maps_path),
            "dates": str(dates_path) if dates_path is not None else "archive:dates",
            "masks": str(masks_path)
            if masks_path is not None
            else ("archive:valid_masks" if masks is not None else "derived"),
            "case_index": case_index,
        }
        return self.from_arrays(
            maps,
            dates,
            volcano_id=volcano_id,
            horizon_days=horizon_days,
            valid_masks=masks,
            metadata=bundle_metadata,
            model_name=model_name,
        )
