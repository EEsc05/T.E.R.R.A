"""Serialize forecast arrays and their input/model metadata."""

from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import os
import numpy as np


def _json_value(value):
    """Create standard JSON values; missing/nonfinite numbers become null."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class ForecastOutput:
    """Wrap a ForecastResult and save it without pickled Python objects."""

    def __init__(self, result, forecast_input=None, model_key=None):
        self.result = result
        self.forecast_input = forecast_input
        self.model_key = model_key

    def to_dict(self, include_arrays=False):
        result = self.result
        prediction = np.asarray(result.prediction_mm)
        finite = prediction[np.isfinite(prediction)]
        summary = {
            "model_name": result.model_name,
            "model_version": result.model_version,
            "model_key": self.model_key,
            "target_date": str(result.target_date),
            "shape": list(prediction.shape),
            "units": "signed LOS mm",
            "valid_pixels": int(finite.size),
            "prediction_min_mm": float(finite.min()) if finite.size else None,
            "prediction_max_mm": float(finite.max()) if finite.size else None,
            "has_uncertainty": result.samples_mm is not None,
            "metadata": result.metadata,
        }
        if self.forecast_input is not None:
            inp = self.forecast_input
            summary["input"] = {
                "volcano_id": inp.volcano_id,
                "horizon_days": inp.horizon_days,
                "origin_date": str(inp.forecast_origin),
                "history_length": inp.history_length,
                "metadata": inp.metadata,
            }
        if include_arrays:
            for key in (
                "prediction_mm",
                "change_mm",
                "samples_mm",
                "std_mm",
                "intervals_mm",
            ):
                summary[key] = getattr(result, key)
        return _json_value(summary)

    @staticmethod
    def _write(path, writer, overwrite):
        """Publish a completed file atomically; existing outputs require opt-in."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                writer(stream)
                stream.flush()
                if overwrite:
                    os.replace(temporary, path)
                else:
                    os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def save(self, path, overwrite=False):
        """Write a .npz with all forecast arrays and a scalar metadata_json string."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Forecast output path must end in .npz")
        result = self.result
        arrays = {
            "prediction_mm": result.prediction_mm,
            "change_mm": result.change_mm,
            "target_date": np.asarray(str(result.target_date)),
            "model_name": np.asarray(result.model_name),
            "model_version": np.asarray(result.model_version),
            "metadata_json": np.asarray(json.dumps(self.to_dict(), allow_nan=False)),
        }
        for key in ("samples_mm", "std_mm"):
            value = getattr(result, key)
            if value is not None:
                arrays[key] = value
        for level, (lower, upper) in (result.intervals_mm or {}).items():
            arrays[f"lower{level}_mm"] = lower
            arrays[f"upper{level}_mm"] = upper
        return self._write(path, lambda f: np.savez_compressed(f, **arrays), overwrite)

    def save_summary(self, path, overwrite=False):
        """Write readable JSON metadata and summary statistics."""
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("Summary output path must end in .json")
        payload = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        return self._write(path, lambda f: f.write(payload.encode()), overwrite)
