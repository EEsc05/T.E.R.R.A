"""Explicit persistence of models trained by this project; no bundled checkpoints."""

from pathlib import Path
import copy
import joblib
import torch
from ..output import ForecastOutput


def save_model(path, payload, overwrite=False):
    path = Path(path)
    if path.suffix.lower() != ".joblib":
        raise ValueError("Trained model path must end in .joblib")
    portable = copy.deepcopy(payload)
    model = portable["estimator"]
    if hasattr(model, "device"):
        model.device = torch.device("cpu")
        model.model = model.model.cpu()
        portable["parameters"]["device"] = "cpu"
    return ForecastOutput._write(
        path, lambda stream: joblib.dump(portable, stream), overwrite
    )


def load_model(path):
    """Load a user-selected trusted joblib artifact (joblib uses Python pickle)."""
    payload = joblib.load(Path(path))
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "terra-trained-model"
        or payload.get("format_version") != 1
    ):
        raise ValueError("Not a model trained and saved by this project")
    return payload
