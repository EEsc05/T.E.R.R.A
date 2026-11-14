"""Synthetic supervised examples; no external data or pretrained artifacts."""

import json
import numpy as np
from src import ForecastInput, TrainingDataset


def dataset(n=6, start="2020-01-01", offset=0.0):
    rng = np.random.default_rng(7)
    inputs = []
    targets = []
    future = []
    yy, xx = np.mgrid[:32, :32]
    for k in range(n):
        dates = (
            np.datetime64(start, "D") + np.arange(6) * 12 + np.timedelta64(k * 100, "D")
        )
        spatial = np.sin(xx / 7 + k * 0.2) + np.cos(yy / 9)
        maps = np.stack(
            [spatial * (1 + t * 0.2) + rng.normal(0, 0.1, (32, 32)) for t in range(6)]
        ).astype("float32")
        md = {
            "units": "mm",
            "spacing_km": [0.2, 0.2],
            "geometry_available": False,
            "grid_convention": "native_center32",
        }
        inp = ForecastInput("site", maps, dates, np.ones_like(maps, dtype=bool), 30, md)
        inputs.append(inp)
        targets.append(maps[-1] + spatial * 0.5 + offset)
        future.append(inp.target_date)
    targets = np.asarray(targets)
    return TrainingDataset(inputs, targets, np.isfinite(targets), np.asarray(future))


def save_dataset(path, data):
    np.savez_compressed(
        path,
        maps=np.asarray([i.maps for i in data.inputs]),
        dates=np.asarray([i.dates for i in data.inputs]),
        valid_masks=np.asarray([i.valid_masks for i in data.inputs]),
        targets=data.targets,
        target_masks=data.target_masks,
        target_dates=data.target_dates,
        volcano_ids=np.asarray([i.volcano_id for i in data.inputs]),
        metadata_json=np.asarray([json.dumps(i.metadata) for i in data.inputs]),
    )
