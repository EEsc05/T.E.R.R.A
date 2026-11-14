"""four_horizon: real checkpoint inference, not lookup of saved predictions."""

import numpy as np
from ..utils.checkpoints import load_joblib, config
from ..preprocessing.interface import prepare, result
from ..preprocessing.shared import prepare_batch
from ..preprocessing.features import build
from ..preprocessing.spatial import predict as spatial_predict
from ..preprocessing.context import predict as context_predict
from .ema30 import EMA30Model
from .smoothing import SmoothingModel


class FourBranchModel:
    def __init__(self):
        self.spatial = load_joblib("checkpoints/spatial_block16_linear.joblib")
        self.context = load_joblib("checkpoints/context_SVR_residual_temporal.joblib")
        self.weights = config("configs/four_horizon_weights.json")
        temporal = config("configs/v9_temporal_selection.json")
        c = temporal["configs"][temporal["best"]]
        if not (
            c["family"] == "EMA"
            and c["half"] == 30
            and c["window"] == 6
            and not c["trend"]
        ):
            raise ValueError(
                "Selected temporal recipe does not match copied EMA30; no silent substitution allowed."
            )
        if (
            self.context["config"]["target"] != "residual_temporal"
            or self.spatial["representation"] != "block16"
            or self.spatial["method"] != "linear"
        ):
            raise ValueError("Unexpected frozen branch configuration.")

    def predict(self, forecast_input):
        p = prepare(forecast_input, fixed_grid=True)
        m, a = prepare_batch(p)
        x = build(m, a)
        temporal = EMA30Model().predict(forecast_input).prediction_mm
        quality = SmoothingModel().predict(forecast_input).prediction_mm
        spatial, _ = spatial_predict(self.spatial, m, a, np.array([0]))
        context = context_predict(
            self.context, a, x, np.array([0]), {"residual_temporal": temporal[None]}
        )
        branches = np.array([temporal, spatial[0], context[0], quality])
        binid = int(np.digitize(p.horizon, self.weights["bin_edges_days"]))
        weights = next(r["weights"] for r in self.weights["bins"] if r["bin"] == binid)
        forecast = np.einsum("b,bhw->hw", weights, branches)
        return result(
            p,
            forecast,
            "Four-branch hybrid",
            "four_horizon",
            dict(
                branch_names=self.weights["branches"],
                branch_weights=list(weights),
                horizon_bin=binid,
                branch_maps_mm=branches,
                training_or_calibration_performed=False,
                calibration="Frozen original2022 horizon-bin shrinkage weights, copied from previously verified selected-model bundle",
            ),
        )
