"""Trainable temporal, spatial, context and quality forecasting ensemble."""

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from ..preprocessing.interface import prepare, result
from ..preprocessing.shared import prepare_batch
from ..preprocessing.features import build
from ..preprocessing.spatial import predict as spatial_predict
from ..preprocessing.context import predict as context_predict
from ..dataset import validate_split
from .base import evaluate
from .ema30 import EMA30Model
from .smoothing import SmoothingModel


class FourBranchModel:
    def __init__(self, seed=42, max_context_pixels=10000, report_training_metrics=True):
        self.seed = seed
        self.max_context_pixels = max_context_pixels
        self.report_training_metrics = report_training_metrics
        self.is_fitted = False

    def _branches(self, inp):
        p = prepare(inp, fixed_grid=True)
        m, a = prepare_batch(p)
        temporal = EMA30Model().predict(inp).prediction_mm
        quality = SmoothingModel().predict(inp).prediction_mm
        spatial, _ = spatial_predict(self.spatial, m, a, np.array([0]))
        context = context_predict(
            self.context,
            a,
            build(m, a),
            np.array([0]),
            {"residual_temporal": temporal[None]},
        )
        return p, np.asarray([temporal, spatial[0], context[0], quality])

    def fit(self, training, validation=None):
        self.is_fitted = False
        if len(training) < 2:
            raise ValueError("Four-branch training requires at least two examples")
        if self.max_context_pixels < 2:
            raise ValueError("max_context_pixels must be at least 2")
        if validation is not None:
            validate_split(training, validation)
        batches, histories, xs, ys = [], [], [], []
        for inp, target, mask in zip(
            training.inputs, training.targets, training.target_masks
        ):
            p = prepare(inp, fixed_grid=True)
            m, a = prepare_batch(p)
            batches.append((m, a))
            histories.extend(np.nan_to_num((p.raw - p.last) / a["scale"][0]))
            features = build(m, a)[0]
            baseline = EMA30Model().predict(inp).prediction_mm
            valid = mask & p.seen & np.isfinite(target)
            xs.append(features[valid.ravel()])
            ys.append((target - baseline)[valid])
        # Learn one spatial mode in each native 8×8 block from training histories only.
        histories = np.asarray(histories)
        basis = []
        for y in range(0, 32, 8):
            for x in range(0, 32, 8):
                patch = histories[:, y : y + 8, x : x + 8].reshape(len(histories), 64)
                component = (
                    PCA(n_components=1, random_state=self.seed)
                    .fit(patch)
                    .components_[0]
                )
                full = np.zeros((32, 32))
                full[y : y + 8, x : x + 8] = component.reshape(8, 8)
                basis.append(full.ravel())
        self.spatial = {
            "representation": "block16",
            "method": "linear",
            "basis": np.asarray(basis),
        }
        x, y = np.concatenate(xs), np.concatenate(ys)
        rng = np.random.default_rng(self.seed)
        if len(x) > self.max_context_pixels:
            ix = rng.choice(len(x), self.max_context_pixels, replace=False)
            x, y = x[ix], y[ix]
        scale = max(float(np.std(y)), 1.0)
        pipeline = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))
        pipeline.fit(x, y / scale)
        self.context = {
            "pipeline": pipeline,
            "columns": np.arange(x.shape[1]),
            "target_scale": scale,
            "target_offset": 0.0,
            "config": {"target": "residual_temporal"},
        }
        self.weights = np.full((3, 4), 0.25)
        self.is_fitted = True
        self.training_report = {
            "training_examples": len(training),
            "context_pixels": len(x),
            "seed": self.seed,
            "fusion": "equal weights without validation",
        }
        if validation is not None:
            rows = [([], []) for _ in range(3)]
            for inp, target, mask in zip(
                validation.inputs, validation.targets, validation.target_masks
            ):
                p, branches = self._branches(inp)
                valid = mask & p.seen & np.isfinite(target)
                bin_id = int(np.digitize(p.horizon, [31, 91]))
                rows[bin_id][0].append(branches[:, valid].T)
                rows[bin_id][1].append(target[valid])
            for bin_id, (features, targets) in enumerate(rows):
                if not targets:
                    continue
                x, y = np.concatenate(features), np.concatenate(targets)
                opt = minimize(
                    lambda w: np.mean((x @ w - y) ** 2),
                    np.full(4, 0.25),
                    bounds=[(0, 1)] * 4,
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                    method="SLSQP",
                )
                if not opt.success:
                    raise ValueError(
                        f"Fusion weight optimization failed: {opt.message}"
                    )
                self.weights[bin_id] = np.maximum(opt.x, 0) / np.maximum(opt.x, 0).sum()
            self.training_report["fusion"] = (
                "horizon-bin weights fitted on validation; these scores are calibration scores"
            )
            self.training_report["calibration_metrics"] = evaluate(self, validation)
        if self.report_training_metrics:
            self.training_report["training_metrics"] = evaluate(self, training)
        return self

    train = fit

    def predict(self, forecast_input):
        if not self.is_fitted:
            raise ValueError(
                "FourBranchModel is untrained; call fit() or explicitly load your own trained model"
            )
        p, branches = self._branches(forecast_input)
        bin_id = int(np.digitize(p.horizon, [31, 91]))
        weights = self.weights[bin_id]
        prediction = np.einsum("b,bhw->hw", weights, branches)
        return result(
            p,
            prediction,
            "Four-branch hybrid",
            "trained_from_scratch",
            {
                "branch_names": ["temporal", "spatial", "context", "quality"],
                "branch_weights": weights.tolist(),
                "horizon_bin": bin_id,
                "branch_maps_mm": branches,
                "training_seed": self.seed,
            },
        )
