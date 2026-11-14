"""Shared baseline fitting and masked evaluation utilities."""

import numpy as np
from ..preprocessing.interface import prepare


def evaluate(model, dataset):
    errors = []
    for inp, target, mask in zip(dataset.inputs, dataset.targets, dataset.target_masks):
        pred = model.predict(inp).prediction_mm
        valid = mask & np.isfinite(target) & np.isfinite(pred)
        if not valid.any():
            raise ValueError("No overlapping prediction/target pixels for evaluation")
        errors.append((pred[valid] - target[valid]).astype(float))
    errors = np.concatenate(errors)
    return {
        "mae_mm": float(np.mean(abs(errors))),
        "rmse_mm": float(np.sqrt(np.mean(errors**2))),
        "bias_mm": float(errors.mean()),
        "observed_pixels": int(len(errors)),
    }


class LocalModel:
    """Local baselines have no global trainable weights; fits use supplied history."""

    def fit(self, training, validation=None):
        for inp in training.inputs:
            prepare(
                inp, min_history=1 if type(self).__name__ == "PersistenceModel" else 6
            )
        self.training_report = {
            "training_examples": len(training),
            "global_weights_learned": False,
            "training_metrics": evaluate(self, training),
        }
        if validation is not None:
            self.training_report["validation_metrics"] = evaluate(self, validation)
        return self

    train = fit
