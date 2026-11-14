"""Train, evaluate, save and run the six forecasting model families."""

from .input import ForecastInput
from .models.persistence import PersistenceModel
from .models.ema30 import EMA30Model
from .models.smoothing import SmoothingModel
from .models.robust import RobustModel
from .models.four_branch import FourBranchModel
from .models.cvae import CVAEModel
from .models.base import evaluate
from .dataset import validate_split
from .utils.checkpoints import save_model, load_model


class TrainingModel:
    MODEL_TYPES = {
        "persistence": PersistenceModel,
        "ema30": EMA30Model,
        "smoothing": SmoothingModel,
        "robust": RobustModel,
        "four_branch": FourBranchModel,
        "cvae": CVAEModel,
    }

    def __init__(self, model_name, parameters=None):
        if model_name not in self.MODEL_TYPES:
            raise ValueError(
                f"Unknown model {model_name!r}; choose from {list(self.MODEL_TYPES)}"
            )
        self.model_name = model_name
        self.parameters = dict(parameters or {})
        self._models = {}
        self.training_report = None

    def __repr__(self):
        return f"TrainingModel(model_name={self.model_name!r}, parameters={self.parameters!r})"

    def to_dict(self):
        return {"model_name": self.model_name, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data):
        return cls(data["model_name"], data.get("parameters"))

    def get_data(self, v_id, maps, dates, masks, horizon_days, metadata=None):
        return ForecastInput(v_id, maps, dates, masks, horizon_days, metadata or {})

    def fit(self, training, validation=None):
        self._models = {}
        self.training_report = None
        if validation is not None:
            validate_split(training, validation)
        # Always construct a fresh estimator. fit() is never a warm start.
        estimator = self.MODEL_TYPES[self.model_name](**self.parameters)
        estimator.fit(training, validation)
        self._models = {self.model_name: estimator}
        self.training_report = dict(estimator.training_report)
        if validation is not None and self.model_name == "cvae":
            self.training_report["validation_metrics"] = evaluate(estimator, validation)
        return self

    train = fit

    def evaluate(self, dataset):
        if self.model_name not in self._models:
            raise ValueError("Fit or explicitly load a trained model before evaluation")
        return evaluate(self._models[self.model_name], dataset)

    def save(self, path, overwrite=False):
        if self.training_report is None:
            raise ValueError("Call fit() before saving a model")
        return save_model(
            path,
            {
                "format": "terra-trained-model",
                "format_version": 1,
                **self.to_dict(),
                "estimator": self._models[self.model_name],
                "training_report": self.training_report,
            },
            overwrite,
        )

    @classmethod
    def load(cls, path):
        payload = load_model(path)
        manager = cls(payload["model_name"], payload["parameters"])
        estimator = payload["estimator"]
        if not isinstance(estimator, cls.MODEL_TYPES[manager.model_name]):
            raise ValueError("Saved model family does not match its estimator")
        manager._models = {manager.model_name: estimator}
        manager.training_report = payload["training_report"]
        return manager

    def _predict(self, name, data):
        if name not in self._models:
            options = self.parameters if name == self.model_name else {}
            self._models[name] = self.MODEL_TYPES[name](**options)
        return self._models[name].predict(data)

    def predict(self, data):
        return self._predict(self.model_name, data)

    def persistence(self, data):
        return self._predict("persistence", data)

    def ema30(self, data):
        return self._predict("ema30", data)

    def smoothing(self, data):
        return self._predict("smoothing", data)

    def robust(self, data):
        return self._predict("robust", data)

    def four_branch(self, data):
        return self._predict("four_branch", data)

    def cvae(self, data):
        return self._predict("cvae", data)
