"""Common entry point for the six frozen forecasting models.

The historical TrainingModel name is retained for callers. These models perform
inference with existing parameters; they do not retrain or save placeholder files.
"""

from .input import ForecastInput
from .models.persistence import PersistenceModel
from .models.ema30 import EMA30Model
from .models.smoothing import SmoothingModel
from .models.robust import RobustModel
from .models.four_branch import FourBranchModel
from .models.cvae import CVAEModel


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

    def __repr__(self):
        return f"TrainingModel(model_name={self.model_name!r}, parameters={self.parameters!r})"

    def to_dict(self):
        return {"model_name": self.model_name, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data):
        return cls(data["model_name"], data.get("parameters"))

    def get_data(self, v_id, maps, dates, masks, horizon_days, metadata=None):
        return ForecastInput(v_id, maps, dates, masks, horizon_days, metadata or {})

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
