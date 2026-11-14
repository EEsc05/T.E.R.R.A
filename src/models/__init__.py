"""Forecast models, organized by model family."""

from .persistence import PersistenceModel
from .ema30 import EMA30Model
from .smoothing import SmoothingModel
from .robust import RobustModel
from .four_branch import FourBranchModel
from .cvae import CVAEModel

__all__ = [
    "PersistenceModel",
    "EMA30Model",
    "SmoothingModel",
    "RobustModel",
    "FourBranchModel",
    "CVAEModel",
    "TrainingModel",
]


def __getattr__(name):
    if name == "TrainingModel":
        from ..training import TrainingModel

        return TrainingModel
    raise AttributeError(name)
