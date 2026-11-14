"""Compatibility exports; the coordinator implementation lives in src.training.

Python resolves src.models to the models package, which exports the same API.
"""

from .training import TrainingModel
from .models import (
    PersistenceModel,
    EMA30Model,
    SmoothingModel,
    RobustModel,
    FourBranchModel,
    CVAEModel,
)

__all__ = [
    "TrainingModel",
    "PersistenceModel",
    "EMA30Model",
    "SmoothingModel",
    "RobustModel",
    "FourBranchModel",
    "CVAEModel",
]
