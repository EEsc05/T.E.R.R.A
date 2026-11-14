"""Exact robust rank3 local spatial fit and ridge coefficient forecast."""

from .base import LocalModel
from ..preprocessing.interface import prepare, result
from ..preprocessing.local import LocalFit


class RobustModel(LocalModel):
    def predict(self, forecast_input):
        p = prepare(forecast_input)
        forecast = LocalFit(
            p.raw,
            p.days,
            dict(
                family="lowrank",
                history=6,
                rank=3,
                robust=True,
                alpha=4.0,
                method="linear",
            ),
        ).predict(p.horizon)
        return result(
            p,
            forecast,
            "Robust low-rank",
            "lr_h6_r3_rob1_linear",
            dict(
                valid_zero_handling="Explicit mask governs validity; finite zero is observed. Original volcanic64 make_history zero-as-missing must be encoded in valid_masks if replaying that cohort."
            ),
        )
