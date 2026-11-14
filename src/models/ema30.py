"""30-day half-life baseline, fitted locally to each supplied history."""

from .base import LocalModel
from ..preprocessing.interface import prepare, result
from ..preprocessing.local import LocalFit


class EMA30Model(LocalModel):
    def predict(self, forecast_input):
        p = prepare(forecast_input)
        forecast = LocalFit(
            p.raw, p.days, dict(family="ema", history=6, half=30)
        ).predict(p.horizon)
        return result(p, forecast, "EMA30", "ema_half30")
