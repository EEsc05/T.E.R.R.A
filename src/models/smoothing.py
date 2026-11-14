"""Quality-aware local smoothing baseline."""

from .base import LocalModel
from ..preprocessing.interface import prepare, result
from ..preprocessing.local import LocalFit


class SmoothingModel(LocalModel):
    def predict(self, forecast_input):
        p = prepare(forecast_input)
        forecast = LocalFit(
            p.raw,
            p.days,
            dict(family="quality", history=6, half=30, gain=0, weighted=False),
        ).predict(p.horizon)
        return result(p, forecast, "Quality-aware smoothing", "quality_h6_half30_w0")
