"""New interface wrapper around copied original last-observation behavior."""

from ..preprocessing.interface import prepare, result


class PersistenceModel:
    def predict(self, forecast_input):
        p = prepare(forecast_input, min_history=1)
        # Original LocalFit persistence branch is exactly the per-pixel last.copy().
        forecast = p.last.copy()
        return result(p, forecast, "Persistence", "persistence")
