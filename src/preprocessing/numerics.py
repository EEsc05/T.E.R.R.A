"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import numpy as np


def fill_past(maps):
    valid = np.isfinite(maps)
    filled = np.zeros_like(maps, dtype="float32")
    for i in range(len(maps)):
        filled[i] = np.where(valid[i], maps[i], filled[i - 1] if i else 0)
    return filled, valid
