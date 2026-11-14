"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import numpy as np


def temporal(c, t, h, kind):
    if kind == "constant":
        return c[-1]
    if kind == "ema":
        w = np.exp(-np.log(2) * (t[-1] - t) / 30)
        return np.average(c, axis=0, weights=w)
    if kind == "kalman":
        level = c[0].copy()
        var = np.ones(c.shape[1])
        noise = np.maximum(np.median(abs(np.diff(c, axis=0)), axis=0) ** 2, 1.0)
        for j, dt in enumerate(np.diff(t), 1):
            var += 0.03 * noise * dt / 30
            gain = var / (var + noise)
            level += gain * (c[j] - level)
            var *= 1 - gain
        return level
    if kind == "robust_trend":
        slopes = np.array(
            [
                (c[j] - c[i]) / (t[j] - t[i])
                for i in range(len(t))
                for j in range(i + 1, len(t))
            ]
        )
        slope = np.median(slopes, 0)
        return np.median(c - (t - t[-1])[:, None] * slope, 0) + slope * h * np.exp(
            -h / 90
        )
    x = np.c_[np.ones(len(t)), (t - t[-1]) / 90]
    beta = np.linalg.solve(x.T @ x + np.diag([0, 4.0]), x.T @ c)
    return beta[0] + beta[1] * h / 90 * (np.exp(-h / 90) if kind == "damped" else 1)
