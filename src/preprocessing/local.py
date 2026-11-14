"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import numpy as np
from scipy.ndimage import uniform_filter
from .numerics import fill_past

NOISE = 1.0  # Default numerical noise floor in mm; not a loaded or fitted parameter.


def smooth(m, t, half):
    v = np.isfinite(m)
    w = np.exp(-np.log(2) * (t[-1] - t) / half)[:, None, None] * v
    return np.divide(
        (np.nan_to_num(m) * w).sum(0),
        w.sum(0),
        out=np.zeros(m.shape[1:]),
        where=w.sum(0) > 0,
    )


class LocalFit:
    def __init__(self, maps, times, cfg):
        k = cfg["history"]
        self.maps = maps[-k:]
        self.t = np.array(times[-k:], dtype=float)
        self.cfg = cfg
        self.f, self.valid = fill_past(self.maps)
        self.seen = self.valid.any(0)
        self.last = self.f[-1]
        self.shape = self.last.shape
        self.proxy = np.maximum(
            np.median(abs(np.diff(self.f, axis=0)), 0) * 1.4826 / np.sqrt(2), NOISE / 3
        )
        if cfg["family"] == "lowrank":
            a = self.f.reshape(k, -1).astype(float)
            if cfg["robust"]:
                med = np.median(a, 0)
                mad = np.maximum(np.median(abs(a - med), 0) * 1.4826, 1)
                a = np.clip(a, med - 3 * mad, med + 3 * mad)
            self.mean = a.mean(0)
            a = a - self.mean
            # Temporal Gram eigenbasis equals the nonzero spatial SVD projection.
            eig, u = np.linalg.eigh(a @ a.T)
            idx = np.argsort(eig)[::-1][: min(cfg["rank"], k - 1)]
            u = u[:, idx]
            s = np.sqrt(np.maximum(eig[idx], 0))
            vt = np.divide(
                u.T @ a,
                s[:, None],
                out=np.zeros((len(s), a.shape[1])),
                where=s[:, None] > 1e-10,
            )
            coeff = u * s
            x = np.stack([np.ones(k), (self.t - self.t[-1]) / 90], 1)
            beta = np.linalg.solve(x.T @ x + np.diag([0, cfg["alpha"]]), x.T @ coeff)
            self.intercept = (self.mean + beta[0] @ vt).reshape(self.shape)
            self.slope = (beta[1] @ vt).reshape(self.shape) / 90
        elif cfg["family"] == "quality":
            med = np.median(self.f, 0)
            mad = np.maximum(np.median(abs(self.f - med), 0) * 1.4826, NOISE / 3)
            clipped = np.clip(self.f, med - 3 * mad, med + 3 * mad)
            w = (
                np.exp(-np.log(2) * (self.t[-1] - self.t) / cfg["half"])[:, None, None]
                * self.valid
            )
            if cfg.get("weighted"):
                w = w / (1 + ((self.f - med) / (3 * mad)) ** 2)
            self.level = np.divide(
                (w * clipped).sum(0),
                w.sum(0),
                out=np.zeros(self.shape),
                where=w.sum(0) > 0,
            )
            rates = np.diff(clipped, axis=0) / np.diff(self.t)[:, None, None]
            self.rate = np.median(rates, 0)

    def predict(self, h):
        c = self.cfg
        fam = c["family"]
        if fam == "persistence":
            out = self.last.copy()
        elif fam == "ema":
            out = smooth(self.maps, self.t, c["half"])
        elif fam == "lowrank":
            out = self.intercept + self.slope * h * (
                np.exp(-h / 90) if c["method"] == "damped90" else 1
            )
        elif fam == "quality":
            out = self.level + c["gain"] * self.rate * h * np.exp(-h / 60)
        else:
            raise ValueError(fam)
        out = np.array(out, dtype="float32")
        out[~self.seen] = np.nan
        return out


def features(m, t, h, sp):
    f, v = fill_past(m[-6:])
    t = np.array(t[-6:], dtype=float)
    last = f[-1]
    rates = np.diff(f, axis=0) / np.diff(t)[:, None, None]
    gy, gx = np.gradient(last, *sp)
    constant = lambda z: np.full(last.shape, z)
    channels = [
        rates[-1] * h,
        np.median(rates, 0) * h,
        (f[-1] - f[0]) / max(t[-1] - t[0], 1) * h,
        (rates[-1] - rates[-2]) * h * h / 180,
        smooth(m[-6:], t, 120) - last,
        f.std(0),
        gy,
        gx,
        uniform_filter(last, size=5) - last,
        v.mean(0),
        constant(h),
        constant(np.log1p(h)),
        constant(t[-1] - t[0]),
        constant(np.median(np.diff(t))),
        constant(np.log(sp[0])),
        constant(np.log(sp[1])),
    ]
    return np.stack(channels, -1).reshape(-1, len(channels))
