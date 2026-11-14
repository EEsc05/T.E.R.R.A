"""Exact research definitions; module imports adapted for isolated inference. See hackathon_stage1/SOURCE_MANIFEST.json for source provenance."""

import numpy as np
from .local import fill_past, smooth


def build(m, a):
    x = a["context"].copy()
    x[:, :, :9] *= a["scale"][:, None, None]
    extras = []
    for i, r in m.iterrows():
        raw = a["history"][i]
        t = a["times"][i]
        f, v = fill_past(raw)
        last = f[-1]
        med = np.median(f, 0)
        mad = np.median(abs(f - med), 0)
        trend = np.median(
            np.array(
                [
                    (f[k] - f[j]) / (t[k] - t[j])
                    for j in range(6)
                    for k in range(j + 1, 6)
                ]
            ),
            0,
        )
        z = f.reshape(6, -1).astype(float)
        z -= z.mean(0)
        eig, u = np.linalg.eigh(z @ z.T)
        ii = np.argsort(eig)[::-1][:3]
        ss = np.sqrt(np.maximum(eig[ii], 0))
        vt = np.divide(
            u[:, ii].T @ z,
            ss[:, None],
            out=np.zeros((3, 1024)),
            where=ss[:, None] > 1e-8,
        )
        coeff = z @ vt.T
        components = coeff[-1, :, None] * vt
        recon = z.mean(0) + coeff @ vt
        residual = (z[-1] - recon[-1]).reshape(32, 32)
        extra = [
            last,
            med,
            mad,
            smooth(raw, t, 30) - last,
            smooth(raw, t, 120) - last,
            trend * r.horizon_days,
            v.sum(0),
            residual,
            *components.reshape(3, 32, 32),
        ]
        extras.append(np.stack(extra, -1).reshape(1024, -1))
    return np.nan_to_num(np.concatenate([x, np.array(extras)], axis=-1)).astype(
        "float32"
    )
