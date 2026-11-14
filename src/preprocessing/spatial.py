"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import numpy as np
from .numerics import fill_past
from .temporal import temporal as latent_temporal


def encode(art, m, a, ids, targets=False):
    xs = []
    ys = []
    bs = []
    coeffs = []
    rec = []
    for i in ids:
        basis = (
            art["geometry"][str(a["context"][i, 0, 16] < 0)]
            if art["representation"] == "geometry4" and a["context"][i, 0, 19] > 0
            else art["basis"]
        )
        f, v = fill_past(a["history"][i])
        h = (f - a["last"][i]) / a["scale"][i]
        h = np.nan_to_num(h).reshape(6, -1)
        c = h @ basis.T
        t = a["times"][i]
        row = np.r_[
            c.ravel(),
            m.iloc[i].horizon_days / 180,
            np.log1p(m.iloc[i].horizon_days) / 6,
            np.ptp(t) / 365,
            np.median(np.diff(t)) / 30,
            np.log(a["spacing"][i]),
            a["context"][i, 0, 16:20],
            np.log1p(a["scale"][i]),
        ]
        xs.append(row)
        bs.append(basis)
        coeffs.append(c)
        rec.append(float(abs(h - c @ basis).mean() * a["scale"][i]))
        if targets:
            valid = (
                np.isfinite(a["truth"][i]).ravel() & np.isfinite(a["last"][i]).ravel()
            )
            b = basis[:, valid]
            dy = ((a["truth"][i] - a["last"][i]) / a["scale"][i]).ravel()[valid]
            ys.append(np.linalg.solve(b @ b.T + np.eye(len(b)) * 0.001, b @ dy))
    return np.array(xs), np.array(ys), np.array(bs), np.array(coeffs), np.array(rec)


def predict(art, m, a, ids):
    x, _, basis, coeff, recon = encode(art, m, a, ids, False)
    method = art["method"]
    if method in ["constant", "linear", "ema", "kalman"]:
        zs = np.array(
            [
                latent_temporal(coeff[j], a["times"][i], m.iloc[i].horizon_days, method)
                for j, i in enumerate(ids)
            ]
        )
    else:
        zs = art["target_scaler"].inverse_transform(
            art["head"].predict(art["scaler"].transform(x))
        )
    pred = (
        np.einsum("nr,nrp->np", zs, basis).reshape(-1, 32, 32)
        * a["scale"][ids, None, None]
        + a["last"][ids]
    )
    return pred.astype("float32"), recon
