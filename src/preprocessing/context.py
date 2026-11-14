"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import numpy as np


def predict(art, a, x, ids, parents):
    out = []
    for i in ids:
        z = (
            art["pipeline"].predict(x[i][:, art["columns"]]).reshape(32, 32)
            * art["target_scale"]
            + art["target_offset"]
            + parents[art["config"]["target"]][i]
        )
        z[~np.isfinite(a["last"][i])] = np.nan
        out.append(z)
    return np.array(out, dtype="float32")
