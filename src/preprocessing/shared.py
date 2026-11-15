"""New single-input adapter replicating examples() normalization/feature layout."""

import numpy as np
import pandas as pd
from .numerics import fill_past
from .local import features


def prepare_batch(prep):
    md = prep.metadata
    if "spacing_km" not in md:
        raise ValueError(
            "metadata.spacing_km=[row_spacing_km,column_spacing_km] is required."
        )
    sp = np.asarray(md["spacing_km"], dtype=float)
    if sp.shape != (2,) or not np.isfinite(sp).all() or (sp <= 0).any():
        raise ValueError(
            "spacing_km must contain two finite positive values in kilometres."
        )
    if "geometry_available" not in md or not isinstance(
        md["geometry_available"], (bool, np.bool_)
    ):
        raise ValueError(
            "Explicit metadata.geometry_available=True/False is required; do not invent known geometry."
        )
    if md["geometry_available"] and "los_vector" not in md:
        raise ValueError(
            "Known geometry requires metadata.los_vector=[east,north,up] static LOS projection means."
        )
    los = np.asarray(md.get("los_vector", [0.0, 0.0, 0.0]), dtype=float)
    if los.shape != (3,) or not np.isfinite(los).all():
        raise ValueError(
            "los_vector must contain three finite static projection means."
        )
    if md.get("grid_convention") not in ("native_center32", "aoi_block_mean32"):
        raise ValueError(
            "Declare grid_convention as native_center32 or aoi_block_mean32; the model does not resample maps."
        )
    m = prep.raw
    t = prep.days
    f, v = fill_past(m)
    s = max(float(np.nanquantile(abs(m[-1] - m[0]), 0.75)), 1.0)
    s = 1.0 if not np.isfinite(s) else s
    scalars = [
        prep.horizon / 180,
        np.log1p(prep.horizon) / 6,
        (t[-1] - t[0]) / 365,
        np.median(np.diff(t)) / 30,
        *np.log(sp),
        *los,
        np.log1p(s) / 5,
        float(md["geometry_available"]),
    ]
    channels = np.concatenate(
        [
            (f - f[-1]) / s,
            v.astype("float32"),
            np.broadcast_to(np.array(scalars)[:, None, None], (len(scalars), 32, 32)),
        ]
    )
    xx = features(m, t, prep.horizon, sp)
    xx[:, :9] /= s
    xx = np.c_[
        xx,
        np.tile(los, (len(xx), 1)),
        np.full(len(xx), float(md["geometry_available"])),
    ]
    a = dict(
        history=m[None],
        times=t[None],
        spacing=sp[None],
        last=prep.last[None],
        scale=np.array([s]),
        context=xx[None],
        cnn=np.nan_to_num(channels)[None].astype("float32"),
    )
    meta = pd.DataFrame(
        [
            dict(
                horizon_days=prep.horizon,
                origin_day=float(t[-1]),
                volcano=prep.volcano_id,
            )
        ]
    )
    return meta, a
