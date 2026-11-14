"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SEED = 42


def conditioning(meta, a):
    change = (
        np.nan_to_num(a["history"][:, -1] - a["history"][:, 0])
        / a["scale"][:, None, None]
    )
    coarse = change.reshape(-1, 8, 4, 8, 4).mean((2, 4))
    return np.c_[coarse.reshape(len(meta), -1), a["cnn"][:, -11:, 0, 0]]


def decode_samples(
    artifact, c, base, scale, sp, h, n=32, seed=SEED, model=None, kind="gaussian"
):
    rng = np.random.default_rng(seed)
    cx = np.clip(artifact["condition_scaler"].transform(c[None]), -6, 6)
    basis = artifact["basis"]
    ls = artifact["latent_scaler"]
    if model is None:
        raise ValueError("A model fitted on your training data is required")
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    with torch.no_grad():
        latent = (
            model.sample(torch.tensor(cx, dtype=torch.float32, device=device), n)
            .cpu()
            .numpy()
        )
    maps = basis.decode(ls.inverse_transform(latent))
    white = rng.normal(size=(n, 32, 32))
    for j in range(n):
        white[j] = gaussian_filter(
            white[j], sigma=max(0.5, 1.0 / max(sp)), mode="reflect"
        )
        white[j] /= max(white[j].std(), 1e-6)
    samples = base[None] + (maps + white * artifact["noise"]) * scale
    return samples.astype("float32")
