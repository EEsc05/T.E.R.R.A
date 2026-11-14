"""Exact research definitions; module imports adapted for isolated inference. See hackathon_stage1/SOURCE_MANIFEST.json for source provenance."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SEED = 20261108


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
    z = artifact["head"].predict(cx)[0]
    rank = len(z)
    if kind == "independent_gaussian":
        binid = min(int(h // 30), 5)
        std = artifact["independent_std"][binid]
        samples = base[None] + rng.normal(size=(n, 32, 32)) * std * scale
    else:
        if model is not None:
            device = next(model.parameters()).device
            torch.manual_seed(seed)
            with torch.no_grad():
                latent = (
                    model.sample(
                        torch.tensor(cx, dtype=torch.float32, device=device), n
                    )
                    .cpu()
                    .numpy()
                )
        else:
            eps = rng.normal(size=(n, rank)) @ artifact["chol"].T
            if kind == "student":
                eps /= np.sqrt(rng.chisquare(4, (n, 1)) / 2)
            latent = z + eps
        maps = basis.decode(ls.inverse_transform(latent))
        white = rng.normal(size=(n, 32, 32))
        for j in range(n):
            white[j] = gaussian_filter(
                white[j], sigma=max(0.5, 1.0 / max(sp)), mode="reflect"
            )
            white[j] /= max(white[j].std(), 1e-6)
        samples = base[None] + (maps + white * artifact["noise"]) * scale
    return samples.astype("float32")
