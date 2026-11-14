"""Exact research definitions; module imports adapted for isolated inference. See hackathon_stage1/SOURCE_MANIFEST.json for source provenance."""

import numpy as np
from scipy.fft import dctn, idctn
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning

SEED = 20261106


class SpatialBasis:
    def __init__(self, kind="block_pca", rank=8):
        self.kind = kind
        self.rank = rank

    def fit(self, images):
        a = np.nan_to_num(np.asarray(images, float))
        self.mean = a.mean(0)
        self.models = []
        if self.kind in ["block_pca", "robust_block_pca"]:
            if self.kind == "robust_block_pca":
                center = np.median(a, axis=0)
                scale = np.median(abs(a - center), axis=0) * 1.4826 + 0.1
                a = np.clip(a, center - 4 * scale, center + 4 * scale)
            for yy in range(0, 32, 8):
                for xx in range(0, 32, 8):
                    self.models.append(
                        PCA(n_components=self.rank, random_state=SEED).fit(
                            a[:, yy : yy + 8, xx : xx + 8].reshape(len(a), 64)
                        )
                    )
        elif self.kind in ["pca", "robust_pca"]:
            if self.kind == "robust_pca":
                center = np.median(a, axis=0)
                scale = np.median(abs(a - center), axis=0) * 1.4826 + 0.1
                a = np.clip(a, center - 4 * scale, center + 4 * scale)
            self.model = PCA(
                n_components=min(self.rank, len(a) - 2),
                svd_solver="randomized",
                random_state=SEED,
            ).fit(a.reshape(len(a), -1))
        elif self.kind == "sparse_dictionary":
            self.model = MiniBatchDictionaryLearning(
                n_components=self.rank,
                batch_size=64,
                max_iter=40,
                random_state=SEED,
                transform_algorithm="lasso_lars",
                transform_alpha=0.1,
            ).fit(a.reshape(len(a), -1))
        return self

    def encode(self, a):
        a = np.nan_to_num(np.asarray(a, float))
        n = len(a)
        if self.kind in ["block_pca", "robust_block_pca"]:
            return np.concatenate(
                [
                    m.transform(a[:, yy : yy + 8, xx : xx + 8].reshape(n, 64))
                    for m, (yy, xx) in zip(
                        self.models,
                        [(y, x) for y in range(0, 32, 8) for x in range(0, 32, 8)],
                    )
                ],
                axis=1,
            )
        if self.kind in ["pca", "robust_pca", "sparse_dictionary"]:
            return self.model.transform(a.reshape(n, -1))
        if self.kind == "dct":
            return dctn(a - self.mean, axes=(-2, -1), norm="ortho")[
                :, : self.rank, : self.rank
            ].reshape(n, -1)
        if self.kind == "haar":
            # Full two-level Haar analysis; retain coarse and spatially localized details at a fixed rank.
            c = a - self.mean
            out = []
            for _ in range(2):
                aa, bb, cc, dd = (
                    c[:, ::2, ::2],
                    c[:, ::2, 1::2],
                    c[:, 1::2, ::2],
                    c[:, 1::2, 1::2],
                )
                out.append(
                    np.stack(
                        [
                            (aa - bb + cc - dd) / 2,
                            (aa + bb - cc - dd) / 2,
                            (aa - bb - cc + dd) / 2,
                        ],
                        1,
                    )
                )
                c = (aa + bb + cc + dd) / 2
            return np.concatenate(
                [c.reshape(n, -1), out[-1].reshape(n, -1)], axis=1
            )  # 256coefficients; omit finest detail
        if self.kind == "multiscale":
            # Coarse DCT plus local residual PCA, preserving locality rather than a global rank truncation.
            return self.sub.encode(a)

    def decode(self, z):
        n = len(z)
        if self.kind in ["block_pca", "robust_block_pca"]:
            out = np.zeros((n, 32, 32))
            for i, (m, (yy, xx)) in enumerate(
                zip(
                    self.models,
                    [(y, x) for y in range(0, 32, 8) for x in range(0, 32, 8)],
                )
            ):
                out[:, yy : yy + 8, xx : xx + 8] = m.inverse_transform(
                    z[:, i * self.rank : (i + 1) * self.rank]
                ).reshape(n, 8, 8)
            return out
        if self.kind in ["pca", "robust_pca"]:
            return self.model.inverse_transform(z).reshape(n, 32, 32)
        if self.kind == "sparse_dictionary":
            return (z @ self.model.components_).reshape(n, 32, 32)
        if self.kind == "dct":
            a = np.zeros((n, 32, 32))
            a[:, : self.rank, : self.rank] = z.reshape(n, self.rank, self.rank)
            return idctn(a, axes=(-2, -1), norm="ortho") + self.mean
        if self.kind == "haar":
            c = z[:, :64].reshape(n, 8, 8)
            details = z[:, 64:].reshape(n, 3, 8, 8)
            for level in range(2):
                hh, vv, dd = details[:, 0], details[:, 1], details[:, 2]
                out = np.zeros((n, c.shape[1] * 2, c.shape[2] * 2))
                out[:, ::2, ::2] = (c + hh + vv + dd) / 2
                out[:, ::2, 1::2] = (c - hh + vv - dd) / 2
                out[:, 1::2, ::2] = (c + hh - vv - dd) / 2
                out[:, 1::2, 1::2] = (c - hh - vv + dd) / 2
                c = out
                details = np.zeros((n, 3, c.shape[1], c.shape[2]))
            return c + self.mean
