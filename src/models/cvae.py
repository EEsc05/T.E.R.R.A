"""Conditional VAE with training-only spatial basis and fresh neural weights."""

import copy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from ..dataset import validate_split
from ..preprocessing.interface import prepare, result
from ..preprocessing.shared import prepare_batch
from ..preprocessing.conditioning import conditioning, decode_samples
from ..preprocessing.basis import SpatialBasis
from .cvae_network import CVAE as CVAENetwork
from .ema30 import EMA30Model


class CVAEModel:
    def __init__(
        self,
        device="cpu",
        n_samples=32,
        seed=42,
        epochs=80,
        batch_size=64,
        learning_rate=0.001,
        patience=15,
        basis_rank=8,
    ):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("CVAE supports cpu or cuda devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; use device='cpu' in this environment"
            )
        self.n_samples = n_samples
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.basis_rank = basis_rank
        self.is_fitted = False

    def _examples(self, dataset):
        conditions, residuals, valid_masks = [], [], []
        for inp, target, mask in zip(
            dataset.inputs, dataset.targets, dataset.target_masks
        ):
            p = prepare(inp, fixed_grid=True)
            m, a = prepare_batch(p)
            conditions.append(conditioning(m, a)[0])
            base = EMA30Model().predict(inp).prediction_mm
            valid = mask & p.seen & np.isfinite(target)
            residuals.append(np.where(valid, (target - base) / a["scale"][0], np.nan))
            valid_masks.append(valid)
        return np.asarray(conditions), np.asarray(residuals), np.asarray(valid_masks)

    def fit(self, training, validation=None):
        self.is_fitted = False
        if len(training) < 2:
            raise ValueError("CVAE training requires at least two examples")
        if (
            min(self.epochs, self.batch_size, self.patience, self.basis_rank) < 1
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "epochs, batch_size, patience, basis_rank and learning_rate must be positive"
            )
        if validation is not None:
            validate_split(training, validation)
        train_c, train_r, valid = self._examples(training)
        # Missing target pixels are imputed with a training-only mean residual for PCA.
        counts = valid.sum(0)
        imputation = np.divide(
            np.nansum(train_r, 0), counts, out=np.zeros((32, 32)), where=counts > 0
        )
        train_r = np.where(valid, train_r, imputation)
        rank = min(self.basis_rank, len(training) - 1, 64)
        basis = SpatialBasis("block_pca", rank).fit(train_r)
        latent_scaler = StandardScaler().fit(basis.encode(train_r))
        condition_scaler = RobustScaler().fit(train_c)
        z = latent_scaler.transform(basis.encode(train_r)).astype("float32")
        c = np.clip(condition_scaler.transform(train_c), -6, 6).astype("float32")
        if validation is not None:
            vc, vr, vmask = self._examples(validation)
            vr = np.where(vmask, vr, imputation)
            vz = latent_scaler.transform(basis.encode(vr)).astype("float32")
            vc = np.clip(condition_scaler.transform(vc), -6, 6).astype("float32")
        else:
            vz, vc = z, c
        reconstruction = basis.decode(basis.encode(train_r))
        noise = np.maximum(
            np.sqrt(
                np.divide(
                    np.sum(np.where(valid, (train_r - reconstruction) ** 2, 0), 0),
                    counts,
                    out=np.zeros((32, 32)),
                    where=counts > 0,
                )
            ),
            0.05,
        )
        self.artifact = {
            "basis": basis,
            "latent_scaler": latent_scaler,
            "condition_scaler": condition_scaler,
            "noise": noise,
            "imputation": imputation,
        }
        devices = [] if self.device.type == "cpu" else [self.device.index or 0]
        rng = np.random.default_rng(self.seed)
        curves = []
        best = None
        wait = 0
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            self.model = CVAENetwork(z.shape[1], c.shape[1]).to(self.device)
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001
            )
            zz = torch.as_tensor(z, device=self.device)
            cc = torch.as_tensor(c, device=self.device)
            vzz = torch.as_tensor(vz, device=self.device)
            vcc = torch.as_tensor(vc, device=self.device)
            for epoch in range(self.epochs):
                self.model.train()
                losses = []
                order = rng.permutation(len(z))
                for start in range(0, len(z), self.batch_size):
                    ix = order[start : start + self.batch_size]
                    optimizer.zero_grad()
                    loss, *_ = self.model.loss(zz[ix], cc[ix])
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "Nonfinite training loss; check inputs and learning rate"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                self.model.eval()
                # Identical validation noise each epoch gives comparable stochastic losses.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(self.seed + 1000)
                    with torch.no_grad():
                        score = float(self.model.loss(vzz, vcc)[0].cpu())
                if not np.isfinite(score):
                    raise ValueError("Nonfinite validation loss")
                curves.append(
                    {
                        "epoch": epoch + 1,
                        "training_loss": float(np.mean(losses)),
                        "validation_loss"
                        if validation is not None
                        else "training_selection_loss": score,
                    }
                )
                if best is None or score < best[0] - 1e-4:
                    best = (score, copy.deepcopy(self.model.state_dict()), epoch + 1)
                    wait = 0
                else:
                    wait += 1
                if validation is not None and wait >= self.patience:
                    break
        self.model.load_state_dict(best[1])
        self.model.eval()
        self.is_fitted = True
        self.training_report = {
            "training_examples": len(training),
            "validation_examples": len(validation) if validation else 0,
            "seed": self.seed,
            "epochs_completed": len(curves),
            "best_epoch": best[2],
            "basis_rank_per_block": rank,
            "curves": curves,
            "missing_target_handling": "Training-only per-pixel residual mean imputation for PCA",
            "uncertainty_calibrated": False,
        }
        return self

    train = fit

    def predict(self, forecast_input):
        if not self.is_fitted:
            raise ValueError(
                "CVAEModel is untrained; call fit() or explicitly load your own trained model"
            )
        if (
            not isinstance(self.n_samples, (int, np.integer))
            or isinstance(self.n_samples, bool)
            or self.n_samples < 2
        ):
            raise ValueError("n_samples must be an integer >=2")
        p = prepare(forecast_input, fixed_grid=True)
        m, a = prepare_batch(p)
        seed = int(p.metadata.get("sampling_seed", self.seed + int(p.days[-1])))
        base = EMA30Model().predict(forecast_input).prediction_mm
        devices = [] if self.device.type == "cpu" else [self.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            samples = decode_samples(
                self.artifact,
                conditioning(m, a)[0],
                base,
                a["scale"][0],
                a["spacing"][0],
                p.horizon,
                n=self.n_samples,
                seed=seed,
                model=self.model,
            )
        samples[:, ~p.seen] = np.nan
        mean = samples.mean(0)
        std = samples.std(0)
        intervals = {
            str(level): tuple(
                np.quantile(
                    samples, [(1 - level / 100) / 2, 1 - (1 - level / 100) / 2], axis=0
                )
            )
            for level in [50, 80, 90, 95]
        }
        return result(
            p,
            mean,
            "Conditional VAE",
            "trained_from_scratch",
            {
                "training_seed": self.seed,
                "sampling_seed": seed,
                "samples": self.n_samples,
                "device": str(self.device),
                "uncertainty_calibrated": False,
            },
            samples_mm=samples,
            std_mm=std,
            intervals_mm=intervals,
        )
