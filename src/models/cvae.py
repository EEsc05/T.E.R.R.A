"""Preferred frozen CVAE and complete calibrated physical-map decoder."""

import numpy as np
import torch
from ..utils.checkpoints import verified_path, load_joblib, config
from ..preprocessing.interface import prepare, result
from ..preprocessing.shared import prepare_batch
from ..preprocessing.conditioning import conditioning, decode_samples
from .cvae_network import CVAE as CVAENetwork
from .ema30 import EMA30Model


class CVAEModel:
    def __init__(self, device="cpu", n_samples=32):
        self.device = torch.device(device)
        self.n_samples = n_samples
        selected = config("configs/v6_cvae_frozen_selection.json")
        self.model_id = "cvae_level15_seed20261109"
        if selected["selected_model"] != self.model_id:
            raise ValueError("The preferred checkpoint is not the frozen selection.")
        self.artifact = load_joblib("checkpoints/level15_probabilistic.joblib")
        checkpoint = torch.load(
            verified_path("checkpoints/" + self.model_id + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        self.training_seed = checkpoint["seed"]
        # Preserve surrounding RNG state while constructing then loading the frozen architecture.
        with torch.random.fork_rng(devices=[]):
            self.model = CVAENetwork(
                checkpoint["latent_dim"], checkpoint["condition_dim"]
            )
            self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device).eval()
        self.factor = float(
            config("configs/v6_frozen_scales.json")["scales"][self.model_id]
        )

    def predict(self, forecast_input):
        p = prepare(forecast_input, fixed_grid=True)
        m, a = prepare_batch(p)
        n = self.n_samples
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
            raise ValueError(
                "CVAE n_samples must be an integer>=2; original evaluation uses32."
            )
        seed = int(p.metadata.get("sampling_seed", 20261108 + int(p.days[-1])))
        c = conditioning(m, a)[0]
        base = EMA30Model().predict(forecast_input).prediction_mm
        devices = (
            []
            if self.device.type == "cpu"
            else [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        )
        with torch.random.fork_rng(devices=devices):
            raw = decode_samples(
                self.artifact,
                c,
                base,
                a["scale"][0],
                a["spacing"][0],
                p.horizon,
                n=n,
                seed=seed,
                model=self.model,
            )
        mean = np.mean(raw, 0)
        samples = mean + (raw - mean) * self.factor
        samples[:, ~p.seen] = np.nan
        mean = np.mean(samples, 0)
        std = np.std(samples, 0)
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
            self.model_id,
            dict(
                training_seed=self.training_seed,
                sampling_seed=seed,
                samples=n,
                device=str(self.device),
                calibration_factor=self.factor,
                calibration_targets="Original2022 validation",
                selection="Original2023development CRPS, frozen before global/Peru assessment",
                probability_semantics="Marginal model sample spread; not physical measurement uncertainty or guaranteed simultaneous-field coverage",
                alternative_coarse_model_used=False,
            ),
            samples_mm=samples,
            std_mm=std,
            intervals_mm=intervals,
        )
