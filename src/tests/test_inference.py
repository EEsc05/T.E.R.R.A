"""Training from scratch, held-out evaluation and explicit model persistence."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src import TrainingModel, CVAEModel, FourBranchModel, TrainingDataset
from src.dataset import validate_split
from src.models.cvae_network import CVAE as Network
from .helpers import dataset, save_dataset


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = dataset()
        cls.validation = dataset(3, start="2023-01-01")
        cls.models = {}
        cls.models["four_branch"] = TrainingModel(
            "four_branch", {"max_context_pixels": 128}
        ).fit(cls.training, cls.validation)
        cls.models["cvae"] = TrainingModel(
            "cvae", {"epochs": 4, "batch_size": 3, "basis_rank": 2, "n_samples": 4}
        ).fit(cls.training, cls.validation)

    def test_constructors_do_not_load_weights(self):
        with patch(
            "joblib.load", side_effect=AssertionError("Implicit checkpoint load")
        ):
            for model in [CVAEModel(), FourBranchModel()]:
                self.assertFalse(model.is_fitted)
                with self.assertRaisesRegex(ValueError, "untrained"):
                    model.predict(self.training.inputs[0])

    def test_all_six_models_fit_and_forecast(self):
        for name in TrainingModel.MODEL_TYPES:
            manager = self.models.get(name) or TrainingModel(name).fit(
                self.training, self.validation
            )
            r = manager.predict(self.validation.inputs[0])
            self.assertEqual(r.prediction_mm.shape, (32, 32))
            self.assertTrue(np.isfinite(r.prediction_mm).all())
            self.assertEqual(r.target_date, self.validation.target_dates[0])
            self.assertTrue(np.isfinite(manager.evaluate(self.validation)["rmse_mm"]))
            if name == "cvae":
                self.assertEqual(r.samples_mm.shape, (4, 32, 32))
                np.testing.assert_allclose(r.prediction_mm, r.samples_mm.mean(0))
            elif name != "four_branch":
                self.assertFalse(manager.training_report["global_weights_learned"])

    def test_cvae_updates_random_initial_weights(self):
        trained = self.models["cvae"]._models["cvae"]
        with torch.random.fork_rng():
            torch.manual_seed(trained.seed)
            fresh = Network(32, 75)
        self.assertTrue(
            any(
                not torch.equal(a.cpu(), b.cpu())
                for a, b in zip(trained.model.parameters(), fresh.parameters())
            )
        )
        self.assertEqual(len(trained.training_report["curves"]), 4)
        self.assertFalse(trained.training_report["uncertainty_calibrated"])

    def test_preprocessors_fit_only_on_training(self):
        trained = self.models["cvae"]._models["cvae"]
        conditions, residuals, valid = trained._examples(self.training)
        np.testing.assert_allclose(
            trained.artifact["condition_scaler"].center_, np.median(conditions, 0)
        )
        self.assertEqual(
            trained.artifact["latent_scaler"].n_samples_seen_, len(self.training)
        )
        self.assertEqual(
            trained.artifact["basis"].models[0].n_samples_, len(self.training)
        )

    def test_four_branch_fits_actual_targets(self):
        shifted = copy.deepcopy(self.training)
        shifted.targets += 30
        model = FourBranchModel(max_context_pixels=128).fit(shifted)
        fitted = self.models["four_branch"]._models["four_branch"]
        x = np.zeros((3, 31))
        self.assertGreater(
            float(
                np.mean(
                    model.context["pipeline"].predict(x) * model.context["target_scale"]
                )
            ),
            float(
                np.mean(
                    fitted.context["pipeline"].predict(x)
                    * fitted.context["target_scale"]
                )
            )
            + 10,
        )
        np.testing.assert_allclose(fitted.weights.sum(1), 1)
        self.assertTrue((fitted.weights >= 0).all())
        self.assertIn("calibration_metrics", fitted.training_report)

    def test_fit_does_not_mutate_inputs_or_global_rng(self):
        original = self.training.inputs[0].maps.copy()
        before = torch.random.get_rng_state().clone()
        TrainingModel("cvae", {"epochs": 1, "basis_rank": 1}).fit(
            self.training, self.validation
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        np.testing.assert_array_equal(original, self.training.inputs[0].maps)

    def test_repeatable_sampling_and_masks(self):
        model = self.models["cvae"]
        inp = copy.deepcopy(self.validation.inputs[0])
        inp.valid_masks[:, 0, 0] = False
        before = torch.random.get_rng_state().clone()
        a = model.predict(inp)
        b = model.predict(inp)
        np.testing.assert_array_equal(a.samples_mm, b.samples_mm)
        self.assertTrue(np.isnan(a.prediction_mm[0, 0]))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_save_load_round_trip_without_bundled_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, model in self.models.items():
                path = Path(temp) / f"{name}.joblib"
                model.save(path)
                loaded = TrainingModel.load(path)
                np.testing.assert_array_equal(
                    model.predict(self.validation.inputs[0]).prediction_mm,
                    loaded.predict(self.validation.inputs[0]).prediction_mm,
                )
                with self.assertRaises(FileExistsError):
                    model.save(path)
            with self.assertRaisesRegex(ValueError, "fit"):
                TrainingModel("cvae").save(Path(temp) / "untrained.joblib")

    def test_training_data_contract_and_temporal_split(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.npz"
            save_dataset(path, self.training)
            restored = TrainingDataset.from_npz(path)
            np.testing.assert_array_equal(restored.targets, self.training.targets)
            self.assertEqual(restored.inputs[0].horizon_days, 30)
        with self.assertRaisesRegex(ValueError, "purged temporal split"):
            validate_split(self.training, self.training)
        with self.assertRaises(ValueError):
            TrainingDataset(
                self.training.inputs,
                self.training.targets,
                np.zeros_like(self.training.target_masks),
                self.training.target_dates,
            )


if __name__ == "__main__":
    unittest.main()
