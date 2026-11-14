"""Real-input equivalence plus interface behavior; no forecast-performance study."""

import importlib, json, unittest, tempfile, subprocess, sys, os, shutil, hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .. import (
    ForecastInput,
    PersistenceModel as Persistence,
    EMA30Model as EMA30,
    SmoothingModel as QualityAwareSmoothing,
    RobustModel as RobustLowRank,
    FourBranchModel as FourBranchHybrid,
    CVAEModel as CVAE,
)
from ..preprocessing.interface import prepare
from ..preprocessing.shared import prepare_batch
from ..preprocessing.features import build
from ..preprocessing.conditioning import conditioning

PKG = Path(__file__).resolve().parents[2] / "hackathon_stage1"


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = np.load(PKG / "tests/fixtures/real_inputs.npz")
        cls.expected = np.load(PKG / "tests/fixtures/original_outputs.npz")
        cls.cases = json.loads((PKG / "tests/fixtures/cases.json").read_text())
        cls.models = {
            "persistence": Persistence(),
            "EMA30": EMA30(),
            "quality": QualityAwareSmoothing(),
            "lowrank": RobustLowRank(),
            "hybrid": FourBranchHybrid(),
            "cvae": CVAE(),
        }

    def inp(self, k=0):
        c = self.cases[k]
        return ForecastInput(
            c["volcano_id"],
            self.inputs["maps"][k].copy(),
            self.inputs["dates"][k].copy(),
            self.inputs["valid_masks"][k].copy(),
            c["horizon_days"],
            dict(c["metadata"]),
        )

    def test_01_copied_sources_checkpoints_have_unchanged_bytes(self):
        manifest = json.loads((PKG / "SOURCE_MANIFEST.json").read_text())
        for row in manifest["copied_files"]:
            with (PKG / row["copied_path"]).open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            self.assertEqual(actual, row["sha256"], row["copied_path"])

    def test_02_all_six_modules_import(self):
        for module in [
            "persistence",
            "ema30",
            "smoothing",
            "robust",
            "four_branch",
            "cvae",
        ]:
            importlib.import_module("src.models." + module)
        self.assertFalse(
            any(m.startswith("volcanic_forecasting_v2") for m in sys.modules),
            "Original namespace must never be imported for inference.",
        )

    def test_03_preprocessing_feature_order_normalization_exact(self):
        for k in range(len(self.cases)):
            p = prepare(self.inp(k), fixed_grid=True)
            m, a = prepare_batch(p)
            for key in ["context", "cnn", "last", "scale"]:
                np.testing.assert_allclose(
                    a[key][0],
                    self.expected[key][k],
                    atol=1e-7,
                    equal_nan=True,
                    err_msg=key,
                )
            np.testing.assert_allclose(
                build(m, a)[0], self.expected["features"][k], atol=1e-7, equal_nan=True
            )
            np.testing.assert_allclose(
                conditioning(m, a)[0],
                self.expected["condition"][k],
                atol=1e-7,
                equal_nan=True,
            )
            self.assertNotIn("truth", a)

    def test_04_five_deterministic_models_match_saved_original_maps(self):
        for k, c in enumerate(self.cases):
            for name, model in self.models.items():
                if name == "cvae":
                    continue
                r = model.predict(self.inp(k))
                np.testing.assert_allclose(
                    r.prediction_mm,
                    self.expected[name][k],
                    atol=1e-4,
                    equal_nan=True,
                    err_msg=name,
                )
                self.assertEqual(r.prediction_mm.shape, (32, 32))
                self.assertEqual(
                    r.target_date.astype("datetime64[D]"),
                    np.datetime64(c["target_day"], "D"),
                )
                p = prepare(self.inp(k))
                self.assertTrue(np.isfinite(r.prediction_mm[p.seen]).all())
                np.testing.assert_allclose(
                    r.change_mm, r.prediction_mm - p.last, equal_nan=True
                )
                self.assertIsNone(r.samples_mm)
                self.assertIsNone(r.std_mm)

    def test_05_preferred_CVAE_matches_original_CPU_inference(self):
        for k in range(len(self.cases)):
            r = self.models["cvae"].predict(self.inp(k))
            np.testing.assert_allclose(
                r.samples_mm,
                self.expected["cvae_cpu_samples"][k],
                atol=1e-4,
                equal_nan=True,
            )
            self.assertEqual(r.samples_mm.shape, (32, 32, 32))
            self.assertEqual(r.metadata["training_seed"], 20261109)
            self.assertEqual(r.metadata["calibration_factor"], 1.0)
            self.assertFalse(r.metadata["alternative_coarse_model_used"])
            np.testing.assert_allclose(
                r.prediction_mm, r.samples_mm.mean(0), equal_nan=True
            )
            np.testing.assert_allclose(r.std_mm, r.samples_mm.std(0), equal_nan=True)
            for interval in r.intervals_mm.values():
                self.assertEqual(interval[0].shape, (32, 32))
                self.assertTrue(
                    np.all(
                        interval[0][prepare(self.inp(k)).seen]
                        <= interval[1][prepare(self.inp(k)).seen]
                    )
                )

    def test_06_prediction_never_fits_parameters(self):
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVR
        from sklearn.linear_model import Ridge
        from contextlib import ExitStack

        with ExitStack() as stack:
            for cls in [StandardScaler, RobustScaler, PCA, SVR, Ridge]:
                stack.enter_context(
                    patch.object(
                        cls,
                        "fit",
                        side_effect=AssertionError("Inference attempted training"),
                    )
                )
            for model in self.models.values():
                model.predict(self.inp())

    def test_07_masks_zeros_reproducibility_and_rng(self):
        import torch

        a = self.inp()
        a.valid_masks[:, 0, 0] = False
        a.maps[:, 0, 0] = 1e9
        for model in self.models.values():
            r = model.predict(a)
            self.assertTrue(np.isnan(r.prediction_mm[0, 0]))
            self.assertTrue(np.isnan(r.change_mm[0, 0]))
        b = self.inp()
        b.maps[:] = 0
        b.valid_masks[:] = True
        for model in [Persistence(), EMA30(), QualityAwareSmoothing(), RobustLowRank()]:
            np.testing.assert_array_equal(
                model.predict(b).prediction_mm, np.zeros((32, 32))
            )
        torch.manual_seed(45)
        before = torch.random.get_rng_state().clone()
        first = self.models["cvae"].predict(self.inp())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = self.models["cvae"].predict(self.inp())
        np.testing.assert_array_equal(first.samples_mm, second.samples_mm)

    def test_08_reject_unsupported_input_instead_of_adapting_model(self):
        for key, mutator in [
            ("short", lambda a: setattr(a, "maps", a.maps[:5])),
            ("unordered", lambda a: setattr(a, "dates", a.dates[::-1])),
            ("bad_horizon", lambda a: setattr(a, "horizon_days", 0)),
            ("units", lambda a: a.metadata.update(units="metres")),
        ]:
            a = self.inp()
            mutator(a)
            with self.assertRaises(ValueError, msg=key):
                self.models["hybrid"].predict(a)
        for model in [self.models["hybrid"], self.models["cvae"]]:
            a = self.inp()
            a.maps = a.maps[:, :16, :16]
            a.valid_masks = a.valid_masks[:, :16, :16]
            with self.assertRaises(ValueError):
                model.predict(a)
            a = self.inp()
            del a.metadata["spacing_km"]
            with self.assertRaises(ValueError):
                model.predict(a)
            a = self.inp()
            a.metadata["geometry_available"] = True
            a.metadata.pop("los_vector")
            with self.assertRaises(ValueError):
                model.predict(a)
            a = self.inp()
            del a.metadata["grid_convention"]
            with self.assertRaises(ValueError):
                model.predict(a)
            a = self.inp()
            a.horizon_days = 181
            with self.assertRaises(ValueError):
                model.predict(a)

    def test_09_fixed_six_acquisitions_and_generic_shape_controls(self):
        a = self.inp()
        reference = self.models["hybrid"].predict(a)
        a.maps = np.concatenate([a.maps[:1], a.maps])
        a.valid_masks = np.concatenate([a.valid_masks[:1], a.valid_masks])
        a.dates = np.r_[a.dates[0] - np.timedelta64(12, "D"), a.dates]
        r = self.models["hybrid"].predict(a)
        np.testing.assert_array_equal(r.prediction_mm, reference.prediction_mm)
        self.assertEqual(r.metadata["older_acquisitions_ignored"], 1)
        for model in [Persistence(), EMA30(), QualityAwareSmoothing(), RobustLowRank()]:
            b = self.inp()
            b.maps = b.maps[:, :12, :13]
            b.valid_masks = b.valid_masks[:, :12, :13]
            self.assertEqual(model.predict(b).prediction_mm.shape, (12, 13))
        b = self.inp()
        b.maps = b.maps[-1:]
        b.valid_masks = b.valid_masks[-1:]
        b.dates = b.dates[-1:]
        np.testing.assert_allclose(
            Persistence().predict(b).prediction_mm,
            np.where(b.valid_masks[0], b.maps[0], np.nan),
            equal_nan=True,
        )

    def test_10_coordinator_and_public_imports(self):
        from src import TrainingModel
        from src.models import TrainingModel as PackageTrainingModel
        from src.training import TrainingModel as DirectTrainingModel

        self.assertIs(TrainingModel, PackageTrainingModel)
        self.assertIs(TrainingModel, DirectTrainingModel)
        for name, reference in [
            ("persistence", "persistence"),
            ("ema30", "EMA30"),
            ("smoothing", "quality"),
            ("robust", "lowrank"),
            ("four_branch", "hybrid"),
            ("cvae", "cvae"),
        ]:
            manager = TrainingModel.from_dict({"model_name": name, "parameters": {}})
            self.assertEqual(manager.to_dict(), {"model_name": name, "parameters": {}})
            result = manager.predict(self.inp())
            np.testing.assert_allclose(
                result.prediction_mm,
                self.models[reference].predict(self.inp()).prediction_mm,
                atol=1e-4,
                equal_nan=True,
            )
            self.assertFalse(result.model_version.startswith(("V6_", "V9_")))
            self.assertEqual(len(manager._models), 1)
            self.assertIsNotNone(getattr(manager, name)(self.inp()))

    def test_11_runtime_does_not_import_hackathon_or_research_code(self):
        script = """import sys, importlib.abc, json, numpy as np
from pathlib import Path
class BlockOriginal(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ['hackathon_stage1', 'volcanic_forecasting_v2', 'selected_models']:
   raise ImportError('Original Python packages are unavailable')
sys.meta_path.insert(0, BlockOriginal())
from src import *
p = Path('hackathon_stage1/tests/fixtures')
z = np.load(p/'real_inputs.npz'); c = json.loads((p/'cases.json').read_text())[0]
i = ForecastInput(c['volcano_id'], z['maps'][0], z['dates'][0], z['valid_masks'][0], c['horizon_days'], c['metadata'])
for model in [PersistenceModel(), EMA30Model(), SmoothingModel(), RobustModel(), FourBranchModel(), CVAEModel()]:
 assert model.predict(i).prediction_mm.shape == (32, 32)
"""
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        run = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PKG.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
