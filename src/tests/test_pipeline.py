"""Terminal training/prediction and preparation/output integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from src import InputPreparation, ForecastOutput, PersistenceModel
from .helpers import dataset, save_dataset

ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.training = dataset(3)
        self.validation = dataset(2, start="2023-01-01")
        self.train_path = self.directory / "train.npz"
        self.validation_path = self.directory / "validation.npz"
        save_dataset(self.train_path, self.training)
        save_dataset(self.validation_path, self.validation)

    def cli(self, *args):
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        return subprocess.run(
            [sys.executable, "-m", "src", *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_terminal_training_and_prediction_all_families(self):
        inp = self.validation.inputs[0]
        history = self.directory / "history.npz"
        np.savez(
            history,
            maps=inp.maps,
            dates=inp.dates,
            valid_masks=inp.valid_masks,
            metadata_json=json.dumps(inp.metadata),
        )
        for name in [
            "persistence",
            "ema30",
            "smoothing",
            "robust",
            "four_branch",
            "cvae",
        ]:
            checkpoint = self.directory / f"{name}.joblib"
            report = self.directory / f"{name}.json"
            run = self.cli(
                "train",
                "--model",
                name,
                "--input",
                self.train_path,
                "--validation",
                self.validation_path,
                "--output",
                checkpoint,
                "--report",
                report,
                "--epochs",
                2,
                "--basis-rank",
                1,
                "--max-context-pixels",
                64,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(report.read_text())["training_examples"], 3)
            out = self.directory / f"{name}.npz"
            args = [
                "predict",
                "--model",
                name,
                "--checkpoint",
                checkpoint,
                "--input",
                history,
                "--volcano-id",
                "site",
                "--horizon-days",
                30,
                "--output",
                out,
            ]
            if name == "cvae":
                args += ["--n-samples", 4]
            prediction = self.cli(*args)
            self.assertEqual(prediction.returncode, 0, prediction.stderr)
            with np.load(out, allow_pickle=False) as a:
                self.assertTrue(np.isfinite(a["prediction_mm"]).all())
                if name == "cvae":
                    self.assertEqual(a["samples_mm"].shape, (4, 32, 32))

    def test_learned_cli_requires_explicit_training_and_checkpoint(self):
        for name in ["four_branch", "cvae"]:
            run = self.cli(
                "train",
                "--model",
                name,
                "--input",
                self.train_path,
                "--output",
                self.directory / "model.joblib",
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("--validation", run.stderr)
            run = self.cli(
                "--model",
                name,
                "--input",
                self.train_path,
                "--case-index",
                0,
                "--volcano-id",
                "site",
                "--horizon-days",
                30,
                "--output",
                self.directory / "forecast.npz",
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("--checkpoint", run.stderr)

    def test_reject_leaking_split_without_output(self):
        destination = self.directory / "model.joblib"
        run = self.cli(
            "train",
            "--model",
            "cvae",
            "--input",
            self.train_path,
            "--validation",
            self.train_path,
            "--output",
            destination,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(destination.exists())

    def test_units_nodata_and_crop(self):
        maps = np.ones((6, 35, 37))
        maps[:, 3, 4] = -999
        dates = self.training.inputs[0].dates
        data = InputPreparation(units="cm", nodata=-999, center_crop=True).from_arrays(
            maps, dates, volcano_id="site", horizon_days=30
        )
        self.assertEqual(data.shape, (32, 32))
        self.assertFalse(data.valid_masks[:, 2, 2].any())
        self.assertTrue(np.all(data.maps[data.valid_masks] == 10))
        self.assertEqual(maps[0, 3, 4], -999)

    def test_output_no_overwrite_and_pickle_free_archive(self):
        inp = self.training.inputs[0]
        result = PersistenceModel().predict(inp)
        output = ForecastOutput(result, inp, "persistence")
        path = output.save(self.directory / "forecast.npz")
        output.save_summary(self.directory / "summary.json")
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a["prediction_mm"], result.prediction_mm)
            self.assertEqual(
                json.loads(a["metadata_json"].item())["input"]["volcano_id"], "site"
            )
        with self.assertRaises(FileExistsError):
            output.save(path)


if __name__ == "__main__":
    unittest.main()
