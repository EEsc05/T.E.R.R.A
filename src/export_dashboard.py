"""Export real prepared maps, saved predictions and model scores to an offline dashboard."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .preparation import InputPreparation
from .training import TrainingModel
from .output import _json_value


def clean(value):
    return _json_value(value)


def write_script(path, assignment, value):
    text = (
        json.dumps(clean(value), separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    path.write_text(assignment + "=" + text + ";\n")


def maps(values):
    return np.round(np.asarray(values, dtype=float), 3).reshape((-1, 1024)).tolist()


def export(manifest_path, run, output):
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    scores = json.loads((run / "scores.json").read_text())
    shutil.copytree(Path(__file__).parent / "dashboard", output, dirs_exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir(exist_ok=True)
    models = {
        name: TrainingModel.load(run / "models" / f"{name}.joblib")
        for name in TrainingModel.MODEL_TYPES
    }
    index = []
    # Pooled order is precisely the order used by train_all: selected sites then extras.
    with np.load(run / "pooled_holdout.npz", allow_pickle=False) as holdout:
        holdout_maps = holdout["maps"]
        holdout_dates = holdout["dates"]
        target_dates = holdout["target_dates"]
        targets = holdout["targets"]
        holdout_ids = holdout["volcano_ids"]
        reporting_horizons = sorted(
            set(
                float(x)
                for x in (target_dates - holdout_dates[:, -1]) / np.timedelta64(1, "D")
            )
        )
    predictions = {}
    for name in models:
        with np.load(
            run / "predictions" / f"{name}.npz", allow_pickle=False
        ) as archive:
            predictions[name] = {k: archive[k] for k in archive.files}
    for item in manifest["datasets"]:
        directory = Path(item["source_dir"])
        id = item["id"]
        with np.load(directory / "series.npz", allow_pickle=False) as series:
            metadata = json.loads(series["metadata_json"].item())
            dates = (
                np.asarray(series["dates"], dtype="datetime64[D]").astype(str).tolist()
            )
            observed = maps(np.where(series["valid"], series["cube"], np.nan))
        history = InputPreparation().from_files(
            input_path=directory / "history.npz",
            volcano_id=item["physical_volcano"],
            horizon_days=30,
            model_name="persistence",
        )
        indices = []
        for i in range(len(target_dates)):
            # Physical labels alone are insufficient when multiple sensor/orbit stacks coexist.
            if holdout_ids[i] != item["physical_volcano"]:
                continue
            with np.load(directory / "validation.npz", allow_pickle=False) as own:
                if any(
                    np.array_equal(holdout_dates[i], d)
                    and np.array_equal(holdout_maps[i], m, equal_nan=True)
                    for d, m in zip(own["dates"], own["maps"])
                ):
                    indices.append(i)
        model_data = {}
        for name, manager in models.items():
            result = manager.predict(history)
            latest = dict(
                origin=str(history.forecast_origin)[:10],
                target=str(history.target_date)[:10],
                prediction=maps(result.prediction_mm)[0],
            )
            if result.std_mm is not None:
                latest["std"] = maps(result.std_mm)[0]
                latest["lower90"] = maps(result.intervals_mm["90"][0])[0]
                latest["upper90"] = maps(result.intervals_mm["90"][1])[0]
            cases = []
            for i in indices:
                p = predictions[name]
                mask = p["scoring_masks"][i]
                prediction = p["prediction_mm"][i]
                case = dict(
                    origin=str(holdout_dates[i, -1])[:10],
                    target=str(target_dates[i])[:10],
                    prediction=maps(prediction)[0],
                    observed=maps(np.where(mask, targets[i], np.nan))[0],
                    mae_mm=float(np.abs(prediction[mask] - targets[i][mask]).mean()),
                )
                if "std_mm" in p:
                    case["std"] = maps(p["std_mm"][i])[0]
                cases.append(case)
            model_data[name] = dict(latest=latest, cases=cases)
        entry = dict(
            id=id,
            name=metadata.get("volcano_name", item["physical_volcano"])
            .replace("_", " ")
            .title(),
            country=metadata.get("country")
            or (
                "Peru"
                if item["physical_volcano"] in {"sabancaya", "ubinas", "misti"}
                else "Unknown"
            ),
            sensor=item["sensor"],
            physical_volcano=item["physical_volcano"],
            center=metadata["center_lonlat"],
            first_date=dates[0],
            last_date=dates[-1],
            epochs=len(dates),
            training_examples=item["training_examples"],
            reporting_examples=item["reporting_examples"],
            stack=metadata.get("stack"),
            half_width_km=metadata.get("half_width_km"),
            mean_valid_fraction=float(np.mean(np.isfinite(observed))),
            file=f"data/{id}.js",
        )
        # Nulls, not zeroes, represent unobserved pixels throughout the export.
        stack = dict(
            **entry,
            dates=dates,
            observed=observed,
            models=model_data,
            metadata=metadata,
        )
        write_script(
            data_dir / f"{id}.js", f"window.TERRA_STACKS[{json.dumps(id)}]", stack
        )
        index.append(entry)
        print("Exported", id, flush=True)
    payload = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        run=run.name,
        datasets=index,
        scores=scores,
        protocol={
            k: v
            for k, v in protocol.items()
            if k not in ["source_code_sha256", "source_sha256"]
        },
        reporting_horizons_days=reporting_horizons,
        training=dict(
            examples=manifest["training_examples"],
            reporting_examples=manifest["reporting_examples"],
            volcanoes=manifest["physical_volcanoes"],
        ),
        caveats=[
            "Retrospective comparison; some training observations are later than reporting forecast origins.",
            "Displacement is relative line-of-sight motion, not vertical motion or an eruption probability.",
            "Full observed histories are retrospective reconstructions. Reporting comparisons use their original paired targets.",
            "CVAE uncertainty is uncalibrated; check measured coverage in the detailed view.",
        ],
        sources=[
            "https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/",
            "https://github.com/aria-tools/ARIA-tools#readme",
        ],
    )
    write_script(data_dir / "index.js", "window.TERRA_INDEX", payload)
    (output / "dataset.json").write_text(
        json.dumps(clean(payload), indent=2, allow_nan=False) + "\n"
    )
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data_dir.glob("*.js"))
    }
    (output / "data_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Dashboard:", output / "index.html")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("data/nisar_aria_training/manifest.json")
    )
    p.add_argument("--run", type=Path, default=Path("outputs/nisar_aria_all_data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/volcano_watch"))
    a = p.parse_args()
    export(a.manifest, a.run, a.output_dir)


if __name__ == "__main__":
    main()
