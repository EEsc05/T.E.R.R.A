"""Train all six families and compare them against persistence on untouched windows.

Selection uses physical-volcano folds within training archives and any explicit
extra-training archives. Reporting data is never passed to fit(); reused and
non-forward-time comparisons are marked in the protocol.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from .dataset import TrainingDataset, validate_split
from .training import TrainingModel
from .preprocessing.interface import prepare
from .output import _json_value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n"
    )


def subset(data, indices):
    indices = list(indices)
    return TrainingDataset(
        [data.inputs[i] for i in indices],
        data.targets[indices],
        data.target_masks[indices],
        data.target_dates[indices],
    )


def concatenate(datasets):
    return TrainingDataset(
        [i for d in datasets for i in d.inputs],
        np.concatenate([d.targets for d in datasets]),
        np.concatenate([d.target_masks for d in datasets]),
        np.concatenate([d.target_dates for d in datasets]),
    )


def temporal_audit(training, reporting):
    cases = [
        {
            "site": inp.volcano_id,
            "origin_date": str(inp.forecast_origin),
            "later_training_targets": int(
                np.sum(training.target_dates > inp.forecast_origin)
            ),
        }
        for inp in reporting.inputs
    ]
    strict = all(case["later_training_targets"] == 0 for case in cases)
    return {
        "latest_training_target_date": str(training.target_dates.max()),
        "strict_global_forward_time_test": strict,
        "per_case": cases,
        "interpretation": "Temporal row holdout"
        if strict
        else "Retrospective row holdout: training includes observations later than reporting forecast origins",
    }


def save_dataset(path, data):
    np.savez_compressed(
        path,
        maps=np.asarray([i.maps for i in data.inputs]),
        dates=np.asarray([i.dates for i in data.inputs]),
        valid_masks=np.asarray([i.valid_masks for i in data.inputs]),
        targets=data.targets,
        target_masks=data.target_masks,
        target_dates=data.target_dates,
        volcano_ids=np.asarray([i.volcano_id for i in data.inputs]),
        metadata_json=np.asarray(
            [json.dumps(_json_value(i.metadata)) for i in data.inputs]
        ),
    )


def common_mask(inp, target, target_mask):
    return target_mask & np.isfinite(target) & prepare(inp, min_history=1).seen


def score_case(result, inp, target, target_mask, model, index):
    mask = common_mask(inp, target, target_mask)
    prediction = np.asarray(result.prediction_mm)
    if prediction.shape != target.shape or not np.isfinite(prediction[mask]).all():
        raise ValueError(
            f"{model} returned invalid predictions on the common scoring mask"
        )
    if not mask.any():
        raise ValueError("No observed pixels on the common scoring mask")
    err = prediction[mask].astype(float) - target[mask].astype(float)
    row = {
        "model": model,
        "case_index": index,
        "site": inp.volcano_id,
        "origin_date": str(inp.forecast_origin),
        "target_date": str(inp.target_date),
        "horizon_days": inp.horizon_days,
        "pixels": int(mask.sum()),
        "mae_mm": float(np.mean(abs(err))),
        "mse_mm2": float(np.mean(err**2)),
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "bias_mm": float(np.mean(err)),
    }
    if result.samples_mm is not None:
        samples = result.samples_mm[:, mask].astype(float)
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite samples on scoring mask")
        # Empirical CRPS: E|X-y| - 0.5 E|X-X'|, computed by sorted sample ranks.
        ordered = np.sort(samples, axis=0)
        n = len(samples)
        ranks = (2 * np.arange(1, n + 1) - n - 1)[:, None]
        crps = (
            np.mean(abs(samples - target[mask]), axis=0)
            - np.sum(ranks * ordered, axis=0) / n**2
        )
        low, high = result.intervals_mm["90"]
        row.update(
            crps_mm=float(crps.mean()),
            coverage90=float(
                np.mean((target[mask] >= low[mask]) & (target[mask] <= high[mask]))
            ),
            interval_width90_mm=float(np.mean(high[mask] - low[mask])),
        )
    return row


def summarize(rows):
    """Equal volcano weight, equal window weight within each volcano."""
    sites = []
    for site in sorted({r["site"] for r in rows}):
        cases = [r for r in rows if r["site"] == site]
        item = {
            "site": site,
            "cases": len(cases),
            "pixels": sum(r["pixels"] for r in cases),
        }
        for key in [
            "mae_mm",
            "mse_mm2",
            "bias_mm",
            "crps_mm",
            "coverage90",
            "interval_width90_mm",
        ]:
            if all(key in r for r in cases):
                item[key] = float(np.mean([r[key] for r in cases]))
        item["rmse_mm"] = float(np.sqrt(item["mse_mm2"]))
        sites.append(item)
    aggregate = {
        "cases": len(rows),
        "sites": len(sites),
        "pixels": sum(r["pixels"] for r in rows),
    }
    for key in [
        "mae_mm",
        "mse_mm2",
        "bias_mm",
        "crps_mm",
        "coverage90",
        "interval_width90_mm",
    ]:
        if all(key in s for s in sites):
            aggregate[key] = float(np.mean([s[key] for s in sites]))
    aggregate["rmse_mm"] = float(np.sqrt(aggregate["mse_mm2"]))
    return aggregate, sites


def fit_fusion(cases):
    weights = np.full((3, 4), 0.25)
    for bin_id in range(3):
        group = [c for c in cases if c["bin"] == bin_id]
        if not group:
            continue
        scale = max(float(np.mean([np.mean(c["y"] ** 2) for c in group])), 1.0)

        def loss(w):
            return float(
                np.mean([np.mean((c["x"] @ w - c["y"]) ** 2) for c in group]) / scale
            )

        def gradient(w):
            return (
                np.mean(
                    [2 * c["x"].T @ (c["x"] @ w - c["y"]) / len(c["y"]) for c in group],
                    axis=0,
                )
                / scale
            )

        opt = minimize(
            loss,
            np.full(4, 0.25),
            jac=gradient,
            method="SLSQP",
            bounds=[(0, 1)] * 4,
            constraints={
                "type": "eq",
                "fun": lambda w: w.sum() - 1,
                "jac": lambda w: np.ones(4),
            },
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if not opt.success:
            raise ValueError(f"Internal fusion fit failed: {opt.message}")
        weights[bin_id] = np.maximum(opt.x, 0) / np.maximum(opt.x, 0).sum()
    return weights


def select_configuration(
    training, directory, seed, max_epochs, max_context_pixels, device
):
    sites = sorted({i.volcano_id for i in training.inputs})
    if len(sites) < 3:
        raise ValueError("This protocol requires at least three training sites")
    cv = []
    fusion = []
    for held_site in sites:
        fit = subset(
            training,
            [k for k, i in enumerate(training.inputs) if i.volcano_id != held_site],
        )
        tune = subset(
            training,
            [k for k, i in enumerate(training.inputs) if i.volcano_id == held_site],
        )
        validate_split(fit, tune)
        print(
            f"Internal CV: fit {len(fit)} examples; tune on {held_site} ({len(tune)})",
            flush=True,
        )
        four = TrainingModel(
            "four_branch",
            {
                "seed": seed,
                "max_context_pixels": max_context_pixels,
                "report_training_metrics": False,
            },
        ).fit(fit)
        estimator = four._models["four_branch"]
        for inp, target, mask in zip(tune.inputs, tune.targets, tune.target_masks):
            prepared, branches = estimator._branches(inp)
            valid = common_mask(inp, target, mask)
            # Removing the common persistence level improves numerical conditioning.
            fusion.append(
                {
                    "bin": int(np.digitize(inp.horizon_days, [31, 91])),
                    "x": (branches[:, valid] - prepared.last[valid]).T.astype(float),
                    "y": (target[valid] - prepared.last[valid]).astype(float),
                }
            )
        cvae = TrainingModel(
            "cvae",
            {
                "seed": seed,
                "epochs": max_epochs,
                "batch_size": 16,
                "basis_rank": 8,
                "device": device,
                "patience": 15,
            },
        ).fit(fit, tune)
        report = cvae.training_report
        write_json(directory / f"cv_cvae_{held_site}.json", report)
        cv.append(
            {
                "held_out_site": held_site,
                "fit_examples": len(fit),
                "tuning_examples": len(tune),
                "best_epoch": report["best_epoch"],
                "epochs_completed": report["epochs_completed"],
            }
        )
    epochs = max(1, int(np.median([c["best_epoch"] for c in cv])))
    selection = {
        "method": "leave-one-physical-volcano-out within all training partitions only",
        "folds": cv,
        "cvae_epochs": epochs,
        "cvae_seed": seed,
        "four_branch_weights": fit_fusion(fusion).tolist(),
        "bin_edges_days": [31, 91],
        "bins_with_oof_data": sorted({c["bin"] for c in fusion}),
        "reporting_partition_used_for_selection": False,
    }
    write_json(directory / "selection.json", selection)
    return selection


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rescore_reference(reference, heldout, destination):
    """Evaluate existing saved weights on the current reporting rows without fitting."""
    destination.mkdir()
    scores = {}
    rows = []
    for name in TrainingModel.MODEL_TYPES:
        manager = TrainingModel.load(reference / "models" / f"{name}.joblib")
        predictions = []
        case_rows = []
        for index, (inp, target, mask) in enumerate(
            zip(heldout.inputs, heldout.targets, heldout.target_masks)
        ):
            result = manager.predict(inp)
            row = score_case(result, inp, target, mask, name, index)
            case_rows.append(row)
            rows.append(row)
            predictions.append(result.prediction_mm)
        scores[name] = summarize(case_rows)[0]
        np.savez_compressed(
            destination / f"{name}.npz",
            prediction_mm=np.asarray(predictions),
            targets=heldout.targets,
            target_dates=heldout.target_dates,
            scoring_masks=np.asarray(
                [
                    common_mask(i, t, m)
                    for i, t, m in zip(
                        heldout.inputs, heldout.targets, heldout.target_masks
                    )
                ]
            ),
            sites=np.asarray([i.volcano_id for i in heldout.inputs]),
        )
    write_csv(destination / "per_case.csv", rows)
    return scores


def run(
    data_root,
    sites,
    output_dir,
    seed=42,
    max_epochs=80,
    max_context_pixels=10000,
    device="cpu",
    extra_training=None,
    compare_to=None,
    rescore_comparison=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    report_dir = output_dir / "reports"
    report_dir.mkdir()
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()
    data_root = Path(data_root)
    training = concatenate(
        [TrainingDataset.from_npz(data_root / site / "train.npz") for site in sites]
    )
    extra_training = [Path(path) for path in (extra_training or [])]
    if extra_training:
        training = concatenate(
            [training] + [TrainingDataset.from_npz(path) for path in extra_training]
        )
    training_counts = {
        site: sum(i.volcano_id == site for i in training.inputs)
        for site in sorted({i.volcano_id for i in training.inputs})
    }
    heldout = concatenate(
        [
            TrainingDataset.from_npz(data_root / site / "validation.npz")
            for site in sites
        ]
    )
    validate_split(training, heldout)
    save_dataset(output_dir / "pooled_train.npz", training)
    save_dataset(output_dir / "pooled_holdout.npz", heldout)
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for site in sites
        for p in [data_root / site / "train.npz", data_root / site / "validation.npz"]
    }
    hashes.update(
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in extra_training
        }
    )
    if compare_to is not None:
        compare_to = Path(compare_to)
    if compare_to is not None and not rescore_comparison:
        with (
            np.load(compare_to / "pooled_holdout.npz", allow_pickle=False) as prior,
            np.load(output_dir / "pooled_holdout.npz", allow_pickle=False) as current,
        ):
            for key in [
                "maps",
                "dates",
                "valid_masks",
                "targets",
                "target_dates",
                "target_masks",
                "volcano_ids",
            ]:
                np.testing.assert_array_equal(
                    prior[key],
                    current[key],
                    err_msg=f"Comparison reporting set differs: {key}",
                )
    reporting_sites = {i.volcano_id for i in heldout.inputs}
    training_only_sites = sorted(set(training_counts) - reporting_sites)
    prior_reporting_matches = 0
    if compare_to is not None:
        with np.load(compare_to / "pooled_holdout.npz", allow_pickle=False) as prior:
            for index, inp in enumerate(heldout.inputs):
                prior_reporting_matches += int(
                    any(
                        prior["volcano_ids"][j] == inp.volcano_id
                        and prior["target_dates"][j] == heldout.target_dates[index]
                        and np.array_equal(prior["dates"][j], inp.dates)
                        and np.array_equal(prior["maps"][j], inp.maps, equal_nan=True)
                        for j in range(len(prior["target_dates"]))
                    )
                )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sites": sites,
        "training_examples": len(training),
        "training_examples_by_volcano": training_counts,
        "extra_training_files": [str(path) for path in extra_training],
        "comparison_run": str(compare_to) if compare_to is not None else None,
        "reporting_set_previously_evaluated": prior_reporting_matches > 0,
        "prior_reporting_window_matches": prior_reporting_matches,
        "training_only_volcanoes": training_only_sites,
        "comparison_method": "rescore_reference_models_on_current_holdout"
        if rescore_comparison
        else "identical_holdout_saved_scores",
        "reporting_examples_by_volcano": {
            site: sum(i.volcano_id == site for i in heldout.inputs)
            for site in sorted({i.volcano_id for i in heldout.inputs})
        },
        "japan_examples_used_only_for_training": training_counts.get("mount_asama", 0)
        > 0,
        "temporal_audit": temporal_audit(training, heldout),
        "reporting_examples": len(heldout),
        "source_reporting_partition": "validation.npz for each selected site, excluded from fitting",
        "seed": seed,
        "max_cvae_epochs": max_epochs,
        "max_context_pixels": max_context_pixels,
        "aggregation": "mean per-window errors within each volcano, then equal volcano weighting; RMSE=sqrt(macro MSE)",
        "mask": "same finite observed target pixels with historical support for every model",
        "selection": "leave-one-volcano-out inside training; final weights never fitted to reporting targets",
        "baseline": "persistence",
        "source_sha256": hashes,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "scikit-learn", "torch", "joblib"]
        },
        "caveat": f"{len(heldout)} reporting windows across {len({i.volcano_id for i in heldout.inputs})} volcanoes; temporal availability audited separately; not a geographic performance guarantee",
        "source_code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.rglob("*.py")
        },
    }
    write_json(output_dir / "protocol.json", protocol)
    print(
        f"Pooled training: {len(training)}; untouched reporting windows: {len(heldout)}",
        flush=True,
    )
    selection = select_configuration(
        training, report_dir, seed, max_epochs, max_context_pixels, device
    )
    rows = []
    scores = {}
    site_rows = []
    for name in TrainingModel.MODEL_TYPES:
        print(
            f"Fitting final {name} on all {len(training)} training examples...",
            flush=True,
        )
        start = time.perf_counter()
        parameters = {}
        if name == "four_branch":
            parameters = {
                "seed": seed,
                "max_context_pixels": max_context_pixels,
                "report_training_metrics": False,
            }
        elif name == "cvae":
            parameters = {
                "seed": seed,
                "epochs": selection["cvae_epochs"],
                "batch_size": 16,
                "basis_rank": 8,
                "n_samples": 32,
                "device": device,
            }
        manager = TrainingModel(name, parameters).fit(training)
        if name == "four_branch":
            estimator = manager._models[name]
            estimator.weights = np.asarray(selection["four_branch_weights"])
            manager.training_report["fusion"] = (
                "training-only leave-one-site-out predictions"
            )
            manager.training_report["weights"] = estimator.weights.tolist()
            manager.training_report["training_metrics"] = manager.evaluate(training)
            estimator.training_report = dict(manager.training_report)
        manager.training_report["selection_protocol"] = (
            "reports/selection.json; no reporting targets used for fitting"
        )
        training_seconds = time.perf_counter() - start
        model_path = manager.save(model_dir / f"{name}.joblib")
        # Score the saved/reloaded artifact, not just the transient training object.
        loaded = TrainingModel.load(model_path)
        predictions = []
        samples = []
        stds = []
        case_rows = []
        for index, (inp, target, mask) in enumerate(
            zip(heldout.inputs, heldout.targets, heldout.target_masks)
        ):
            result = loaded.predict(inp)
            row = score_case(result, inp, target, mask, name, index)
            case_rows.append(row)
            rows.append(row)
            predictions.append(result.prediction_mm)
            if result.samples_mm is not None:
                samples.append(result.samples_mm)
                stds.append(result.std_mm)
        score, per_site = summarize(case_rows)
        score.update(training_seconds=training_seconds, artifact=str(model_path))
        scores[name] = score
        for row in per_site:
            site_rows.append({"model": name, **row})
        arrays = {
            "prediction_mm": np.asarray(predictions),
            "targets": heldout.targets,
            "scoring_masks": np.asarray(
                [
                    common_mask(i, t, m)
                    for i, t, m in zip(
                        heldout.inputs, heldout.targets, heldout.target_masks
                    )
                ]
            ),
            "target_dates": heldout.target_dates,
            "sites": np.asarray([i.volcano_id for i in heldout.inputs]),
        }
        if samples:
            arrays.update(samples_mm=np.asarray(samples), std_mm=np.asarray(stds))
        np.savez_compressed(prediction_dir / f"{name}.npz", **arrays)
        write_json(report_dir / f"{name}_training.json", manager.training_report)
        write_json(output_dir / "scores.json", scores)
        print(
            f"{name}: MAE {score['mae_mm']:.4f} mm; RMSE {score['rmse_mm']:.4f} mm",
            flush=True,
        )
    baseline = scores["persistence"]["mae_mm"]
    for score in scores.values():
        score["mae_skill_vs_persistence_pct"] = (
            100 * (1 - score["mae_mm"] / baseline) if baseline else None
        )
    baseline_sites = {
        r["site"]: r["mae_mm"] for r in site_rows if r["model"] == "persistence"
    }
    for row in site_rows:
        base = baseline_sites[row["site"]]
        row["mae_skill_vs_persistence_pct"] = (
            100 * (1 - row["mae_mm"] / base) if base else None
        )
    write_json(output_dir / "scores.json", scores)
    write_csv(
        output_dir / "scores.csv",
        [{"model": name, **score} for name, score in scores.items()],
    )
    write_csv(output_dir / "per_site.csv", site_rows)
    write_csv(output_dir / "per_case.csv", rows)
    lines = [
        "# NISAR training comparison",
        "",
        f"Trained from scratch on {len(training)} examples across {len(training_counts)} volcanoes. "
        f"Reported on {len(heldout)} reporting windows excluded from all fitting and selection.",
        "",
        "| Model | MAE (mm) | RMSE (mm) | MAE skill vs persistence |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, score in scores.items():
        lines.append(
            f"| {name} | {score['mae_mm']:.4f} | {score['rmse_mm']:.4f} | {score['mae_skill_vs_persistence_pct']:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "Positive skill means lower MAE than persistence. Scores weight volcanoes equally and "
            "windows equally within each volcano. Identical scoring masks are used for all models.",
            "",
            f"CVAE duration selected from training-only folds: {selection['cvae_epochs']} epochs. "
            "Ensemble weights use training-only out-of-fold branch predictions.",
            "",
            "Persistence, EMA30, smoothing and robust low-rank are local baselines with no global neural weights. "
            "Four-branch and CVAE learn new parameters. No hackathon checkpoints were loaded.",
            "",
            f"Observed target horizons in this run: {sorted({i.horizon_days for i in training.inputs + heldout.inputs})} days. Reporting volcanoes: {sorted({i.volcano_id for i in heldout.inputs})}. "
            "CVAE uncertainty is uncalibrated. A reused reporting set is not a new independent test.",
        ]
    )
    if "crps_mm" in scores["cvae"]:
        s = scores["cvae"]
        lines.extend(
            [
                "",
                f"CVAE empirical CRPS: {s['crps_mm']:.4f} mm; "
                f"90% interval coverage: {s['coverage90']:.1%}; mean interval width: {s['interval_width90_mm']:.4f} mm.",
            ]
        )
    audit = temporal_audit(training, heldout)
    if not audit["strict_global_forward_time_test"]:
        lines.extend(
            [
                "",
                "**Retrospective comparison:** Some training targets occur after the reporting forecast origins. "
                f"The latest training target is {audit['latest_training_target_date']}. "
                "These scores are not a strict global forward-time test and cannot establish prospective improvement.",
            ]
        )
    if extra_training:
        lines.extend(
            [
                "",
                f"Training counts by physical volcano: {training_counts}. "
                f"Training-only volcanoes without a reporting score: {training_only_sites}. "
                "Different viewing directions and sensors remain separate histories and share their physical-volcano grouping.",
            ]
        )
    if compare_to is not None:
        if rescore_comparison:
            previous = rescore_reference(
                compare_to, heldout, output_dir / "reference_predictions"
            )
            write_json(output_dir / "comparison_reference_scores.json", previous)
            comparison_note = "Reference saved models were rescored on the current reporting windows and identical masks. No fitting was performed on the reporting rows."
        else:
            previous = json.loads((compare_to / "scores.json").read_text())
            comparison_note = "The same previously evaluated windows and scoring masks are reused. No reporting targets were used for fitting or internal configuration selection."

        comparison = []
        lines.extend(
            [
                "",
                "## Comparison with the previous model set",
                "",
                comparison_note,
                "",
                "| Model | Previous MAE | New MAE | MAE change | Previous RMSE | New RMSE |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, score in scores.items():
            old = previous[name]
            row = {
                "model": name,
                "original_mae_mm": old["mae_mm"],
                "combined_mae_mm": score["mae_mm"],
                "mae_change_mm": score["mae_mm"] - old["mae_mm"],
                "mae_improvement_vs_original_pct": 100
                * (1 - score["mae_mm"] / old["mae_mm"])
                if old["mae_mm"]
                else None,
                "original_rmse_mm": old["rmse_mm"],
                "combined_rmse_mm": score["rmse_mm"],
                "rmse_change_mm": score["rmse_mm"] - old["rmse_mm"],
            }
            comparison.append(row)
            lines.append(
                f"| {name} | {old['mae_mm']:.4f} | {score['mae_mm']:.4f} | {row['mae_change_mm']:+.4f} | {old['rmse_mm']:.4f} | {score['rmse_mm']:.4f} |"
            )
        write_json(output_dir / "comparison.json", comparison)
        write_csv(output_dir / "comparison.csv", comparison)
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Completed. Report: {output_dir / 'REPORT.md'}", flush=True)
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/nisar"))
    parser.add_argument("--sites", nargs="+", default=["sabancaya", "ubinas", "misti"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs")
        / ("nisar_training_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument(
        "--extra-training",
        nargs="+",
        type=Path,
        help="Additional supervised NPZ archives used only for training",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Existing training run to compare on identical reporting data",
    )
    parser.add_argument(
        "--rescore-comparison",
        action="store_true",
        help="Rescore reference models on the current reporting data instead of requiring identical prior data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--max-context-pixels", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.rescore_comparison and args.compare_to is None:
        parser.error("--rescore-comparison requires --compare-to")
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    torch.set_num_threads(2)
    run(
        args.data_root,
        args.sites,
        args.output_dir,
        args.seed,
        args.max_epochs,
        args.max_context_pixels,
        args.device,
        extra_training=args.extra_training,
        compare_to=args.compare_to,
        rescore_comparison=args.rescore_comparison,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
