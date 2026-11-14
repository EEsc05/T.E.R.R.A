"""Terminal training entry point: python -m src train --help."""

import argparse
import json
from pathlib import Path
from .dataset import TrainingDataset, validate_split
from .training import TrainingModel
from .output import ForecastOutput, _json_value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a fresh forecasting model from supervised NumPy data."
    )
    parser.add_argument("--model", required=True, choices=TrainingModel.MODEL_TYPES)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Training NPZ with maps, dates, targets, target_dates",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        help="Separate, later validation NPZ; required for learned models",
    )
    parser.add_argument(
        "--test",
        type=Path,
        help="Optional untouched final test NPZ, later than training/validation",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New trained .joblib model"
    )
    parser.add_argument(
        "--report", type=Path, help="Optional training metrics and curves .json"
    )
    parser.add_argument("--units", choices=["mm", "cm", "m"])
    parser.add_argument("--nodata", type=float)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--basis-rank", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-context-pixels", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.model in ["four_branch", "cvae"] and args.validation is None:
        parser.error("--validation is required for four_branch and cvae")
    if args.output.suffix.lower() != ".joblib":
        parser.error("--output must end in .joblib")
    if args.report and args.report.suffix.lower() != ".json":
        parser.error("--report must end in .json")
    sources = [
        p.resolve() for p in [args.input, args.validation, args.test] if p is not None
    ]
    if len(sources) != len(set(sources)):
        parser.error("Training, validation and test must be separate files")
    for dest in [args.output, args.report]:
        if dest is not None:
            if dest.resolve() in sources:
                parser.error("Output must differ from input files")
            if dest.exists() and not args.overwrite:
                parser.error(f"Output exists: {dest}; use --overwrite")

    def read(path):
        return TrainingDataset.from_npz(
            path, units=args.units, nodata=args.nodata, center_crop=args.center_crop
        )

    try:
        training = read(args.input)
        validation = read(args.validation) if args.validation else None
        test = read(args.test) if args.test else None
        if test is not None:
            validate_split(training, test)
            if validation is not None:
                validate_split(validation, test)
        parameters = {}
        if args.model == "four_branch":
            parameters = {
                "seed": args.seed,
                "max_context_pixels": args.max_context_pixels,
            }
        if args.model == "cvae":
            parameters = {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "patience": args.patience,
                "basis_rank": args.basis_rank,
                "device": args.device,
            }
        print(
            f"Training {args.model} from scratch on {len(training)} examples...",
            flush=True,
        )
        manager = TrainingModel(args.model, parameters).fit(training, validation)
        if test is not None:
            manager.training_report["test_metrics"] = manager.evaluate(test)
        manager.save(args.output, overwrite=args.overwrite)
        if args.report:
            content = (
                json.dumps(
                    _json_value(manager.training_report), indent=2, allow_nan=False
                )
                + "\n"
            )
            ForecastOutput._write(
                args.report,
                lambda stream: stream.write(content.encode()),
                args.overwrite,
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "model": args.model,
                "output": str(args.output),
                "training_examples": len(training),
                "report": str(args.report) if args.report else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
