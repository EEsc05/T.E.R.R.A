"""Command-line forecasting: python -m src.main --help."""

import argparse
import json
import sys
from pathlib import Path
from .preparation import InputPreparation
from .output import ForecastOutput
from .training import TrainingModel


def build_parser():
    parser = argparse.ArgumentParser(
        description="Forecast signed LOS displacement from NumPy history."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit path to your own trained .joblib model",
    )
    parser.add_argument("--model", required=True, choices=InputPreparation.MODEL_NAMES)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--input",
        type=Path,
        help="NPZ with maps, dates, optional valid_masks/metadata_json",
    )
    sources.add_argument(
        "--maps", type=Path, help="Map history .npy with shape (T,H,W)"
    )
    parser.add_argument(
        "--dates",
        type=Path,
        help="Dates .npy (UTC dates or days since 1970); overrides NPZ dates",
    )
    parser.add_argument(
        "--masks", type=Path, help="Boolean masks .npy; overrides NPZ valid_masks"
    )
    parser.add_argument(
        "--case-index", type=int, help="Select a case from batched (N,T,H,W) maps"
    )
    parser.add_argument("--volcano-id", required=True)
    parser.add_argument("--horizon-days", type=float, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="JSON object with spacing, geometry and other metadata",
    )
    parser.add_argument(
        "--units",
        choices=InputPreparation.UNIT_SCALES,
        help="Source units; defaults to metadata.units or mm",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        help="Missing-value sentinel in source units; zero is otherwise valid",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Extract native central 32×32 pixels without resizing",
    )
    parser.add_argument(
        "--grid-convention",
        choices=["native_center32", "aoi_block_mean32"],
        help="Confirm an already-prepared native center32 grid",
    )
    parser.add_argument("--spacing-km", type=float, nargs=2, metavar=("ROW", "COLUMN"))
    parser.add_argument(
        "--geometry-available",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Declare known or unknown geometry; required by learned models (or in metadata)",
    )
    parser.add_argument(
        "--los-vector", type=float, nargs=3, metavar=("EAST", "NORTH", "UP")
    )
    parser.add_argument("--sampling-seed", type=int, help="CVAE sampling seed override")
    parser.add_argument(
        "--n-samples", type=int, help="CVAE sample count, integer >=2 (default 32)"
    )
    parser.add_argument(
        "--device",
        help="CVAE device (default cpu; installed environment uses CPU PyTorch)",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination forecast .npz"
    )
    parser.add_argument("--summary", type=Path, help="Optional readable summary .json")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files"
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare-data":
        from .prepare_nisar import main as prepare_main

        return prepare_main(argv[1:])
    if argv and argv[0] == "train":
        from .train import main as train_main

        return train_main(argv[1:])
    if argv and argv[0] == "predict":
        argv = argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model in ["four_branch", "cvae"] and args.checkpoint is None:
        parser.error(
            "--checkpoint is required for learned models; first run python -m src train"
        )
    if args.model != "cvae" and any(
        v is not None for v in (args.device, args.n_samples, args.sampling_seed)
    ):
        parser.error("--device, --n-samples and --sampling-seed apply only to cvae")
    if args.n_samples is not None and args.n_samples < 2:
        parser.error("--n-samples must be at least 2")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.summary is not None and args.summary.suffix.lower() != ".json":
        parser.error("--summary must end in .json")
    # Reject existing destinations and source/output collisions before inference.
    sources = [
        args.input,
        args.maps,
        args.dates,
        args.masks,
        args.metadata,
        args.checkpoint,
    ]
    destinations = [args.output, args.summary]
    for path in destinations:
        if path is None:
            continue
        if any(path.resolve() == src.resolve() for src in sources if src is not None):
            parser.error(f"Output must differ from input files: {path}")
        if path.exists() and not args.overwrite:
            parser.error(
                f"Output already exists: {path}; use --overwrite to replace it"
            )
    metadata = {
        key: value
        for key, value in {
            "spacing_km": args.spacing_km,
            "los_vector": args.los_vector,
            "geometry_available": args.geometry_available,
            "grid_convention": args.grid_convention,
            "sampling_seed": args.sampling_seed,
        }.items()
        if value is not None
    }
    try:
        data = InputPreparation(args.units, args.nodata, args.center_crop).from_files(
            input_path=args.input,
            maps_path=args.maps,
            dates_path=args.dates,
            masks_path=args.masks,
            metadata_path=args.metadata,
            metadata=metadata,
            case_index=args.case_index,
            volcano_id=args.volcano_id,
            horizon_days=args.horizon_days,
            model_name=args.model,
        )
        parameters = {}
        if args.model == "cvae":
            parameters = {
                "device": args.device or "cpu",
                "n_samples": args.n_samples or 32,
            }
        if args.checkpoint is not None:
            manager = TrainingModel.load(args.checkpoint)
            if manager.model_name != args.model:
                raise ValueError("--model does not match the selected checkpoint")
            if args.model == "cvae":
                estimator = manager._models["cvae"]
                if args.device is not None:
                    import torch

                    estimator.device = torch.device(args.device)
                    estimator.model.to(estimator.device)
                if args.n_samples is not None:
                    estimator.n_samples = args.n_samples
        else:
            manager = TrainingModel(args.model, parameters)
        result = manager.predict(data)
        output = ForecastOutput(result, data, args.model)
        output.save(args.output, overwrite=args.overwrite)
        if args.summary is not None:
            output.save_summary(args.summary, overwrite=args.overwrite)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": str(args.summary) if args.summary else None,
                "model": args.model,
                "volcano_id": data.volcano_id,
                "target_date": str(result.target_date),
                "shape": list(result.prediction_mm.shape),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
