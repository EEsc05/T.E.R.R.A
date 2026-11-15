"""Query/download NISAR GUNW and export causal forecasting/training inputs.

Based on the project's NISAR_forecast_pipeline_v2.ipynb. Run:
    python -m src.prepare_nisar --help
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
import json
import os
import re
import shutil
from pathlib import Path
import sys
import asf_search as asf
import numpy as np
import pandas as pd
from shapely.geometry import box
from . import nisar_processing as processing
from .dataset import TrainingDataset, validate_split
from .input import ForecastInput
from .output import _json_value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")


def product_size(product, filename):
    sizes = product.properties.get("bytes", {})
    if isinstance(sizes, dict):
        entry = sizes.get(filename, {})
        return int(entry.get("bytes", 0)) if isinstance(entry, dict) else 0
    return int(sizes or 0)


def invert_network(pair_disp, pair_valid, i_idx, j_idx, n_epochs):
    """Notebook least-squares inversion on each pixel's datum-connected component.

    Disconnected epochs stay missing, rather than discarding every epoch whenever
    one pair is masked. No future pair may be passed for an historical snapshot.
    """
    pairs, height, width = pair_disp.shape
    cube = np.full((n_epochs, height * width), np.nan)
    valid = np.zeros_like(cube, dtype=bool)
    rms = np.full(height * width, np.nan)
    patterns, inverse = np.unique(
        pair_valid.reshape(pairs, -1).T, axis=0, return_inverse=True
    )
    y = pair_disp.reshape(pairs, -1)
    i_idx, j_idx = np.asarray(i_idx), np.asarray(j_idx)
    for k, pattern in enumerate(patterns):
        edges = np.flatnonzero(pattern)
        if not len(edges):
            continue
        connected = {0}
        while True:
            previous = len(connected)
            for edge in edges:
                i, j = int(i_idx[edge]), int(j_idx[edge])
                if i in connected or j in connected:
                    connected.update([i, j])
            if len(connected) == previous:
                break
        nodes = sorted(connected - {0})
        if not nodes:
            continue
        used = [e for e in edges if i_idx[e] in connected and j_idx[e] in connected]
        columns = {epoch: col for col, epoch in enumerate(nodes)}
        a = np.zeros((len(used), len(nodes)))
        for row, edge in enumerate(used):
            for epoch, sign in [(i_idx[edge], -1), (j_idx[edge], 1)]:
                if epoch != 0:
                    a[row, columns[epoch]] = sign
        pixels = np.flatnonzero(np.asarray(inverse).ravel() == k)
        values = y[np.ix_(used, pixels)]
        solution = np.linalg.lstsq(a, values, rcond=None)[0]
        cube[np.ix_([0] + nodes, pixels)] = np.vstack([np.zeros(len(pixels)), solution])
        valid[np.ix_([0] + nodes, pixels)] = True
        if len(used) > len(nodes):
            rms[pixels] = np.sqrt(np.mean((values - a @ solution) ** 2, axis=0))
    return (
        cube.reshape(n_epochs, height, width),
        valid.reshape(n_epochs, height, width),
        rms.reshape(height, width),
    )


class PairNetwork:
    """Keep pair observations so each history can be inverted as of its origin."""

    def __init__(self, pairs, grid, site, metadata):
        self.pairs = pairs
        self.grid = grid
        self.site = site
        self.metadata = metadata
        self.dates = np.asarray(
            sorted({p["reference"] for p in pairs} | {p["secondary"] for p in pairs}),
            dtype="datetime64[D]",
        )
        self.cache = {}

    def asof(self, end):
        if end not in self.cache:
            dates = self.dates[: end + 1]
            indices = {str(d): i for i, d in enumerate(dates)}
            pairs = [
                p for p in self.pairs if np.datetime64(p["secondary"], "D") <= dates[-1]
            ]
            if not pairs:
                raise ValueError("No pairs available for requested historical cutoff")
            cube, valid, rms = invert_network(
                np.asarray([p["disp"] for p in pairs]),
                np.asarray([p["ok"] for p in pairs]),
                [indices[p["reference"]] for p in pairs],
                [indices[p["secondary"]] for p in pairs],
                len(dates),
            )
            self.cache[end] = (cube.astype("float32"), valid, rms.astype("float32"))
        return self.cache[end]

    def history(self, end, history=6, horizon=30):
        cube, valid, _ = self.asof(end)
        sl = slice(end - history + 1, end + 1)
        if sl.start < 1:
            raise ValueError("Need enough epochs after the zero-displacement datum")
        if not valid[sl].any(axis=(1, 2)).all():
            raise ValueError("History contains an entirely unobserved epoch")
        metadata = dict(
            self.metadata,
            window_epoch_indices=list(range(sl.start, sl.stop)),
            inversion_asof=str(self.dates[end]),
            future_pairs_used=False,
        )
        return ForecastInput(
            self.metadata.get("volcano_id", self.site),
            np.where(valid[sl], cube[sl], np.nan),
            self.dates[sl],
            valid[sl],
            horizon,
            metadata,
        )

    def examples(self, history=6, horizon=30, tolerance=12):
        """Nearest actual future acquisition; never synthesize interpolated labels."""
        examples = []
        for end in range(history, len(self.dates) - 1):
            lags = (self.dates[end + 1 :] - self.dates[end]) / np.timedelta64(1, "D")
            local = int(np.argmin(abs(lags - horizon)))
            target = end + 1 + local
            lag = float(lags[local])
            if abs(lag - horizon) > tolerance or not 1 <= lag <= 180:
                continue
            try:
                inp = self.history(end, history, lag)
                cube, valid, _ = self.asof(target)
                mask = valid[target] & np.any(inp.valid_masks, axis=0)
                if not mask.any():
                    continue
                examples.append((inp, cube[target], mask, self.dates[target]))
            except ValueError:
                continue
        return examples


def save_examples(path, examples):
    if not examples:
        return None
    data = TrainingDataset(
        [e[0] for e in examples],
        np.asarray([e[1] for e in examples]),
        np.asarray([e[2] for e in examples]),
        np.asarray([e[3] for e in examples]),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
            [json.dumps(_json_value(i.metadata), allow_nan=False) for i in data.inputs]
        ),
    )
    TrainingDataset.from_npz(
        path
    )  # Verify the artifact with the actual training reader.
    return data


def split_examples(network, examples):
    """Purged temporal holdout. Export all examples even if the stack is too short."""
    if len(examples) < 4:
        return examples, []
    # Select a cutoff that retains >=2 train examples and >=1 disjoint validation window.
    candidates = []
    for cutoff in network.dates[1:-1]:
        train = [e for e in examples if e[3] < cutoff]
        validation = [e for e in examples if e[0].dates[0] >= cutoff]
        if len(train) >= 2 and validation:
            candidates.append(
                (
                    abs(len(validation) / (len(train) + len(validation)) - 0.2),
                    train,
                    validation,
                )
            )
    if not candidates:
        return examples, []
    _, train, validation = min(candidates, key=lambda item: item[0])
    return train, validation


def select_stack(catalog, site_poly, args):
    if catalog.empty:
        raise ValueError("No parseable NISAR GUNW products found")
    candidates = catalog[
        (catalog.baseline_days > 0) & (catalog.baseline_days <= args.max_baseline_days)
    ]
    if candidates.empty:
        raise ValueError("No products satisfy the baseline limit")
    # Prefer actual AOI coverage, then connected dates; do not mix viewing geometries.
    scores = []
    for key, group in candidates.groupby(["track", "direction", "frame"]):
        if args.stack and key != (args.stack[0], args.stack[1], args.stack[2]):
            continue
        component = processing._largest_component(group)
        coverage = []
        for index in group.result_idx:
            try:
                from shapely.geometry import shape

                coverage.append(
                    shape(args.results[index].geometry).intersection(site_poly).area
                    / site_poly.area
                )
            except (ValueError, TypeError, AttributeError):
                coverage.append(0.0)
        score = (
            int(max(coverage, default=0) >= args.min_aoi_coverage),
            len(component),
            max(coverage, default=0),
        )
        scores.append((score, key, group))
    if not scores:
        raise ValueError("Requested stack is absent from the catalog")
    _, key, group = max(scores, key=lambda row: row[0])
    count = (
        len(processing._largest_component(group))
        if args.max_epochs is None
        else args.max_epochs
    )
    selected = processing.select_pairs(group, count)
    if selected.empty:
        raise ValueError("Selected network is empty")
    return key, selected


def make_network(selected, raw_dir, site, config, args):
    grid = processing.make_grid(
        Path(raw_dir) / selected.file.iloc[0],
        config["lon"],
        config["lat"],
        config["half_km"],
        32,
        32,
    )
    pairs = []
    first = None
    for row in selected.itertuples():
        path = Path(raw_dir) / row.file
        print(f"Processing {site}: {row.ref_date} -> {row.sec_date}", flush=True)
        result = processing.process_pair(path, grid)
        if result is None:
            continue
        first = first or path
        pairs.append(
            {
                "reference": str(row.ref_date),
                "secondary": str(row.sec_date),
                "file": row.file,
                "disp": result["disp"],
                "ok": result["ok"],
                "coherence": result["coh_mean"],
            }
        )
    if not pairs:
        raise ValueError("No usable pairs after AOI/coherence filtering")
    # Remove disconnected acquisitions at the network level after dropped products.
    connected = processing._largest_component(
        pd.DataFrame(
            {
                "ref_date": [p["reference"] for p in pairs],
                "sec_date": [p["secondary"] for p in pairs],
            }
        )
    )
    pairs = [
        p for p in pairs if p["reference"] in connected and p["secondary"] in connected
    ]
    info = processing.read_product_info(first)
    geometry = processing.read_los_geometry(first, grid)
    metadata = {
        "units": "mm",
        "grid_convention": "aoi_block_mean32",
        "spacing_km": [abs(grid.dy) / 1000, abs(grid.dx) / 1000],
        "geometry_available": False,
        "geometry_note": "Product XY vectors are retained raw; no unverified ENU conversion",
        "raw_geometry": geometry,
        "source": "NISAR L2 GUNW via ASF",
        "source_collections": sorted(
            {str(c) for c in selected.collection if c is not None}
        ),
        "beta_products_included": any(
            "BETA" in str(c).upper() for c in selected.collection
        ),
        "volcano_id": config.get("volcano_id", site),
        "volcano_name": config.get("volcano_name", site),
        "country": config.get("country"),
        "volcano_number": config.get("volcano_number"),
        "coordinate_source": config.get("coordinate_source"),
        "stack_id": config.get("stack_id", site),
        "stack": config.get("stack"),
        "source_notebook": "NISAR_forecast_pipeline_v2.ipynb",
        "source_analysis": config.get("source_analysis"),
        "source_native_roi": config.get("source_native_roi"),
        "reference_product": config.get("reference_product"),
        "grid": asdict(grid),
        "center_lonlat": [config["lon"], config["lat"]],
        "half_width_km": config["half_km"],
        "phase_to_los_sign": args.phase_sign,
        "los_sign_convention": "positive_toward_satellite"
        if args.phase_sign == -1
        else "positive_away_from_satellite",
        "sign_verified_from": "https://github.com/aria-tools/ARIA-tools#readme",
        "native_phase_convention": "opposite ARIA S1; earlier reference and later secondary",
        "coherence_threshold": args.coherence_threshold,
        "min_pair_valid_fraction": args.min_pair_valid_fraction,
        "min_block_valid_fraction": processing.MIN_BLOCK_VALID_FRACTION,
        "reference_area": "per-pair valid-pixel median",
        "iono_mode": "none",
        "ramp_removed": False,
        "product_info": info,
        "pairs": [
            {
                "reference": p["reference"],
                "secondary": p["secondary"],
                "file": p["file"],
                "valid_pixel_fraction": float(p["ok"].mean()),
            }
            for p in pairs
        ],
    }
    return PairNetwork(pairs, grid, site, metadata)


def export_network(network, directory, args):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cube, valid, rms = network.asof(len(network.dates) - 1)
    np.savez_compressed(
        directory / "series.npz",
        dates=network.dates,
        cube=cube,
        valid=valid,
        resid_rms=rms,
        metadata_json=json.dumps(_json_value(network.metadata)),
    )
    write_json(directory / "metadata.json", network.metadata)
    history = None
    for end in range(len(network.dates) - 1, args.history - 1, -1):
        try:
            history = network.history(end, args.history, args.horizon_days)
            break
        except ValueError:
            continue
    if history is None:
        raise ValueError("Insufficient usable acquisitions for a history window")
    np.savez_compressed(
        directory / "history.npz",
        maps=history.maps,
        dates=history.dates,
        valid_masks=history.valid_masks,
        volcano_id=np.asarray(history.volcano_id),
        horizon_days=history.horizon_days,
        metadata_json=json.dumps(_json_value(history.metadata)),
    )
    examples = network.examples(
        args.history, args.horizon_days, args.target_tolerance_days
    )
    save_examples(directory / "examples.npz", examples)
    train, validation = split_examples(network, examples)
    if validation:
        a = save_examples(directory / "train.npz", train)
        b = save_examples(directory / "validation.npz", validation)
        validate_split(a, b)
    return {
        "site": network.site,
        "status": "generated",
        "epochs": len(network.dates),
        "pairs": len(network.pairs),
        "examples": len(examples),
        "training_examples": len(train) if validation else 0,
        "validation_examples": len(validation),
        "history_path": str(directory / "history.npz"),
        "training_ready": bool(validation),
        "note": None
        if validation
        else "No leakage-free train/validation split; collect a longer connected time series",
    }


def local_search_results(raw, poly, catalogue_path):
    """Preserve maturity labels when processing cached subsets offline."""
    from types import SimpleNamespace

    known = {}
    if catalogue_path.exists():
        known = {
            row["file"]: row.get("collection")
            for row in json.loads(catalogue_path.read_text())
        }
    results = []
    for path in sorted(raw.glob("*.h5")):
        collection = known.get(path.name)
        with processing.h5py.File(path, "r") as file:
            value = file.attrs.get("source_collection")
            if isinstance(value, bytes):
                value = value.decode()
            if value and value != "unknown":
                collection = str(value)
            if not collection:
                match = re.search(
                    r"/(NISAR_L2_GUNW_[^/]+)/", str(file.attrs.get("source_url", ""))
                )
                if match:
                    collection = match[1]
        results.append(
            SimpleNamespace(
                properties={"fileName": path.name, "collectionName": collection},
                geometry=poly.__geo_interface__,
            )
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Site IDs from the built-in catalogue or --sites-config",
    )
    parser.add_argument(
        "--sites-config",
        type=Path,
        help="JSON mapping site IDs to lon, lat, half_km and optional stack/source metadata",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/nisar"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Existing/downloaded HDF5 product directory; shared across sites",
    )
    parser.add_argument("--start-date", default="2025-08-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--horizon-days", type=float, default=30)
    parser.add_argument("--target-tolerance-days", type=float, default=12)
    parser.add_argument(
        "--max-epochs", type=int, help="Default: all connected dates, for training"
    )
    parser.add_argument("--max-baseline-days", type=int, default=24)
    parser.add_argument("--stack", nargs=3, metavar=("TRACK", "DIRECTION", "FRAME"))
    parser.add_argument("--coherence-threshold", type=float, default=0.4)
    parser.add_argument("--min-aoi-coverage", type=float, default=0.9)
    parser.add_argument(
        "--min-pair-valid-fraction",
        type=float,
        default=0.2,
        help="Minimum fraction of valid output pixels to retain a pair; pixel masks are unchanged",
    )
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument(
        "--include-beta",
        action="store_true",
        help="Explicitly allow pre-calibration BETA products; excluded by default",
    )
    parser.add_argument(
        "--token-file", type=Path, help="Local Earthdata token file; never logged"
    )
    parser.add_argument(
        "--survey", action="store_true", help="Query/catalog only; no downloads"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Process matching local .h5 files without querying/downloading",
    )
    parser.add_argument(
        "--full-products",
        action="store_true",
        help="Download full HDF5 products instead of compact AOI subsets",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent AOI subset downloads"
    )
    args = parser.parse_args(argv)
    configurations = dict(processing.SITES)
    configurations["ridgecrest_test"] = {"lon": -117.75, "lat": 35.60, "half_km": 25}
    if args.sites_config:
        try:
            custom = json.loads(args.sites_config.read_text())
            if not isinstance(custom, dict) or not custom:
                raise ValueError("Site configuration must be a nonempty object")
            for site, config in custom.items():
                if not isinstance(config, dict) or not all(
                    k in config for k in ["lon", "lat", "half_km"]
                ):
                    raise ValueError(f"Missing lon/lat/half_km for {site}")
                if (
                    not -180 <= float(config["lon"]) <= 180
                    or not -90 < float(config["lat"]) < 90
                    or float(config["half_km"]) <= 0
                ):
                    raise ValueError(f"Invalid coordinates/half_km for {site}")
                if "stack" in config and (
                    len(config["stack"]) != 3 or config["stack"][1] not in ["A", "D"]
                ):
                    raise ValueError(f"Invalid stack for {site}")
            configurations.update(custom)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
        if args.sites is None:
            args.sites = list(custom)
    if args.sites is None:
        args.sites = ["sabancaya", "ubinas", "misti"]
    if any(site not in configurations for site in args.sites):
        parser.error("Unknown site; provide --sites-config for custom regions")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.history < 1 or args.horizon_days <= 0 or args.target_tolerance_days < 0:
        parser.error("Invalid history/horizon/tolerance")
    if args.max_epochs is not None and args.max_epochs < args.history + 1:
        parser.error("--max-epochs must include the datum plus the historical epochs")
    if not 0 <= args.coherence_threshold <= 1 or not 0 < args.min_aoi_coverage <= 1:
        parser.error("Coherence/coverage thresholds must be fractions between 0 and 1")
    if not 0 <= args.min_pair_valid_fraction <= 1:
        parser.error("--min-pair-valid-fraction must be between 0 and 1")
    try:
        if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
            parser.error("--start-date must precede --end-date")
    except ValueError:
        parser.error("Dates must use YYYY-MM-DD")
    if args.stack:
        try:
            args.stack = (int(args.stack[0]), args.stack[1].upper(), int(args.stack[2]))
        except ValueError:
            parser.error("--stack must be TRACK A|D FRAME")
        if args.stack[1] not in ["A", "D"]:
            parser.error("Stack direction must be A or D")
    raw_base = args.raw_dir or args.output_dir / "raw"
    raw_base.mkdir(parents=True, exist_ok=True)
    try:
        token = (
            args.token_file.read_text().strip()
            if args.token_file
            else os.environ.get("EARTHDATA_TOKEN")
        )
    except OSError:
        parser.error("Cannot read --token-file")
    session = None
    manifest = []
    for site in args.sites:
        directory = args.output_dir / site
        directory.mkdir(parents=True, exist_ok=True)
        raw = (
            raw_base if args.local_only or args.full_products else directory / "subsets"
        )
        raw.mkdir(parents=True, exist_ok=True)
        config = configurations[site]
        poly = box(
            *processing.aoi_lonlat(config["lon"], config["lat"], config["half_km"])
        )
        processing.CENTER_LON = config["lon"]
        processing.CENTER_LAT = config["lat"]
        processing.HALF_WIDTH_KM = config["half_km"]
        processing.COHERENCE_THRESHOLD = args.coherence_threshold
        processing.MIN_AOI_COVERAGE = args.min_aoi_coverage
        processing.MIN_PAIR_VALID_FRACTION = args.min_pair_valid_fraction
        processing.PHASE_TO_LOS_SIGN = args.phase_sign
        try:
            print(f"Querying {site}...", flush=True)
            if args.local_only:
                args.results = local_search_results(
                    raw, poly, directory / "catalog.json"
                )
            else:
                args.results = asf.search(
                    platform="NISAR",
                    processingLevel="GUNW",
                    intersectsWith=poly.wkt,
                    start=args.start_date,
                    end=args.end_date,
                    maxResults=3000,
                )
            catalog = processing.build_catalog(args.results, poly)
            if args.local_only and not catalog.empty:
                catalog = catalog[
                    (catalog.sec_date >= date.fromisoformat(args.start_date))
                    & (catalog.sec_date <= date.fromisoformat(args.end_date))
                ]
            write_json(
                directory / "catalog.json", catalog.astype(object).to_dict("records")
            )
            eligible_catalog = catalog
            if not args.include_beta:
                eligible_catalog = catalog[
                    ~catalog.collection.astype(str).str.upper().str.contains("BETA")
                ]
            selection_args = argparse.Namespace(**vars(args))
            if not args.stack and "stack" in config:
                selection_args.stack = tuple(config["stack"])
            key, selected = select_stack(eligible_catalog, poly, selection_args)
            write_json(
                directory / "selected_pairs.json",
                selected.astype(object).to_dict("records"),
            )
            row = {
                "site": site,
                "catalog_products": len(catalog),
                "stack": list(key),
                "selected_pairs": len(selected),
                "selected_epochs": len(set(selected.ref_date) | set(selected.sec_date)),
                "selected_bytes": sum(
                    product_size(args.results[i], filename)
                    for i, filename in zip(selected.result_idx, selected.file)
                ),
            }
            if args.survey:
                row["status"] = "surveyed"
                manifest.append(row)
                print(json.dumps(_json_value(row)), flush=True)
                continue
            missing = [
                i
                for i, f in zip(selected.result_idx, selected.file)
                if not (raw / f).is_file()
            ]
            if missing and args.local_only:
                raise ValueError("Selected local files disappeared")
            if missing:
                remaining_bytes = sum(
                    product_size(
                        args.results[i], args.results[i].properties["fileName"]
                    )
                    for i in missing
                )
                row["download_bytes"] = remaining_bytes
                if not token:
                    row.update(
                        status="authentication_required",
                        missing_files=len(missing),
                        message="Set EARTHDATA_TOKEN or provide --token-file, then rerun the same command",
                    )
                    manifest.append(row)
                    print(json.dumps(_json_value(row)), flush=True)
                    continue
                if args.full_products and remaining_bytes > shutil.disk_usage(raw).free:
                    raise ValueError(
                        "Not enough disk space for the selected raw products"
                    )
                if session is None:
                    session = asf.ASFSession().auth_with_token(token)
                if args.full_products:
                    asf.ASFSearchResults([args.results[i] for i in missing]).download(
                        path=str(raw), session=session, processes=2
                    )
                else:
                    from .nisar_remote import download_subset_worker

                    transferred = 0
                    with ProcessPoolExecutor(max_workers=args.workers) as pool:
                        pending = [
                            pool.submit(
                                download_subset_worker,
                                args.results[index].properties,
                                dict(session.headers),
                                raw,
                                config,
                            )
                            for index in missing
                        ]
                        for count, future in enumerate(as_completed(pending), 1):
                            path, fetched = future.result()
                            transferred += fetched
                            print(
                                f"{site}: AOI subset {count}/{len(missing)}; fetched {transferred / 1024**2:.1f} MiB",
                                flush=True,
                            )
                    row["range_bytes_fetched"] = transferred
                row["transfer_mode"] = (
                    "full_products" if args.full_products else "aoi_ranges"
                )
            network = make_network(selected, raw, site, config, args)
            row.update(export_network(network, directory, args))
            manifest.append(row)
            print(json.dumps(_json_value(row)), flush=True)
        except Exception as error:
            # Preserve useful diagnostics while redacting the credential itself.
            message = str(error).replace(token, "[redacted]") if token else str(error)
            row = {"site": site, "status": "failed", "error": message}
            manifest.append(row)
            print(json.dumps(_json_value(row)), file=sys.stderr, flush=True)
        write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "manifest.json", manifest)
    return (
        0 if all(row["status"] in ["surveyed", "generated"] for row in manifest) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
