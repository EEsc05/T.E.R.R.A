"""Prepare archived Sentinel-1 ARIA GUNW as causal TERRA NumPy inputs.

Uses the prior discovery inventory; no on-demand processing is ordered. Native
ARIA phase is positive toward the sensor for later-minus-earlier displacement.
NISAR native phase has the opposite sign. Source references:
https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/
https://github.com/aria-tools/ARIA-tools#readme
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re

import asf_search as asf
import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

from .nisar_processing import Grid, _largest_component
from .nisar_remote import RangeReader
from .prepare_nisar import PairNetwork, export_network, write_json

DATA = "science/grids/data"
META = "science/radarMetaData"
PATTERN = re.compile(r"^S1-GUNW-([AD])-([LR])-(\d+)-tops-(\d{8})_(\d{8})-")
GUIDE = "https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/"
ADAPTER_VERSION = 1


def product_dates(name):
    match = PATTERN.match(name)
    if not match:
        raise ValueError("Unrecognized ARIA product name")
    later, earlier = [
        str(np.datetime64(datetime.strptime(d, "%Y%m%d").date()))
        for d in match.group(4, 5)
    ]
    if later <= earlier:
        raise ValueError(
            "ARIA reference must be newer than secondary; refusing ambiguous phase sign"
        )
    return earlier, later


def make_grid(config):
    lon, lat = config["lon"], config["lat"]
    epsg = (32600 if lat >= 0 else 32700) + min(60, max(1, int((lon + 180) // 6) + 1))
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = config["half_km"] * 1000
    return Grid(epsg, x - half, x + half, y - half, y + half, 32, 32)


def text_value(value):
    a = np.asarray(value)
    value = a.reshape(-1)[0] if a.size else ""
    return value.decode() if isinstance(value, bytes) else str(value)


def read_values(dataset, selection):
    values = np.asarray(dataset[selection], dtype=float)
    valid = np.isfinite(values)
    fill = dataset.attrs.get("_FillValue")
    if fill is not None:
        valid &= values != float(np.asarray(fill).reshape(-1)[0])
    values = values * float(
        np.asarray(dataset.attrs.get("scale_factor", 1)).reshape(-1)[0]
    )
    values += float(np.asarray(dataset.attrs.get("add_offset", 0)).reshape(-1)[0])
    return values, valid


def reduce_pair(source, config, grid, coherence_threshold=0.4, min_block_fraction=0.5):
    """Read only native AOI samples, reproject pixel centers, and area-bin them.

    Keep one connected unwrapping component; disconnected components have
    independent arbitrary phase offsets. Reference the retained pixels' median,
    matching the current NISAR preprocessing (not an absolute displacement datum).
    """
    latitude = np.asarray(source[f"{DATA}/latitude"][()])
    longitude = np.asarray(source[f"{DATA}/longitude"][()])
    if len(latitude) < 2 or len(longitude) < 2:
        raise ValueError("Invalid geographic coordinate vectors")
    transform = Transformer.from_crs(4326, grid.epsg, always_xy=True)
    inverse = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    # Include curved projected edges by sampling the boundary, not only corners.
    edges = np.linspace(0, 1, 33)
    ex = np.r_[
        grid.x0 + edges * (grid.x1 - grid.x0),
        grid.x0 + edges * (grid.x1 - grid.x0),
        np.full(33, grid.x0),
        np.full(33, grid.x1),
    ]
    ey = np.r_[
        np.full(33, grid.y0),
        np.full(33, grid.y1),
        grid.y0 + edges * (grid.y1 - grid.y0),
        grid.y0 + edges * (grid.y1 - grid.y0),
    ]
    lon, lat = inverse.transform(ex, ey)
    rows = np.flatnonzero((latitude >= min(lat)) & (latitude <= max(lat)))
    cols = np.flatnonzero((longitude >= min(lon)) & (longitude <= max(lon)))
    if not len(rows) or not len(cols):
        raise ValueError("Product misses the physical AOI")
    selection = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    phase_ds = source[f"{DATA}/unwrappedPhase"]
    if text_value(phase_ds.attrs.get("units", "")).lower() not in (
        "rad",
        "radian",
        "radians",
    ):
        raise ValueError("Unwrapped phase units must be radians")
    phase, phase_ok = read_values(phase_ds, selection)
    coherence, coh_ok = read_values(source[f"{DATA}/coherence"], selection)
    components, comp_ok = read_values(source[f"{DATA}/connectedComponents"], selection)
    ok = (
        phase_ok
        & coh_ok
        & comp_ok
        & (coherence >= coherence_threshold)
        & (coherence <= 1)
        & (components > 0)
    )
    if not ok.any():
        raise ValueError("No coherent connected pixels in AOI")
    labels, counts = np.unique(components[ok], return_counts=True)
    component = labels[np.argmax(counts)]
    ok &= components == component
    wavelength_ds = source[f"{META}/wavelength"]
    if text_value(wavelength_ds.attrs.get("units", "")).lower() not in (
        "meter",
        "meters",
        "metre",
        "m",
    ):
        raise ValueError("Wavelength units must be meters")
    wavelength = float(np.asarray(wavelength_ds[()]).reshape(-1)[0])
    if not 0.04 < wavelength < 0.07:
        raise ValueError("Unexpected Sentinel-1 C-band wavelength")
    displacement = phase * wavelength * 1000 / (4 * np.pi)
    displacement -= np.median(displacement[ok])
    xx, yy = transform.transform(
        *np.meshgrid(longitude[selection[1]], latitude[selection[0]])
    )
    rb = np.floor((grid.y1 - yy) / grid.dy).astype(int)
    cb = np.floor((xx - grid.x0) / grid.dx).astype(int)
    inside = (rb >= 0) & (rb < grid.H) & (cb >= 0) & (cb < grid.W)
    flat = rb * grid.W + cb
    use = ok & inside
    size = grid.H * grid.W
    count = np.bincount(flat[use], minlength=size)
    sums = np.bincount(flat[use], weights=displacement[use], minlength=size)
    # Expected count derives from nominal native pixel area, so cropped product
    # edges cannot pass coverage simply because their missing pixels weren't read.
    xc, yc = transform.transform(config["lon"], config["lat"])
    x2, y2 = transform.transform(
        config["lon"] + np.median(np.diff(longitude)), config["lat"]
    )
    x3, y3 = transform.transform(
        config["lon"], config["lat"] + np.median(np.diff(latitude))
    )
    native_area = abs((x2 - xc) * (y3 - yc) - (y2 - yc) * (x3 - xc))
    expected = grid.dx * grid.dy / native_area
    valid = (count >= min_block_fraction * expected) & (count > 0)
    means = np.full(size, np.nan)
    np.divide(sums, count, out=means, where=valid)
    return dict(
        disp=means.reshape(32, 32).astype("float32"),
        ok=valid.reshape(32, 32),
        wavelength_m=wavelength,
        valid_pixel_fraction=float(valid.mean()),
        mean_native_coherence=float(coherence[use].mean()) if use.any() else 0,
        retained_component=int(component),
        units="mm",
        corrections_applied=[],
        phase_to_los_sign=1,
        los_sign_convention="positive_toward_satellite",
        source_native_shape=list(phase.shape),
    )


def cache_key(config, options):
    return hashlib.sha256(
        json.dumps(
            dict(config=config, options=options, adapter_version=ADAPTER_VERSION),
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]


def fetch_pair(product, session, config, grid, directory, options):
    name = product.properties["fileName"]
    earlier, later = product_dates(name)
    key = cache_key(config, options)
    destination = directory / key / (name + ".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            info = json.loads(archive["metadata_json"].item())
            result = dict(info, disp=archive["disp"], ok=archive["ok"])
    else:
        with RangeReader(
            session, product.properties["url"], block_size=8 * 2**20, max_blocks=16
        ) as remote:
            # Legacy NetCDF chunks are scattered: many small range requests can
            # cost more than fetching a modest file once. Bound memory explicitly;
            # large products continue using cached ranges. Full files stay in RAM.
            if remote.size <= 128 * 2**20:
                content, _ = remote._fetch(0, remote.size - 1)
                backing = io.BytesIO(content)
                strategy = "bounded_full_file_in_memory"
            else:
                backing = remote
                strategy = "http_ranges"
            with h5py.File(backing, "r") as source:
                result = reduce_pair(source, config, grid, **options)
            info = {k: v for k, v in result.items() if k not in ("disp", "ok")}
            info.update(
                source_url=product.properties["url"],
                source_file=name,
                bytes_fetched=remote.bytes_fetched,
                adapter_version=ADAPTER_VERSION,
                transfer_strategy=strategy,
                cache_key=key,
            )
        temporary = destination.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                disp=result["disp"],
                ok=result["ok"],
                metadata_json=json.dumps(info),
            )
        temporary.replace(destination)
        result.update(info)
    return dict(result, reference=earlier, secondary=later, file=name)


def choose_stacks(rows, max_epochs, max_baseline):
    groups = defaultdict(list)
    for row in rows:
        if 0 < row["baseline_days"] <= max_baseline:
            groups[(row["track"], row["direction"], row["tile"])].append(row)
    selected = []
    for key, group in groups.items():
        # Deterministically select one product for duplicate pairs within one footprint.
        unique = {}
        for row in sorted(group, key=lambda r: r["file"]):
            unique[(row["earlier_date"], row["later_date"])] = row
        frame = pd.DataFrame(unique.values()).rename(
            columns={"earlier_date": "ref_date", "later_date": "sec_date"}
        )
        component = _largest_component(frame)
        dates = sorted(component)[-max_epochs:] if max_epochs else sorted(component)
        frame = frame[frame.ref_date.isin(dates) & frame.sec_date.isin(dates)]
        if frame.empty:
            continue
        component = _largest_component(frame)
        frame = frame[frame.ref_date.isin(component) & frame.sec_date.isin(component)]
        selected.append((len(component), key, frame.to_dict("records")))
    return sorted(selected, key=lambda item: item[0], reverse=True)


def process_site(site, config, rows, directory, args, token):
    """One worker per site; h5py serializes reads within a process."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stacks = choose_stacks(rows, args.max_epochs, args.max_baseline_days)
    if not stacks:
        return dict(site=site, status="no_usable_catalog_stack", training_ready=False)
    session = asf.ASFSession().auth_with_token(token)
    options = dict(
        coherence_threshold=args.coherence_threshold,
        min_block_fraction=args.min_block_fraction,
    )
    grid = make_grid(config)
    attempts = []
    for _, stack, selected in stacks[: args.max_stack_attempts]:
        names = [r["file"].removesuffix(".nc") for r in selected]
        products = {}
        for start in range(0, len(names), 100):
            for product in asf.granule_search(names[start : start + 100]):
                products[product.properties["fileName"]] = product
        ordered = sorted(selected, key=lambda r: (r["sec_date"], r["ref_date"]))
        # Probe representative dates before downloading a complete unsuitable stack.
        probes = sorted(set([0, len(ordered) // 2, len(ordered) - 1]))
        order = [ordered[i] for i in probes] + [
            r for i, r in enumerate(ordered) if i not in probes
        ]
        pairs = []
        failures = []
        tested = 0
        stack_bad = False
        for row in order:
            name = row["file"]
            try:
                if name not in products:
                    raise ValueError("Product no longer resolves in ASF catalogue")
                pair = fetch_pair(
                    products[name], session, config, grid, directory / "pairs", options
                )
                if pair["valid_pixel_fraction"] < args.min_pair_fraction:
                    raise ValueError(
                        f"AOI valid fraction {pair['valid_pixel_fraction']:.3f} below {args.min_pair_fraction}"
                    )
                pairs.append(pair)
            except Exception as error:
                message = str(error).replace(token, "[redacted]")
                failures.append(dict(file=name, error=message))
            tested += 1
            if tested == len(probes) and not pairs:
                stack_bad = True
                break
            if tested % 10 == 0 or tested == len(order):
                print(
                    f"{site}: processed {tested}/{len(order)} pairs; usable={len(pairs)}",
                    flush=True,
                )
        attempt = dict(
            stack=list(stack),
            selected_pairs=len(selected),
            resolved_products=len(products),
            processed_pairs=tested,
            usable_pairs=len(pairs),
            failures=failures,
        )
        attempts.append(attempt)
        write_json(directory / "attempts.json", attempts)
        if stack_bad or not pairs:
            continue
        frame = pd.DataFrame(
            [dict(ref_date=p["reference"], sec_date=p["secondary"]) for p in pairs]
        )
        connected = _largest_component(frame)
        pairs = [
            p
            for p in pairs
            if p["reference"] in connected and p["secondary"] in connected
        ]
        if len(connected) < args.history + 2:
            continue
        meta = dict(
            units="mm",
            sensor="Sentinel-1",
            source="ARIA S1 GUNW via ASF",
            polarization="VV",
            volcano_id=config.get("volcano_id", site),
            volcano_name=config["volcano_name"],
            country=config["country"],
            volcano_number=config.get("volcano_number"),
            coordinate_source=config.get("coordinate_source"),
            stack_id=f"aria_{site}_{stack[0]}_{stack[1]}_{stack[2]}",
            stack=list(stack),
            grid_convention="aoi_block_mean32",
            grid=asdict(grid),
            spacing_km=[grid.dy / 1000, grid.dx / 1000],
            center_lonlat=[config["lon"], config["lat"]],
            half_width_km=config["half_km"],
            aoi_note=config.get("aoi_note"),
            geometry_available=False,
            geometry_note="LOS measurements retained; no conversion to vertical or ENU displacement",
            phase_to_los_sign=1,
            los_sign_convention="positive_toward_satellite",
            sign_verified_from=GUIDE,
            source_date_order="reference=newer; secondary=older",
            network_date_order="earlier to later",
            wavelength_m=sorted({p["wavelength_m"] for p in pairs}),
            reference_area="per-pair median over dominant coherent connected component",
            reference_caveat="Relative AOI deformation; changing reference pixels can introduce offsets; broad uniform deformation removed",
            coherence_threshold=args.coherence_threshold,
            min_block_valid_fraction=args.min_block_fraction,
            min_pair_valid_fraction=args.min_pair_fraction,
            corrections_applied=[],
            correction_note="Raw unwrapped phase; atmospheric, ionospheric and tide corrections not applied, matching existing raw NISAR inputs",
            max_epochs_requested=args.max_epochs,
            max_baseline_days=args.max_baseline_days,
            source_catalog_truncated=args.catalog_truncated,
            causality="Each input inversion uses only pairs with both acquisitions no later than forecast origin; archive publication latency is not modeled",
            pairs=[
                {
                    k: p[k]
                    for k in [
                        "file",
                        "reference",
                        "secondary",
                        "source_url",
                        "valid_pixel_fraction",
                        "wavelength_m",
                        "retained_component",
                    ]
                }
                for p in pairs
            ],
        )
        network = PairNetwork(pairs, grid, site, meta)
        result = export_network(network, directory, args)
        result.update(
            country=config["country"],
            stack=list(stack),
            catalog_products=len(rows),
            selected_pairs=len(selected),
            first_date=str(network.dates[0]),
            last_date=str(network.dates[-1]),
            sensor="Sentinel-1 ARIA GUNW",
            max_epochs=args.max_epochs,
            mean_pair_valid_fraction=float(
                np.mean([p["valid_pixel_fraction"] for p in pairs])
            ),
        )
        write_json(directory / "result.json", result)
        return result
    result = dict(
        site=site,
        country=config["country"],
        status="insufficient_quality_or_connected_history",
        training_ready=False,
        attempts=len(attempts),
    )
    write_json(directory / "result.json", result)
    return result


def safe_process(site, config, rows, directory, args, token):
    try:
        return process_site(site, config, rows, directory, args, token)
    except Exception as error:
        result = dict(
            site=site,
            status="error",
            training_ready=False,
            error=str(error).replace(token, "[redacted]"),
        )
        write_json(Path(directory) / "result.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-root", type=Path, default=Path("data/gunw_discovery")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/aria"))
    parser.add_argument("--sites", nargs="+")
    parser.add_argument(
        "--sites-config",
        type=Path,
        help="Optional site definitions overriding the discovery configurations",
    )
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=60,
        help="Latest connected acquisitions per stack; 0 means entire inventoried component",
    )
    parser.add_argument("--max-baseline-days", type=int, default=48)
    parser.add_argument("--max-stack-attempts", type=int, default=3)
    parser.add_argument("--coherence-threshold", type=float, default=0.4)
    parser.add_argument("--min-block-fraction", type=float, default=0.5)
    parser.add_argument("--min-pair-fraction", type=float, default=0.2)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--horizon-days", type=float, default=30)
    parser.add_argument("--target-tolerance-days", type=float, default=12)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.max_epochs < 0
        or args.max_stack_attempts < 1
        or args.history < 1
    ):
        parser.error(
            "workers, stack attempts and history must be positive; max-epochs nonnegative"
        )
    if not all(
        0 < x <= 1
        for x in [
            args.coherence_threshold,
            args.min_block_fraction,
            args.min_pair_fraction,
        ]
    ):
        parser.error("Quality fractions must lie in (0,1]")
    token = (
        args.token_file.read_text().strip()
        if args.token_file
        else os.environ.get("EARTHDATA_TOKEN", "").strip()
    )
    if not token:
        parser.error("Set EARTHDATA_TOKEN or --token-file; token is never logged")
    configs = json.loads((args.discovery_root / "candidate_sites.json").read_text())
    if args.sites_config:
        configs.update(json.loads(args.sites_config.read_text()))
    survey = json.loads((args.discovery_root / "aria_survey.json").read_text())
    requested = (
        set(args.sites) if args.sites else {r["site"] for r in survey["results"]}
    )
    unknown = requested - {r["site"] for r in survey["results"]}
    if unknown:
        parser.error("Sites absent from ARIA survey: " + ", ".join(sorted(unknown)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {}
        for row in survey["results"]:
            site = row["site"]
            if site not in requested:
                continue
            if not row.get("products"):
                results.append(
                    dict(
                        site=site,
                        status="no_archived_products_in_survey",
                        training_ready=False,
                    )
                )
                continue
            rows = json.loads(
                (args.discovery_root / "aria_catalogues" / f"{site}.json").read_text()
            )
            site_args = argparse.Namespace(**vars(args))
            site_args.catalog_truncated = row.get("truncated_at_limit", False)
            jobs[
                executor.submit(
                    safe_process,
                    site,
                    configs[site],
                    rows,
                    args.output_dir / site,
                    site_args,
                    token,
                )
            ] = site
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            write_json(
                args.output_dir / "summary.json",
                dict(
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    complete=False,
                    results=results,
                ),
            )
    write_json(
        args.output_dir / "summary.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            complete=True,
            max_epochs_per_site=args.max_epochs,
            sites_requested=len(requested),
            results=results,
        ),
    )
    ready = [r for r in results if r.get("training_ready")]
    write_json(
        args.output_dir / "ready_sites.json",
        {
            r["site"]: dict(configs[r["site"]], sensor="Sentinel-1", stack=r["stack"])
            for r in ready
        },
    )
    print(
        f"Completed: {len(ready)} training-ready sites; {sum(r.get('training_examples', 0) for r in ready)} training examples",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
