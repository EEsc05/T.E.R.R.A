"""Inventory existing Sentinel-1 ARIA GUNW separately from NISAR inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import asf_search as asf
import pandas as pd
from shapely.geometry import box
from . import nisar_processing as p

PATTERN = re.compile(r"^S1-GUNW-[AD]-[LR]-(\d+)-tops-(\d{8})_(\d{8})-[^-]+-([^-.]+)-")


def search(site, config, root, end):
    poly = box(*p.aoi_lonlat(config["lon"], config["lat"], config["half_km"]))
    products = asf.search(
        dataset=asf.DATASET.ARIA_S1_GUNW,
        processingLevel="GUNW_STD",
        intersectsWith=poly.wkt,
        start="2014-01-01",
        end=end,
        maxResults=5000,
    )
    rows = []
    groups = {}
    for product in products:
        props = product.properties
        name = props.get("fileName", "")
        match = PATTERN.search(name)
        if not match:
            continue
        a, b = sorted(
            [
                datetime.strptime(match[2], "%Y%m%d").date(),
                datetime.strptime(match[3], "%Y%m%d").date(),
            ]
        )
        direction = props.get("flightDirection")
        track = int(match[1])
        location = match[4]
        row = {
            "file": name,
            "track": track,
            "direction": direction,
            "tile": location,
            "earlier_date": str(a),
            "later_date": str(b),
            "baseline_days": (b - a).days,
        }
        rows.append(row)
        if 0 < row["baseline_days"] <= 48:
            groups.setdefault((track, direction, location), []).append(row)
    (root / "aria_catalogues" / f"{site}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    stacks = []
    for key, group in groups.items():
        unique = {(r["earlier_date"], r["later_date"]): r for r in group}
        component = p._largest_component(
            pd.DataFrame(unique.values()).rename(
                columns={"earlier_date": "ref_date", "later_date": "sec_date"}
            )
        )
        stacks.append(
            {
                "track": key[0],
                "direction": key[1],
                "nominal_tile": key[2],
                "connected_dates": len(component),
                "first_date": min(component),
                "last_date": max(component),
                "unique_pairs": sum(
                    r["earlier_date"] in component and r["later_date"] in component
                    for r in unique.values()
                ),
            }
        )
    stacks.sort(key=lambda s: s["connected_dates"], reverse=True)
    return {
        "site": site,
        "country": config["country"],
        "volcano_name": config["volcano_name"],
        "products": len(products),
        "truncated_at_limit": len(products) >= 5000,
        "parsed_products": len(rows),
        "sensor": "Sentinel-1 ARIA GUNW",
        "format": "netCDF",
        "requires_separate_input_adapter": True,
        "dates_for_connectivity_only": "Chronological endpoints, not the product phase/reference convention",
        "stacks": stacks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-root", type=Path, default=Path("data/gunw_discovery")
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    root = args.discovery_root
    (root / "aria_catalogues").mkdir(exist_ok=True)
    candidates = json.loads((root / "candidate_sites.json").read_text())
    # Survey one volcano per country first; preserve sensors as separate inventories.
    chosen = {}
    countries = set()
    for site, config in candidates.items():
        if config["country"] in countries:
            continue
        countries.add(config["country"])
        chosen[site] = config
    summaries = []
    end = datetime.now(timezone.utc).date().isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(search, site, config, root, end): site
            for site, config in chosen.items()
        }
        for future in as_completed(pending):
            site = pending[future]
            try:
                row = future.result()
                summaries.append(row)
                dates = max([s["connected_dates"] for s in row["stacks"]], default=0)
                print(
                    f"{site}: {row['products']} ARIA GUNW products; connected dates={dates}",
                    flush=True,
                )
            except Exception as error:
                summaries.append({"site": site, "error": str(error)})
    (root / "aria_survey.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "start": "2014-01-01",
                "end": end,
                "site_count": len(chosen),
                "max_baseline_days_for_connectivity": 48,
                "note": "Existing archive only; separate sensor, format, wavelength and grid. Not fed to NISAR models.",
                "results": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
