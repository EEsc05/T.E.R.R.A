"""Sample earliest/middle/latest NISAR GUNW rasters before a larger preparation run."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import asf_search as asf
import numpy as np
import pandas as pd
from . import nisar_processing as p
from .nisar_remote import download_subset_worker
from .output import _json_value


def probe(properties, headers, root, config):
    path, fetched = download_subset_worker(properties, headers, root, config)
    grid = p.make_grid(path, config["lon"], config["lat"], config["half_km"], 32, 32)
    result = p.read_pair(path, grid)
    return {
        "file": path.name,
        "range_bytes_fetched": fetched,
        "valid_fraction": float(result["ok"].mean()) if result else 0.0,
        "valid_pixels": int(result["ok"].sum()) if result else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-config", type=Path, required=True)
    parser.add_argument(
        "--catalogue-root", type=Path, default=Path("data/gunw_discovery")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/nisar_global"))
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    sites = json.loads(args.sites_config.read_text())
    jobs = []
    for site, config in sites.items():
        rows = json.loads(
            (args.catalogue_root / "catalogues" / f"{site}.json").read_text()
        )
        rows = [
            r
            for r in rows
            if [r["track"], r["direction"], r["frame"]] == config["stack"]
            and r["polarization"] == "HH"
            and 0 < r["baseline_days"] <= 36
            and "BETA" not in str(r["collection"]).upper()
        ]
        unique = {
            (r["ref_date"], r["sec_date"]): r
            for r in sorted(rows, key=lambda r: (str(r["processing_date"]), r["file"]))
        }
        component = p._largest_component(pd.DataFrame(unique.values()))
        selected = sorted(
            [
                r
                for r in unique.values()
                if r["ref_date"] in component and r["sec_date"] in component
            ],
            key=lambda r: (r["ref_date"], r["sec_date"]),
        )
        for index in sorted({0, len(selected) // 2, len(selected) - 1}):
            jobs.append((site, config, selected[index]["file"]))
    products = asf.granule_search([job[2].removesuffix(".h5") for job in jobs])
    by_name = {
        product.properties["fileName"]: product.properties for product in products
    }
    token = args.token_file.read_text().strip()
    session = asf.ASFSession().auth_with_token(token)
    summaries = {
        site: {"config": config, "samples": []} for site, config in sites.items()
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                probe,
                by_name[name],
                dict(session.headers),
                args.data_root / site / "subsets",
                config,
            ): site
            for site, config, name in jobs
        }
        for future in as_completed(pending):
            site = pending[future]
            try:
                row = future.result()
                summaries[site]["samples"].append(row)
                print(
                    f"{site}: sample valid pixels={row['valid_fraction']:.1%}",
                    flush=True,
                )
            except Exception as error:
                summaries[site]["samples"].append(
                    {"error": str(error).replace(token, "[redacted]")}
                )
    for site, record in summaries.items():
        samples = record["samples"]
        record["passes_three_sample_coverage_check"] = len(samples) == 3 and all(
            s.get("valid_fraction", 0) >= 0.2 for s in samples
        )
    path = args.catalogue_root / "quality_probes.json"
    path.write_text(
        json.dumps(_json_value(summaries), indent=2, allow_nan=False) + "\n"
    )
    print("Wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
