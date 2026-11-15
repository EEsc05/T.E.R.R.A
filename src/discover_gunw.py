"""Discover NISAR GUNW training opportunities across countries without downloading rasters."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
import requests
import numpy as np
import asf_search as asf
from shapely.geometry import box, shape
from . import nisar_processing as p
from .output import _json_value

GVP_URL = "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows"
GVP_LAYER = "GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes"
# Broad, geographically diverse initial survey; candidates are resolved from GVP, not guessed coordinates.
CANDIDATES = [
    "Kilauea",
    "Mauna Loa",
    "Etna",
    "Vesuvio",
    "Campi Flegrei",
    "Piton de la Fournaise",
    "Merapi",
    "Agung",
    "Semeru",
    "Krakatau",
    "Sinabung",
    "Taal",
    "Mayon",
    "Pinatubo",
    "Kanlaon",
    "Ruapehu",
    "Tongariro",
    "Taupo",
    "Askja",
    "Hekla",
    "Grimsvotn",
    "Reykjanes",
    "Katla",
    "Popocatepetl",
    "Colima",
    "Fuego",
    "Pacaya",
    "Santa Maria",
    "Masaya",
    "Concepcion",
    "Cotopaxi",
    "Tungurahua",
    "Sierra Negra",
    "Wolf",
    "Fernandina",
    "Lascar",
    "Villarrica",
    "Nevados de Chillan",
    "Lastarria",
    "Uturuncu",
    "Corbetti",
    "Aluto",
    "Erta Ale",
    "Dabbahu",
    "Fentale",
    "Nyiragongo",
    "Nyamuragira",
    "Ol Doinyo Lengai",
    "Soufriere Hills",
    "Soufriere St. Vincent",
    "Nevado del Ruiz",
    "Galeras",
    "Rabaul",
    "Bagana",
    "Manam",
    "Yasur",
    "Ambrym",
    "Ticsani",
    "Huaynaputina",
    "Tutupaca",
    "Yucamane",
    "Coropuna",
    "Sabancaya",
    "Ubinas",
    "Misti",
    "Asamayama",
]


def normalized(name):
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    key = "".join(sorted(re.findall("[a-z0-9]+", ascii_name.lower())))
    return {"elmisti": "misti"}.get(key, key)


def write(path, value):
    Path(path).write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n"
    )


def candidate_sites(root):
    path = root / "gvp_holocene.geojson"
    if not path.exists():
        response = requests.get(
            GVP_URL,
            params={
                "service": "WFS",
                "version": "1.0.0",
                "request": "GetFeature",
                "typeName": GVP_LAYER,
                "maxFeatures": 2000,
                "outputFormat": "application/json",
            },
            timeout=60,
        )
        response.raise_for_status()
        write(path, response.json())
    features = json.loads(path.read_text())["features"]
    wanted = {normalized(n) for n in CANDIDATES}
    sites = {}
    for feature in features:
        row = feature["properties"]
        key = normalized(row["Volcano_Name"])
        if key not in wanted:
            continue
        number = int(row["Volcano_Number"])
        slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            unicodedata.normalize("NFKD", row["Volcano_Name"])
            .encode("ascii", "ignore")
            .decode()
            .lower(),
        ).strip("_")
        # Only terrestrial targets; ocean grids cannot provide this displacement history.
        if float(row.get("Elevation") or 0) < 0:
            continue
        sites[slug] = {
            "volcano_id": {
                354006: "sabancaya",
                354020: "ubinas",
                354010: "misti",
                283110: "mount_asama",
            }.get(number, slug),
            "volcano_name": row["Volcano_Name"],
            "country": row["Country"],
            "lon": float(row["Longitude"]),
            "lat": float(row["Latitude"]),
            "half_km": 15.0,
            "volcano_number": number,
            "coordinate_source": f"https://volcano.si.edu/volcano.cfm?vn={number}",
        }
    write(root / "candidate_sites.json", sites)
    write(
        root / "candidate_selection.json",
        {
            "source": GVP_URL,
            "layer": GVP_LAYER,
            "global_volcano_records": len(features),
            "resolved_candidates": len(sites),
            "unresolved_requested_names": sorted(
                wanted - {normalized(s["volcano_name"]) for s in sites.values()}
            ),
        },
    )
    return sites


def catalogue_site(site, config, root, known, start, end, max_baseline):
    poly = box(*p.aoi_lonlat(config["lon"], config["lat"], config["half_km"]))
    results = asf.search(
        dataset="NISAR",
        processingLevel="GUNW",
        intersectsWith=poly.wkt,
        start=start,
        end=end,
        maxResults=5000,
    )
    rows = []
    for product in results:
        props = product.properties
        name = props.get("fileName", "")
        match = p.GUNW_RE.search(name)
        if not match:
            continue
        ref = datetime.strptime(match["ref_start"][:8], "%Y%m%d").date()
        sec = datetime.strptime(match["sec_start"][:8], "%Y%m%d").date()
        pol = (
            "HH"
            if "_SH_" in name
            else "VV"
            if "_SV_" in name
            else str(props.get("polarization", "unknown"))
        )
        try:
            coverage = shape(product.geometry).intersection(poly).area / poly.area
        except (TypeError, ValueError, AttributeError):
            coverage = 0.0
        rows.append(
            {
                "file": name,
                "track": int(match["track"]),
                "direction": match["dir"],
                "frame": int(match["frame"]),
                "polarization": pol,
                "ref_date": ref.isoformat(),
                "sec_date": sec.isoformat(),
                "baseline_days": (sec - ref).days,
                "aoi_coverage": float(coverage),
                "collection": props.get("collectionName"),
                "processing_date": props.get("processingDate"),
                "known_product_id": name in known,
            }
        )
    write(root / "catalogues" / f"{site}.json", rows)
    groups = {}
    for row in rows:
        if (
            not 0 < row["baseline_days"] <= max_baseline
            or "BETA" in str(row["collection"]).upper()
        ):
            continue
        key = (row["track"], row["direction"], row["frame"], row["polarization"])
        groups.setdefault(key, []).append(row)
    stacks = []
    import pandas as pd

    for (track, direction, frame, polarization), group in groups.items():
        group = sorted(
            group,
            key=lambda r: (
                r["ref_date"],
                r["sec_date"],
                str(r["processing_date"]),
                r["file"],
            ),
        )
        unique = {(r["ref_date"], r["sec_date"]): r for r in group}
        frame_data = pd.DataFrame(unique.values())
        component = p._largest_component(frame_data)
        selected = [
            r
            for r in unique.values()
            if r["ref_date"] in component and r["sec_date"] in component
        ]
        epochs = sorted(component)
        # Candidate supervised windows use actual target dates and exclude the zero datum.
        examples = 0
        for origin in range(6, len(epochs) - 1):
            lags = np.asarray(
                [
                    (
                        datetime.fromisoformat(d)
                        - datetime.fromisoformat(epochs[origin])
                    ).days
                    for d in epochs[origin + 1 :]
                ]
            )
            if np.min(abs(lags - 30)) <= 12:
                examples += 1
        stacks.append(
            {
                "site": site,
                **config,
                "stack": [track, direction, frame],
                "polarization": polarization,
                "pairs": len(selected),
                "epochs": len(epochs),
                "first_date": epochs[0],
                "last_date": epochs[-1],
                "catalogue_only_potential_examples": examples,
                "new_product_ids": sum(not r["known_product_id"] for r in selected),
                "minimum_aoi_coverage": min(r["aoi_coverage"] for r in selected),
                "collections": sorted({str(r["collection"]) for r in selected}),
                "already_sampled_physical_site": config["volcano_number"]
                in {354006, 354020, 354010, 283110},
                "pixel_quality_verified": False,
            }
        )
    return {
        "site": site,
        **config,
        "products": len(results),
        "parsed_products": len(rows),
        "truncated_at_limit": len(results) >= 5000,
        "stacks": stacks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/gunw_discovery"))
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument(
        "--end-date", default=datetime.now(timezone.utc).date().isoformat()
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-baseline-days", type=int, default=36)
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Verify Earthdata authentication; never saved or printed",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing same-date site results from this survey",
    )
    args = parser.parse_args(argv)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "catalogues").mkdir(exist_ok=True)
    authentication = "not requested"
    if args.token_file:
        token = args.token_file.read_text().strip()
        try:
            asf.ASFSession().auth_with_token(token)
            authentication = "successful"
        except Exception:
            authentication = "failed or expired"
        finally:
            del token
    sites = candidate_sites(root)
    known = {
        f.name
        for base in [Path("data/nisar"), Path("data/nisar_japan")]
        for f in base.rglob("*.h5")
    }
    summaries = []
    cached = {}
    previous_path = root / "survey.json"
    if args.reuse_cache and previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if (
            previous.get("start") == args.start_date
            and previous.get("end") == args.end_date
            and previous.get("max_baseline_days") == args.max_baseline_days
        ):
            cached = {
                row["site"]: row for row in previous["results"] if "error" not in row
            }
    summaries.extend(cached[site] for site in sites if site in cached)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                catalogue_site,
                site,
                config,
                root,
                known,
                args.start_date,
                args.end_date,
                args.max_baseline_days,
            ): site
            for site, config in sites.items()
            if site not in cached
        }
        for future in as_completed(pending):
            site = pending[future]
            try:
                summary = future.result()
                summaries.append(summary)
                best = max((s["epochs"] for s in summary["stacks"]), default=0)
                print(
                    f"{site}: {summary['country']}; {summary['products']} GUNW products; best connected dates={best}",
                    flush=True,
                )
            except Exception as error:
                summaries.append({"site": site, "error": str(error)})
                print(f"{site}: query failed ({type(error).__name__})", flush=True)
    stacks = [stack for summary in summaries for stack in summary.get("stacks", [])]
    stacks.sort(
        key=lambda s: (
            s["catalogue_only_potential_examples"],
            s["epochs"],
            s["minimum_aoi_coverage"],
        ),
        reverse=True,
    )
    write(root / "all_stacks.json", stacks)
    write(
        root / "survey.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "authentication": authentication,
            "sensor": "NISAR",
            "processing_level": "GUNW",
            "start": args.start_date,
            "end": args.end_date,
            "max_baseline_days": args.max_baseline_days,
            "candidates": len(sites),
            "known_product_ids": len(known),
            "countries": sorted({s["country"] for s in sites.values()}),
            "results": summaries,
            "note": "Catalogue availability only. Pixel coherence, masks, component connectivity and training yield require actual subset processing.",
        },
    )
    shortlist = {}
    countries = set()
    for stack in stacks:
        if (
            stack["country"] in countries
            or stack["country"] in ["Peru", "Japan"]
            or stack["polarization"] != "HH"
        ):
            continue
        if stack["epochs"] < 18 or stack["minimum_aoi_coverage"] < 0.9:
            continue
        slug = stack["site"]
        shortlist[slug] = {
            k: stack[k]
            for k in [
                "lon",
                "lat",
                "half_km",
                "stack",
                "volcano_id",
                "volcano_name",
                "country",
                "volcano_number",
                "coordinate_source",
            ]
        }
        countries.add(stack["country"])
        if len(shortlist) == 5:
            break
    write(root / "shortlist_sites.json", shortlist)
    print(
        json.dumps(
            {
                "authentication": authentication,
                "surveyed_volcanoes": len(sites),
                "countries": len({s["country"] for s in sites.values()}),
                "stacks": len(stacks),
                "shortlisted_new_countries": list(countries),
                "shortlisted_sites": list(shortlist),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
