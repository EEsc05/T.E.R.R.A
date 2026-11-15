"""Create a human-readable, product-level inventory for a saved training run."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import h5py
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def day(value):
    return str(np.datetime64(value, "D"))


def coordinate(value, positive, negative):
    return f"{abs(value):.6f} degrees {positive if value >= 0 else negative}"


def public_url(value):
    if isinstance(value, bytes):
        value = value.decode()
    parts = urlsplit(str(value))
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Source product lacks a public HTTPS URL")
    # Do not export signed query strings or other transient authentication values.
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def archive_path(item, partition, protocol):
    if item["training_only"]:
        if partition == "reporting":
            return None
        path = Path(item["source_dir"]) / "examples.npz"
    else:
        name = "train.npz" if partition == "training" else "validation.npz"
        matches = [
            Path(p)
            for p in protocol["source_sha256"]
            if Path(p).parent.name == item["id"] and Path(p).name == name
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous archive for " + item["id"])
        path = matches[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != protocol["source_sha256"].get(str(path)):
        raise ValueError("Input hash differs from the actual model run: " + str(path))
    return path


def load_windows(path):
    if path is None:
        return []
    with np.load(path, allow_pickle=False) as a:
        return [
            dict(
                index=i,
                dates=[day(d) for d in dates],
                origin=day(dates[-1]),
                target=day(target),
                horizon_days=float((target - dates[-1]) / np.timedelta64(1, "D")),
                observed_target_pixels=int(mask.sum()),
            )
            for i, (dates, target, mask) in enumerate(
                zip(a["dates"], a["target_dates"], a["target_masks"])
            )
        ]


def product_url(pair, item, metadata):
    if pair.get("source_url"):
        return public_url(pair["source_url"])
    name = pair["file"]
    directory = Path(item["source_dir"])
    candidates = [directory / "subsets" / name]
    if item["physical_volcano"] == "ubinas":
        candidates.append(Path("data/nisar/ubinas/subsets") / name)
    for path in candidates:
        if path.is_file():
            with h5py.File(path, "r") as f:
                if "source_url" in f.attrs:
                    return public_url(f.attrs["source_url"])
    raise ValueError("Cannot verify source URL from saved provenance: " + name)


def export(run, output):
    manifest = read_json(run / "input_manifest.json")
    protocol = read_json(run / "protocol.json")
    countries = read_json("data/gunw_discovery/candidate_sites.json")
    features = read_json("data/gunw_discovery/gvp_holocene.geojson")["features"]
    gvp = {int(f["properties"]["Volcano_Number"]): f["properties"] for f in features}
    misti = next(p for p in gvp.values() if "misti" in p["Volcano_Name"].lower())
    sections = []
    window_sections = []
    product_sections = []
    all_products = []
    training_count = 0
    reporting_count = 0
    country_names = set()
    for number, item in enumerate(manifest["datasets"], 1):
        directory = Path(item["source_dir"])
        meta = read_json(directory / "metadata.json")
        config = countries.get(item["physical_volcano"], {})
        volcano_number = (
            meta.get("volcano_number")
            or config.get("volcano_number")
            or (
                misti["Volcano_Number"] if item["physical_volcano"] == "misti" else None
            )
        )
        location = gvp.get(int(volcano_number), {}) if volcano_number else {}
        country = (
            meta.get("country") or config.get("country") or location.get("Country")
        )
        if not country:
            raise ValueError("Missing country for " + item["id"])
        country_names.add(country)
        name = (
            meta.get("volcano_name")
            or location.get("Volcano_Name")
            or item["physical_volcano"]
        )
        if name == item["physical_volcano"]:
            name = name.replace("_", " ").title()
        lon, lat = meta["center_lonlat"]
        half = meta["half_width_km"]
        paths = {
            role: archive_path(item, role, protocol)
            for role in ["training", "reporting"]
        }
        windows = {role: load_windows(path) for role, path in paths.items()}
        assert len(windows["training"]) == item["training_examples"]
        assert len(windows["reporting"]) == item["reporting_examples"]
        training_count += len(windows["training"])
        reporting_count += len(windows["reporting"])
        cutoffs = {
            role: max((w["target"] for w in values), default=None)
            for role, values in windows.items()
        }
        pairs = []
        for pair in meta["pairs"]:
            earlier, later = pair["reference"], pair["secondary"]
            if earlier >= later:
                raise ValueError("Expected normalized chronological pair dates")
            url = product_url(pair, item, meta)
            if item["sensor"] == "NISAR" and "BETA" in url:
                raise ValueError("Unexpected BETA product in current run")
            entry = dict(
                file=pair["file"],
                url=url,
                earlier=earlier,
                later=later,
                training=bool(cutoffs["training"] and later <= cutoffs["training"]),
                reporting=bool(cutoffs["reporting"] and later <= cutoffs["reporting"]),
            )
            pairs.append(entry)
            all_products.append(entry)
        used = [p for p in pairs if p["training"] or p["reporting"]]
        counts = Counter(
            "training + reporting"
            if p["training"] and p["reporting"]
            else "training"
            if p["training"]
            else "reporting"
            if p["reporting"]
            else "prepared only"
            for p in pairs
        )
        acquisitions = sorted({p[k] for p in pairs for k in ["earlier", "later"]})
        description = (
            (
                "NISAR L-band GUNW"
                if item["sensor"] == "NISAR"
                else "Sentinel-1 C-band ARIA GUNW"
            )
            + " observations converted to relative line-of-sight ground-displacement maps around "
            + name
            + "."
        )
        title = f"{number:02d}. {name} -- {country} [{item['id']}]"
        lines = [
            title,
            "-" * len(title),
            "Mini description: " + description,
            "Regional context: "
            + str(
                location.get("Subregion") or location.get("Region") or "Not recorded"
            ),
            "GVP volcano type: "
            + str(location.get("Primary_Volcano_Type", "Not recorded")),
            f"Processed AOI center: {coordinate(lat, 'N', 'S')}; {coordinate(lon, 'E', 'W')}.",
            f"Processed AOI: {2 * half:g} x {2 * half:g} km; 32 x 32 cells; spacing {meta['spacing_km'][0]:g} x {meta['spacing_km'][1]:g} km.",
            "Sensor / polarization: "
            + item["sensor"]
            + " / "
            + str(
                meta.get("polarization")
                or ("HH" if item["sensor"] == "NISAR" else "VV")
            ),
            "Orbit/stack: "
            + json.dumps(
                meta.get("stack")
                or {
                    k: meta.get("product_info", {}).get(k)
                    for k in ["track_number", "orbit_pass_direction", "frame_number"]
                }
            ),
            f"Prepared reconstruction source: {acquisitions[0]} to {acquisitions[-1]}; {len(acquisitions)} dates; {len(pairs)} retained GUNW pairs.",
            f"GUNW pairs available to training or reporting inversions: {len(used)}; roles: "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            f"Supervised windows: {item['training_examples']} training; {item['reporting_examples']} held-out reporting.",
            "Evaluation role: "
            + (
                "TRAINING ONLY; no independent reporting score for this history."
                if item["training_only"]
                else "Separate training and purged temporal reporting partitions."
            ),
            "Original prepared folder: " + str(directory.resolve()),
            "Metadata: " + str((directory / "metadata.json").resolve()),
            "GVP location / volcano information: https://volcano.si.edu/volcano.cfm?vn="
            + str(volcano_number),
            f"Map of processed AOI center: https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=10/{lat}/{lon}",
        ]
        if location:
            lines.append(
                "GVP catalogue coordinate (may differ slightly from processed AOI center): "
                + coordinate(location["Latitude"], "N", "S")
                + "; "
                + coordinate(location["Longitude"], "E", "W")
                + "."
            )
        for role, values in windows.items():
            if not values:
                continue
            lines.extend(
                [
                    role.title()
                    + " archive actually read by the run: "
                    + str(paths[role].resolve()),
                    "SHA256: " + protocol["source_sha256"][str(paths[role])],
                    f"{role.title()} supervised-window coverage: input dates {min(w['dates'][0] for w in values)} onward; target dates {min(w['target'] for w in values)} to {max(w['target'] for w in values)}.",
                    f"{role.title()} last inversion cutoff: {cutoffs[role]}; observed horizons: "
                    + ", ".join(
                        f"{x:g}" for x in sorted({w["horizon_days"] for w in values})
                    )
                    + " days.",
                ]
            )
        lines.append(
            "Representative source links (complete product list in Appendix B):"
        )
        samples = sorted(set([0, len(used) // 2, len(used) - 1])) if used else []
        for i in samples:
            lines.append(
                "  "
                + used[i]["earlier"]
                + " -> "
                + used[i]["later"]
                + ": "
                + used[i]["url"]
            )
        sections.extend(lines + [""])
        window_sections.extend([title])
        for role, values in windows.items():
            for w in values:
                window_sections.append(
                    f"  {role.upper()} archive row {w['index']:03d} | input acquisitions: "
                    + ", ".join(w["dates"])
                    + f" | target: {w['target']} | lead: {w['horizon_days']:g} days | observed target-mask pixels: {w['observed_target_pixels']}"
                )
        window_sections.append("")
        product_sections.extend([title])
        for p in pairs:
            role = (
                "TRAINING + REPORTING"
                if p["training"] and p["reporting"]
                else "TRAINING"
                if p["training"]
                else "REPORTING"
                if p["reporting"]
                else "PREPARED ONLY: not used for this run's fitting or reporting"
            )
            product_sections.extend(
                [
                    f"  [{role}] {p['earlier']} -> {p['later']}",
                    "  Product: " + p["file"],
                    "  Download: " + p["url"],
                    "",
                ]
            )
    assert training_count == manifest["training_examples"] == 254
    assert reporting_count == manifest["reporting_examples"] == 62
    used_unique = {p["file"] for p in all_products if p["training"] or p["reporting"]}
    train_unique = {p["file"] for p in all_products if p["training"]}
    report_unique = {p["file"] for p in all_products if p["reporting"]}
    heading = [
        "T.E.R.R.A -- DATA SOURCES FOR THE LATEST MODEL SET",
        "=" * 55,
        "Model run: " + str(run.resolve()),
        "Generated (UTC): " + datetime.now(timezone.utc).isoformat(),
        f"Scope: {len(manifest['datasets'])} separate sensor/orbit histories; {manifest['physical_volcanoes']} physical volcanoes; {len(country_names)} countries.",
        f"Model inputs: {training_count} supervised training examples; {reporting_count} held-out reporting examples across {len(protocol['reporting_examples_by_volcano'])} volcanoes.",
        f"Unique source GUNW product files available to fitting/reconstruction: {len(train_unique)}.",
        f"Unique source GUNW product files available to reporting/reconstruction: {len(report_unique)} (overlap with training is expected).",
        f"Unique source GUNW files available to either role: {len(used_unique)}.",
        "Countries: " + ", ".join(sorted(country_names)),
        "Models: persistence, EMA30, smoothing, robust low-rank, four-branch ensemble, CVAE.",
        "",
        "CONTENTS",
        "1. Data representation, sources and interpretation",
        "2. Dataset-by-dataset inventory",
        "Appendix A. Every supervised training/reporting example",
        "Appendix B. Every retained source GUNW product, its link and use classification",
        "",
        "1. DATA REPRESENTATION, SOURCES AND INTERPRETATION",
        "-" * 50,
        "The actual model inputs are NumPy .npz archives, not CSV files. Each example contains six dated 32 x 32 displacement maps in millimetres, validity masks, and a real future acquisition as the target. Observed leads in this run are 18, 24, 30 and 36 days. No target labels were interpolated in time.",
        "Source measurements are NISAR GUNW HDF5 and Sentinel-1 ARIA GUNW NetCDF products. These are unwrapped radar interferograms representing pairs of acquisition dates. Raw Sentinel-1/NISAR SLC files were not ingested directly.",
        "Maps represent relative line-of-sight displacement, positive toward the satellite. They are not vertical displacement, official alerts or eruption probabilities. Histories retain separate sensor and viewing-direction metadata; matching units and dimensions do not eliminate sensor differences.",
        "Each history/target is reconstructed using retained pairs no later than its acquisition cutoff. Each pair is referenced to its retained-pixel median. Additional atmospheric/tide corrections were not applied by this pipeline. Product publication latency is not modeled.",
        "Training means inputs used for fitting and internal model selection. Reporting means held-out inputs/targets used only for final scoring. ARIA Ambrym and both Mount Asama histories are training only. The full comparison is retrospective; 12 reporting windows were reused from the previous expanded run.",
        "Product-use labels in Appendix B are based on the retained pair network and the latest target cutoff in each partition. A label means that product was eligible for at least one relevant cumulative inversion, not that every native pixel contributed. Some retained products may be prepared-only; these are explicitly marked and excluded from the used-product totals.",
        "ASF product links point to original full products and generally require Earthdata sign-in. Local cached HDF5 subsets/reduced pairs are derivatives. This inventory includes no credentials or signed URL query strings.",
        "GVP metadata supplies location/country/type context; it is not a displacement observation or an eruption label. AOI centers below are the coordinates actually used, which may differ slightly from GVP summit coordinates. Map links are for orientation only.",
        "Only the prepared, quality-screened portion of the archive entered this run. ARIA processing was capped at 60 connected acquisitions per selected stack; this is not a list of every product found in the broader survey. Manam and Lastarria did not produce usable training histories. Earlier Ubinas BETA pairs, exploratory Japan pixel-window surveys and historical hackathon checkpoints were not fitting inputs. Older model sets were loaded only for post-training comparison.",
        "Source/reference links:",
        "  ASF product search: https://search.asf.alaska.edu/",
        "  NISAR GUNW guide: https://nisar-docs.asf.alaska.edu/gunw/",
        "  ARIA GUNW guide: https://hyp3-docs.asf.alaska.edu/guides/gunw_product_guide/",
        "  Shared phase convention: https://github.com/aria-tools/ARIA-tools#readme",
        "  Smithsonian GVP web services: https://volcano.si.edu/database/webservices.cfm",
        "  GVP coordinate/type snapshot used for this inventory: "
        + str(Path("data/gunw_discovery/gvp_holocene.geojson").resolve()),
        "  Input manifest: " + str((run / "input_manifest.json").resolve()),
        "  Actual run protocol / input hashes: "
        + str((run / "protocol.json").resolve()),
        "  Model scores and limitations: " + str((run / "REPORT.md").resolve()),
        "  ARIA preparation report: " + str(Path("data/aria/REPORT.md").resolve()),
        "",
        "2. DATASET-BY-DATASET INVENTORY",
        "=" * 50,
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            heading
            + sections
            + [
                "APPENDIX A. EVERY SUPERVISED EXAMPLE",
                "=" * 50,
                "Rows are zero-based indices in the exact archives named above. Target-mask counts describe observed target pixels before the additional common historical-support scoring mask.",
                "",
            ]
            + window_sections
            + [
                "APPENDIX B. COMPLETE RETAINED PRODUCT LIST",
                "=" * 50,
                "Dates are chronological earlier -> later. ARIA native filenames list the newer reference date first; the displayed dates are normalized for the forecasting reconstruction.",
                "",
            ]
            + product_sections
        )
        + "\n"
    )
    summary = dict(
        output=str(output),
        datasets=len(manifest["datasets"]),
        training_examples=training_count,
        reporting_examples=reporting_count,
        countries=len(country_names),
        unique_training_products=len(train_unique),
        unique_reporting_products=len(report_unique),
        unique_used_products=len(used_unique),
        retained_dataset_product_references=len(all_products),
        prepared_only_references=sum(
            not (p["training"] or p["reporting"]) for p in all_products
        ),
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("outputs/nisar_aria_all_data"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    print(json.dumps(export(a.run, a.output or a.run / "DATA_SOURCES.txt"), indent=2))


if __name__ == "__main__":
    main()
