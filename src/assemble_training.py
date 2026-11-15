"""Assemble preserved NISAR and successfully prepared ARIA training partitions."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .dataset import TrainingDataset, validate_split
from .prepare_nisar import write_json


def assemble(nisar_root, aria_roots, output_dir, japan_root):
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((nisar_root / "input_sources.json").read_text())
    datasets = []
    training_only = []
    sites = []
    sources = {}
    candidates = []
    for directory in sorted(nisar_root.iterdir()):
        if not (directory / "train.npz").exists():
            continue
        record = provenance["files"][str(directory / "train.npz")]
        original = Path(record["source"])
        full_source = original.parent if original.is_file() else directory
        candidates.append(("nisar_" + directory.name, directory, full_source, "NISAR"))
    for root in aria_roots:
        for result_file in sorted(root.glob("*/result.json")):
            result = json.loads(result_file.read_text())
            directory = result_file.parent
            if result.get("training_ready"):
                candidates.append(
                    ("aria_" + directory.name, directory, directory, "Sentinel-1")
                )
            elif result.get("status") == "generated" and result.get("examples", 0) > 0:
                training_only.append(
                    ("aria_" + directory.name, directory, "Sentinel-1")
                )
    for label, directory, full_source, sensor in candidates:
        if label in sites:
            raise ValueError("Duplicate prepared stack: " + label)
        train = TrainingDataset.from_npz(directory / "train.npz")
        validation = TrainingDataset.from_npz(directory / "validation.npz")
        validate_split(train, validation)
        destination = output_dir / label
        destination.mkdir()
        for name in ["train.npz", "validation.npz"]:
            source = directory / name
            target = destination / name
            shutil.copy2(source, target)
            sources[str(target)] = dict(
                source=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        sites.append(label)
        datasets.append(
            dict(
                id=label,
                source_dir=str(full_source),
                sensor=sensor,
                physical_volcano=train.inputs[0].volcano_id,
                training_examples=len(train),
                reporting_examples=len(validation),
                training_only=False,
            )
        )
    extras = []
    for name in ["mount_asama_ascending", "mount_asama_descending"]:
        training_only.append(("nisar_" + name, japan_root / name, "NISAR"))
    for label, directory, sensor in training_only:
        path = directory / "examples.npz"
        data = TrainingDataset.from_npz(path)
        extras.append(str(path))
        datasets.append(
            dict(
                id=label,
                source_dir=str(directory),
                sensor=sensor,
                physical_volcano=data.inputs[0].volcano_id,
                training_examples=len(data),
                reporting_examples=0,
                training_only=True,
            )
        )
        sources[str(path)] = dict(
            source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    result = dict(
        sites=sites,
        extra_training=extras,
        datasets=datasets,
        files=sources,
        training_examples=sum(d["training_examples"] for d in datasets),
        reporting_examples=sum(d["reporting_examples"] for d in datasets),
        physical_volcanoes=len({d["physical_volcano"] for d in datasets}),
        note="Preserved input copies; sensor/orbit histories remain separate. Partitioning is retrospective across sites.",
    )
    write_json(output_dir / "manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nisar-root", type=Path, default=Path("data/nisar_training_global")
    )
    p.add_argument(
        "--aria-roots",
        type=Path,
        nargs="+",
        default=[Path("data/aria"), Path("data/aria_island")],
    )
    p.add_argument("--japan-root", type=Path, default=Path("data/nisar_japan"))
    p.add_argument("--output-dir", type=Path, default=Path("data/nisar_aria_training"))
    a = p.parse_args()
    r = assemble(a.nisar_root, a.aria_roots, a.output_dir, a.japan_root)
    print(
        json.dumps(
            {
                k: r[k]
                for k in [
                    "training_examples",
                    "reporting_examples",
                    "physical_volcanoes",
                    "sites",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
