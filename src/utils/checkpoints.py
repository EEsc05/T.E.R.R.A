"""Verify byte-identical copied checkpoints and remap the one historical pickle class."""

from pathlib import Path
import json, hashlib
from joblib.numpy_pickle import NumpyUnpickler

# Parameter files stay in their original location outside src.
ROOT = Path(__file__).resolve().parents[2] / "hackathon_stage1"


def verified_path(relative):
    path = ROOT / relative
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    row = next(
        (r for r in manifest["copied_files"] if r["copied_path"] == relative), None
    )
    if row is None:
        raise ValueError(
            "Checkpoint/config is not registered in SOURCE_MANIFEST.json: " + relative
        )
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != row["sha256"]:
        raise ValueError("Copied research file checksum mismatch: " + relative)
    return path


def config(relative):
    return json.loads(verified_path(relative).read_text())


class CheckpointUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if (
            module == "volcanic_forecasting_v2.src.research_v4.representation"
            and name == "SpatialBasis"
        ):
            from ..preprocessing.basis import SpatialBasis

            return SpatialBasis
        return super().find_class(module, name)


def load_joblib(relative):
    path = verified_path(relative)
    # The original checkpoints are uncompressed joblib dumps. Read them without importing the old repository.
    with path.open("rb") as stream:
        return CheckpointUnpickler(
            str(path), stream, ensure_native_byte_order=True, mmap_mode=None
        ).load()
