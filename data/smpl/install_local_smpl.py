"""Validate and link an existing licensed local SMPL neutral model.

No models are downloaded. The source remains unchanged, and existing targets
are never overwritten. Runtime loading uses CrowdSim's restricted unpickler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from CrowdSim.tools.smpl_mesh_visualizer import (  # noqa: E402
    inspect_smpl_pickle,
    load_smpl_model_data,
)

LOCAL_NEUTRAL = ROOT.parent / (
    "arena_sangfor/arena_ws/arena_assets/smpl/SMPL_python_v.1.1.0/"
    "smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
)


def install(source: Path, destination: Path) -> dict:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    start = time.perf_counter()
    audit = inspect_smpl_pickle(source)
    arrays = load_smpl_model_data(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if destination.exists() or destination.is_symlink():
        if not destination.exists() or not destination.samefile(source):
            raise FileExistsError(f"Refusing to replace existing SMPL target: {destination}")
        action = "already-linked"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
        action = "linked"
    return {
        "action": action,
        "source": str(source),
        "destination": str(destination),
        "sha256": digest.hexdigest(),
        "bytes": audit["bytes"],
        "pickle_globals": audit["globals"],
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "validation_seconds": time.perf_counter() - start,
        "license": "Existing local model only; retain its original SMPL license.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=LOCAL_NEUTRAL)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = install(args.source, args.destination)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
