"""Future actual4x200s 45D diagnostic using saved AMASS centers, never refit."""

from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "pose_diversity_candidate_v1"
BASIS = BASE / "actual_evaluation_v1/reference_basis.npz"
BASIS_SHA = "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-admission", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("CPU-only and new evidence directory required")
    evidence = load_module(
        "protected_pose45_actual_source",
        HERE / "terrain_protected_evaluation_evidence_v1.py",
    )
    result = json.loads((args.input / "result.json").read_text())
    source = evidence.admit_actual_evaluation(
        result, "navigation", args.checkpoint_admission, args.import_report
    )
    ledger_bytes, events_bytes = (
        source.pop("ledger_bytes"),
        source.pop("event_prefix_bytes"),
    )
    if evidence.record(BASIS)["sha256"] != BASIS_SHA:
        raise ValueError("Actual frozen AMASS-only45D reference centers changed")
    original = json.loads((BASE / "actual_evaluation_v1/evaluation.json").read_text())
    path = BASE / "evaluate_pose_diversity.py"
    # Bind the frozen implementation before source-load; the producer is never edited.
    freeze = json.loads((BASE / "source_freeze_v1.json").read_text())
    rows = freeze["files"]
    if isinstance(rows, dict):
        expected = rows[str(path.relative_to(ROOT))]["sha256"]
    else:
        candidates = [r for r in rows if Path(r["path"]).name == path.name]
        if len(candidates) != 1:
            raise ValueError("Frozen45D evaluator identity ambiguous")
        expected = candidates[0]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Original45D score implementation changed")
    with np.load(BASIS, allow_pickle=False) as data:
        if set(data.files) != {"centers_50", "centers_100"}:
            raise ValueError("Fixed cluster keys differ")
        centers = {k: data[f"centers_{k}"].copy() for k in (50, 100)}
    for k, value in centers.items():
        if (
            value.dtype != np.float64
            or value.shape != (k, 45)
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid frozen45D center tensor")
    tool = load_module("unchanged_pose45_with_saved_basis", path)

    def reuse_reference(reference, **kwargs):
        if len(reference) != original["method"]["fit"]["available_reference_poses"]:
            raise ValueError("Actual frozen AMASS reference population differs")
        fit = deepcopy(original["method"]["fit"])
        fit["refit_performed_for_this_new_run"] = False
        fit["saved_basis_artifact"] = evidence.record(BASIS)
        return centers, fit

    tool.fit_reference = reuse_reference
    tool.main(
        [
            "--run",
            "protected2000=" + str(args.input.resolve()),
            "--output",
            str(args.output.resolve()),
        ]
    )
    report = {
        "schema": "protected_terrain.pose45_saved_basis_actual_provenance.v1",
        "CPU_only": True,
        "GPU_used": False,
        "SDK_started": False,
        "reference_fit_repeated": False,
        "primary90D_replaced": False,
        "saved_basis": evidence.record(BASIS),
        "actual_source_admission": source,
        "original45D_evaluator": evidence.record(path),
        "wrapper": evidence.record(__file__),
        "generated_sample_fit_used": False,
    }
    (args.output / "protected_actual_source.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output / "GPU_import_ledger_as_observed.json").write_bytes(ledger_bytes)
    (args.output / "GPU_import_events_as_observed.jsonl").write_bytes(events_bytes)


if __name__ == "__main__":
    main()
