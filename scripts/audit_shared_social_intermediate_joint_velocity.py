#!/usr/bin/env python3
"""Audit actual intermediate batch coordinates; never infer training completion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import joint_velocity_consistency as kernel  # noqa: E402
from scripts import diagnose_shared_social_task_context as phase  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

RUN = ROOT / "output/diffusion_amp/shared_social_episode_reference_pilot2000_v1"
XML = ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"
OBSERVATIONS = ROOT / "CrowdSim/humanoid/amp/observations.py"
OBSERVATIONS_SHA = "000f3980a8f4d3c1fac929c57029dd05a9ce19b1d8ce82423dad273c9315e599"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--update", type=int, choices=(500, 1000, 1500), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    here = args.output.resolve()
    kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not here.exists(), "Fresh CPU-only output required")
    sources = [identity(path) for path in (Path(__file__), Path(kernel.__file__), Path(phase.__file__),
                                         OBSERVATIONS, ROOT / "scripts/diagnose_shared_social_training.py")]
    kernel.require(sources[3]["sha256"] == OBSERVATIONS_SHA, "Canonical AMP body/DOF order changed")
    rollout_path = RUN / f"rollout_update_{args.update:06d}.pt"
    inputs = [identity(rollout_path), identity(XML)]
    # Read only the closed prefix. The active writer may append a partial final
    # row; no progress-file hash or completion claim is made from this ledger.
    with (RUN / "metrics.jsonl").open() as stream:
        for index in range(args.update + 1):
            row = json.loads(next(stream))
            kernel.require(type(row["update_count"]) is int and row["update_count"] == index + 1,
                           "Require contiguous completed updates beyond the selected archive")
    import torch
    from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES, _DISC_DOFS

    packet = torch.load(rollout_path, weights_only=True, map_location="cpu", mmap=True)
    collection = packet["collection"]
    rollout, diagnostics = ({key: value.numpy() for key, value in collection[name].items()}
                            for name in ("rollout", "diagnostics"))
    metadata = collection["metadata"]
    checked_phase = phase.verify_phase(rollout, diagnostics, metadata)
    kernel.require(metadata["controls"] == 32 and metadata["actors"] == 256
                   and metadata["training_update_count"] == args.update - 1
                   and metadata["training_physical_transition_count"] == (args.update - 1) * 8192
                   and metadata["control_step_end"] == args.update * 32
                   and metadata["physics_step_end"] == args.update * 128,
                   "Actual saved batch differs from its selected update")
    parent_names = {}

    def walk(body, parent=None):
        name = body.attrib["name"]
        kernel.require(name not in parent_names, "Duplicate MJCF body")
        parent_names[name] = parent
        for child in body.findall("body"):
            walk(child, name)

    for body in ET.parse(XML).getroot().find("worldbody").findall("body"):
        walk(body)
    kernel.require(set(parent_names) == set(PACER_BODY_NAMES) and parent_names["Pelvis"] is None,
                   "Require the complete named SMPL body tree")
    parents = [PACER_BODY_NAMES.index(parent_names[name]) for name in PACER_BODY_NAMES[1:]]
    native, child, parent = kernel.reconstruct(diagnostics["common_rigid_body_rot"],
        diagnostics["common_rigid_body_ang_vel"], diagnostics["common_dof_vel"], parents)
    child_error, parent_error = native - child, native - parent
    selected_bodies = sorted({index // 3 for index in _DISC_DOFS})
    kernel.require(len(selected_bodies) == 19 and len(_DISC_DOFS) == 57
                   and all(index in _DISC_DOFS for body in selected_bodies for index in range(body * 3, body * 3 + 3)),
                   "Require nineteen complete AMP velocity triples")
    kernel.require(np.max(np.abs(child_error)) <= 1e-5, "Native69 does not match same-instant child angular velocities")
    amp_velocity = diagnostics["post_amp_history"][..., 0, 126:183]
    kernel.require(np.array_equal(amp_velocity, diagnostics["common_dof_vel"][..., _DISC_DOFS]),
                   "POST AMP57 is not the exact actual mapped native velocity")
    kernel.require(all(identity(row["path"]) == row for row in sources + inputs), "Archive or sources changed")
    kernel.require(not torch.cuda.is_initialized() and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"}
                                                              for name in sys.modules), "CPU audit initialized CUDA or imported an SDK")
    report = {"schema": "shared_social.actual_intermediate_joint_velocity_CPU_audit.v1", "status": "passed_sameinstant_batch_coordinate_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "selected_archive_update": args.update,
        "sources": sources, "inputs": inputs, "complete_closed_metric_prefix_updates": args.update + 1,
        "phase_verification": checked_phase, "physical_actor_controls": 8192,
        "body_order": list(PACER_BODY_NAMES), "nonroot_parent_indices": parents,
        "sameinstant_all69_native_minus_child": kernel.error_statistics(child_error),
        "sameinstant_all69_native_minus_parent": kernel.error_statistics(parent_error),
        "sameinstant_AMP57_native_minus_child": kernel.error_statistics(child_error[..., selected_bodies, :]),
        "sameinstant_AMP57_native_minus_parent": kernel.error_statistics(parent_error[..., selected_bodies, :]),
        "per_joint_child_max_abs_rad_s": dict(zip(PACER_BODY_NAMES[1:], np.max(np.abs(child_error), axis=(0, 1, 3)).tolist(), strict=True)),
        "POST_AMP57_equals_mapped_native_exactly": True, "coordinate_tolerance_rad_s": 1e-5,
        "calculation": "relative_world=omega_child_world-omega_parent_world; inverse current body XYZW rotation; no time differencing",
        "scope": "This intermediate256-person batch and its captured timestamps/body order; not arbitrary assets, training completion, policy quality or a finite-difference accuracy claim.",
        "all_failure_reset_controls_retained": True, "NN_constructed": False, "SDK_started": False,
        "CUDA_initialized": False, "training_restarted": False, "training_completed_inferred": False,
        "heldout_quality_evaluation": False, "default_model_promoted": False, "paper_effect_achieved": False}
    here.mkdir(parents=True)
    np.savez_compressed(here / "sameinstant_errors.npz", child_error_rad_s=child_error, parent_error_rad_s=parent_error)
    report["errors"] = identity(here / "sameinstant_errors.npz")
    result = here / "result.json"
    result.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for path in here.iterdir():
        path.chmod(0o444)
    print(json.dumps({"result": identity(result), "sameinstant_all69_native_minus_child": report["sameinstant_all69_native_minus_child"],
                      "sameinstant_AMP57_native_minus_child": report["sameinstant_AMP57_native_minus_child"],
                      "sameinstant_AMP57_native_minus_parent": report["sameinstant_AMP57_native_minus_parent"],
                      "phase_verification": checked_phase, "SDK_started": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
