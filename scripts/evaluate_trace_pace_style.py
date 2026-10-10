#!/usr/bin/env python3
"""Frozen held-out AMASS style diagnostics of actual TRACE/PACER physics.

Reuse the existing15-keypoint/30Hz reference and cluster basis without fitting
the generated data. Local MMD-squared/proxies differ from Nav's private scores.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "CrowdSim").is_dir():
    ROOT = Path("/project/EXP/SangforAndSaut/ProtoMotions")
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.execution.read_text())
    result_path = args.run / "result.json"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    if (
        receipt.get("actual_returncode") != 0
        or receipt.get("actual_returncode_source") != "subprocess.Popen.wait"
        or receipt.get("result_sha256") != hashlib.sha256(result_bytes).hexdigest()
        or not receipt.get("sources_unchanged")
        or receipt.get("inputs_unchanged") is False
        or result.get("status") != "passed"
        or result.get("cleanup_succeeded") is not True
    ):
        raise ValueError("Require an actual completed source-bound physical run")
    import numpy as np
    from CrowdSim.humanoid.style_evaluation import (
        sequence_style_samples,
        compare_speed_conditioned,
        StyleSpec,
    )
    from scripts.evaluate_shared_social_navigation_style import load_frozen_reference

    reference, pose_tool, centers, provenance, identities = load_frozen_reference()
    physical_path = args.run / "physical.npz"
    physical_sha = hashlib.sha256(physical_path.read_bytes()).hexdigest()
    bound_sha = receipt.get("capture_sha256", {}).get("physical.npz")
    if bound_sha is not None and bound_sha != physical_sha:
        raise ValueError("Physical capture differs from the parent's completed receipt")
    with np.load(physical_path, allow_pickle=False) as data:
        body = data["body_positions"].copy()
        rotation = data["root_rotations_xyzw"].copy()
    if len(body) != result["actual_controls"] or len(body) < 31:
        raise ValueError("Incomplete actual pose/control alignment")
    resets = np.zeros(body.shape[:2], dtype=bool)
    resets[0] = True
    measured_inputs = {}
    support_path = args.run / "ground_support.npz"
    if support_path.exists():
        if (
            receipt.get("capture_sha256", {}).get("ground_support.npz")
            != hashlib.sha256(support_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Measured support is not bound by the actual parent")
        with np.load(support_path, allow_pickle=False) as data:
            if tuple(data["common_body_names"]) != tuple(
                result["physics"]["body_names"]
            ) or len(data["control_step"]) != len(body):
                raise ValueError("Measured COM/contact body order or clock differs")
            pelvis = list(data["common_body_names"]).index("Pelvis")
            if not (
                np.array_equal(body, data["body_link_pos_w"][:, -1])
                and np.array_equal(rotation, data["body_link_quat_w"][:, -1, :, pelvis])
                and np.array_equal(data["control_step"], np.arange(len(body)))
                and np.array_equal(data["control_step_after"], np.arange(1, len(body) + 1))
                and (data["substep_valid_count"] == 4).all()
                and data["valid"].all()
                and not data["reset_mask"][1:].any()
            ):
                raise ValueError("Style body/rotation/clock differs from the native final substep")
            measured_inputs = {
                "body_masses_kg": data["body_masses_kg"][0, -1].copy(),
                "body_com_world": data["body_com_pos_w"][:, -1].copy(),
                "foot_contact_mask": data["side_contact_substeps"][:, -1].copy(),
            }
    generated = sequence_style_samples(
        body,
        rotation,
        episode_ids=np.ones(body.shape[:2], dtype=np.int64),
        reset_mask=resets,
        body_names=tuple(result["physics"]["body_names"]),
        spec=StyleSpec(),
        **measured_inputs,
    )
    report = {
        "schema": "crowdsim.trace_pace_heldout_style.v1",
        "paper_scores": False,
        "test34_read": False,
        "measured_pose_COM_contact_same_final_substep": bool(measured_inputs),
        "input_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "physical_sha256": physical_sha,
        "capture_parent_hash_available": bound_sha is not None,
        "evaluation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "reference": provenance,
        "frozen_source_identities": identities,
        "local_training_parents_disjoint": True,
        "public_PACER_training_overlap_known": False,
        "actual_controls": len(body),
        "actors": body.shape[1],
        "all_actual_frames_loaded": True,
        "all_same_episode_transitions_included": True,
        "style_transition_count": int(
            generated["summary"]["same_episode_transition_count"]
        ),
        "first_frame_used_as_derivative_context": True,
        "style_sample_domain": "All T-1 legal intervals per actor; pose-only uses the interval end frames",
        "cluster_basis_refitted": False,
        "generated_physical_summary": generated["summary"],
        "speed_conditioned": compare_speed_conditioned(generated, reference),
        "pose_only": pose_tool.compare_pose_only(
            generated["local_pose"], reference["local_pose"], centers
        ),
        "limitations": [
            "Short controlled fixture is not the paper8-scene200s population",
            "Near-ground toe sliding is a kinematic proxy; no measured contact-slip score",
            "This checks local held-out motion; author public PACER training IDs remain unknown",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "actual_controls": len(body),
                "actors": body.shape[1],
                "speed_conditioned_MMD2": report["speed_conditioned"][
                    "equal_supported_bin_mean_pose_velocity_mmd2"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
