"""Recorded S1 post-control0 -> SFM raw51 -> existing causal151 CPU linkage.

No policy/weight loading, simulation, trajectory success or quality assertion.
The saved physical state is a fixed recorded input, not a new live rollout.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_shared_social_locomotion as admission  # noqa: E402
from CrowdSim.humanoid.trajectory.social_force_runtime import SCENE, SocialForceSharedPlanner  # noqa: E402


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CUDA-hidden process required")
    here = Path(__file__).resolve().parent
    paths = [here / "snapshot_CPU.json", here / "snapshot_CPU_arrays.npz"]
    if any(p.exists() or p.is_symlink() for p in paths):
        raise ValueError("Preserve existing recorded-input linkage evidence")
    post = ROOT / "output/diffusion_amp/shared_social_popart_terminal2000_evaluation_v1"
    receipts = ROOT / "docs/style_improvement_20261005/shared_social_evaluation_v1"
    result, audit, arrays, inputs = admission.admit(post / "navigation200s_execution.json", receipts / "popart_post_training_execution.json")
    helpers, sources = admission.load_helpers()
    verified = admission.analyze(result, audit, arrays, helpers)
    with np.load(arrays[0]["path"], allow_pickle=False) as raw:
        root_xy = raw["root_pos"][0, :, :2].copy()
        quaternion = raw["root_rot"][0].copy()
        waypoints = raw["global_waypoints_xy"][0].copy()
    with np.load(arrays[1]["path"], allow_pickle=False) as raw:
        velocity = raw["body_link_lin_vel_w"][0, -1, :, 0, :2].copy()
    import torch
    from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig
    q = torch.from_numpy(quaternion)
    heading = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1].square() + q[:, 2].square()))
    states = torch.cat((torch.from_numpy(root_xy), torch.from_numpy(velocity), heading[:, None]), -1)
    goals = torch.from_numpy(waypoints)
    before = states.clone(), goals.clone(), torch.get_rng_state().clone()
    source_paths = [Path(__file__), ROOT / "scripts/diagnose_shared_social_locomotion.py",
                    ROOT / "CrowdSim/humanoid/trajectory/social_force_runtime.py", ROOT / "CrowdSim/humanoid/trajectory/social_force.py",
                    ROOT / "CrowdSim/humanoid/training/reference.py", SCENE]
    sources.extend(admission.record(p) for p in source_paths)
    planner = SocialForceSharedPlanner.for_CPU_snapshot(scene_source=SCENE, seed=7)
    prediction = planner.predict_snapshot(states, goals)
    config = ReferenceAdapterConfig(max_speed_mps=1.5, max_acceleration_mps2=2., initial_speed_policy="brake")
    adapter = CausalReferenceAdapter(config)
    dense = adapter.build(prediction.xy, states[:, :2], states[:, 2:4])
    if (not torch.equal(states, before[0]) or not torch.equal(goals, before[1])
            or not torch.equal(torch.get_rng_state(), before[2]) or torch.cuda.is_initialized()
            or any(name.startswith(("isaacsim", "isaaclab", "omni", "pxr")) for name in sys.modules)):
        raise ValueError("Recorded input/RNG changed or GPU/SDK initialized")
    acceleration = torch.linalg.vector_norm(torch.diff(dense.velocity, dim=1), dim=-1) * 30
    if dense.xy.shape != (4, 151, 2) or float(acceleration.max()) > 2.0001:
        raise ValueError("Common reference shape or acceleration bound failed")
    for identity in sources + inputs:
        if admission.record(identity["path"]) != identity:
            raise ValueError("Source or recorded input changed during linkage")
    with paths[1].open("xb") as stream:
        np.savez_compressed(stream, input_states=states.numpy(), registered_waypoints=goals.numpy(),
                            raw_xy=prediction.xy.numpy(), reference_xy=dense.xy.numpy(), reference_velocity=dense.velocity.numpy(),
                            raw_times_s=dense.raw_times_s.numpy(), reference_times_s=dense.times_s.numpy(),
                            initial_overspeed=dense.initial_overspeed.numpy(), speed_violation=dense.speed_violation.numpy())
    report = {"schema": "shared_social.recorded_input_SFM_reference_CPU_linkage.v1", "status": "completed_CPU_linkage",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs, "sources": sources,
              "actual_completed_control_selected": 1, "frame_selection": "Fixed first complete control0, all four actors",
              "original_raw_capture_controls_verified": verified["all_actor_controls"], "planner": planner.describe(),
              "reference_adapter": adapter.describe(), "raw_shape": list(prediction.xy.shape), "reference_shape": list(dense.xy.shape),
              "max_reference_acceleration_mps2": float(acceleration.max()),
              "max_reference_speed_mps": float(torch.linalg.vector_norm(dense.velocity, dim=-1).max()),
              "initial_overspeed_by_actor": dense.initial_overspeed.tolist(), "arrays": admission.record(paths[1]),
              "input_storage_unchanged": True, "CPU_RNG_preserved": True, "GPU_used": False, "SDK_started": False,
              "policy_weights_loaded": False, "live_sfm_rollout_executed": False, "physical_quality_established": False,
              "scope": "Recorded post-step input -> future raw plan -> reference initialization only; ordinary live replan/actor/physics integration remains untested"}
    with paths[0].open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(paths[0]), "reference_shape": report["reference_shape"],
                      "max_acceleration_mps2": report["max_reference_acceleration_mps2"], "live_sfm_rollout_executed": False}))


if __name__ == "__main__":
    main()
