#!/usr/bin/env python3
"""Real Sim6.1 floor/wall/peer contact controls and complete N4 shutdown.

Known placements are sensor fixtures, not autonomous navigation or gait quality.
An external parent must additionally observe this process's actual exit status.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate(app, args, owner, report, lifecycle):
    owner.physics_frame = inspect.currentframe()
    import torch
    from isaaclab.sim.utils import get_current_stage
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp_world import AMPWorld
    from CrowdSim.humanoid.foot_capture import FootCaptureAdapter
    from CrowdSim.humanoid.foot_support import FOOT_NAMES
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
    from protomotions.simulator.isaaclab.utils.usd_body_paths import resolve_body_prim_paths

    torch.set_num_threads(4)
    corpus = AMASSCorpus(args.manifest, split="validation", device="cuda:0", seed=7)
    origins = [[6, 6], [18, 18], [6, 18], [18, 6]]
    scene = {"schema": "navisaaclab.reconstruction_scene.v1", "bounds_xy": [2, 2, 22, 22],
             "origins_xy": origins, "routes_xy": [[[x, y], [x, y + 1]] for x, y in origins],
             "cuboids": [{"center": [10, 10, 1], "size": [1, 4, 2]}]}
    world = AMPWorld(app, corpus, num_envs=4, shared_world=True, scene_description=scene, contact_reporting=True)
    owner.world = world
    lifecycle.verify_loaded()
    report["physics"] = world.physics_manifest()
    if args.mode != "floor":
        report["peer_contract"] = world.enable_pairwise_contacts(batched_sensor_bodies=True)
    relative = resolve_body_prim_paths(world.simulator._robot.cfg.spawn.usd_path, list(FOOT_NAMES))
    feet = [{name: f"/World/envs/env_{actor}/Robot/{relative[name]}" for name in FOOT_NAMES} for actor in range(4)]
    if args.mode != "peer":
        capture = FootCaptureAdapter.from_world(world, actor_foot_paths=feet, stage=get_current_stage(),
            floor_roots=["/World/ground"], contact_reporting_enabled=True,
            peer_monitor=getattr(world.simulator, "pedestrian_contacts", None))
        world.simulator.pedestrian_contacts = capture
        report["floor_contract"] = capture.manifest
    else:
        capture = world.simulator.pedestrian_contacts
    counter = world.simulator._sim.get_physics_step_count
    cases = {"floor_only": origins, "static_wall": [[9.5, 10], *origins[1:]],
             "other_actor": [origins[0], [6.2, 6], *origins[2:]]}
    if args.mode == "floor":
        cases = {"floor_only": origins}
    report["cases"] = {}
    zero = torch.zeros(4, 69, device=world.device)
    for name, xy in cases.items():
        if args.mode != "peer":
            capture.reset_episode_boundary()
        world.simulator.reset_envs(ResetState(
            root_pos=torch.tensor([[x, y, .95] for x, y in xy], device=world.device),
            root_rot=torch.tensor([[0., 0., 0., 1.]] * 4, device=world.device),
            root_vel=torch.zeros(4, 3, device=world.device), root_ang_vel=torch.zeros(4, 3, device=world.device),
            dof_pos=zero, dof_vel=zero, state_conversion=StateConversion.COMMON))
        static, peer = torch.zeros(4, device=world.device), torch.zeros(4, 4, device=world.device)
        max_com_error = 0.
        before = counter()
        for step in range(12):
            if args.mode != "peer":
                capture.prepare_control_step(step)
            world.step(zero)
            measured = capture.measurements()
            if args.mode != "peer":
                assert measured["extra_physics_steps"] == 0
            assert counter() == before + (step + 1) * 4
            for value in measured.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert bool(torch.isfinite(value).all()), "Nonfinite contact/body channel"
            if args.mode != "peer":
                max_com_error = max(max_com_error, float(measured["com_composition_max_error_m"].max()))
            if args.mode != "floor":
                static = torch.maximum(static, measured["static_contact_max_force_n"])
                peer = torch.maximum(peer, measured["peer_contact_max_force_n"])
        report["cases"][name] = {"static_max_force_n": static.tolist(), "peer_max_force_n": peer.tolist(),
                                  "controls": 12, "physics_substeps": 48,
                                  "max_com_error_m": max_com_error if args.mode != "peer" else None,
                                  "peer_and_static_channels_measured": args.mode != "floor"}
    if args.mode != "floor":
        floor, wall, peer = (report["cases"][name] for name in cases)
        assert max(floor["static_max_force_n"]) <= 1 and max(map(max, floor["peer_max_force_n"])) <= 1
        assert wall["static_max_force_n"][0] > 1 and max(wall["static_max_force_n"][1:]) <= 1
        assert peer["peer_max_force_n"][0][1] > 1 and peer["peer_max_force_n"][1][0] > 1
        assert all(peer["peer_max_force_n"][i][i] == 0 for i in range(4))
    if args.mode != "peer":
        assert all(row["max_com_error_m"] <= 1e-4 for row in report["cases"].values())
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("combined", "floor", "peer"), default="combined")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from CrowdSim.humanoid.runtime.social_navigation_lifecycle import FourActorNavigationLifecycle
    from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    report = {"status": "starting", "fixture_only": True, "navigation_quality_verified": False, "mode": args.mode}
    lifecycle = FourActorNavigationLifecycle(report) if args.mode == "combined" else SDK61Lifecycle(report)
    app = import_simulator_before_torch("isaaclab")({"headless": True, "device": "cuda:0",
        "fast_shutdown": False, "kit_args": lifecycle.kit_args}).app
    owner = SimpleNamespace(world=None, physics_frame=None)
    try:
        validate(app, args, owner, report, lifecycle)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        from CrowdSim.humanoid.runtime.trace_pace_scope import clear_completed_physics_frame
        report["frame_release"] = clear_completed_physics_frame(owner, validate)
        gc.collect()
        lifecycle.close(app, owner.world)
        owner.world = None
        if not report["cleanup_succeeded"]:
            report["status"] = "failed_cleanup"
        (args.output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({key: report.get(key) for key in ("status", "error", "cleanup_succeeded")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
