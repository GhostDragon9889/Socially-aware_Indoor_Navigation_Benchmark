#!/usr/bin/env python3
"""Validate filtered PhysX contacts against known floor/wall/peer fixtures.

Resets deliberately place bodies near contact partners for sensor validation.
This is not a navigation or autonomous locomotion demonstration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batched-contact-views", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output directory")
    output.mkdir(parents=True)
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    app = import_simulator_before_torch("isaaclab")({"headless": True, "device": "cuda:0",
        "kit_args": "--/app/renderer/enabled=false --/renderer/enabled=false"}).app
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp_world import AMPWorld
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
    world = None
    result = {"status": "failed", "method": "contact sensor fixture probe; resets intentionally impose known contacts"}
    try:
        torch.set_num_threads(4)
        corpus = AMASSCorpus("data/diffusion_amp/amass_manifest.json", "validation", device="cuda:0", seed=7)
        scene = {"schema": "navisaaclab.reconstruction_scene.v1", "bounds_xy": [2, 2, 22, 22],
            "origins_xy": [[6, 6], [18, 18]], "routes_xy": [[[6, 18], [6, 6]], [[18, 6], [18, 18]]],
            "cuboids": [{"center": [10, 10, 1], "size": [1, 4, 2]}]}
        world = AMPWorld(app, corpus, num_envs=2, shared_world=True, scene_description=scene,
                         contact_reporting=True)
        result["contract"] = world.enable_pairwise_contacts(batched_sensor_bodies=args.batched_contact_views)
        cases = {"floor_only": [[6, 6], [18, 18]], "static_wall": [[9.5, 10], [18, 18]],
                 "other_actor": [[6, 6], [6.2, 6]]}
        result["cases"] = {}
        for name, xy in cases.items():
            position = torch.tensor([[x, y, .95] for x, y in xy], device=world.device)
            rotations = torch.tensor([[0., 0., 0., 1.]]*2, device=world.device)
            zero_dofs = torch.zeros(2, 69, device=world.device)
            world.simulator.reset_envs(ResetState(root_pos=position, root_rot=rotations,
                root_vel=torch.zeros(2, 3, device=world.device), root_ang_vel=torch.zeros(2, 3, device=world.device),
                dof_pos=zero_dofs, dof_vel=zero_dofs, state_conversion=StateConversion.COMMON))
            max_static = torch.zeros(2, device=world.device)
            max_peer = torch.zeros(2, 2, device=world.device)
            for _ in range(12):
                world.step(zero_dofs)
                observed = world.simulator.pedestrian_contacts.measurements()
                max_static = torch.maximum(max_static, observed["static_contact_max_force_n"])
                max_peer = torch.maximum(max_peer, observed["peer_contact_max_force_n"])
            result["cases"][name] = {"static_max_force_n": max_static.cpu().tolist(),
                "peer_max_force_n": max_peer.cpu().tolist(), "physics_substeps": 48}
        f, wall, peer = (result["cases"][key] for key in cases)
        assert max(f["static_max_force_n"]) <= 1 and max(map(max, f["peer_max_force_n"])) <= 1, "Floor contact leaked into collision filters"
        assert wall["static_max_force_n"][0] > 1 and wall["static_max_force_n"][1] <= 1, "Known wall contact missing/misassigned"
        assert peer["peer_max_force_n"][0][1] > 1 and peer["peer_max_force_n"][1][0] > 1, "Known actor pair contact missing"
        assert peer["peer_max_force_n"][0][0] == peer["peer_max_force_n"][1][1] == 0, "Self contact leaked"
        result["status"] = "passed"
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({key: result.get(key) for key in ("status", "error", "cases")}), flush=True)
        if world is not None:
            world.simulator._shutdown_exit_code = 0 if result["status"] == "passed" else 1
            world.close()
        else:
            app.close()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
