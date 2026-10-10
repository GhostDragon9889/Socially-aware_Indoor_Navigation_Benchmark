"""CPU exact-seed comparison of real learned Diffusion weights, not physics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from CrowdSim.humanoid.diffusion.paper import PaperDiffusionPlanner


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run():
    if torch.cuda.is_initialized():
        raise RuntimeError("This verification is CPU-only")
    torch.set_num_threads(4)
    name = "CrowdSim.humanoid.diffusion.scene_callback_actual_before"
    spec = importlib.util.spec_from_file_location(name, HERE / "before/paper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    before = module.PaperDiffusionPlanner(checkpoint_path=CHECKPOINT, device="cpu")
    current = PaperDiffusionPlanner(checkpoint_path=CHECKPOINT, device="cpu")
    states = torch.tensor([[-4., -3., .4, .1, .2], [-1., -3., .3, 0., -.1]])
    history = states[:, None].repeat(1, 31, 1)
    raster = torch.zeros(2, 2, 224, 224)
    raster[:, 0] = 1.
    times = torch.arange(1, 11) * .5
    waypoints = states[:, None, :2] + torch.stack((times*.4, times*0.), -1)[None]
    arguments = dict(states=states, history=history, neighbors=None, raster=raster,
        waypoints=waypoints, waypoint_times=times, num_samples=2, seed=91, guidance_weight=.01)
    cases = []
    for mode in ("legacy_posterior", "trace_clean"):
        extras = {} if mode == "legacy_posterior" else dict(scene_ids=[0, 0], agent_collision_weight=.0001,
            scene_collision_weight=.0001, scene_bounds_xy=[-100., -100., 100., 100.],
            obstacle_centers_xy=[[40., 40.]], obstacle_sizes_xy=[[1., 1.]],
            trajectory_speed_weight=.00001, trajectory_accel_weight=.00001)
        rng = torch.get_rng_state().clone()
        expected = before.plan(**arguments, guidance_mode=mode, **extras)
        actual = current.plan(**arguments, guidance_mode=mode, scene_collision_cost_fn=None, **extras)
        results = {}
        for field in ("xy", "yaw", "vel", "velocity", "time", "sample_states", "selected_sample"):
            left, right = getattr(expected, field), getattr(actual, field)
            equal = torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
            assert equal, field
            results[field] = {"bytewise_equal": equal, "shape": list(left.shape),
                "tensor_sha256": hashlib.sha256(left.contiguous().numpy().tobytes()).hexdigest()}
        assert torch.equal(rng, torch.get_rng_state())
        assert all(parameter.grad is None for parameter in current.model.parameters())
        assert all(torch.equal(value, before.model.state_dict()[name]) for name, value in current.model.state_dict().items())
        cases.append({"guidance_mode": mode, "arguments": "Fixed mathematical2actor observations; samples2/private seed91",
            "outputs": results, "global_Torch_CPU_RNG_unchanged": True, "all_parameter_grads_none": True})
    result = {"schema": "native_grid_diffusion_actual_legacy_parity.v1", "status": "passed",
        "device": "cpu", "threads": 4, "checkpoint": record(CHECKPOINT),
        "before_source": record(HERE / "before/paper.py"),
        "new_source": record(ROOT / "CrowdSim/humanoid/diffusion/paper.py"),
        "learned_weight_update_count": current.trained_steps, "cases": cases,
        "exact_output_tensor_comparisons": 14, "weights_changed": False,
        "SDK_or_Kit_started": False, "CUDA_used": False, "physical_rollouts": 0,
        "limitation": "Existing learned weights and default/analytic branches have exact CPU parity; optional grid callback has separate mathematical/provider tests, no native physical navigation claim."}
    with (HERE / "actual_legacy_parity.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checkpoint_sha256": result["checkpoint"]["sha256"],
        "learned_steps": current.trained_steps, "exact_output_tensor_comparisons": 14}))


if __name__ == "__main__":
    run()
