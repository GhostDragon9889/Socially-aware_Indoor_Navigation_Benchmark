"""Bounded CPU replay from frozen guide20 contexts; never a new physics run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from CrowdSim.humanoid.diffusion.paper import PaperDiffusionPlanner
from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene
from CrowdSim.humanoid.waypoint_weighting import near_horizon_weighting_description
from scripts.benchmark_paper_clean_guidance import capture_context, scene_plan_metrics


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT/'output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_200s_v1'
CHECKPOINT = ROOT/'output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt'
EXPECTED_SHA = '63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("Hide CUDA before this CPU-only audit")
    output = ROOT/'docs/style_improvement_20261002/near_horizon_waypoint_real_cpu_replay.json'
    if output.exists():
        raise ValueError("Refuse to overwrite historical CPU replay evidence")
    if sha(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("Frozen Diffusion5000 checkpoint SHA mismatch")
    runner = json.loads((CAPTURE/'result.json').read_text())
    if runner['status'] != 'passed':
        raise ValueError("Complete frozen guide20 capture required")
    source = ROOT/'docs/style_improvement_20261002/near_horizon_waypoint_before/CrowdSim/humanoid/diffusion/paper.py'
    name = 'CrowdSim.humanoid.diffusion.replay_near_before'
    spec = importlib.util.spec_from_file_location(name, source)
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    spec.loader.exec_module(old)
    previous = old.PaperDiffusionPlanner(checkpoint_path=CHECKPOINT, device='cpu')
    current = PaperDiffusionPlanner(checkpoint_path=CHECKPOINT, device='cpu')
    with np.load(CAPTURE/'physical_states.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in ('root_pos', 'root_rot', 'episode_ids', 'route_vertex_xy')}
    times = torch.arange(.5, 5.1, .5)
    record = {'status': 'running', 'scope': 'two fixed captured contexts, not physical navigation evaluation',
              'cpu_only': True, 'sdk_started': False, 'cuda_used': False, 'cpu_threads': 4,
              'precision': 'FP32', 'diffusion_steps': 100, 'noise_seed': 91,
              'checkpoint_sha256': EXPECTED_SHA, 'trained_steps': current.trained_steps,
              'capture_sha256': sha(CAPTURE/'physical_states.npz'), 'capture_result_sha256': sha(CAPTURE/'result.json'),
              'sources': {str(path.relative_to(ROOT)): sha(path) for path in [source,
                  ROOT/'CrowdSim/humanoid/diffusion/paper.py', ROOT/'CrowdSim/humanoid/waypoint_weighting.py']},
              'velocity_source': 'same-episode backward30Hz root finite difference; not stored native velocity',
              'goals': 'fresh static A* timed waypoints toward captured true route vertex; no future oracle or path replacement',
              'conditions': [], 'limitations': ['Only two contexts and one fixed seed; not proof of physical improvement',
                  '0.4m collision discs are geometric proxies, not native contacts',
                  'Causal reference adapter and AMP controller are not executed here']}
    started = time.monotonic()
    for frame in (900, 4500):
        context = capture_context(arrays, frame)
        scene = ReconstructionScene(runner['evaluation_scene'], device='cpu')
        states = torch.tensor(context['states'])
        destination = torch.tensor(arrays['route_vertex_xy'][frame])
        distance = (scene.routes-destination[:, None]).norm(dim=-1)
        if not bool((distance.min(-1).values < 1e-4).all()):
            raise ValueError("Captured route vertex not found in declared scene")
        goals, _, _ = scene.route_waypoints(states[:, :2], distance.argmin(-1), times=times, advance=False)
        context['raster'] = scene.raster(states)
        flags = dict(waypoints=goals, waypoint_times=times, num_samples=1, seed=91,
            guidance_weight=20., guidance_mode='trace_clean', scene_ids=torch.zeros(4, dtype=torch.long),
            agent_collision_weight=5., agent_collision_radius=.6,
            scene_collision_weight=10., scene_collision_radius=.8,
            scene_bounds_xy=scene.bounds, obstacle_centers_xy=scene.centers[:, :2], obstacle_sizes_xy=scene.sizes[:, :2])
        before = previous.plan(**context, **flags)
        disabled = current.plan(**context, **flags, near_horizon_mass=0.)
        fields = ('xy', 'yaw', 'vel', 'velocity', 'time', 'sample_states', 'selected_sample')
        equality = {field: bool(torch.equal(getattr(before, field), getattr(disabled, field))) for field in fields}
        if not all(equality.values()):
            raise AssertionError("Default actual100DDPM planner replay drift")
        for mass, prediction in [(0., disabled), (.25, current.plan(**context, **flags, near_horizon_mass=.25))]:
            xy = prediction.xy.cpu().numpy()
            condition = {'frame': frame, 'captured_time_s': (frame+1)/30,
                'near_horizon_mass': mass, 'context_default_regression': {
                    'scope': 'before original mean versus current near_horizon_mass=0; not eta=.25 equality',
                    'bitwise_equality': equality},
                'weighting': near_horizon_weighting_description(times.tolist(), mass),
                'selected_sample': prediction.selected_sample.tolist(),
                'xy_sha256': hashlib.sha256(xy.tobytes()).hexdigest(),
                'metrics': scene_plan_metrics(context['states'], xy, goals.numpy(), times.numpy(), runner['evaluation_scene'])}
            record['conditions'].append(condition)
            print(json.dumps({'frame': frame, 'eta': mass, 'metrics': condition['metrics']}), flush=True)
    record.update(status='passed', elapsed_s=time.monotonic()-started,
                  source_bank_and_weights_modified=False)
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'elapsed_s': record['elapsed_s'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
