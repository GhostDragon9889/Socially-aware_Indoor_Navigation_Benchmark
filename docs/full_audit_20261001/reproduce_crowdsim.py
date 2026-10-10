"""CPU correctness audit against current source; no Isaac Sim/Kit is launched.

Tensor-backed articulation fixtures verify math/API boundaries only. The
subset-context check uses unmodified production AST methods and the actual
MaskedMimicContext; the path generator is zeroed to isolate indexing.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import copy
import hashlib
import importlib
import json
import math
import tempfile

import numpy as np
import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[2]
out = Path(__file__).with_name("crowdsim_evidence.json")
results = {"method": "Current production code, CPU only; no Kit/physics", "tests": {}}

try:
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
except ImportError as error:
    results["tests"]["runtime_missing_import"] = str(error)

for module_name in ("CrowdSim.ppo.ppo_policy", "crowdsim_ppo.ppo_policy"):
    module = importlib.import_module(module_name)
    for method in ("update", "update_with_kl"):
        torch.manual_seed(42)
        trainer = module.RobotPPOTrainer(module.RobotPPOConfig(
            obs_dim=5, hidden_dims=(8,), map_enabled=False, depth_enabled=False,
            num_neighbors=1, minibatch_size=2, ppo_epochs=1), torch.device("cpu"))
        buffer = module.RobotRolloutBuffer(3, 1, 5, 2, torch.device("cpu"), num_neighbors=1)
        buffer.obs.normal_()
        buffer.raw_actions.normal_()
        buffer.returns[:, 0] = torch.tensor([1., 2., 3.])
        buffer.advantages[:, 0] = torch.tensor([-1., 0., 1.])
        with torch.no_grad():
            lp, _, values = trainer.model.evaluate_actions(
                buffer.obs[:, 0], buffer.raw_actions[:, 0],
                neighbors=buffer.neighbors[:, 0], neighbor_mask=buffer.neighbor_mask[:, 0])
            buffer.log_probs[:, 0] = lp
            buffer.values[:, 0] = values
        kwargs = {"ref_model": copy.deepcopy(trainer.model), "kl_beta": .1} if method == "update_with_kl" else {}
        stats = getattr(trainer, method)(buffer, **kwargs)
        results["tests"][module_name + "." + method] = {
            "value_loss_finite": math.isfinite(stats["value_loss"]),
            "nonfinite_parameter_tensors": sum(not bool(torch.isfinite(v).all()) for v in trainer.model.parameters()),
            "samples": 3, "minibatch_size": 2,
        }

from CrowdSim.control.drive import DriveController
drive = object.__new__(DriveController)
drive.config = NS(num_robots=1)
drive.robot = NS(data=NS(root_quat_w=torch.tensor([[0., 0., 0., 1.]])))
results["tests"]["drive_xyzw_boundary"] = {
    "identity_actual_yaw_degrees": float(np.degrees(drive.yaws()[0])),
    "identity_expected_yaw_degrees": 0.,
    "yaw_zero_quat_actual": DriveController._yaw_to_quat(torch.tensor([0.])).tolist(),
    "yaw_zero_quat_expected_xyzw": [[0., 0., 0., 1.]],
}
from CrowdSim.tools.smpl_mesh_visualizer import quat_apply_wxyz
results["tests"]["mesh_xyzw_boundary"] = {
    "actual_vector": quat_apply_wxyz(torch.tensor([[0.,0.,0.,1.]]), torch.tensor([[1.,0.,0.]])).tolist(),
    "expected_vector": [[1.,0.,0.]],
}

from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig
torch.manual_seed(42)
env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=10000, num_neighbors=0,
    min_start_goal_distance=2., max_start_goal_distance=4.,
    rl_initial_heading_min_offset_degrees=0., rl_initial_heading_max_offset_degrees=15.), torch.device("cpu"))
delta = env.goal - env.position
angle = torch.atan2(delta[:,1], delta[:,0]) - env.heading
offset = torch.atan2(angle.sin(), angle.cos()).abs() * 180 / math.pi
distance = delta.norm(dim=-1)
results["tests"]["point_goal_reset_bounds"] = {
    "seed": 42, "samples": 10000, "distance_below_min_2m": int((distance < 2.).sum()),
    "heading_offset_above_max_15deg": int((offset > 15.0001).sum()),
    "minimum_distance": float(distance.min()), "maximum_heading_offset": float(offset.max()),
}
try:
    empty_env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=1, num_neighbors=0), torch.device("cpu"))
    empty_env.step(torch.zeros(1,2))
except Exception as error:
    results["tests"]["point_goal_zero_neighbors_step"] = {"error_type": type(error).__name__, "error": str(error)}

from protomotions.envs.context_views import MaskedMimicContext
path = root / "CrowdSim/nav_manager.py"
node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "CrowdNavigationManager")
node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ("_override_masked_mimic_context", "_yaw_to_quat_xyzw_tensor")]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
namespace = {"np": np, "torch": torch}
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
manager = namespace["CrowdNavigationManager"]()
manager.config = NS(num_humanoids=2)
manager.reached = np.array([True, False])
manager._humanoid_target_offsets = lambda n: np.array([.5], dtype=np.float32)
manager._humanoid_future_targets_from_path = lambda xy, offsets: (np.zeros((1,1,2), np.float32), np.zeros((1,1), np.float32))
base = MaskedMimicContext(mimic=None, ref_pos=torch.zeros(1,1,1,3), ref_rot=torch.tensor([[[[0.,0.,0.,1.]]]]),
    target_times=torch.zeros(1,1), time_offsets=torch.zeros(1,1), target_poses_masks=torch.ones(1,1), target_bodies_masks=torch.ones(1,2))
ctx = NS(masked_mimic=base, env_ids=torch.tensor([1]), current=NS(rigid_body_pos=torch.zeros(1,1,3)))
component = NS(conditionable_body_ids=torch.tensor([0]), num_conditionable_bodies=1,
    env=NS(robot_config=NS(anchor_body_index=0), motion_manager=NS(motion_times=torch.tensor([10.,20.]))))
manager._override_masked_mimic_context(component, ctx)
results["tests"]["subset_reset"] = {"env_ids": [1], "actual_target_time": float(ctx.masked_mimic.target_times.item()),
    "expected_target_time": 20.5, "actual_pose_mask": float(ctx.masked_mimic.target_poses_masks.item()), "expected_pose_mask": 1.}

from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import NormObsBaseConfig
normalizer = NormObsBase(NormObsBaseConfig(normalize_obs=True, norm_ema_decay=.999))
normalizer.running_obs_norm._lazy_init(torch.zeros(2,1))
before = normalizer.running_obs_norm.mean.clone()
with torch.no_grad():
    normalizer(torch.tensor([[10.],[10.]]))
after_train = normalizer.running_obs_norm.mean.clone()
normalizer.eval()
with torch.no_grad():
    normalizer(torch.tensor([[20.],[20.]]))
results["tests"]["no_grad_does_not_freeze_normalizer"] = {
    "initial_mean": before.tolist(), "after_training_mode_no_grad": after_train.tolist(),
    "after_eval_mode_no_grad": normalizer.running_obs_norm.mean.tolist(),
    "training_mode_updates_statistics": not torch.equal(before, after_train),
    "eval_mode_preserves_statistics": torch.equal(after_train, normalizer.running_obs_norm.mean),
}

from CrowdSim.pref.pref_train_reward import CrowdSimRewardNet
torch.manual_seed(42)
network = CrowdSimRewardNet(system_dim=7, vec_dim=5, depth_size=16, state_embed_dim=16, action_embed_dim=16).eval()
def episode(length):
    return {"steps": [{"sys_state": np.full(5, .3, dtype=np.float32), "action": np.array([.2,.1], np.float32), "depth": np.full((16,16), .4, np.float32)} for _ in range(length)]}
short, long = episode(2), episode(4)
with torch.no_grad():
    alone = float(network([short])[0])
    paired = float(network([short,long])[0])
    real_steps = float(network.forward_per_step([short,long])[0,:2].sum())
results["tests"]["legacy_preference_padding"] = {
    "short_length": 2, "other_length": 4, "mode": "eval", "alone_score": alone,
    "batched_score": paired, "real_steps_only_score": real_steps, "score_difference": paired-alone,
}

from PIL import Image
from CrowdSim.utils.map_metadata import load_occupancy_map_metadata
from CrowdSim.world.map import NavigationTask
with tempfile.TemporaryDirectory(prefix="crowdsim-map-audit-") as name:
    directory = Path(name)
    Image.fromarray(np.array([[0,255],[0,255]], dtype=np.uint8)).save(directory / "map.png")
    (directory / "map.yaml").write_text("image: map.png\nresolution: 0.1\norigin: [0, 0, 0]\nnegate: 1\nfree_thresh: 0.196\noccupied_thresh: 0.65\n")
    meta = load_occupancy_map_metadata(directory / "map.yaml")
    task = object.__new__(NavigationTask)
    task.config = NS(free_threshold=meta.free_threshold)
    free, obstacle = task._load_map(meta.image_path)
    results["tests"]["map_negate_one"] = {
        "loaded_negate": meta.negate, "actual_first_row_free": free[0].tolist(),
        "expected_first_row_free": [True, False], "computed_threshold": meta.free_threshold,
    }

    # Real builder exits in validation before simulator initialization. A guard
    # prevents a future code change from accidentally launching Fabric/Kit.
    from CrowdSim.world import builder
    old_fabric = builder.create_fabric
    def forbidden_simulator_initialization():
        raise RuntimeError("Audit guard: simulator initialization is prohibited")
    builder.create_fabric = forbidden_simulator_initialization
    config = {"scene": {"scene_usd": str(directory / "scene.usda"), "scene_map": str(directory / "map.yaml")},
        "humanoid": {"checkpoint": str(directory / "absent.ckpt"), "motion_file": str(directory / "absent.pt")},
        "navigation": {"num_humanoids": 0, "num_robots": 0}}
    (directory / "scene.usda").write_text("#usda 1.0\n")
    try:
        builder.build_env(config, empty_mode=True, headless=True)
    except Exception as error:
        results["tests"]["empty_mode_prevalidation"] = {"error_type": type(error).__name__, "error": str(error),
            "simulator_guard_reached": "Audit guard" in str(error)}
    finally:
        builder.create_fabric = old_fabric

from CrowdSim.utils.config_loader import load_config
from CrowdSim.ppo.goal_curriculum import PPOGoalCurriculum
env_config = load_config(root / "CrowdSim/config/env.yaml")
nav_config = NS(min_start_goal_distance=7., max_start_goal_distance=10.,
    rl_initial_heading_min_offset_degrees=0., rl_initial_heading_max_offset_degrees=0.)
curriculum = PPOGoalCurriculum(nav_config, env_config["navigation"]["rl"]["goal_curriculum"])
results["tests"]["default_eval_curriculum"] = {"declared_path_distance": [7.,10.],
    "effective_initial_distance": [nav_config.min_start_goal_distance,nav_config.max_start_goal_distance],
    "enabled": curriculum.spec.enabled, "stage": curriculum.stage.name}

train_path = root / "CrowdSim/ppo/train_ppo.py"
logger = next(n for n in ast.parse(train_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "RobotTrainingLogger")
logger.body = [n for n in logger.body if isinstance(n, ast.FunctionDef) and n.name == "_outcome_stats"]
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[logger], type_ignores=[])), str(train_path), "exec"), namespace)
counts = {"reached": 1, "timeout": 1, "collision": 0, "stuck": 0}
results["tests"]["overlapping_terminal_events"] = {"counts_for_one_episode": counts,
    "logger_output": namespace["RobotTrainingLogger"]._outcome_stats(counts), "expected_done_count": 1}

nav_tree = ast.parse((root / "CrowdSim/nav_manager.py").read_text())
nav_class = next(n for n in nav_tree.body if isinstance(n, ast.ClassDef) and n.name == "CrowdNavigationManager")
attach = next(n for n in nav_class.body if isinstance(n, ast.FunctionDef) and n.name == "attach")
reset_hook = next(n for n in attach.body if isinstance(n, ast.FunctionDef) and n.name == "reset_with_navigation")
select_ids = next(n for n in nav_class.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_env_ids_from_args")
capture = []
spawn = torch.tensor([[1.,2.,0.,0.,0.,0.,1.], [3.,4.,0.,0.,0.,0.,1.]])
manual = torch.tensor([[9.,9.,0.,0.,0.,0.,1.]])
manager = NS(env=NS(num_envs=2, _crowdsim_robot_spawn_poses=spawn), _has_completed_initial_reset=False,
    config=NS(num_robots=1, device=torch.device("cpu")),
    robot=NS(write_root_pose_to_sim=lambda poses: capture.append(poses.clone()), write_root_velocity_to_sim=lambda vel: None),
    drive=NS(sync_teleported_poses=lambda poses: None),
    _read_agent_state=lambda: (np.zeros((3,2)),np.zeros((3,2))),
    _compute_sfm_reference_waypoints=lambda p,v: None, _update_local_target_markers=lambda: None,
    _record_trajectory_frame=lambda p,v: None)
namespace = {"self": manager, "env": manager.env, "np": np, "torch": torch, "original_reset": lambda *a,**kw: ({}, {})}
exec(compile(ast.fix_missing_locations(ast.Module(body=[select_ids, reset_hook], type_ignores=[])),
    str(root / "CrowdSim/nav_manager.py"), "exec"), namespace)
manager._reset_env_ids_from_args = lambda args, kwargs: namespace["_reset_env_ids_from_args"](manager,args,kwargs)
capture.append(manual)
namespace["reset_with_navigation"](manager.env, None)
results["tests"]["first_full_reset_overwrites_eval_teleport"] = {
    "pre_reset_requested_robot_xy": capture[0][0,:2].tolist(),
    "reset_hook_written_robot_xy": capture[-1][0,:2].tolist(),
    "method": "Actual nested reset hook AST; articulation writes recorded by fixture; no physics",
}

files = [p for folder in (root / "CrowdSim", root / "src/crowdsim_ppo") for p in folder.rglob("*.py")]
files.append(root / "protomotions/simulator/base_simulator/record.py")
results["source_hashes"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
results["python"] = __import__("sys").version
results["torch"] = torch.__version__
out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
print(json.dumps(results["tests"], indent=2, allow_nan=False))
