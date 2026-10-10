"""CPU review reproductions; execute from the repository root with PYTHONPATH=src:.

MotionManager is executed from its unchanged AST class body to isolate missing
OmegaConf/MotionLib imports. Tensor-backed simulator inputs are synthetic;
these checks do not claim to run IsaacLab physics.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import contextlib
import copy
import io
import json
import math
import importlib
import numpy as np
import torch

torch.set_num_threads(1)
results = {}

# A singleton tail batch must leave losses and parameters finite.
for module in ('CrowdSim.ppo.ppo_policy', 'crowdsim_ppo.ppo_policy'):
    p = importlib.import_module(module)
    for method in ('update', 'update_with_kl'):
        torch.manual_seed(42)
        trainer = p.RobotPPOTrainer(p.RobotPPOConfig(obs_dim=5, hidden_dims=(8,),
            map_enabled=False, depth_enabled=False, num_neighbors=1,
            minibatch_size=2, ppo_epochs=1), torch.device('cpu'))
        buffer = p.RobotRolloutBuffer(3, 1, 5, 2, torch.device('cpu'), num_neighbors=1)
        buffer.obs.normal_()
        buffer.raw_actions.normal_()
        buffer.returns[:, 0] = torch.tensor([1., 2., 3.])
        buffer.advantages[:, 0] = torch.tensor([-1., 0., 1.])
        with torch.no_grad():
            lp, _, values = trainer.model.evaluate_actions(buffer.obs[:, 0], buffer.raw_actions[:, 0], neighbors=buffer.neighbors[:, 0], neighbor_mask=buffer.neighbor_mask[:, 0])
            buffer.log_probs[:, 0] = lp
            buffer.values[:, 0] = values
        kwargs = {'ref_model': copy.deepcopy(trainer.model), 'kl_beta': 0.1} if method=='update_with_kl' else {}
        stats = getattr(trainer, method)(buffer, **kwargs)
        results[module+'.'+method] = {
            'value_loss_finite': math.isfinite(stats['value_loss']),
            'nonfinite_parameter_tensors': sum(not bool(torch.isfinite(v).all()) for v in trainer.model.parameters()),
            'rollout_samples': 3, 'minibatch_size': 2, 'executed_minibatches': int(stats['updates']) if 'updates' in stats else 'not returned',
        }

# A target backend identity quaternion should yield zero robot yaw.
from CrowdSim.control.drive import DriveController
controller = object.__new__(DriveController)
controller.config = NS(num_robots=1)
controller.robot = NS(data=NS(root_quat_w=torch.tensor([[0.,0.,0.,1.]])))
results['xyzw_identity_yaw'] = {'expected_degrees':0, 'actual_degrees':float(np.degrees(controller.yaws()[0]))}

# Requested reset distances/headings must still hold near map boundaries.
from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig
torch.manual_seed(42)
env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=10000,num_neighbors=0,
    min_start_goal_distance=2., max_start_goal_distance=4.,
    rl_initial_heading_min_offset_degrees=0.,rl_initial_heading_max_offset_degrees=15.),torch.device('cpu'))
delta=env.goal-env.position
angles=torch.atan2(delta[:,1],delta[:,0])-env.heading
offsets=torch.atan2(angles.sin(),angles.cos()).abs()*180/math.pi
dist=delta.norm(dim=-1)
results['point_goal_reset_bounds']={'seed':42,'samples':10000,'distance_below_2m':int((dist<2).sum()),
    'heading_offset_above_15deg':int((offsets>15.0001).sum()),'min_distance':float(dist.min()),'max_heading_offset':float(offsets.max())}

# Preserve length eligibility after weight updates; respect explicit repeated IDs.
from protomotions.envs.motion_manager.config import MotionManagerConfig
path=Path('protomotions/envs/motion_manager/motion_manager.py')
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MotionManager')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
namespace={'torch':torch,'np':np,'ListConfig':list}
exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
MotionManager=namespace['MotionManager']
lib=NS(motion_weights=torch.ones(2), motion_lengths=torch.tensor([.1,10.]),motion_file='synthetic.yaml',num_motions=lambda:2)
mm=MotionManager(MotionManagerConfig(sample_time_truncate_s=.5),1,.1,torch.device('cpu'),lib)
before=mm.motion_weights.tolist()
with contextlib.redirect_stdout(io.StringIO()):mm.update_sampling_weights(mm.motion_weights.clamp(min=.5))
results['short_motion_weights']={'before':before,'after':mm.motion_weights.tolist(),'expected_short_motion_weight':0.}
mm2=MotionManager(MotionManagerConfig(subset_method=[1,1,1]),3,.1,torch.device('cpu'),lib)
results['explicit_motion_ids']={'expected':[1,1,1],'actual':None if mm2.available_motion_ids is None else mm2.available_motion_ids.tolist()}

out=Path(__file__).with_name('reproductions.json')
out.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps(results,indent=2,allow_nan=False))
