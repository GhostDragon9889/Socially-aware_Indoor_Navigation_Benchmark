"""Run with the AMASS venv; use unchanged production methods and real context class.
Only path-target generation is replaced with zeros to isolate global index selection.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from protomotions.envs.context_views import MaskedMimicContext
path=Path('CrowdSim/nav_manager.py')
cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CrowdNavigationManager')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_override_masked_mimic_context','_yaw_to_quat_xyzw_tensor')]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
ns={'np':np,'torch':torch}
exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
manager=ns['CrowdNavigationManager']()
manager.config=NS(num_humanoids=2)
manager.reached=np.array([True,False])
manager._humanoid_target_offsets=lambda n:np.array([.5],dtype=np.float32)
manager._humanoid_future_targets_from_path=lambda xy,offsets:(np.zeros((1,1,2),np.float32),np.zeros((1,1),np.float32))
base=MaskedMimicContext(mimic=None,ref_pos=torch.zeros(1,1,1,3),ref_rot=torch.tensor([[[[0.,0.,0.,1.]]]]),
    target_times=torch.zeros(1,1),time_offsets=torch.zeros(1,1),target_poses_masks=torch.ones(1,1),target_bodies_masks=torch.ones(1,2))
ctx=NS(masked_mimic=base,env_ids=torch.tensor([1]),current=NS(rigid_body_pos=torch.zeros(1,1,3)))
component=NS(conditionable_body_ids=torch.tensor([0]),num_conditionable_bodies=1,
    env=NS(robot_config=NS(anchor_body_index=0),motion_manager=NS(motion_times=torch.tensor([10.,20.]))))
manager._override_masked_mimic_context(component,ctx)
result={'env_ids':[1],'expected_target_time':20.5,'actual_target_time':float(ctx.masked_mimic.target_times.item()),
        'expected_pose_mask':1.,'actual_pose_mask':float(ctx.masked_mimic.target_poses_masks.item()),
        'real_MaskedMimicContext':True,'method':'Unchanged AST production methods; zero path generator isolates time/mask indexing'}
Path(__file__).with_name('subset_reset.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
