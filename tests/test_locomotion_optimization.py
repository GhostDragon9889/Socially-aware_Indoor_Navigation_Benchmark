from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance,PedestrianGuidanceConfig
from CrowdSim.humanoid.training.locomotion_objective import locomotion_reward

def guide(config):
 image=torch.zeros(1,2,224,224);image[:,0]=1
 return PedestrianGuidance(torch.zeros(1,4),{'extent':torch.tensor([[.5,.5,1.7]]),'image':image},config,scene_ids=torch.zeros(1,dtype=torch.long))

def test_turn_speed_objective_prefers_slower_turns_and_keeps_finite_derivatives():
 cfg=PedestrianGuidanceConfig(target_speed_m_s=1.,turn_speed_decay_s=1.)
 g=guide(cfg)
 data=torch.zeros(1,2,20,6)
 data[:,0,:,2]=1.;data[:,1,:,2]=1/(2**.5)
 data[:,:,:,3]=torch.arange(1,21)*.1
 data.requires_grad_()
 speed=g.components(data)['target_speed']
 assert speed[0,1] < speed[0,0]
 gradient=torch.autograd.grad(speed.sum(),data)[0]
 assert torch.isfinite(gradient).all() and gradient[...,2].abs().sum()>0

def test_start_ramp_does_not_reward_instantaneous_speed_jump():
 g=guide(PedestrianGuidanceConfig(target_speed_m_s=1.,speed_ramp_time_s=1.))
 data=torch.zeros(1,2,10,6);data[:,0,:,2]=1.;data[:,1,:,2]=torch.arange(1,11)*.1
 cost=g.components(data)['target_speed']
 assert cost[0,1] < cost[0,0]

def test_posture_speed_reward_distinguishes_correct_training_reference_and_excludes_inactive_dofs():
 ref=SimpleNamespace(dof_pos=torch.zeros(2,69),root_vel=torch.tensor([[1.,0.,0.],[1.,0.,0.]]))
 q=torch.zeros(2,69);q[1,0]=.8
 state=SimpleNamespace(dof_pos=q,root_vel=torch.tensor([[1.,0.,0.],[.3,0.,0.]]))
 mask=torch.ones(69);mask[-1]=0
 action=torch.zeros(2,69);action[0,-1]=999
 settings={'posture_weight':.35,'velocity_weight':.25,'action_change_weight':.01}
 reward,metrics=locomotion_reward(state,ref,torch.zeros_like(action),action,mask,settings)
 assert reward[0] > reward[1] and metrics['action_change'][0] == 0
 assert metrics['velocity_error_m_s'][0] == 0
