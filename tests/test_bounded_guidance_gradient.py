import torch,pytest
from CrowdSim.humanoid.diffusion.trace_helpers import unicyle_forward_dynamics
from CrowdSim.humanoid.diffusion.dynamics import Unicycle
from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidanceConfig

def integrate(actions,leak):
 return unicyle_forward_dynamics(Unicycle('test',acce_bound=[-4,4],vbound=[0,.75]),torch.zeros(1,4),actions,.1,pedestrian_speed_ceiling=.75,pedestrian_yaw_rate_ceiling=1.2,bounded_guidance_gradient=leak)

def test_saturated_stop_has_identical_forward_bounds_but_goal_can_escape():
 a=torch.zeros(1,52,2);a[:,:,0]=-10.;a.requires_grad_()
 old=integrate(a,0.);new=integrate(a,.1)
 assert torch.equal(old,new)
 assert torch.equal(new[...,:3],torch.zeros_like(new[...,:3]))
 old_grad=torch.autograd.grad((old[...,0]-2).square().mean(),a,retain_graph=True)[0]
 new_grad=torch.autograd.grad((new[...,0]-2).square().mean(),a)[0]
 assert old_grad[:,:,0].abs().max()==0
 assert (new_grad[:,:,0]<0).any() and torch.isfinite(new_grad).all()

def test_no_grad_sampling_is_identical_and_upper_limits_remain_exact():
 torch.manual_seed(1);a=torch.randn(1,52,2)*10
 with torch.no_grad():
  old=integrate(a,0.);new=integrate(a,.1)
 assert torch.equal(old,new)
 assert (new[:,:,2]>=0).all() and (new[:,:,2]<=.75).all()
 assert torch.diff(torch.cat([torch.zeros(1,1),new[:,:,3]],1),dim=1).abs().max()<=.120001
 with pytest.raises(ValueError,match=r'in\[0,1\]' ):PedestrianGuidanceConfig(bounded_guidance_gradient=1.1)
