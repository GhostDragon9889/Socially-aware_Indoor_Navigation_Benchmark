import torch,pytest
from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance,PedestrianGuidanceConfig,TrajectoryAdmissionError
from CrowdSim.humanoid.diffusion.refinement import refine_samples
from CrowdSim.humanoid.diffusion.trace_helpers import unicyle_forward_dynamics
from CrowdSim.humanoid.diffusion.dynamics import Unicycle

def guide(config,state):
 image=torch.zeros(len(state),2,224,224);image[:,0]=1
 return PedestrianGuidance(state,{'extent':torch.tensor([[.5,.5,1.7]]).expand(len(state),3),'image':image},config,scene_ids=torch.zeros(len(state),dtype=torch.long),local_targets=torch.stack([torch.linspace(.2,2.,10),torch.zeros(10)],-1)[None].expand(len(state),10,2),target_indices=torch.arange(4,50,5))

def test_refinement_escapes_saturated_stop_without_changing_forward_bounds():
 torch.set_num_threads(2)
 cfg=PedestrianGuidanceConfig(final_refinement_steps=96,goal_weight=100.,map_weight=0.,agent_weight=0.,dynamics_weight=5.,target_speed_weight=8.,target_speed_m_s=.5,max_speed_m_s=.75,max_acceleration_m_s2=4.,max_yaw_rate_rad_s=1.2)
 state=torch.zeros(1,4);dyn=Unicycle('test',acce_bound=[-4.,4.],vbound=[0.,.75])
 raw=torch.zeros(1,52,2);raw[:,:,0]=-10.
 integrated=unicyle_forward_dynamics(dyn,state,raw,.1,pedestrian_speed_ceiling=.75,pedestrian_yaw_rate_ceiling=1.2)
 samples=torch.cat([integrated,raw],-1)[:,None]
 original=samples.clone();result,meta=refine_samples(samples,state,guide(cfg,state),dyn)
 assert torch.equal(samples,original)
 assert (result[...,2]>=0).all() and (result[...,2]<=.75).all()
 assert torch.diff(torch.cat([torch.zeros(1,1,1),result[...,3]],-1),dim=-1).abs().max()<=.120001
 assert result[0,0,5,0]>.03 and result[0,0,-1,0]>1.5
 assert meta['guidance_cost_sum_after']<meta['guidance_cost_sum_before']
 assert not meta['model_parameters_optimized'] and not meta['body_pose_modified']
 assert result[...,4].abs().max()<=4 and result[...,5].abs().max()<=1.2

def test_disabled_refinement_returns_identical_samples_and_unsafe_selection_still_fails():
 cfg=PedestrianGuidanceConfig();state=torch.zeros(2,4);state[1,0]=.2
 g=guide(cfg,state);samples=torch.zeros(2,1,52,6)
 out,meta=refine_samples(samples,state,g,None)
 assert out is samples and meta is None
 with pytest.raises(TrajectoryAdmissionError):g.select(samples)
