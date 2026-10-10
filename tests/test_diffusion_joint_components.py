import torch
from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance,PedestrianGuidanceConfig

def test_unrelated_choices_cannot_prune_only_feasible_shared_pair():
 # The64 cheapest global prefixes all choose a colliding trajectory for actor2.
 # Actor2's costly stopping candidate permits actor3 to proceed safely.
 state=torch.tensor([[0.,50.,0.,0.],[0.,60.,0.,0.],[0.,0.,0.,0.],[2.,0.,0.,0.]])
 cfg=PedestrianGuidanceConfig(goal_weight=1.,map_weight=0.,agent_weight=0.,dynamics_weight=0.)
 context={'extent':torch.tensor([[.5,.5,1.7]]).expand(4,3),'image':torch.stack([torch.ones(4,224,224),torch.zeros(4,224,224)],1)}
 guide=PedestrianGuidance(state,context,cfg,scene_ids=torch.zeros(4,dtype=torch.long))
 trajectory=torch.zeros(4,17,52,6)
 progress=torch.linspace(.02,1.,52)
 trajectory[2,:16,:,0]=progress;trajectory[3,:,:,0]=-progress
 cost=torch.arange(17).float().repeat(4,1)*.001;cost[2,:16]=0.;cost[2,16]=100.
 parts={k:torch.zeros(4,17) for k in ('map','dynamics','target_speed','yaw_smoothness','speed_smoothness','minimum_speed_m_s','maximum_speed_limit_violation_m_s','maximum_acceleration_m_s2','maximum_yaw_rate_rad_s')}
 parts['goal']=cost;parts['minimum_map_clearance_m']=torch.ones(4,17)
 guide.score=lambda x:(cost,parts)
 selected,diagnostic=guide.select(trajectory)
 assert selected[2]==16
 picked=trajectory[torch.arange(4),selected]
 root=state[:,None,:2]+picked[:,:,:2]
 assert (root[2]-root[3]).norm(dim=-1).min()>=.99
 assert diagnostic['selected_joint_maximum_physical_overlap_m'].max()==0
