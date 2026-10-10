from types import SimpleNamespace
import torch
from CrowdSim.humanoid.initial_reference_pool import build_initial_reference_pool
from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES

def fixture():
 body=torch.zeros(10,24,3);body[:,0,2]=.97
 for side in ('L','R'):
  body[:,PACER_BODY_NAMES.index(side+'_Shoulder'),2]=1.4
  body[:,PACER_BODY_NAMES.index(side+'_Wrist'),2]=.9
 body[3,PACER_BODY_NAMES.index('R_Wrist'),2]=1.6
 q=torch.zeros(10,24,4);q[...,3]=1
 data={'length_starts':torch.tensor([0]),'motion_num_frames':torch.tensor([10]),'gts':body,'grs':q,'gvs':torch.zeros(10,24,3),'gavs':torch.zeros(10,24,3),'dvs':torch.zeros(10,69)}
 return SimpleNamespace(ids=torch.tensor([0]),data=data)

def test_disabled_matches_existing_pool_and_complete_histories_exclude_raised_arms():
 c=fixture();original=c.data['gts'].clone()
 legacy,r=build_initial_reference_pool(c,3,.25)
 assert legacy[0].tolist()==list(range(2,9)) and r['eligible_frame_count']==7
 neutral,r=build_initial_reference_pool(c,3,.25,25)
 assert neutral[0].tolist()==[2,6,7,8] and neutral[2].tolist()==[2,6,7,8]
 assert torch.equal(c.data['gts'],original) and r['runtime_pose_override'] is False
 assert r['arm_condition_applied_to_every_history_frame'] is True

def test_empty_or_invalid_pool_fails_and_unsupported_pose_is_not_replaced():
 import pytest
 c=fixture()
 c.data['gts'][:,PACER_BODY_NAMES.index('R_Wrist'),2]=1.6
 with pytest.raises(ValueError,match='No complete real'):build_initial_reference_pool(c,3,.25,25)
 for value in (True,float('nan'),4,46):
  with pytest.raises(ValueError):build_initial_reference_pool(c,3,.25,value)
