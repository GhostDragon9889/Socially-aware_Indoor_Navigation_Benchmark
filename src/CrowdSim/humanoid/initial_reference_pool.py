"""Select complete, real AMASS histories for bounded native-scene starts."""
import math
import torch
from .amp.observations import PACER_BODY_NAMES


def build_initial_reference_pool(corpus, history_frames, max_speed_m_s, arm_cone_deg=None):
    if type(history_frames) is not int or history_frames<1:
        raise ValueError('Positive complete history required')
    if type(max_speed_m_s) not in (int,float) or not math.isfinite(max_speed_m_s) or not 0<max_speed_m_s<=.5:
        raise ValueError('Initial reference speed must be in(0,.5]')
    if arm_cone_deg is not None and (type(arm_cone_deg) not in (int,float) or not math.isfinite(arm_cone_deg) or not 5<=arm_cone_deg<=45):
        raise ValueError('Arm cone must be5..45 degrees or None')
    pool=[];sources=[];local_frames=[];legacy_count=0
    for motion_id in corpus.ids.tolist():
        begin=int(corpus.data['length_starts'][motion_id]);count=int(corpus.data['motion_num_frames'][motion_id])
        ix=torch.arange(begin+history_frames-1,begin+count-1)
        speed=corpus.data['gvs'][ix,0,:2].norm(dim=-1)
        height=corpus.data['gts'][ix,0,2];q=corpus.data['grs'][ix,0]
        good=(speed<=max_speed_m_s)&(height>.9)&(height<1.05)&(1-2*q[:,:2].square().sum(-1)>.98)
        feet=corpus.data['gts'][ix][:,[4,8],2]
        joint_rate=corpus.data['dvs'][ix].square().mean(-1).sqrt()
        root_rate=corpus.data['gavs'][ix,0].norm(dim=-1)
        good &= (feet.max(-1).values<.07)&(joint_rate<1.5)&(root_rate<.8)
        legacy_count+=int(good.sum())
        if arm_cone_deg is not None:
            body=corpus.data['gts'][begin:begin+count]
            wrist=[PACER_BODY_NAMES.index(side+'_Wrist') for side in ('L','R')]
            shoulder=[PACER_BODY_NAMES.index(side+'_Shoulder') for side in ('L','R')]
            delta=body[:,wrist]-body[:,shoulder]
            cone=torch.atan2(delta[:,:,:2].norm(dim=-1),-delta[:,:,2])*180/math.pi
            arm_good=(torch.isfinite(cone)&(cone<=arm_cone_deg)&(delta[:,:,2]<0)).all(-1)
            prefix=torch.cat((torch.zeros(1,dtype=torch.long),arm_good.long().cumsum(0)))
            local=ix-begin
            good &= prefix[local+1]-prefix[local+1-history_frames]==history_frames
        selected=ix[good];pool.extend(selected.tolist());sources.extend([motion_id]*len(selected));local_frames.extend((selected-begin).tolist())
    if not pool:raise ValueError('No complete real AMASS history meets the declared native-scene initial conditions')
    report={'schema':'crowdsim.real_AMASS_initial_reference_pool.v1','max_speed_m_s':max_speed_m_s,'history_frames':history_frames,'arm_cone_deg':arm_cone_deg,'arm_condition_applied_to_every_history_frame':arm_cone_deg is not None,'low_speed_upright_grounded_frame_count':legacy_count,'eligible_frame_count':len(pool),'eligible_motion_count':len(set(sources)),'pose_source':'unmodified same-frame AMASS root and69DOF state','runtime_pose_override':False}
    return (torch.tensor(pool),torch.tensor(sources),torch.tensor(local_frames)),report
