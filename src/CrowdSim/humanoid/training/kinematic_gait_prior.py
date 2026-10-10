"""Training-only AMASS gait regression, conditioned on measured physical legs.

It supplies rewards, never poses/actions at deployment. Both sides share a
fit, so amplitudes are not prescribed by a hand-built animation or sine wave.
AMP source: https://xbpeng.github.io/projects/AMP/
"""
import torch

def features(state, names):
 q=state.root_rot;x,y,z,w=q.unbind(-1)
 yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));c,s=yaw.cos(),yaw.sin()
 forward=lambda v:c*v[...,0]+s*v[...,1]
 speed=state.root_vel[:,:2].norm(dim=-1)
 inputs=[];targets=[]
 for side,other in (('L','R'),('R','L')):
  hip,knee,ankle,toe=[state.rigid_body_pos[:,names.index(side+'_'+n)] for n in ('Hip','Knee','Ankle','Toe')]
  f=forward(toe-hip);v=forward(state.rigid_body_vel[:,names.index(side+'_Toe')]-state.root_vel)
  phase=torch.sigmoid((v-.1)/.1)
  inputs.append(torch.stack((speed,f,v,phase,f.square(),speed*phase,f*phase,v*phase),-1))
  a,b=hip-knee,ankle-knee
  flex=torch.pi-torch.acos(((a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1))).clamp(-1+1e-6,1-1e-6))
  lift=(toe[:,2]-state.rigid_body_pos[:,names.index(other+'_Toe'),2]).clamp_min(0)
  # The arm opposite to this leg naturally advances with its swing.
  arm=state.rigid_body_pos[:,names.index(other+'_Wrist')]-state.rigid_body_pos[:,names.index(other+'_Shoulder')]
  angle=torch.atan2(forward(arm),-arm[:,2])
  targets.append(torch.stack((lift,flex,angle),-1))
 return torch.stack(inputs,1),torch.stack(targets,1)

def compact_arm_score(state,names,limits,scale):
 q=state.root_rot;x,y,z,w=q.unbind(-1)
 yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));rows=[]
 for side in ('L','R'):
  shoulder=state.rigid_body_pos[:,names.index(side+'_Shoulder')]
  wrist=state.rigid_body_pos[:,names.index(side+'_Wrist')]-shoulder
  elbow=state.rigid_body_pos[:,names.index(side+'_Elbow')]-shoulder
  lateral=(-yaw.sin()*wrist[:,0]+yaw.cos()*wrist[:,1]).abs()
  rows.append(torch.stack((lateral,wrist[:,2],elbow[:,2]),-1))
 actual=torch.stack(rows,1)
 deficit=(actual-limits).clamp_min(0)
 score=torch.exp(-scale*deficit.square().mean((1,2)))
 return score,deficit

class KinematicGaitPrior:
 def __init__(self,data,device):
  if data['schema']!='crowdsim.real_amass_gait_regression.v1' or data['fit_split']!='train' or data['train_clip_count']!=2848:
   raise ValueError('Fixed real training bank gait prior required')
  self.data=data
  self.compact_arm_limits=None
  compact=data.get('compact_arms')
  if compact is not None:
   if compact['schema']!='crowdsim.real_amass_compact_arm_envelope.v1' or compact['fit_split']!='train' or not 0<compact['unique_motion_ids']<=2848:
    raise ValueError('Compact-arm limits must come from real training walking windows')
   limits=[compact['lateral_wrist_limit_m'],compact['wrist_height_above_shoulder_limit_m'],compact['elbow_height_above_shoulder_limit_m']]
   if not (0<limits[0]<1 and -1<limits[1]<0 and -1<limits[2]<0):raise ValueError('Finite natural compact-arm envelope required')
   self.compact_arm_limits=torch.tensor(limits,dtype=torch.float32,device=device)
   self.compact_arm_weight=compact['reward_weight'];self.compact_arm_scale=compact['squared_deficit_scale_per_m2']
   if not 0<self.compact_arm_weight<=10 or not 0<self.compact_arm_scale<=1000:raise ValueError('Finite bounded compact-arm reward required')
  for key in ('input_mean','input_std','coefficients','target_min','target_max','residual_scale'):
   value=torch.tensor(data[key],dtype=torch.float32,device=device)
   if not torch.isfinite(value).all():raise ValueError('Finite gait prior required')
   setattr(self,key,value)
  if self.coefficients.shape!=(9,3) or self.input_mean.shape!=(8,) or (self.input_std<=0).any() or (self.residual_scale<=0).any():raise ValueError('Wrong gait regression dimensions/scales')
 def reward(self,state,names, *, arm_weight=.7):
  f,y=features(state,names)
  # Detach conditioning: this is an RL reward, not inverse kinematics.
  f=f.detach();pred=torch.cat((torch.ones_like(f[...,:1]),((f-self.input_mean)/self.input_std).clamp(-4,4)),-1)@self.coefficients
  pred=pred.clamp(self.target_min,self.target_max)
  moving=((f[...,0]-.25)/.15).clamp(0,1)
  swing=f[...,3]
  error=(y-pred)/self.residual_scale
  scores=torch.exp(-error.square())
  # Lift and knee targets are used only for a physically swinging foot;
  # arm coupling remains active over the complete moving cycle.
  values=(scores[...,0]*swing*.5+scores[...,1]*swing*.3+scores[...,2]*arm_weight)*moving
  reward=values.mean(-1)
  metrics={'gait_prior_reward':reward,'gait_prior_lift_error_m':(y[...,0]-pred[...,0]).abs().mean(-1),'gait_prior_knee_error_rad':(y[...,1]-pred[...,1]).abs().mean(-1),'gait_prior_arm_error_rad':(y[...,2]-pred[...,2]).abs().mean(-1)}
  if self.compact_arm_limits is not None:
   compact_score,deficit=compact_arm_score(state,names,self.compact_arm_limits,self.compact_arm_scale)
   reward=reward+self.compact_arm_weight*compact_score
   metrics.update(compact_arm_reward=compact_score,compact_arm_mean_deficit_m=deficit.mean((1,2)),compact_arm_lateral_deficit_m=deficit[...,0].mean(1),compact_arm_wrist_height_deficit_m=deficit[...,1].mean(1))
  return reward,metrics
