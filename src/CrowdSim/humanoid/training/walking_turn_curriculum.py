"""Training-only real walking resets and tight-turn XY command exercises.

The AMP discriminator continues to draw all2848 train clips. This course uses
source-bound walking windows for physical episode initialization/reward phases.
Only high-level XY intentions are analytic; every body transition stays PhysX.
"""
import hashlib,json,math
from pathlib import Path
import torch

class WalkingTurnCurriculum:
 def __init__(self,corpus,window_path,num_envs):
  if corpus.split!='train' or len(corpus.ids)!=2848:raise ValueError('Fixed2848 train bank required')
  data=json.loads(Path(window_path).read_text());self.corpus=corpus;self.num_envs=num_envs
  if data.get('fit_split',data.get('split'))!='train':raise ValueError('Train-only walking windows required')
  for key,expected in [('manifest_sha256',hashlib.sha256(corpus.manifest_path.read_bytes()).hexdigest()),('source_bank_sha256',corpus.entry['sha256'])]:
   if data[key]!=expected:raise ValueError('Walking windows must bind the exact corpus')
  ids=set(corpus.ids.tolist());rows=[]
  for win in data['windows']:
   mid,first=win['motion_id'],win['first'];length=win.get('frames',60);last=first+length-1
   if type(length)!=int or not 60<=length<=180:raise ValueError('Real reset windows require60..180 source frames')
   if type(mid)!=int or mid not in ids or type(first)!=int or first<0 or last>=int(corpus.data['motion_num_frames'][mid]):raise ValueError('Every walking window must stay within a real training clip')
   start=max(first,9)
   if last-start>=31:rows.append([mid,start,last])
  if not rows:raise ValueError('Walking reset course has no legal history/future')
  self.windows=torch.tensor(rows,dtype=torch.long);self.path=Path(window_path)
  self.anchor=torch.zeros(num_envs,2,device=corpus.device);self.yaw=torch.zeros(num_envs,device=corpus.device)
  self.bound=torch.zeros(num_envs,dtype=torch.bool,device=corpus.device);self.resets=0;self.source_window_held_knots=0
  actors=torch.arange(num_envs,device=corpus.device)
  self.group=actors%4;self.radius=.28+.14*((actors//4)%2)
  self.speed=.45+.1*((actors//8)%2);self.sign=torch.where(self.group==1,1.,-1.)
  self.speed_course=data.get('curriculum')
  self.initial_speed=torch.zeros(num_envs,device=corpus.device)
  self.stop_return_mask=torch.zeros(num_envs,dtype=torch.bool,device=corpus.device)
  if self.speed_course is not None:
   cfg=self.speed_course
   if cfg.get('schema')!='crowdsim.balanced_walking_speed_course.v1':raise ValueError('Unknown explicit speed course')
   levels=cfg['speed_levels_m_s'];radii=cfg['turn_radius_offsets_m'];ramp=cfg['transition_s']
   if (levels!=[.5,.75,1.,1.25,1.5] or len(radii)!=2 or any(type(r) not in (int,float) or not math.isfinite(r) or r<1 for r in radii)
       or type(ramp) not in (int,float) or not math.isfinite(ramp) or not .2<=ramp<=1.):raise ValueError('Finite safe walking-speed curriculum required')
   if num_envs%160:raise ValueError('160-actor multiples fully cross4 horizons,4 categories,5 speeds,2 radii')
   self.group=(actors//4)%4
   self.speed=torch.tensor(levels,device=corpus.device)[(actors//16)%5]
   self.radius=torch.tensor(radii,device=corpus.device)[(actors//80)%2]+.5*self.speed
   self.sign=torch.where(self.group==1,1.,-1.)
   if cfg.get('stop_return_course') is not None:
    if cfg['stop_return_course']!={'schema':'crowdsim.stop_return_intent.v1','brake_start_s':1.2,'brake_duration_s':.4,'return_start_s':2.,'return_ramp_s':.5,'return_speed_fraction':.55}:raise ValueError('Unknown finite stop-return clock')
    self.stop_return_mask=(self.group==2)&((actors//80)%2==1)
    tight=((self.group==1)|(self.group==3))&((actors//80)%2==1)
    self.speed=torch.where(tight,(self.speed*.65).clamp_max(.65),self.speed)
    self.radius=torch.where(tight,self.radius.new_full((num_envs,),.42),self.radius)
 def sample_reference(self,count):
  picked=self.windows[torch.randint(len(self.windows),(count,),generator=self.corpus.generator)]
  # The first legal phase retains ten actual history frames and31 real future
  # transitions. The source's declared60-frame walking window ends the episode.
  frames=picked[:,1].clone()
  mid=picked[:,0];self.last_remaining=picked[:,2]-frames
  absolute=self.corpus.data['length_starts'][mid]+frames
  history=absolute[:,None]-torch.arange(10)
  return self.corpus.state_at(absolute),self.corpus.state_at(history.reshape(-1)),self.corpus.data['dps'][absolute+1].to(self.corpus.device),mid,frames
 def bind_reset(self,world,actors):
  curriculum=world.amass_root_curriculum
  if curriculum is None or len(actors)!=len(self.last_remaining):raise ValueError('Bind the exact reset batch')
  curriculum.remaining_steps[actors]=torch.minimum(curriculum.remaining_steps[actors],self.last_remaining.to(world.device))
  st=world.state();q=st.root_rot[actors];x,y,z,w=q.unbind(-1)
  self.anchor[actors]=st.root_pos[actors,:2].detach();self.yaw[actors]=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));self.bound[actors]=True;self.resets+=len(actors)
  self.initial_speed[actors]=st.root_vel[actors,:2].norm(dim=-1).detach()
 def commands(self,world,offsets):
  if not bool(self.bound.all()):raise ValueError('Every actor needs its actual source reset')
  curriculum=world.amass_root_curriculum
  raw=(curriculum.initial_frames+world.age).cpu()[:,None]+offsets
  ends=(curriculum.initial_frames+curriculum.remaining_steps).cpu()[:,None]
  self.source_window_held_knots+=int(((raw>ends)&(self.group.cpu()==0)[:,None]).sum())
  raw=torch.minimum(raw,ends)
  real,_,beyond=curriculum._sample_xy(curriculum.motion_ids.cpu(),raw,curriculum.offset_xy,curriculum.end_velocity_xy)
  seconds=(world.age[:,None]+offsets.to(world.device)).float()/30
  distance=self.speed[:,None]*seconds
  if self.speed_course is not None:
   ramp=self.speed_course['transition_s'];early=seconds.clamp_max(ramp)
   distance=self.initial_speed[:,None]*early+.5*(self.speed-self.initial_speed)[:,None]*early.square()/ramp+self.speed[:,None]*(seconds-ramp).clamp_min(0)
  course=circle_commands(self.anchor,self.yaw,self.radius,self.speed,self.sign,distance/self.speed[:,None])
  # Global actor group0 follows the actual walking source; group2 walks straight;
  # groups1/3 practice both turn directions. No group ID enters the policy.
  straight_distance=distance if self.speed_course is not None else seconds.clamp_max(.8)*self.speed[:,None]
  if bool(self.stop_return_mask.any()):
   stop_distance,_=stop_return_course(seconds,self.initial_speed[:,None],self.speed[:,None],self.speed_course['transition_s'])
   straight_distance=torch.where(self.stop_return_mask[:,None],stop_distance,straight_distance)
  straight=self.anchor[:,None]+straight_distance[...,None]*torch.stack((self.yaw.cos(),self.yaw.sin()),-1)[:,None]
  course=torch.where((self.group==2)[:,None,None],straight,course)
  xy=torch.where((self.group==0)[:,None,None],real,course)
  now=world.age.float()/30;current_speed=self.speed;current_distance=self.speed*now
  if self.speed_course is not None:
   ramp=self.speed_course['transition_s'];early=now.clamp_max(ramp)
   current_speed=self.initial_speed+(self.speed-self.initial_speed)*(now/ramp).clamp(0,1)
   current_distance=self.initial_speed*early+.5*(self.speed-self.initial_speed)*early.square()/ramp+self.speed*(now-ramp).clamp_min(0)
  angle=self.yaw+self.sign*current_distance/self.radius
  velocity=current_speed[:,None]*torch.stack((angle.cos(),angle.sin()),-1)
  straight_velocity=current_speed[:,None]*torch.stack((self.yaw.cos(),self.yaw.sin()),-1)
  if self.speed_course is None:straight_velocity=straight_velocity*(now<.8)[:,None]
  if bool(self.stop_return_mask.any()):
   _,stop_speed=stop_return_course(now,self.initial_speed,self.speed,self.speed_course['transition_s'])
   stop_velocity=stop_speed[:,None]*torch.stack((self.yaw.cos(),self.yaw.sin()),-1)
   straight_velocity=torch.where(self.stop_return_mask[:,None],stop_velocity,straight_velocity)
  velocity=torch.where((self.group==2)[:,None],straight_velocity,velocity)
  # Actual real reference velocity remains the target for group0 only.
  absolute=self.corpus.data['length_starts'][curriculum.motion_ids.cpu()]+(curriculum.initial_frames+world.age).cpu()
  reference=self.corpus.state_at(absolute)
  velocity=torch.where((self.group==0)[:,None],reference.root_vel[:,:2],velocity)
  return xy,beyond,velocity
 def describe(self):
  if self.speed_course is not None:
   return {'schema':'crowdsim.balanced_walking_speed_course.v1','source_windows':str(self.path),'source_windows_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'fit_split':'train','walking_windows':len(self.windows),'physical_reset_unique_source_ids':len(self.windows[:,0].unique()),'expert_population_unchanged':2848,'actor_assignment':'horizon actor%4; category actor//4%4; speed actor//16%5; radius actor//80%2; complete160-actor product','course':self.speed_course,'source_reset_velocity_unmodified':True,'real_reference_pose_unmodified':True,'velocity_reward_target':'actual real source for group0; exact derivative of integrated XY speed ramp for other groups','physical_state_or_action_override':False,'source_episode_bound':'end of declared real source window;10 history frames retained','source_reset_count':self.resets,'source_window_held_knots':self.source_window_held_knots,'author_recipe':False}
  return {'schema':'crowdsim.walking_turn_training_course.v1','source_windows':str(self.path),'source_windows_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'fit_split':'train','walking_windows':len(self.windows),'physical_reset_unique_source_ids':len(self.windows[:,0].unique()),'expert_population_unchanged':2848,'actor_assignment':'actor_id modulo4: real walking, left circle, straight then stop at0.8s, right circle','circle_radius_m':[.28,.42],'command_speed_m_s':[.45,.55],'real_reference_pose_unmodified':True,'velocity_reward_target':'real source for group0; derivative of declared XY command for procedural groups','physical_state_or_action_override':False,'source_episode_bound':'end of declared60-frame real walking window; reset at first legal history10 phase, not a random later phase','source_reset_count':self.resets,'source_window_held_knots':self.source_window_held_knots,'author_recipe':False}
 def state_dict(self):
  return {'schema':'crowdsim.walking_turn_training_course.v1','description':self.describe(),'anchor':self.anchor.detach().cpu().clone(),'yaw':self.yaw.detach().cpu().clone(),'bound':self.bound.detach().cpu().clone(),'initial_speed':self.initial_speed.detach().cpu().clone(),'resets':self.resets}

def circle_commands(anchor,yaw,radius,speed,sign,seconds):
 for x in (anchor,yaw,radius,speed,sign,seconds):
  if not torch.isfinite(x).all():raise ValueError('Finite command geometry required')
 if bool((radius<=0).any()) or bool((speed<0).any()) or not bool((sign.abs()==1).all()):raise ValueError('Positive radius and explicit turn sides required')
 angle=seconds*speed[:,None]/radius[:,None]
 local=torch.stack((radius[:,None]*angle.sin(),sign[:,None]*radius[:,None]*(1-angle.cos())),-1)
 c,s=yaw.cos()[:,None],yaw.sin()[:,None]
 world=torch.stack((c*local[...,0]-s*local[...,1],s*local[...,0]+c*local[...,1]),-1)
 return anchor[:,None]+world


def stop_return_course(seconds,initial_speed,speed,ramp):
 """Continuous high-level out/stop/return XY; derivative is the reward target."""
 out=seconds.clamp_max(1.2);early=out.clamp_max(ramp)
 distance=initial_speed*early+.5*(speed-initial_speed)*early.square()/ramp+speed*(out-ramp).clamp_min(0)
 braking=(seconds-1.2).clamp(0,.4)
 distance=distance+speed*(braking-.5*braking.square()/.4)
 returning=(seconds-2.).clamp_min(0);early_return=returning.clamp_max(.5)
 distance=distance-.55*speed*(.5*early_return.square()/.5+(returning-.5).clamp_min(0))
 velocity=initial_speed+(speed-initial_speed)*(seconds/ramp).clamp(0,1)
 velocity=torch.where(seconds>=1.2,speed*(1-(seconds-1.2)/.4).clamp(0,1),velocity)
 velocity=torch.where(seconds>=2.,-.55*speed*((seconds-2.)/.5).clamp(0,1),velocity)
 return distance,velocity
