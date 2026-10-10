"""A one-time measured holding intent after an actual stationary braking interval.

Changes a desired point only. Never writes a physical state or relaxes admission.
"""
from collections import deque
import torch,math
class SettledBrakingTarget:
 def __init__(self,count):
  self.history=[deque(maxlen=31) for _ in range(count)];self.settled=[False]*count
 def update(self,roots,braking,current_body_gap):
  if roots.shape!=(len(self.history),2) or braking.shape!=(len(self.history),):raise ValueError('Actual per-actor roots/masks required')
  if braking.dtype!=torch.bool or len(current_body_gap)!=len(self.history) or not all(math.isfinite(float(g)) for g in current_body_gap):raise ValueError('Finite measured body gaps and boolean braking masks required')
  if not torch.isfinite(roots).all():raise ValueError('Finite measured roots required')
  events=[]
  for actor,history in enumerate(self.history):
   if not bool(braking[actor]):history.clear();self.settled[actor]=False;continue
   history.append(roots[actor].detach().clone())
   if self.settled[actor] or len(history)<31 or current_body_gap[actor]<=0:continue
   positions=torch.stack(list(history));diam=torch.linalg.vector_norm(positions.amax(0)-positions.amin(0))
   if float(diam)<.05:
    self.settled[actor]=True;events.append({'actor':actor,'actual_root_xy':roots[actor].detach().clone(),'measured1s_bbox_diagonal_m':float(diam)})
  return events
