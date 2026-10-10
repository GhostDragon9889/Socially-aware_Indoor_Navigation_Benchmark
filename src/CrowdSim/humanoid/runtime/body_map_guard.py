"""Actual oriented collision-shape projections and causal rigid-body forecasts.

Uses source MJCF capsules/box corners, not a height-independent3D sphere radius.
Only braking intent changes; actual poses/actions still come from AMP and PhysX.
The2D native map remains conservative; actual force archives are mandatory.
"""

from protomotions.repository import repository_root
from pathlib import Path
import hashlib,xml.etree.ElementTree as ET,itertools
import numpy as np
import torch
ASSET=repository_root(__file__)/'protomotions/data/assets/mjcf/smpl_humanoid.xml'

def collision_spheres(body_names, *, path=ASSET, contact_offset_m=.02):
 tree=ET.parse(path);bodies={b.attrib['name']:b for b in tree.findall('.//body')};centers=[];radii=[];vertices=[];surface=[]
 if len(set(body_names))!=len(body_names) or not np.isfinite(contact_offset_m) or contact_offset_m<0:raise ValueError('Unique identities and finite contact margin required')
 signs=np.array(list(itertools.product((-1,1),repeat=3)))
 for name in body_names:
  geoms=bodies[name].findall('geom')
  if len(geoms)!=1:raise ValueError('Exactly one source collision shape per SMPL body required')
  g=geoms[0];size=np.fromstring(g.attrib['size'],sep=' ')
  if g.attrib['type']=='box':
   center=np.fromstring(g.attrib.get('pos','0 0 0'),sep=' ');radius=np.linalg.norm(size);corners=signs*size
   qw,qx,qy,qz=np.fromstring(g.attrib.get('quat','1 0 0 0'),sep=' ');qv=np.array([qx,qy,qz]);corners=corners+2*np.cross(qv,np.cross(qv,corners)+qw*corners)
   points=corners;round_radius=0.
  elif g.attrib['type']=='capsule':
   endpoints=np.fromstring(g.attrib['fromto'],sep=' ').reshape(2,3);center=endpoints.mean(0);radius=np.linalg.norm(endpoints[1]-endpoints[0])/2+size[0]
   # Repeated endpoints let the same8-vertex projection cover both shape kinds.
   points=np.tile(endpoints-center,(4,1));round_radius=size[0]
  else:raise ValueError('Unsupported collision shape')
  if center.shape!=(3,) or points.shape!=(8,3) or not np.isfinite(points).all() or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:raise ValueError('Finite authored collision bounds required')
  centers.append(center);radii.append(radius+contact_offset_m);vertices.append(points);surface.append(round_radius+contact_offset_m)
 return {'centers':np.array(centers),'radii':np.array(radii),'vertices':np.array(vertices),'surface_radii':np.array(surface),'asset_path':str(path),'asset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'contact_offset_m':contact_offset_m,'scope':'Actual oriented capsule endpoints/box corners projected on native2D map, with constant measured-velocity forecast; actual forces remain mandatory'}

def rotate_quaternion(vectors,quaternions):
 xyz=quaternions[...,:3];w=quaternions[...,3:4]
 return vectors+2*torch.cross(xyz,torch.cross(xyz,vectors,dim=-1)+w*vectors,dim=-1)

def world_centers(positions,quaternions,local_centers):
 local=torch.as_tensor(local_centers,dtype=positions.dtype,device=positions.device).expand_as(positions)
 rotated=rotate_quaternion(local,quaternions)
 return positions+rotated,rotated

def rotate_forecast(vectors,omega,times):
 # vectors[B,L,V,3], omega[B,L,3], exact rigid rotation for constant omega.
 speed=omega.norm(dim=-1);axis=omega/speed.clamp_min(1e-8)[...,None]
 axis=axis[...,None,None,:];v=vectors[...,None,:];angle=speed[...,None,None]*times
 angle=angle[...,None];c,s=angle.cos(),angle.sin()
 return v*c+torch.cross(axis.expand_as(v),v,dim=-1)*s+axis*(axis*v).sum(-1,keepdim=True)*(1-c)

def shape_forecasts(state,geometry,times):
 p,q=state.rigid_body_pos,state.rigid_body_rot
 local=torch.as_tensor(geometry['centers'],device=p.device,dtype=p.dtype).expand_as(p)
 center_offset=rotate_quaternion(local,q)
 corners=p.new_tensor(geometry['vertices']).expand(*p.shape[:-1],8,3)
 rotated=rotate_quaternion(corners,q[...,None,:].expand(*corners.shape[:-1],4))
 dynamic_center=rotate_forecast(center_offset[...,None,:],state.rigid_body_ang_vel,times).squeeze(-3)
 centers=p[...,None,:]+state.rigid_body_vel[...,None,:]*times[:,None]+dynamic_center
 future_vertices=rotate_forecast(rotated,state.rigid_body_ang_vel,times)
 radii=future_vertices[...,:2].norm(dim=-1).amax(-2)+p.new_tensor(geometry['surface_radii'])[None,:,None]
 return centers,radii

def body_map_hazard(scene,state,geometry, *, horizon_s=.3,gap_m=.025):
 times=state.rigid_body_pos.new_tensor([0.,horizon_s/3,2*horizon_s/3,horizon_s]);center,radii=shape_forecasts(state,geometry,times)
 clearance=scene.signed_distance(center[...,:2])-radii
 current=clearance[...,0];predicted=clearance[...,1:].amin(-1)
 hazard=(predicted<gap_m)&(predicted<current-.01)
 return hazard.any(-1),{'body_parts_at_risk':hazard.tolist(),'current_minimum_sphere_map_clearance_m':current.amin(-1).tolist(),'predicted_minimum_sphere_map_clearance_m':predicted.amin(-1).tolist(),'horizon_s':horizon_s,'gap_m':gap_m,'geometry_source_sha256':geometry['asset_sha256'],'scope':geometry['scope']}
