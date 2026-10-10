#!/usr/bin/env python3
"""Source-preserving static fixture for pedestrian simulation in collected YOHO.

Original doors/toilet seats have dynamic hinges. This explicitly fixes their
initial transforms without disabling collisions. Box colliders become exact
six-face Mesh boxes, compatible with the measured full-height map contract.
All changes are in a fresh stronger reference layer; original assets are read-only.
"""
from pathlib import Path
import argparse, hashlib, json
from pxr import Usd, UsdGeom, UsdPhysics, Gf
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve()
if a.output.exists():p.error('Fresh output required')
s=Usd.Stage.Open(str(a.source));assert s and not s.GetCompositionErrors()
assert str(UsdGeom.GetStageUpAxis(s))=='Z' and UsdGeom.GetStageMetersPerUnit(s)==1.
def rec(path):
 path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
sources=[rec(l.realPath) for l in s.GetUsedLayers() if l.realPath]
a.output.parent.mkdir(parents=True,exist_ok=True)
t=Usd.Stage.CreateNew(str(a.output));UsdGeom.SetStageUpAxis(t,'Z');UsdGeom.SetStageMetersPerUnit(t,1.)
root=UsdGeom.Xform.Define(t,'/YOHO');root.GetPrim().GetReferences().AddReference(str(a.source),str(s.GetDefaultPrim().GetPath()));t.SetDefaultPrim(root.GetPrim())
cache=UsdGeom.XformCache();before={str(x.GetPath()):cache.GetLocalToWorldTransform(x) for x in t.Traverse() if x.IsA(UsdGeom.Xformable)}
boxes=[];bodies=[];joints=[];scenes=[]
for prim in list(t.Traverse()):
 if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
  UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False);bodies.append(str(prim.GetPath()))
 if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False);joints.append(str(prim.GetPath()))
 if prim.IsA(UsdPhysics.Scene):prim.SetActive(False);scenes.append(str(prim.GetPath()))
 if prim.IsA(UsdGeom.Cube) and prim.HasAPI(UsdPhysics.CollisionAPI):
  size=float(UsdGeom.Cube(prim).GetSizeAttr().Get());h=size/2
  points=[(-h,-h,-h),(h,-h,-h),(h,h,-h),(-h,h,-h),(-h,-h,h),(h,-h,h),(h,h,h),(-h,h,h)]
  faces=[0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
  prim.SetTypeName('Mesh');mesh=UsdGeom.Mesh(prim);mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr([4]*6);mesh.CreateFaceVertexIndicesAttr(faces);mesh.CreateExtentAttr([Gf.Vec3f(-h),Gf.Vec3f(h)]);mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateOrientationAttr('rightHanded');UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none');boxes.append(str(prim.GetPath()))
cache.Clear()
assert all(cache.GetLocalToWorldTransform(t.GetPrimAtPath(path))==value for path,value in before.items())
t.GetRootLayer().Save();assert all(rec(row['path'])==row for row in sources)
report={'schema':'crowdsim.collected_YOHO_static_fixture.v1','source':rec(a.source),'source_layers':sources,'fixture':rec(a.output),'fixed_initial_fixture_bodies':bodies,'disabled_fixture_hinges':joints,'disabled_source_physics_scenes':scenes,'exact_cube_mesh_equivalents':boxes,'source_geometry_and_transforms_preserved':True,'source_nonfloor_collisions_removed':False,'fixture_articulation_enabled':False,'original_sources_unchanged':True,'floor_owner':'explicit finite reference ground, declared separately by map preflight','author_scene':False}
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'prepared','bodies_fixed':len(bodies),'box_mesh_equivalents':len(boxes),'fixture':str(a.output)}))
