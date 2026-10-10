"""Freeze composed collected USD while verifying all collider geometry unchanged."""
from pathlib import Path
import argparse,hashlib,json
from pxr import Usd,UsdGeom,UsdPhysics
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def record(path):
 path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
if a.output.exists():p.error('Fresh snapshot required')
s=Usd.Stage.Open(str(a.source));assert s and not s.GetCompositionErrors()
original=[record(l.realPath) for l in s.GetUsedLayers() if l.realPath]
def geometry_signature(stage):
 cache=UsdGeom.XformCache();rows=[]
 for prim in stage.Traverse():
  if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  row={'path':str(prim.GetPath()),'type':prim.GetTypeName(),'enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),'transform':[list(v) for v in cache.GetLocalToWorldTransform(prim)]}
  for key in ('points','faceVertexCounts','faceVertexIndices','extent','size','radius','height','physics:approximation'):
   attr=prim.GetAttribute(key)
   if attr:
    value=attr.Get();row[key]=hashlib.sha256(str(value).encode()).hexdigest()
  rows.append(row)
 return {'collider_count':len(rows),'sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
before=geometry_signature(s);flat=s.Flatten(False);a.output.parent.mkdir(parents=True,exist_ok=True);assert flat.Export(str(a.output))
frozen=Usd.Stage.Open(str(a.output));assert frozen and not frozen.GetCompositionErrors();after=geometry_signature(frozen)
assert before==after,'Flattening changed collider geometry/transforms'
assert all(record(r['path'])==r for r in original),'Source changed during snapshot'
# Flatten resolves the authored layer-relative file anchors. Verify every
# resolved default texture/asset remains identical in the frozen composition.
assets=0
for prim in s.Traverse():
 for attr in prim.GetAttributes():
  if str(attr.GetTypeName()) not in ('asset','asset[]'):continue
  value=attr.Get();other=frozen.GetAttributeAtPath(attr.GetPath()).Get()
  values=value if str(attr.GetTypeName())=='asset[]' else [value]
  others=other if str(attr.GetTypeName())=='asset[]' else [other]
  if values is None:continue
  assert len(values)==len(others)
  for left,right in zip(values,others):
   if left and left.resolvedPath:
    assert left.resolvedPath==right.resolvedPath,(str(attr.GetPath()),left,right)
    assets+=1
report={'schema':'crowdsim.collected_scene_frozen_snapshot.v1','source_layers':original,'snapshot':record(a.output),'geometry_before':before,'geometry_after':after,'resolved_asset_attributes_verified':assets,'original_sources_unchanged':True,'composition_flattened':True,'geometry_or_physics_edited':False}
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'frozen','colliders':before['collider_count'],'asset_attributes':assets,'snapshot':str(a.output)}))
