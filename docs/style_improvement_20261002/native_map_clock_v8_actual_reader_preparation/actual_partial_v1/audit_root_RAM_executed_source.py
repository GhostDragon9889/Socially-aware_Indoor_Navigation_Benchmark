from pathlib import Path
from copy import deepcopy
import hashlib,json
from pxr import Sdf, Usd
root=Path('/project/EXP/SangforAndSaut/ProtoMotions')
doc=root/'docs/style_improvement_20261002/native_map_clock_v8_actual_reader_preparation/actual_partial_v1'
doc.mkdir(exist_ok=False)
def rec(p):
 b=Path(p).read_bytes();return {'path':str(Path(p).resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
rows=[]
for scene in ['Office','Warehouse','Full_Warehouse','Hospital']:
 p=root/'output/scene_maps/native_reference_four_scenes_session_clock_v8'/scene/'result.json'
 data=json.loads(p.read_text());pref=json.loads(Path(data['preflight']['path']).read_text())
 source=Sdf.Layer.FindOrOpen(data['source']['path'])
 text=source.ExportToString();baseline=hashlib.sha256(text.encode()).hexdigest()
 info=data['runtime_time_range_override']
 matches=[]
 for start in [None,0.,-1.]:
  for end in [None,0.,60.,100.]:
   layer=Sdf.Layer.CreateAnonymous('range_contract.usda');assert layer.ImportFromString(text)
   if start is not None:layer.startTimeCode=start
   if end is not None:layer.endTimeCode=end
   digest=hashlib.sha256(layer.ExportToString().encode()).hexdigest()
   if digest==info['source_root_in_memory_sha256_at_query']:
    matches.append({'set_start_time_code':start,'set_end_time_code':end,'only_global_start_end_metadata_changed':True})
 layers=[rec(row['path']) for row in pref['layers']]
 assert layers==pref['layers']
 assert source.ExportToString()==text
 rows.append({'scene':scene,'actual_result':rec(p),'preflight':data['preflight'],'status':data['status'],'error':data.get('error'),'original_source':data['source'],'all_source_layer_disk_fullSHA_unchanged':True,'actual_runtime_range':deepcopy(info),'original_root_RAM_SHA_from_readonly_standalone_USD':baseline,'baseline_matches_actual_helper_entry_SHA':baseline==info['source_root_in_memory_sha256'],'query_SHA_matches_metadata_only_reconstruction':matches,'actual_initial_query_clock':data.get('actual_query_clock_initial'),'root_RAM_mutation_observed':not info['source_root_in_memory_unchanged_at_query'],'no_source_layer_saved_by_CPU_reader':True,'native_resource_promoted_by_this_diagnostic':False})
report={'schema':'dynbench.v8.actual_root_RAM_partial_CPU_audit.v1','CPU_only':True,'SDK_started':False,'GPU_used':False,'native_v8_run_executed_by_root':True,'all_four_native_passed':False,'rows':rows,'reader_source':rec('/tmp/audit_v8_root_RAM.py'),'limitations':['Source end-range-only SHA reconstruction can identify exact bytes but does not establish which asynchronous SDK operation authored them','Failed scenes did not retain actual_query_clock_initial, so their current timeline values cannot be asserted from range metadata','No failed scene promoted; source disk SHA equality does not remove strict source RAM ownership failure']}
(doc/'root_RAM_and_source_layers.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print({'scene':row['scene'],'baseline_matches_entry':row['baseline_matches_actual_helper_entry_SHA'],'query_matches':row['query_SHA_matches_metadata_only_reconstruction'],'status':row['status']})
