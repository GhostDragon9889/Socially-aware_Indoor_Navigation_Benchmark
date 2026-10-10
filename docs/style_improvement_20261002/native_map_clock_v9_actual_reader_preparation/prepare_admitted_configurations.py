"""CPU-only 3..15 actor configurations, after exact actual V9 scene admission."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from CrowdSim.humanoid.application.native_scene_preparation import prepare_configuration, publish_configuration
from CrowdSim.humanoid.runtime.dynbench_scene import file_record

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report',type=Path,required=True)
    parser.add_argument('--actors',type=int,nargs='+',default=list(range(3,16)))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or any(type(n) is not int or n<3 or n>15 for n in args.actors) or len(set(args.actors))!=len(args.actors):
        parser.error('Require a new output and unique counts3..15; no fallback/overwrite')
    report=json.loads(args.audit_report.read_text())
    if report.get('native_v9_execution_observed') is not True or report['registration']['all_required_candidate_bytes_installed'] is not True:
        parser.error('Prepared registrations/synthetic fixtures cannot generate dependent actual configurations')
    admitted=[row for row in report['scenes'] if row.get('status')=='actual_native_v9_resource_admitted_by_CPU_reader']
    if not admitted:
        parser.error('No scene passed actual complete resource admission')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in admitted:
        if file_record(row['actual_result']['path'])!=row['actual_result'] or file_record(row['preflight']['path'])!=row['preflight']:
            raise ValueError('Actual audited source bytes changed')
        for n in args.actors:
            item={'scene':row['scene'],'actors':n,'physical_navigation_validated':False}
            try:
                config,proof=prepare_configuration(row['preflight']['path'],row['actual_result']['path'],actors=n,
                    probe_script_snapshot=report['registration']['producer_snapshot']['path'],clearance_m=.8,
                    spawn_clearance_m=3.,minimum_spawn_separation_m=1.6)
                config_path=args.output/(row['scene']+'_actors'+str(n)+'.yaml')
                proof_path=args.output/(row['scene']+'_actors'+str(n)+'_proof.json')
                published=publish_configuration(config,proof,config_path,proof_path)
                item.update(status='CPU_spawns_and_initial_wrap_Astar_passed',configuration=published['configuration'],proof=file_record(proof_path))
            except Exception as error:
                item.update(status='CPU_configuration_rejected_no_fallback',error={'type':type(error).__name__,'message':str(error)})
            rows.append(item)
    summary={'schema':'native_v9_admitted_scene_configuration_generation.v1','CPU_only':True,'SDK_started':False,'GPU_used':False,
             'actual_audit':file_record(args.audit_report),'results':rows,'physical_navigation_quality_claim':False,
             'all_requested_counts_prepared':all(row['status']=='CPU_spawns_and_initial_wrap_Astar_passed' for row in rows)}
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'prepared':sum(row['status']=='CPU_spawns_and_initial_wrap_Astar_passed' for row in rows),'rejected':sum(row['status']!='CPU_spawns_and_initial_wrap_Astar_passed' for row in rows)}))
    return 0 if summary['all_requested_counts_prepared'] else 1


if __name__=='__main__':
    raise SystemExit(main())
