"""Metadata-only producer checkpoint identity adapter around immutable raw math."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CORE=ROOT/'docs/style_improvement_20261003/state_prior_pacer_reward_native_reader_preparation_v1'
CORE_FREEZE_SHA='bbc81cc438188ab5b9f23c31a229d51c9cf043300874d304f8cc01c6d186d6f3'


def require(value,message):
    if not value:raise ValueError(message)


def record(path):
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def load_core():
    require(record(CORE/'source_freeze_v1.json')['sha256']==CORE_FREEZE_SHA,'Immutable prepared v1 core required')
    freeze=read(CORE/'source_freeze_v1.json')
    for row in freeze['files']+[freeze['archive']]:
        require(record(row['path'])==row and not Path(row['path']).stat().st_mode&0o222,'Immutable source changed')
    helper_path=HERE/'native_evidence_checkpoint_v2.py'
    source=helper_path.read_text()
    normalized=source.replace('ROOT=Path(__file__).resolve().parents[3]\nHERE=ROOT/"docs/style_improvement_20261003/state_prior_pacer_reward_native_reader_preparation_v1"','HERE=Path(__file__).resolve().parent\nROOT=HERE.parents[2]')
    normalized=normalized.replace('all(request["base_checkpoint"].get(key)==checkpoint[key] for key in ("path", "sha256"))','all(request["base_checkpoint"].get(key)==value for key,value in checkpoint.items())')
    original=(CORE/'native_evidence.py').read_text()
    require(normalized==original and ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False),'Only HERE relocation and producer path/SHA comparison may change')
    specification=importlib.util.spec_from_file_location('native_evidence',helper_path)
    helper=importlib.util.module_from_spec(specification);sys.modules['native_evidence']=helper;specification.loader.exec_module(helper)
    specification=importlib.util.spec_from_file_location('immutable_pacer_reward_native_raw_math',CORE/'audit_state_prior_native.py')
    core=importlib.util.module_from_spec(specification);specification.loader.exec_module(core)
    require(core.HERE==CORE and Path(sys.modules['native_evidence'].__file__).resolve()==helper_path,'No metadata helper fallback')
    return core


def registration(actors):
    core=load_core()
    registered=read(CORE/f'native{actors}_registration.json')
    require(core.prepare_evidence(actors)==registered['evidence'],'Pending source/cases evidence changed')
    _,proof=core.readers(actors)
    require(proof==registered['adapter_proof'],'Original JSON mathematics proof must remain exact')
    return {'schema':'pacer_reward.native_checkpoint_metadata_revision_registration.v2','status':'prepared_actual_native_raw_pending',
            'actors':actors,'created_utc':datetime.now(timezone.utc).isoformat(),'CPU_only':True,'GPU_used':False,'SDK_started':False,
            'native_execution_audited':False,'revision_entry':record(__file__),'metadata_helper':record(HERE/'native_evidence_checkpoint_v2.py'),
            'immutable_core_freeze':record(CORE/'source_freeze_v1.json'),'immutable_core_registration':record(CORE/f'native{actors}_registration.json'),
            'original_math_executor':record(CORE/'audit_state_prior_native.py'),'producer_identity_fields':['path','sha256'],
            'actual_sealed_checkpoint_bytes_still_independently_hashed':True,'raw_math_changed':False,
            'future_checkpoint_sha256':None,'first_metadata_ABI_issue':record(HERE/'first_metadata_ABI_issue.json')}


def audit(actors,registered):
    require(registered['schema']=='pacer_reward.native_checkpoint_metadata_revision_registration.v2'
            and registered['actors']==actors and registered['native_execution_audited'] is False
            and registered['revision_entry']==record(__file__)
            and registered['metadata_helper']==record(HERE/'native_evidence_checkpoint_v2.py')
            and registered['immutable_core_freeze']==record(CORE/'source_freeze_v1.json')
            and registered['immutable_core_registration']==record(CORE/f'native{actors}_registration.json'),
            'Exact frozen revision identity required')
    core=load_core()
    report=core.audit(actors,read(CORE/f'native{actors}_registration.json'))
    require(report['execution_reader']==registered['original_math_executor'],'Original raw mathematics executor identity differs')
    report['execution_entry']=record(__file__)
    report['metadata_admission_helper']=registered['metadata_helper']
    report['metadata_admission_revision_registration']=registered
    report['immutable_raw_math_core_freeze']=registered['immutable_core_freeze']
    report['producer_checkpoint_identity_path_SHA_exact']=True
    report['sealed_checkpoint_bytes_hash_readonly_verified_separately']=True
    report['compiled_raw_math_changed']=False
    report['actual_source_admission']['metadata_admission_helper']=registered['metadata_helper']
    report['actual_source_admission']['metadata_admission_revision_entry']=registered['revision_entry']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actors',type=int,choices=(4,15),required=True)
    parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve existing evidence')
    require(args.register_only or args.registration is not None,'Frozen revision registration required')
    report=registration(args.actors) if args.register_only else audit(args.actors,read(args.registration))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'actual_native_audited':report.get('native_200s_artifacts_independently_audited',False),'future_checkpoint_read':False if args.register_only else None}))


if __name__=='__main__':
    main()
