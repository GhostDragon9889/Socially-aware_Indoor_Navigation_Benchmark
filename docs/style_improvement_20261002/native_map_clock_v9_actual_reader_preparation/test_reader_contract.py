"""Docs-only reader control contracts; fake rows are never native evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

DOC=Path(__file__).resolve().parent


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_reader_keeps_all_scene_statuses_and_rejects_failed_scene(tmp_path,monkeypatch):
    reader=load('v9_docs_reader_contract',DOC/'audit_actual_four_scenes.py')
    native=tmp_path/'native'
    for scene in reader.SCENES:
        path=native/scene
        path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps({'scene':scene,'status':'failed' if scene=='Office' else 'scene_resource_gate_passed','native_started':True,'CPU_contract_fixture_only':True}))
    monkeypatch.setattr(reader,'registration',lambda *_:{'all_required_candidate_bytes_installed':True,'CPU_contract_test_only':True})
    observed=[]
    def fake_audit(folder,_):
        scene=folder.name
        observed.append(scene)
        if scene=='Office':
            raise ValueError('Actual failure must remain rejected')
        return {'scene':scene,'status':'actual_native_v9_resource_admitted_by_CPU_reader','CPU_contract_test_only':True}
    monkeypatch.setattr(reader,'audit_scene',fake_audit)
    output=tmp_path/'report'
    assert reader.main(['--native-output',str(native),'--output',str(output)])==1
    report=json.loads((output/'report.json').read_text())
    assert observed==list(reader.SCENES)
    assert len(report['scenes'])==4
    assert report['failed_scenes']==['Office']
    assert report['scenes'][0]['original_native_status_preserved']=='failed'
    assert report['scenes'][0]['admitted'] is False
    assert report['status']=='actual_partial_v9_evidence_rejected'
    assert report['routes'] is None


def test_prepared_registration_cannot_generate_dependent_configs(tmp_path):
    utility=load('v9_preparation_contract',DOC/'prepare_admitted_configurations.py')
    report=tmp_path/'pending.json'
    report.write_text(json.dumps({'native_v9_execution_observed':False,'registration':{'all_required_candidate_bytes_installed':False},'scenes':[]}))
    output=tmp_path/'configurations'
    with pytest.raises(SystemExit) as error:
        utility.main(['--audit-report',str(report),'--output',str(output)])
    assert error.value.code==2
    assert not output.exists()


def test_no_admitted_scene_generates_no_configuration(tmp_path):
    utility=load('v9_no_admission_contract',DOC/'prepare_admitted_configurations.py')
    report=tmp_path/'partial.json'
    report.write_text(json.dumps({'native_v9_execution_observed':True,'registration':{'all_required_candidate_bytes_installed':True},'scenes':[{'status':'rejected_actual_or_missing_evidence'}]}))
    output=tmp_path/'configurations'
    with pytest.raises(SystemExit) as error:
        utility.main(['--audit-report',str(report),'--output',str(output)])
    assert error.value.code==2
    assert not output.exists()
