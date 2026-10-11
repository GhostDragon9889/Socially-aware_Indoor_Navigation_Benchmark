"""Test process isolation and failure receipts, without claiming GPU physics."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_project_tests as runner


def test_xml_summary_keeps_skips_and_failures_separate(tmp_path):
    xml = tmp_path / 'result.xml'
    xml.write_text('<testsuites><testsuite tests="9"><testcase name="pass"/>'
                   '<testcase name="fail"><failure/></testcase>'
                   '<testcase name="error"><error/></testcase>'
                   '<testcase classname="test_x" name="skip"><skipped message="CUDA unavailable"/></testcase>'
                   '</testsuite></testsuites>')
    result = runner.read_result(xml)
    assert result['ordinary_cases'] == 4 and result['passed'] == 1 and result['failures'] == 2
    assert result['skips'][0]['reason'] == 'CUDA unavailable'


@pytest.mark.parametrize('classname,node', [('tests.test_gpu', 'tests/test_gpu.py::case[cuda]'),
                                           ('tests.test_gpu.TestGPU', 'tests/test_gpu.py::TestGPU::case[cuda]')])
def test_cuda_retry_maps_module_classes_and_parameter_identity(tmp_path, classname, node):
    (tmp_path / 'tests').mkdir(); (tmp_path / 'tests/test_gpu.py').touch()
    cases = [{'classname': classname, 'name': 'case[cuda]', 'reason': 'requires CUDA'},
             {'classname': 'other', 'name': 'legacy', 'reason': 'IsaacGym unavailable'}]
    assert runner.cuda_nodes(tmp_path, cases) == [node]


@pytest.mark.parametrize('duplicate', [False, True])
def test_unresolvable_or_duplicate_cuda_skip_is_not_silently_omitted(tmp_path, duplicate):
    case = {'classname': 'test_gpu', 'name': 'case', 'reason': 'GPU unavailable'}
    if duplicate: (tmp_path / 'test_gpu.py').touch()
    with pytest.raises(RuntimeError, match='Duplicate|Cannot locate'):
        runner.cuda_nodes(tmp_path, [case, case] if duplicate else [case])


def checkout(tmp_path, *, cpu_fails=False, gpu_fails=False):
    root = tmp_path / "fixture's 目录"; root.mkdir()
    script = root / 'test_contract.py'
    script.write_text('import os, pytest\n'
                      'def test_cpu_environment():\n'
                      ' assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
                      ' assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"\n'
                      f' assert {not cpu_fails!r}\n'
                      'def test_gpu_environment_contract():\n'
                      ' if not os.environ.get("CUDA_VISIBLE_DEVICES"): pytest.skip("CUDA process fixture")\n'
                      f' assert {not gpu_fails!r}\n')
    return root


@pytest.mark.parametrize('cpu_only', [False, True])
def test_fresh_processes_preserve_visibility_and_report_unverified_skips(tmp_path, monkeypatch, cpu_only):
    root = checkout(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'orchestration-fixture-only')
    result = runner.run(root, tmp_path / 'receipts', [], cpu_only=cpu_only)
    assert result['status'] == ('passed_with_gaps' if cpu_only else 'passed')
    assert result['passed_ordinary_cases'] == (1 if cpu_only else 2)
    assert len(result['phases']) == (1 if cpu_only else 2)
    assert result['phases'][0]['CUDA_VISIBLE_DEVICES'] == ''
    if not cpu_only:
        assert result['phases'][1]['CUDA_VISIBLE_DEVICES'] == 'orchestration-fixture-only'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'orchestration-fixture-only'
    with pytest.raises(FileExistsError): runner.run(root, tmp_path / 'receipts', [])


def test_no_gpu_environment_remains_an_explicit_gap(tmp_path, monkeypatch):
    root = checkout(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    result = runner.run(root, tmp_path / 'receipts', [])
    assert result['status'] == 'passed_with_gaps' and result['passed_ordinary_cases'] == 1
    assert len(result['remaining_skips']) == 1


@pytest.mark.parametrize('cpu_fails', [False, True])
def test_real_child_failure_retains_receipt_and_stops_pipeline(tmp_path, monkeypatch, cpu_fails):
    root = checkout(tmp_path, cpu_fails=cpu_fails, gpu_fails=not cpu_fails)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'orchestration-fixture-only')
    output = tmp_path / 'receipts'
    with pytest.raises(RuntimeError, match='Failed test phase preserved'):
        runner.run(root, output, [])
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'failed_preserved'
    assert len(result['phases']) == (1 if cpu_fails else 2)
    assert result['phases'][-1]['actual_returncode'] == 1
    assert Path(result['phases'][-1]['log']).is_file()


@pytest.mark.parametrize('missing_xml', [False, True])
def test_success_status_without_complete_gpu_results_is_rejected(tmp_path, monkeypatch, missing_xml):
    root = checkout(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'orchestration-fixture-only')
    actual_run = runner.subprocess.run
    def incomplete_child(command, **kwargs):
        if kwargs['env']['CUDA_VISIBLE_DEVICES'] == '':
            return actual_run(command, **kwargs)
        if not missing_xml:
            xml = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
            xml.write_text('<testsuites><testsuite/></testsuites>')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', incomplete_child)
    output = tmp_path / 'receipts'
    with pytest.raises(RuntimeError, match='did not execute|Failed test phase'):
        runner.run(root, output, [])
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'failed_preserved'
    assert result['phases'][-1]['actual_returncode'] == 0
