"""Run CPU contracts first, then retry CUDA skips in a fresh SDK process.

Some historical contracts require that CUDA has never been initialized. A
single mixed process cannot verify those contracts after a real CUDA test.
JUnit skips remain visible; an unavailable backend never counts as a pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def read_result(path: Path) -> dict:
    cases = list(ET.parse(path).iter('testcase'))
    return {
        'ordinary_cases': len(cases),
        'passed': sum(not any(c.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for c in cases),
        'failures': sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
        'skips': [{'classname': c.get('classname'), 'name': c.get('name'),
                   'reason': c.find('skipped').get('message', '')}
                  for c in cases if c.find('skipped') is not None],
    }


def cuda_nodes(root: Path, skipped: list[dict]) -> list[str]:
    nodes = []
    for case in skipped:
        if re.search(r'\b(?:CUDA|GPU)\b', case['reason'], flags=re.I) is None:
            continue
        parts = case['classname'].split('.')
        for end in range(len(parts), 0, -1):
            path = Path(*parts[:end]).with_suffix('.py')
            if (root / path).is_file():
                nodes.append('::'.join([path.as_posix(), *parts[end:], case['name']]))
                break
        else:
            raise RuntimeError('Cannot locate CUDA-skipped test: ' + case['classname'])
    if len(set(nodes)) != len(nodes):
        raise RuntimeError('Duplicate CUDA-skipped test identities')
    return nodes


def run(root: Path, output: Path, targets: list[str], *, cpu_only: bool = False) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    record = {'schema': 'social_navigation.isolated_tests.v1',
              'python': sys.executable, 'root': str(root), 'status': 'running',
              'phases': [], 'remaining_skips': []}
    receipt = output / 'result.json'

    def save():
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

    def phase(name, selected, env):
        xml = output / (name + '.xml')
        command = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-q',
                   '--junitxml=' + str(xml), *selected]
        item = {'name': name, 'command': command, 'actual_returncode': None,
                'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES'),
                'xml': str(xml), 'log': str(output / (name + '.log'))}
        record['phases'].append(item)
        save()
        started = time.monotonic()
        print('Running ' + name + '; log: ' + item['log'], flush=True)
        with Path(item['log']).open('x') as stream:
            child = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT)
        item.update(actual_returncode=child.returncode,
                    elapsed_s=time.monotonic() - started)
        if xml.is_file():
            item['result'] = read_result(xml)
        save()
        if child.returncode != 0 or 'result' not in item or item['result']['failures']:
            raise RuntimeError('Failed test phase preserved: ' + name)
        return item['result']

    try:
        cpu = phase('cpu', targets, dict(environment, CUDA_VISIBLE_DEVICES=''))
        nodes = cuda_nodes(root, cpu['skips'])
        record['cuda_retry_nodes'] = nodes
        remaining = cpu['skips']
        passed = cpu['passed']
        if nodes and not cpu_only:
            gpu = phase('cuda', nodes, environment)
            if gpu['ordinary_cases'] != len(nodes):
                raise RuntimeError('CUDA retry did not execute every selected test')
            remaining = [case for case in remaining
                         if re.search(r'\b(?:CUDA|GPU)\b', case['reason'], re.I) is None]
            remaining += gpu['skips']
            passed += gpu['passed']
        record.update(passed_ordinary_cases=passed, remaining_skips=remaining,
                      status='passed_with_gaps' if remaining else 'passed')
    except BaseException as error:
        record.update(status='failed_preserved', error=type(error).__name__ + ': ' + str(error))
        save()
        raise
    save()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('targets', nargs='*', help='Optional pytest paths or node IDs')
    args = parser.parse_args()
    result = run(ROOT, args.output_dir, args.targets, cpu_only=args.cpu_only)
    print(json.dumps({key: result[key] for key in ('status', 'passed_ordinary_cases', 'remaining_skips')},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
