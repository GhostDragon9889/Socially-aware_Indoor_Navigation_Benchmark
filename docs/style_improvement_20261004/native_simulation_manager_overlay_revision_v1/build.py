#!/usr/bin/env python3
"""Isolated four-CPP diagnostic build. No plugin/SDK/Python binding is loaded."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REVIEW_SHA = 'a6901351e97ce092cd2b328265c2e7b14225188c64892879d8bb87064fd59f5d'
PLAN_SHA = 'ad85c92dc51bc6c0e0e2bc32139867b79f795ff9663eb286fa7c887b903c7dc8'
BASENAME = 'libisaacsim.core.simulation_manager.plugin.so'


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    stat = path.stat()
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'bytes': stat.st_size, 'sha256': digest.hexdigest()}


def exact(expected: dict) -> dict:
    actual = record(Path(expected['path']))
    if (actual['bytes'], actual['sha256']) != (expected['bytes'], expected['sha256']):
        raise ValueError(f"Changed input: {expected['path']}")
    return actual


def run(argv: list[str], output: Path, label: str, report: dict) -> str:
    entry = {'argv': argv, 'returncode': None}
    report['commands'].append(entry)
    with (output / (label + '.log')).open('xb') as log:
        process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    entry['returncode'] = process.returncode
    entry['log'] = record(output / (label + '.log'))
    if process.returncode != 0:
        raise RuntimeError(f'{label} actually failed with exit {process.returncode}')
    return (output / (label + '.log')).read_text(errors='replace')


def validate_output(path: Path) -> Path:
    destination = path.resolve()
    allowed = [Path('/tmp').resolve(), (REPO / 'output').resolve()]
    if not any(base in destination.parents for base in allowed):
        raise ValueError('Output must be a new child directory under /tmp or repo/output')
    if destination.exists():
        raise FileExistsError('Output must not exist; failed builds must be retained')
    return destination


def build(output: Path, report: dict) -> None:
    review_record = record(HERE / 'source_review.json')
    if review_record['sha256'] != REVIEW_SHA:
        raise ValueError('Reviewed source contract changed')
    review = json.loads((HERE / 'source_review.json').read_text())
    plan_record = record(HERE / 'standalone_build_arguments.json')
    if plan_record['sha256'] != PLAN_SHA:
        raise ValueError('Reviewed standalone argument plan changed')
    plan = json.loads((HERE / 'standalone_build_arguments.json').read_text())
    report['execution_source'] = record(Path(__file__).resolve())
    report['argument_plan'] = record(HERE / 'standalone_build_arguments.json')
    report['source_contract'] = review_record
    before = [review_record, report['execution_source'], report['argument_plan']]
    for expected in review['source_records'].values():
        before.append(exact(expected))
    for pair in review['header_comparison'].values():
        before.append(exact(pair['installed']))
    for key in ('installed_stage_update_header', 'installed_extension_config', 'installed_plugin'):
        before.append(exact(review[key]))
    before.append(exact(review['patch']))
    for library in plan['link_libraries']:
        before.append(record(Path(library)))
    report['immutable_inputs_before'] = before

    compiler = plan['compiler']
    version = run([compiler, '--version'], output, 'compiler_version', report)
    if not version.splitlines() or '11.' not in version.splitlines()[0]:
        raise ValueError('Expected source-compatible GCC11 compiler')
    report['compiler_version_output'] = version
    original = Path(review['source_records']['PluginInterface.cpp']['path']).read_bytes()
    derived = original.decode()
    for edit in review['derivation']['edits']:
        if derived.count(edit['original']) != 1:
            raise ValueError('Source substitution count changed')
        derived = derived.replace(edit['original'], edit['replacement'], 1)
    if hashlib.sha256(derived.encode()).hexdigest() != review['derivation']['derived_sha256']:
        raise ValueError('Derived source differs from reviewed three edits')
    reverse = derived
    for edit in reversed(review['derivation']['edits']):
        if reverse.count(edit['replacement']) != 1:
            raise ValueError('Inverse substitution count changed')
        reverse = reverse.replace(edit['replacement'], edit['original'], 1)
    if reverse.encode() != original:
        raise ValueError('Full source inverse failed')
    local_cpp = output / 'PluginInterface.cpp'
    local_cpp.write_text(derived)
    report['derived_cpp'] = record(local_cpp)
    before.append(report['derived_cpp'])
    sources = [local_cpp] + [Path(review['source_records'][name]['path'])
                            for name in plan['translation_units'][1:]]
    objects = []
    headers = {}
    flags = plan['common_compile_arguments']
    # Preprocess dependency lists before compilation; then hash every used header.
    # This is compiler preprocessing, not an SDK/native-library import.
    for source in sources:
        depfile = output / (source.stem + '.d')
        run([compiler, *flags, '-M', '-MT', 'probe', '-MF', str(depfile), str(source)],
            output, source.stem + '_dependencies', report)
        deptext = depfile.read_text().replace('\\\n', ' ')
        if not deptext.startswith('probe:'):
            raise ValueError('Unexpected compiler dependency format')
        for token in shlex.split(deptext[len('probe:'):]):
            dependency = Path(token).resolve()
            headers[str(dependency)] = record(dependency)
    report['actual_used_headers_before'] = list(headers.values())
    for source in sources:
        obj = output / (source.stem + '.o')
        run([compiler, *flags, '-c', str(source), '-o', str(obj)],
            output, source.stem + '_compile', report)
        objects.append(str(obj))
    binary = output / BASENAME
    library_dirs = sorted({str(Path(p).parent) for p in plan['link_libraries']})
    linker_paths = [item for directory in library_dirs
                    for item in ('-Wl,-rpath-link,' + directory, '-Wl,-rpath,' + directory)]
    run([compiler, *plan['link_arguments'], '-Wl,-rpath,$ORIGIN',
         *linker_paths, *objects, *plan['link_libraries'], '-o', str(binary)],
        output, 'link', report)
    elf = run(['/usr/bin/readelf', '-W', '-h', '-d', '-V', '-s', str(binary)],
              output, 'ELF_metadata', report)
    if 'ELF64' not in elf or 'Advanced Micro Devices X86-64' not in elf:
        raise ValueError('Unexpected ELF architecture')
    if '(SONAME)' not in elf or f'[{BASENAME}]' not in elf:
        raise ValueError('ELF SONAME differs from installed plugin name')
    for expected in [*before, *headers.values()]:
        exact(expected)
    report['immutable_inputs_after_equal'] = True
    report['built_binary'] = record(binary)
    report['actual_build_succeeded'] = True
    report['ABI_proven_by_actual_load'] = False
    report['overlay_loaded'] = False
    report['SDK_or_plugin_executed'] = False
    report['process_shutdown_failure_resolved'] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New exclusive directory under /tmp or repo/output')
    args = parser.parse_args()
    output = validate_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'local.isaacsim.native_plugin_standalone_build.v1',
              'target': 'isaacsim.core.simulation_manager.plugin', 'status': 'failed',
              'actual_build_succeeded': False, 'commands': [], 'error': None,
              'SDK_or_plugin_executed': False, 'actual_loaded_overlay_identity': None,
              'actual_shutdown_or_exit_result': None}
    exitcode = 1
    try:
        build(output, report)
        report['status'] = 'compiled_and_ELF_checked_not_loaded'
        exitcode = 0
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error),
                           'traceback': traceback.format_exc()}
    finally:
        with (output / 'build_report.json').open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    return exitcode


if __name__ == '__main__':
    raise SystemExit(main())
