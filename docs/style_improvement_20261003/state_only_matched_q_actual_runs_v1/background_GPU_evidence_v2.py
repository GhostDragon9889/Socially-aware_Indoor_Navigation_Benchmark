"""Observe real NVIDIA compute processes; permit only the known ToDesk video session."""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BINDING = HERE / 'background_v2_source_binding.json'
QUERY = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid',
         '--format=csv,noheader,nounits']
VIDEO_PID = 135883
VIDEO_EXE = '/opt/todesk/bin/ToDesk_Session'
VIDEO_ARGV = [VIDEO_EXE, '--isVideoSession=true']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'Regular v2 source required')
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def decode_query(raw):
    rows = []
    for columns in csv.reader(io.StringIO(raw)):
        if not columns or not any(x.strip() for x in columns):
            continue
        require(len(columns) == 4, 'Exact real NVIDIA pid/name/memory/UUID columns required')
        pid, name, memory, uuid = (x.strip() for x in columns)
        require(pid.isdecimal() and memory.isdecimal(), 'Numeric NVIDIA PID/memory required')
        row = {'pid': int(pid), 'process_name': name, 'used_memory_MiB': int(memory), 'gpu_uuid': uuid}
        require(row['pid'] > 0 and uuid.startswith('GPU-'), 'Actual positive PID and GPU UUID required')
        rows.append(row)
    require(len({(r['pid'], r['gpu_uuid']) for r in rows}) == len(rows), 'Duplicate NVIDIA rows rejected')
    return rows


def validate_observation(evidence):
    require(evidence['schema'] == 'state_only_matched_q.actual_background_GPU_observation.v2'
            and evidence['query_command'] == QUERY and type(evidence['query_exit_code']) is int
            and evidence['query_exit_code'] == 0 and bool(evidence['observed_utc']),
            'Real successful NVIDIA query evidence required')
    parsed = decode_query(evidence['query_stdout'])
    rows = evidence['rows']
    require(parsed == [{k: row[k] for k in ('pid', 'process_name', 'used_memory_MiB', 'gpu_uuid')} for row in rows],
            'Saved NVIDIA stdout and process rows disagree')
    require(len(rows) <= 1, 'Only the single observed ToDesk video task may coexist')
    for row in rows:
        proc = row['proc']
        argv = bytes.fromhex(proc['cmdline_hex']).rstrip(b'\0').split(b'\0')
        argv = [x.decode('utf-8', errors='strict') for x in argv]
        stat_fields = proc['stat_raw'].rsplit(') ', 1)[1].split()
        require(row['pid'] == VIDEO_PID and row['process_name'] == VIDEO_EXE
                and proc['exe'] == VIDEO_EXE and proc['cmdline'] == argv == VIDEO_ARGV
                and type(proc['ppid']) is int and proc['ppid'] == int(stat_fields[1]) == 1
                and proc['start_time_ticks'] == stat_fields[19]
                and proc['stat_raw'].split(' ', 1)[0] == str(VIDEO_PID),
                'Only exact known PPID1 ToDesk_Session --isVideoSession=true is allowed')
    require(evidence['no_parallel_training_GPU_at_launch'] is True
            and evidence['no_parallel_GPU_at_launch'] is (not rows),
            'GPU-idle and no-training claims must distinguish the observed video task')
    return not rows


def observe_background_gpu():
    # Do not replace nvidia-smi output or manipulate PATH. Failure is fail-closed.
    completed = subprocess.run(QUERY, capture_output=True, text=True, check=False, timeout=15)
    require(completed.returncode == 0, 'Real NVIDIA query failed: ' + completed.stderr.strip())
    rows = decode_query(completed.stdout)
    for row in rows:
        require(row['pid'] == VIDEO_PID and row['process_name'] == VIDEO_EXE,
                'Another GPU compute process is running; only known ToDesk video may coexist')
        path = Path('/proc') / str(row['pid'])
        stat_before = (path / 'stat').read_text()
        cmdline = (path / 'cmdline').read_bytes()
        exe = str((path / 'exe').resolve(strict=True))
        stat_after = (path / 'stat').read_text()
        before_fields = stat_before.rsplit(') ', 1)[1].split()
        after_fields = stat_after.rsplit(') ', 1)[1].split()
        require(before_fields[19] == after_fields[19], 'PID identity changed during GPU observation')
        row['proc'] = {'exe': exe, 'cmdline_hex': cmdline.hex(),
                       'cmdline': [x.decode('utf-8', errors='strict') for x in cmdline.rstrip(b'\0').split(b'\0')],
                       'ppid': int(after_fields[1]), 'start_time_ticks': after_fields[19], 'stat_raw': stat_after}
    evidence = {'schema': 'state_only_matched_q.actual_background_GPU_observation.v2',
                'observed_utc': datetime.datetime.now(datetime.UTC).isoformat(),
                'query_command': QUERY, 'query_exit_code': completed.returncode,
                'query_stdout': completed.stdout, 'query_stderr': completed.stderr,
                'rows': rows, 'no_parallel_training_GPU_at_launch': True,
                'no_parallel_GPU_at_launch': not rows,
                'atomic_GPU_lock_claimed': False, 'background_process_terminated': False}
    validate_observation(evidence)
    return evidence


def validate_launch_evidence(launch, source_record, check):
    binding = json.loads(BINDING.read_text())
    for row in binding['new_files']:
        require(record(row['path']) == row and not Path(row['path']).stat().st_mode & 0o222,
                'Readonly v2 source changed')
    evidence = launch['background_GPU_evidence']
    idle = validate_observation(evidence)
    owner = binding['owner']
    check('actual_owner_background_GPU_evidence_and_driver_identity',
          launch['driver_sha256'] == owner['sha256'] and source_record(owner['path']) == owner
          and launch['no_parallel_training_GPU_at_launch'] is True
          and launch['no_parallel_GPU_at_launch'] is idle
          and launch['background_GPU_evidence_source'] == binding['background_helper'])
    return evidence
