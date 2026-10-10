#!/usr/bin/env python3
"""Wait for the actual episode owner, then run the preregistered CPU comparison.

No training/SDK process is launched or restarted. While waiting, require the
recorded live owner's exact command and process start identity. The final
comparison must still admit every actual completed training/evaluation exit.
Source: https://docs.python.org/3.12/library/subprocess.html#subprocess.Popen.wait
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_shared_social_episode_evidence as reader  # noqa: E402


def process_identity(pid):
    """Linux /proc start ticks prevent treating a reused PID as this owner."""
    directory = Path("/proc") / str(pid)
    try:
        command = directory.joinpath("cmdline").read_bytes().split(b"\0")
        stat = directory.joinpath("stat").read_text()
    except FileNotFoundError:
        return None
    fields = stat[stat.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": fields[19],
            "command": [item.decode() for item in command if item]}


def now():
    return datetime.now(timezone.utc).isoformat()


def run(registration_path):
    registered, identity = reader.admit_registration(registration_path)
    here = registration_path.parent
    if any((here / name).exists() for name in ("comparison_owner_state.json", "comparison_completion.json")):
        raise FileExistsError("Comparison owner already started")
    original = reader.kernel.read(reader.PLAN.parent / "owner_state.json")
    pid = original["owner_pid"]
    expected_command = [sys.executable, str(Path(reader.owner.__file__).resolve()), "--plan", str(reader.PLAN)]
    live = process_identity(pid)
    if not reader.COMPLETION.exists():
        reader.require(live is not None and live["command"] == expected_command,
                       "Require the specific actual live episode pipeline owner; never restart it")
    state = {"schema": "shared_social.episode_comparison_owner_state.v1", "status": "waiting_for_actual_episode_completion",
             "owner_pid": os.getpid(), "started_at_utc": now(), "registration": identity,
             "observed_episode_owner": live, "SDK_started": False, "training_started": False,
             "paper_effect_achieved": False, "default_model_promoted": False}
    anchors = [identity, registered["reader"], registered["candidate_owner_plan"],
               *registered["metric_sources"].values(), reader.kernel.record(__file__)]

    def save():
        state["updated_at_utc"] = now()
        reader.owner.io.write_json(here / "comparison_owner_state.json", state)

    def source_check():
        for row in anchors:
            reader.require(reader.kernel.record(row["path"]) == row, "Comparison observer/source changed while waiting")

    save()
    try:
        while not reader.COMPLETION.exists():
            source_check()
            observed = process_identity(pid)
            if reader.COMPLETION.exists():
                break
            reader.require(observed == live and observed is not None,
                           "The actual episode owner is absent or changed without a completion; no restart")
            state["live_episode_owner_last_confirmed_at_utc"] = now()
            save()
            time.sleep(15)
        source_check()
        reader.admit_candidate(registered)
        command = [sys.executable, str(Path(reader.__file__).resolve()), "--registration", str(registration_path)]
        state.update(status="CPU_comparison_running", command=command)
        save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        with (here / "CPU_comparison.log").open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            state["CPU_comparison_pid"] = child.pid
            save()
            code = child.wait()
        state.update(actual_CPU_comparison_exit_code=code, exit_observation="subprocess.Popen.wait")
        reader.require(type(code) is int and code == 0, "Actual preregistered CPU comparison failed")
        result = reader.kernel.read(reader.OUTPUT)
        reader.require(result["status"] == "passed_comparison" and result["registration"] == identity
                       and result["paper_effect_achieved"] is False, "Comparison exit/result/provenance disagree")
        source_check()
        state.update(status="completed_independent_comparison_quality_unclassified", result=reader.kernel.record(reader.OUTPUT))
    except BaseException as error:
        state.update(status="failed_preserved", error={"type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()})
    state["finished_at_utc"] = now()
    save()
    reader.owner.io.write_json(here / "comparison_completion.json", state, readonly=True)
    return 0 if state["status"] == "completed_independent_comparison_quality_unclassified" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--registration", type=Path, default=reader.REGISTRATION)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    path = args.registration.resolve()
    reader.admit_registration(path)
    if args.detach:
        if any((path.parent / name).exists() for name in ("comparison_owner.log", "comparison_owner_state.json", "comparison_completion.json")):
            raise FileExistsError("Comparison observer already launched")
        with (path.parent / "comparison_owner.log").open("xb") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--registration", str(path)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        print(json.dumps({"status": "episode_comparison_observer_launched", "owner_pid": child.pid,
                          "SDK_started": False, "actual_comparison_exit_observed": False}))
        return 0
    return run(path)


if __name__ == "__main__":
    raise SystemExit(main())
