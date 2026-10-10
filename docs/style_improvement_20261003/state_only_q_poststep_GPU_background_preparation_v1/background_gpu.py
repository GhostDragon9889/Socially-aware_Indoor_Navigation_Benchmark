"""One-shot real NVIDIA observation; no training launcher or process termination.

Import and validate_observation are pure stdlib. observe_background_gpu runs only
when root explicitly calls it; sanitized evidence omits all port/cmdline values.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
from pathlib import Path
import re
import subprocess

SCHEMA = "state_only_q.poststep_actual_background_GPU_observation.v1"
QUERY = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
         "--format=csv,noheader,nounits"]
VIDEO_EXE = "/opt/todesk/bin/ToDesk_Session"
VIDEO_NAME = "ToDesk_Session"
EXE_LIMITATION = "proc_exe_permission_denied; NVIDIA process_name and argv0 agree; exe target not independently read"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, *, positive=False):
    return type(value) is int and (value > 0 if positive else value >= 0)


def decode_query(raw):
    require(type(raw) is str, "NVIDIA CSV text required")
    rows = []
    for fields in csv.reader(io.StringIO(raw)):
        if not fields or not any(value.strip() for value in fields):
            continue
        require(len(fields) == 4, "Exact NVIDIA pid/name/memory/UUID columns required")
        pid, name, memory, uuid = (value.strip() for value in fields)
        require(re.fullmatch(r"[1-9][0-9]*", pid) and int(pid) <= 2147483647
                and re.fullmatch(r"[0-9]+", memory), "Safe numeric NVIDIA PID/memory required")
        require(name == VIDEO_EXE, "Unrecognized GPU compute process; launch must wait")
        require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid),
                "Actual NVIDIA GPU UUID required")
        rows.append({"pid": int(pid), "process_name": name, "used_memory_MiB": int(memory), "gpu_uuid": uuid})
    require(len({(row["pid"], row["gpu_uuid"]) for row in rows}) == len(rows), "Duplicate NVIDIA rows rejected")
    return rows


def process_snapshot(pid):
    require(integer(pid, positive=True) and pid <= 2147483647, "Safe process PID required")
    path = Path("/proc") / str(pid)
    stat = (path / "stat").read_text()
    prefix, separator, suffix = stat.rpartition(") ")
    require(bool(separator) and prefix.startswith(str(pid) + " ("), "Actual proc stat PID required")
    fields = suffix.split()
    status = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)
    name = status["Name"].strip()
    ppid = int(status["PPid"].strip())
    require(len(fields) >= 20 and name == prefix.split(" (", 1)[1] == VIDEO_NAME
            and ppid == int(fields[1]) == 1 and re.fullmatch(r"[0-9]+", fields[19])
            and int(fields[19]) > 0, "Only stable PPID1 ToDesk video identity allowed")
    return {"pid": pid, "name": name, "ppid": ppid, "start_time_ticks": int(fields[19])}


def validate_video_argv(raw):
    # Values stay local to this function; no raw argv or port digest is persisted.
    require(raw.endswith(b"\0"), "Actual null-terminated proc cmdline required")
    argv = [value.decode("utf-8", errors="strict") for value in raw[:-1].split(b"\0")]
    require(argv and argv[0] == VIDEO_EXE and len(argv) in (3, 4), "Exact ToDesk argv0/video/port shape required")
    arguments = argv[1:]
    require(arguments.count("--isVideoSession=true") == 1, "Explicit true video-session flag required")
    arguments.remove("--isVideoSession=true")
    if len(arguments) == 1 and arguments[0].startswith("--localPort="):
        port = arguments[0].split("=", 1)[1]
    else:
        require(len(arguments) == 2 and arguments[0] == "--localPort", "Only numeric localPort argument allowed")
        port = arguments[1]
    require(re.fullmatch(r"[0-9]+", port) and 1 <= int(port) <= 65535, "Valid numeric video localPort required")


def validate_observation(evidence):
    require(type(evidence) is dict and set(evidence) == {
        "schema", "observed_utc", "query_command", "query_exit_code", "query_stdout", "rows",
        "no_parallel_training_GPU_at_launch", "no_parallel_GPU_at_launch", "atomic_GPU_lock_claimed",
        "background_process_terminated", "sensitive_port_and_cmdline_values_persisted"}, "Exact sanitized observation fields required")
    require(evidence["schema"] == SCHEMA and evidence["query_command"] == QUERY
            and type(evidence["query_exit_code"]) is int and evidence["query_exit_code"] == 0,
            "Real successful NVIDIA query evidence required")
    require(type(evidence["observed_utc"]) is str
            and datetime.datetime.fromisoformat(evidence["observed_utc"]).utcoffset() == datetime.timedelta(0),
            "Actual UTC observation time required")
    rows = evidence["rows"]
    require(type(rows) is list, "Actual NVIDIA process rows required")
    require(decode_query(evidence["query_stdout"]) == [
        {key: row[key] for key in ("pid", "process_name", "used_memory_MiB", "gpu_uuid")} for row in rows],
        "Actual NVIDIA CSV and persisted rows differ")
    for row in rows:
        require(set(row) == {"pid", "process_name", "used_memory_MiB", "gpu_uuid", "proc"}, "Exact sanitized row fields required")
        require(integer(row["pid"], positive=True) and integer(row["used_memory_MiB"]), "Actual integer NVIDIA PID/memory required")
        proc = row["proc"]
        require(type(proc) is dict and set(proc) == {"before", "after", "argv0", "explicit_video_flag",
            "local_port_numeric_present", "local_port_value_not_recorded", "cmdline_stable_during_observation",
            "exe", "process_left_running"}, "Raw cmdline/port values must not be persisted")
        require(proc["before"] == proc["after"], "Process identity changed during observation")
        for snapshot in (proc["before"], proc["after"]):
            require(type(snapshot) is dict and set(snapshot) == {"pid", "name", "ppid", "start_time_ticks"}
                    and integer(snapshot["pid"], positive=True) and snapshot["pid"] == row["pid"]
                    and snapshot["name"] == VIDEO_NAME and type(snapshot["ppid"]) is int and snapshot["ppid"] == 1
                    and integer(snapshot["start_time_ticks"], positive=True), "Stable actual ToDesk proc identity required")
        require(proc["argv0"] == VIDEO_EXE and all(proc[key] is True for key in (
            "explicit_video_flag", "local_port_numeric_present", "local_port_value_not_recorded",
            "cmdline_stable_during_observation", "process_left_running")), "Exact sanitized video/port proof required")
        exe = proc["exe"]
        require(exe == {"status": "verified", "target": VIDEO_EXE}
                or exe == {"status": "read_denied", "limitation": EXE_LIMITATION}, "Executable proof or explicit read-denied limitation required")
    require(evidence["no_parallel_training_GPU_at_launch"] is True
            and evidence["no_parallel_GPU_at_launch"] is (not rows)
            and evidence["atomic_GPU_lock_claimed"] is False
            and evidence["background_process_terminated"] is False
            and evidence["sensitive_port_and_cmdline_values_persisted"] is False,
            "GPU-idle/video coexistence and process-preservation claims must remain truthful")
    return not rows


def observe_background_gpu():
    # Root executes this once immediately before a future serial launch. No PATH
    # override, substituted query result, signal, launcher or historical PID list.
    query = subprocess.run(QUERY, capture_output=True, text=True, check=False, timeout=15)
    require(query.returncode == 0, "Actual NVIDIA query failed; launch must wait")
    rows = decode_query(query.stdout)
    for row in rows:
        pid = row["pid"]
        path = Path("/proc") / str(pid)
        before = process_snapshot(pid)
        cmdline = (path / "cmdline").read_bytes()
        validate_video_argv(cmdline)
        try:
            target = os.readlink(path / "exe")
            require(target == VIDEO_EXE, "Actual executable differs; launch must wait")
            exe = {"status": "verified", "target": target}
        except PermissionError:
            exe = {"status": "read_denied", "limitation": EXE_LIMITATION}
        require(cmdline == (path / "cmdline").read_bytes(), "Process argv changed during observation")
        after = process_snapshot(pid)
        require(before == after, "Process PID/start/PPID/Name changed during observation")
        row["proc"] = {"before": before, "after": after, "argv0": VIDEO_EXE,
                       "explicit_video_flag": True, "local_port_numeric_present": True,
                       "local_port_value_not_recorded": True, "cmdline_stable_during_observation": True,
                       "exe": exe, "process_left_running": True}
    evidence = {"schema": SCHEMA, "observed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "query_command": QUERY, "query_exit_code": query.returncode, "query_stdout": query.stdout,
                "rows": rows, "no_parallel_training_GPU_at_launch": True,
                "no_parallel_GPU_at_launch": not rows, "atomic_GPU_lock_claimed": False,
                "background_process_terminated": False, "sensitive_port_and_cmdline_values_persisted": False}
    validate_observation(evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = observe_background_gpu()
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "no_parallel_training_GPU_at_launch": True,
                      "no_parallel_GPU_at_launch": evidence["no_parallel_GPU_at_launch"]}))


if __name__ == "__main__":
    main()
