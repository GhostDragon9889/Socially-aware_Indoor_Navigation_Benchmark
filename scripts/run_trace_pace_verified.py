#!/usr/bin/env python3
"""Run TRACE/PACER with a real parent wait, source digests and immutable receipts.

Use the Isaac SDK Python interpreter. Output directory must be fresh.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_name(output.name + "_execution.json")
    log_path = output.with_name(output.name + ".log")
    inputs_path = output.with_name(output.name + "_inputs")
    source_path = output.with_name(output.name + "_sources")
    if any(p.exists() for p in (output, receipt, log_path, inputs_path, source_path)):
        parser.error("Fresh output, receipt and log required")
    extra = args.child_args[1:] if args.child_args[:1] == ["--"] else args.child_args
    if any(arg == "--output" or arg.startswith("--output=") for arg in extra):
        parser.error("Child output must be owned by this parent")
    inputs_path.mkdir()
    input_record = {}
    for flag, default, filename in (
        (
            "--config",
            root / "configs/diffusion_amp/trace_pace_reliable.json",
            "config.json",
        ),
        ("--scene", None, "scene.yaml"),
    ):
        occurrences = [
            i
            for i, arg in enumerate(extra)
            if arg == flag or arg.startswith(flag + "=")
        ]
        if len(occurrences) > 1:
            parser.error("Repeated input flag is not supported: " + flag)
        original = default
        if occurrences:
            index = occurrences[0]
            arg = extra[index]
            if arg == flag:
                if index + 1 >= len(extra):
                    parser.error("Missing value: " + flag)
                original = Path(extra[index + 1])
            else:
                original = Path(arg.split("=", 1)[1])
        if original is None:
            continue
        original = original if original.is_absolute() else root / original
        snapshot = inputs_path / filename
        shutil.copyfile(original, snapshot)
        input_record[flag] = {
            "original": str(original),
            "snapshot": str(snapshot),
            "sha256": digest(snapshot),
        }
        if not occurrences:
            extra.extend([flag, str(snapshot)])
        elif extra[occurrences[0]] == flag:
            extra[occurrences[0] + 1] = str(snapshot)
        else:
            extra[occurrences[0]] = flag + "=" + str(snapshot)
    sources = {root / "scripts/run_trace_pace.py", Path(__file__).resolve(),
               root / "src/protomotions/repository.py"}
    sources.update((root / "src/social_navigation").glob("*.py"))
    sources.update((root / "configs/reproduction").rglob("*.json"))
    sources.update((root / "CrowdSim/humanoid/runtime/sealed").glob("*.py"))
    if (root / "data/external_resources.json").is_file():
        sources.add(root / "data/external_resources.json")
    for relative in ("CrowdSim/humanoid/diffusion", "CrowdSim/humanoid/amp"):
        sources.update((root / relative).rglob("*.py"))
    sources.update(
        root / relative
        for relative in (
            "CrowdSim/humanoid/amp_world.py",
            "CrowdSim/humanoid/amass_corpus.py",
            "CrowdSim/humanoid/initial_reference_pool.py",
            "CrowdSim/humanoid/application/resources.py",
            "protomotions/utils/simulator_imports.py",
            "protomotions/robot_configs/smpl.py",
            "protomotions/components/scene_lib.py",
            "protomotions/components/terrains/terrain.py",
            "protomotions/components/terrains/config.py",
            "protomotions/simulator/base_simulator/simulator.py",
            "protomotions/simulator/base_simulator/simulator_state.py",
            "protomotions/simulator/base_simulator/config.py",
            "protomotions/simulator/isaaclab/simulator.py",
            "protomotions/simulator/isaaclab/config.py",
            "protomotions/simulator/isaaclab/utils/usd_only_scene.py",
            "CrowdSim/humanoid/reconstruction_scene.py",
            "CrowdSim/world/planning.py",
            "CrowdSim/humanoid/runtime/keep_right_planning.py",
            "CrowdSim/humanoid/runtime/trace_pace.py",
            "CrowdSim/humanoid/runtime/trace_pace_scope.py",
            "CrowdSim/humanoid/runtime/trace_pace_lifecycle.py",
            "CrowdSim/humanoid/runtime/trace_pace_measured_lifecycle.py",
            "CrowdSim/humanoid/runtime/trace_pace_native_lifecycle.py",
            "CrowdSim/humanoid/runtime/native_grid_scene.py",
            "CrowdSim/humanoid/runtime/dynbench_scene.py",
            "CrowdSim/humanoid/runtime/native_map_v9.py",
            "CrowdSim/humanoid/runtime/social_evaluation_lifecycle.py",
            "CrowdSim/humanoid/runtime/social_navigation_lifecycle.py",
            "CrowdSim/humanoid/contact_monitor.py",
            "CrowdSim/humanoid/foot_support.py",
            "CrowdSim/humanoid/foot_capture.py",
            "CrowdSim/humanoid/ground_support_archive.py",
            "CrowdSim/humanoid/physics_views.py",
            "CrowdSim/humanoid/analysis/trace_pace_capture.py",
        )
    )
    sources.update((root / "CrowdSim/humanoid/runtime").glob("*.py"))
    sources.add(root / "CrowdSim/utils/map_metadata.py")
    source_hashes = {str(p.resolve()): digest(p) for p in sorted(sources)}
    source_path.mkdir()
    source_snapshot_layout = {}
    for source in sorted(sources):
        relative = (
            source.relative_to(root)
            if source.is_relative_to(root)
            else Path("execution/owner.py")
        )
        snapshot = source_path / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        if digest(snapshot) != source_hashes[str(source.resolve())]:
            raise ValueError("Source changed while snapshotting")
        source_snapshot_layout[str(source.resolve())] = str(relative)
    command = [
        sys.executable,
        str(root / "scripts/run_trace_pace.py"),
        "--output",
        str(output),
        *extra,
    ]
    record = {
        "schema": "crowdsim.trace_pace_actual_execution.v1",
        "status": "running",
        "command": command,
        "actual_returncode": None,
        "source_sha256": source_hashes,
        "input_snapshot": input_record,
        "source_snapshot_root": str(source_path),
        "source_snapshot_layout": source_snapshot_layout,
        "source_scope": "listed planner/AMP/runtime sources; resource hashes are in child result",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2)
    started = time.monotonic()
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    with log_path.open("xb") as log:
        child = subprocess.Popen(
            command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        record["pid"] = child.pid
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        code = child.wait()
    record.update(
        status="completed" if code == 0 else "failed_preserved",
        actual_returncode=code,
        actual_returncode_source="subprocess.Popen.wait",
        duration_seconds=time.monotonic() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        sources_unchanged=all(
            digest(Path(path)) == sha for path, sha in source_hashes.items()
        ),
        inputs_unchanged=all(
            digest(Path(item["snapshot"])) == item["sha256"]
            for item in input_record.values()
        ),
    )
    result_path = output / "result.json"
    if result_path.exists():
        record["result_sha256"] = digest(result_path)
    record["capture_sha256"] = {
        name: digest(output / name)
        for name in (
            "physical.npz",
            "physical_states.npz",
            "planning.json",
            "ground_support.npz",
            "contact_measurements.npz",
            "native_static_residual.npz",
        )
        if (output / name).exists()
    }
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "status",
                    "actual_returncode",
                    "actual_returncode_source",
                    "sources_unchanged",
                    "duration_seconds",
                )
            }
        )
    )
    return (
        0
        if code == 0 and record["sources_unchanged"] and record["inputs_unchanged"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
