"""Summarize closed OLD/S1, NEW/S1 and NEW/S10; reuse original idle math."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
IDLE = ROOT / "docs/style_improvement_20261002/summarize_navigation_style.py"
PAIR = ROOT / "docs/style_improvement_20261003/default_newDP_physical_actual_runs_v1"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    specification = importlib.util.spec_from_file_location("samples10_original_idle_math", IDLE)
    original = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(original)
    assert original.record(IDLE)["sha256"] == "ee0d13c3f345cf11aa2981002c21b070bf53eb4b6129d86d4a6f851443ac56d7"
    selected = (
        ("OLD63e_S1", PAIR / "fixture_old_audit.json", PAIR / "fixture_old_CPU_actual_launch.json", PAIR / "fixture_old_actual_launch.json"),
        ("NEWb563_S1", PAIR / "fixture_new_audit.json", PAIR / "fixture_new_CPU_actual_launch.json", PAIR / "fixture_new_actual_launch.json"),
        ("NEWb563_S10", HERE / "fixture_new_audit.json", HERE / "fixture_new_CPU_actual_launch.json", HERE / "fixture_new_actual_launch.json"),
    )
    runs = {}
    for label, audit_path, cpu_path, gpu_path in selected:
        audit, cpu, gpu = read(audit_path), read(cpu_path), read(gpu_path)
        for owner in (cpu, gpu):
            assert owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
            assert owner["child_exit_observed"] is True and owner["driver_completed"] is True
            assert owner["binding_before"] == owner["binding_after"]
        assert cpu["audit_artifact"] == original.record(audit_path)
        assert audit["actual_source_admission"] == cpu["actual_capture_admission"]
        assert audit["all_failed_contact_idle_reset_samples_retained"] is True
        assert audit["style_score_used_for_admission"] is False
        directory = Path(audit["artifacts"]["result"]["path"]).parent
        artifact_rows = {row["path"]: row for row in gpu["artifacts"]}
        for role in ("result", "physical_states", "ground_support"):
            assert artifact_rows[audit["artifacts"][role]["path"]] == audit["artifacts"][role]
            assert not Path(audit["artifacts"][role]["path"]).stat().st_mode & 0o222
        result = read(directory / "result.json")
        assert audit["counts"]["controls"] == result["actual_control_steps"] == 6000
        assert audit["counts"]["actors"] == 4 and audit["counts"]["duration_s"] == 200.0
        # Original unmodified function retains all legal same-episode intervals and checks every saved target phase.
        idle = original.stagnation(directory, audit)
        with np.load(directory / "physical_states.npz", allow_pickle=False) as saved:
            terminated, episodes = saved["terminated"], saved["episode_ids"]
        events = [{"source_frame": int(frame), "completed_control": int(frame + 1), "derived_time_s": float((frame + 1) / 30),
                   "actor": int(actor), "episode_id": int(episodes[frame, actor])}
                  for frame, actor in np.argwhere(terminated)]
        assert len(events) == audit["counts"]["falls"] == result["fall_resets"]
        valid = sum(row["valid_same_episode_intervals"] for row in idle["per_actor"])
        idle_count = sum(row["idle_intervals"] for row in idle["per_actor"])
        runs[label] = {"actual_GPU_owner": original.record(gpu_path), "actual_CPU_owner": original.record(cpu_path),
                       "actual_raw_audit": original.record(audit_path), "result": audit["artifacts"]["result"],
                       "original_raw_counts": audit["counts"], "original_contact_counts": audit["original_obstacle_peer_channels"],
                       "route_vertices_total": sum(audit["counts"]["route_vertices_reached"]),
                       "physical_zero_gate_passed": audit["physical_zero_gate_passed"],
                       "whole_original_idle_diagnostics": idle,
                       "pooled_valid_idle_intervals": idle_count, "pooled_legal_intervals": valid,
                       "pooled_idle_fraction": idle_count / valid, "fall_events": events,
                       "actors_with_falls": sorted({row["actor"] for row in events}),
                       "GPU_elapsed_seconds": gpu["elapsed_seconds"], "CPU_raw_elapsed_seconds": cpu["elapsed_seconds"],
                       "all_failed_contact_idle_reset_samples_retained": True}
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    report = {"schema": "default_NEW.samples10.actual_complete_comparison.v1", "status": "observed_actual_closed_results",
              "completed_utc": datetime.now(timezone.utc).isoformat(), "execution_source": original.record(__file__),
              "original_idle_source": original.record(IDLE), "original_idle_function_called_unmodified": True,
              "runs": runs, "causality_established": False, "default_promotion": False,
              "S10_native_condition_failed": not runs["NEWb563_S10"]["physical_zero_gate_passed"],
              "style90_45_recomputed_for_this_summary": False, "style_scores_used_as_filter": False,
              "weights_read_or_models_loaded": False, "GPU_used": False, "SDK_started": False,
              "sealed34_outputs_read": False, "CPU_elapsed_seconds": time.monotonic() - started,
              "limits": ["Candidate count changes RNG consumption; S10 first candidate is not a bitwise paired S1 future.",
                         "Falls are events; actors_with_falls is a separate count. Derived control time is not native engine time.",
                         "Idle uses original same-episode root XY displacement speed<0.1m/s; only reset jumps excluded.",
                         "Pair-controls and actor-controls have different denominators; simultaneous pairs are retained.",
                         "The original raw pass establishes capture integrity, not physical success or paper superiority."]}
    output = HERE / "actual_complete_comparison.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    output.chmod(0o444)
    print(json.dumps({"status": report["status"], "output": original.record(output), "CPU_elapsed_seconds": report["CPU_elapsed_seconds"],
                      "runs": {label: {"falls": row["original_raw_counts"]["falls"], "actors_with_falls": row["actors_with_falls"],
                                       "routes": row["route_vertices_total"], "pooled_idle_fraction": row["pooled_idle_fraction"],
                                       "idle": [{"actor": actor["actor"], "count": actor["idle_intervals"], "valid": actor["valid_same_episode_intervals"],
                                                 "fraction": actor["idle_fraction"], "longest_seconds": actor["longest_idle"]["duration_s"]}
                                                for actor in row["whole_original_idle_diagnostics"]["per_actor"]]}
                               for label, row in runs.items()}}, indent=2))


if __name__ == "__main__":
    main()
