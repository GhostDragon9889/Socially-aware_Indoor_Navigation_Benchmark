"""Read sealed NEW physical records only; keep original contact/idle formulas."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ACTUAL = ROOT / "docs/style_improvement_20261003/default_newDP_physical_actual_runs_v1"
IDLE = ROOT / "docs/style_improvement_20261002/summarize_navigation_style.py"
CONTACTS = ROOT / "docs/style_improvement_20261002/navigation_near_horizon_eta025_comparison_v1/analyze_eta025_navigation.py"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verified(row):
    path = Path(row["path"])
    assert not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444
    assert record(path) == row
    return path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    cpu_path = ACTUAL / "fixture_new_CPU_actual_launch.json"
    cpu_record = record(cpu_path)
    assert cpu_record["sha256"] == "ebdbf67cd3ff730d612d60eda43cfc153519534e8ce28229f31105e32bc27b93"
    verified(cpu_record)
    cpu = read(cpu_path)
    assert cpu["schema"] == "default_newDP.actual_CPU_audit_launch.v2"
    assert cpu["status"] == "closed" and type(cpu["exit_code"]) is int and cpu["exit_code"] == 0
    assert cpu["child_exit_observed"] is cpu["driver_completed"] is True
    assert cpu["binding_before"] == cpu["binding_after"]
    audit_path = verified(cpu["audit_artifact"])
    audit = read(audit_path)
    assert audit["actual_capture_independently_audited"] is True
    assert audit["actual_source_admission"] == cpu["actual_capture_admission"]
    admission = audit["actual_source_admission"]
    assert admission["production_source"]["source_count"] == 185
    assert admission["controller"]["sha256"] == "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
    assert admission["planners"]["new"]["sha256"] == "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90"
    gpu_path = verified(admission["root_closed_launch"])
    gpu = read(gpu_path)
    assert gpu["status"] == "closed" and type(gpu["exit_code"]) is int and gpu["exit_code"] == 0
    assert gpu["child_exit_observed"] is gpu["driver_completed"] is True
    assert gpu["binding_before"] == gpu["binding_after"]
    wanted = {Path(audit["artifacts"][k]["path"]).name for k in ("result", "ground_support", "physical_states")}
    wanted.add("reference_plans.npz")
    rows = {Path(row["path"]).name: row for row in gpu["artifacts"] if Path(row["path"]).name in wanted}
    assert set(rows) == wanted
    for row in rows.values():
        verified(row)
    for key in ("result", "ground_support", "physical_states"):
        assert rows[Path(audit["artifacts"][key]["path"]).name] == audit["artifacts"][key]
    directory = Path(audit["artifacts"]["result"]["path"]).parent
    result = read(directory / "result.json")
    with np.load(directory / "physical_states.npz", allow_pickle=False) as saved:
        physical = {key: saved[key] for key in saved.files}
    selected_ground = ("control_step", "control_step_after", "episode_ids", "terminated", "physics_step_end",
                       "double_support_fraction", "no_support_fraction", "foot_normal_load_mean_n",
                       "foot_normal_load_peak_n", "mass_weighted_world_com", "body_link_lin_vel_w")
    with np.load(directory / "ground_support.npz", allow_pickle=False) as saved:
        ground = {key: saved[key] for key in selected_ground}
    with np.load(directory / "reference_plans.npz", allow_pickle=False) as saved:
        plans = {key: saved[key] for key in saved.files}
    assert physical["root_pos"].shape == (6000, 4, 3)
    assert np.array_equal(physical["episode_ids"], ground["episode_ids"])
    assert np.array_equal(physical["terminated"], ground["terminated"])
    assert np.array_equal(ground["control_step_after"], np.arange(1, 6001))
    assert np.array_equal(ground["physics_step_end"], np.arange(4, 24001, 4))
    specification = importlib.util.spec_from_file_location("original_saved_navigation_idle", IDLE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert record(IDLE)["sha256"] == "ee0d13c3f345cf11aa2981002c21b070bf53eb4b6129d86d4a6f851443ac56d7"
    idle = module.stagnation(directory, audit)
    # Exact original event function, without executing the old-pair entry point.
    tree = ast.parse(CONTACTS.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "contact_events"]
    assert len(functions) == 1
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(CONTACTS), "exec"), namespace)
    contact_events = namespace["contact_events"]
    episodes = physical["episode_ids"]
    static_events, peer_events = [], []
    for actor in range(4):
        static_events.extend(contact_events(physical["static_contact"][:, actor], episodes,
                                           physical["static_contact_max_force_n"][:, actor], [actor]))
        for peer in range(actor + 1, 4):
            peer_events.extend(contact_events(physical["peer_contact"][:, actor, peer], episodes,
                                             physical["peer_contact_max_force_n"][:, actor, peer], [actor, peer]))
    valid = ((episodes[1:] == episodes[:-1]) & ~physical["reset_mask"][1:])
    speed = np.linalg.norm(np.diff(physical["root_pos"][..., :2].astype(np.float64), axis=0), axis=-1) * 30
    fall_indices = list(zip(*np.nonzero(physical["terminated"]), strict=True))
    assert len(fall_indices) == audit["counts"]["falls"] == result["fall_resets"] == 1
    fall_records, context_rows = [], []
    for frame, actor in fall_indices:
        frame, actor = int(frame), int(actor)
        episode = int(episodes[frame, actor])
        first, stop = max(0, frame - 60), min(6000, frame + 61)
        selected = np.arange(first, stop)
        same = episodes[selected, actor] == episode
        kept = selected[same]
        actor_plans = np.flatnonzero(plans["actor_ids"] == actor)
        controls = plans["control_step"][actor_plans]
        plan_row = int(actor_plans[np.searchsorted(controls, frame, side="right") - 1])
        age = int(frame - plans["control_step"][plan_row] + 1)
        assert np.allclose(plans["reference_xy"][plan_row, age], physical["target_xy"][frame, actor], atol=1e-5, rtol=0)
        for f in selected:
            f = int(f)
            com = ground["mass_weighted_world_com"][f, -1, actor]
            row = {"source_frame": f, "completed_control": f + 1, "time_s": (f + 1) / 30,
                   "actor": actor, "episode_id": int(episodes[f, actor]), "same_episode_as_fall": bool(episodes[f, actor] == episode),
                   "root_x_m": float(physical["root_pos"][f, actor, 0]), "root_y_m": float(physical["root_pos"][f, actor, 1]),
                   "root_height_m": float(physical["root_pos"][f, actor, 2]), "terminated": bool(physical["terminated"][f, actor]),
                   "reset_mask": bool(physical["reset_mask"][f, actor]),
                   "executed_target_error_m": float(np.linalg.norm(physical["target_xy"][f, actor] - physical["root_pos"][f, actor, :2])),
                   "raw_target_error_m": float(np.linalg.norm(physical["raw_target_xy"][f, actor] - physical["root_pos"][f, actor, :2])),
                   "reference_target_speed_mps": float(np.linalg.norm(physical["reference_target_velocity_xy"][f, actor])),
                   "native_end_substep_root_speed_mps": float(np.linalg.norm(ground["body_link_lin_vel_w"][f, -1, actor, 0, :2])),
                   "previous_interval_speed_mps": None if f == 0 or not valid[f - 1, actor] else float(speed[f - 1, actor]),
                   "previous_interval_idle": None if f == 0 or not valid[f - 1, actor] else bool(speed[f - 1, actor] < .1),
                   "com_x_m": float(com[0]), "com_y_m": float(com[1]), "com_height_m": float(com[2]),
                   "double_support_fraction_120hz": float(ground["double_support_fraction"][f, actor]),
                   "no_support_fraction_120hz": float(ground["no_support_fraction"][f, actor]),
                   "sum_foot_mean_normal_load_n": float(ground["foot_normal_load_mean_n"][f, actor].sum()),
                   "maximum_foot_peak_normal_load_n": float(ground["foot_normal_load_peak_n"][f, actor].max()),
                   "static_contact": bool(physical["static_contact"][f, actor]),
                   "static_peak_force_n": float(physical["static_contact_max_force_n"][f, actor])}
            for peer in range(4):
                row[f"peer{peer}_contact"] = bool(physical["peer_contact"][f, actor, peer])
                row[f"peer{peer}_peak_force_n"] = float(physical["peer_contact_max_force_n"][f, actor, peer])
            context_rows.append(row)
        terminal = next(row for row in context_rows if row["source_frame"] == frame and row["actor"] == actor)
        intervals = kept[kept > 0]
        intervals = intervals[valid[intervals - 1, actor] & (episodes[intervals - 1, actor] == episode)]
        fall_records.append({"terminal": terminal, "physics_step_end": int(ground["physics_step_end"][frame]),
                             "fixed_context_radius_s": 2.0, "requested_source_frames_inclusive": [first, stop - 1],
                             "same_episode_context_frames": int(len(kept)), "cross_episode_context_frames_retained_in_CSV": int((~same).sum()),
                             "same_episode_context_first_last_frame": [int(kept[0]), int(kept[-1])],
                             "same_episode_peer_contact_control_frames": int(physical["peer_contact"][kept, actor].any(-1).sum()),
                             "same_episode_maximum_peer_force_n": float(physical["peer_contact_max_force_n"][kept, actor].max()),
                             "same_episode_root_height_m": module.summary(physical["root_pos"][kept, actor, 2]),
                             "same_episode_executed_target_error_m": module.summary(np.linalg.norm(physical["target_xy"][kept, actor] - physical["root_pos"][kept, actor, :2], axis=-1)),
                             "same_episode_COM_height_m": module.summary(ground["mass_weighted_world_com"][kept, :, actor, 2]),
                             "same_episode_double_support_fraction": float(ground["double_support_fraction"][kept, actor].mean()),
                             "same_episode_no_support_fraction": float(ground["no_support_fraction"][kept, actor].mean()),
                             "same_episode_idle_intervals": int((speed[intervals - 1, actor] < .1).sum()),
                             "same_episode_valid_intervals": int(len(intervals)),
                             "last_executed_saved_plan": {"archive_row": plan_row, "control_step": int(plans["control_step"][plan_row]),
                                "reference_age_after_control": age, "selected_sample": int(plans["selected_sample"][plan_row]),
                                "target_reproduces_saved_reference": True, "initial_overspeed": bool(plans["initial_overspeed"][plan_row]),
                                "raw_xy_half_second": plans["raw_xy"][plan_row, 5].tolist(), "raw_xy_five_second": plans["raw_xy"][plan_row, -1].tolist()},
                             "next_source_frame": None if frame + 1 == 6000 else {"frame": frame + 1,
                                "episode_id": int(episodes[frame + 1, actor]), "reset_mask": bool(physical["reset_mask"][frame + 1, actor])}})
    assert int(np.triu(physical["peer_contact"], 1).sum()) == audit["original_obstacle_peer_channels"]["peer_contact_unordered_pair_frames"] == 70
    assert int(physical["peer_contact"].any(-1).sum()) == 140
    assert sum(event["control_frames"] for event in peer_events) == 70
    assert int(physical["static_contact"].sum()) == 0
    csv_path = HERE / "fall_fixed_2s_context.csv"
    with csv_path.open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(context_rows[0]))
        writer.writeheader()
        writer.writerows(context_rows)
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    report = {"schema": "default_newDP.actual_failed_event_diagnostics.v1", "status": "observed_saved_NEW_records",
              "execution_source": record(__file__), "actual_CPU_owner": cpu_record, "actual_GPU_owner": record(gpu_path),
              "actual_raw_audit": record(audit_path), "captures": rows, "source_epoch": admission["production_source"],
              "controller_sha256_from_source_admission": admission["controller"]["sha256"],
              "deployed_planner_sha256_from_source_admission": admission["runtime_planner_identity"]["deployed_diffusion_checkpoint"]["sha256"],
              "original_idle_math": record(IDLE), "original_contact_event_math": record(CONTACTS),
              "original_contact_function_AST_reused": True, "original_idle_function_called_unmodified": True,
              "fall_records": fall_records, "all_peer_contact_events": peer_events, "all_static_contact_events": static_events,
              "global_original_idle_diagnostics": idle, "original_physical_counts": audit["counts"],
              "original_contact_counts": audit["original_obstacle_peer_channels"], "fixed_context_CSV": record(csv_path),
              "all_falls_contacts_idle_reset_records_retained": True,
              "time_semantics": "source frame i is post-control i, derived simulation elapsed=(i+1)/30; not native engine timestamp",
              "force_semantics": "original >1N strict normal-force flags and 4-substep max; force matrix entries are saved producer aggregates, not independent raw force reconstruction",
              "COM_semantics": "saved mass-weighted world COM already independently checked by raw auditor; end-substep CSV, all4substep heights in context summary",
              "causality_established": False, "weights_read_or_model_run": False, "GPU_used": False, "SDK_started": False,
              "sealed34_outputs_read": False, "CPU_elapsed_s": time.monotonic() - started}
    with (HERE / "diagnostics.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "CPU_elapsed_s": report["CPU_elapsed_s"], "falls": len(fall_records),
                      "peer_events": len(peer_events), "terminal": fall_records[0]["terminal"],
                      "idle_per_actor": [{"actor": row["actor"], "idle_fraction": row["idle_fraction"],
                                          "longest_duration_s": row["longest_idle"]["duration_s"]} for row in idle["per_actor"]]}))


if __name__ == "__main__":
    main()
