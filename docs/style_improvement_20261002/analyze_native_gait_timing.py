"""Read-only120Hz floor-support timing over already audited navigation captures.

All recorded samples, including resets/falls, remain in the report. No contact
debounce, posture filtering, metric/model selection or AMASS force GT is used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_ground_support_capture import fingerprint


POLICY = {
    "force_threshold_n": 1.0,
    "foot_contact": "norm(sum floor-filter normal-force vectors)>1N",
    "side_contact": "any ankle or toe contact; fixed LAnkle,LToe,RAnkle,RToe order",
    "debounce": None,
    "speed_source": "actual120Hz Pelvis body_link_lin_vel_w horizontal norm",
    "speed_bins_mps": {"slow": [0., .1], "transition": [.1, .5], "moving": [.5, None]},
    "speed_boundaries": "lower inclusive, upper exclusive",
    "bout_speed_bin": "mean Pelvis speed over the entire original raw bout; bouts never cut at speed changes",
    "transition_speed_bin": "speed of destination substep; pairs crossing episodes excluded",
    "boundary": "episode boundaries split bouts and transitions; boundary-truncated bouts marked censored",
    "failed_frame_policy": "all recorded samples retained; terminal controls separately flagged without guessing exact substep of fall",
}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()), "min": float(values.min()),
            "p50": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)), "max": float(values.max())}


def bin_name(value):
    return "slow" if value < .1 else "transition" if value < .5 else "moving"


def bouts(codes, episodes, speed, dt, *, signal, actor, run):
    """Constant raw runs, clipped at actual episode boundaries, including censor."""
    boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(episodes)]
    result = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        starts = np.r_[low, low + np.flatnonzero(np.diff(codes[low:high])) + 1]
        stops = np.r_[starts[1:], high]
        for start, stop in zip(starts, stops, strict=True):
            value = int(codes[start])
            if signal != "support_state" and not value:
                continue
            average_speed = float(speed[start:stop].mean())
            result.append({"run": run, "actor": actor, "episode_id": int(episodes[start]),
                           "signal": signal, "value": value, "first_substep_index": int(start),
                           "stop_substep_index_exclusive": int(stop),
                           "first_sample_elapsed_s": float((start+1)*dt),
                           "stop_interval_elapsed_s": float(stop*dt), "duration_s": float((stop-start)*dt),
                           "left_censored": bool(start == low), "right_censored": bool(stop == high),
                           "mean_root_speed_mps": average_speed, "speed_bin": bin_name(average_speed)})
    return result


def exposure(contact, speed, episodes, mask, dt):
    left, right = contact[:, 0], contact[:, 1]
    same = episodes[1:] == episodes[:-1]
    included = same & mask[1:]
    transitions = {
        "left_touchdown": int((included & ~left[:-1] & left[1:]).sum()),
        "left_liftoff": int((included & left[:-1] & ~left[1:]).sum()),
        "right_touchdown": int((included & ~right[:-1] & right[1:]).sum()),
        "right_liftoff": int((included & right[:-1] & ~right[1:]).sum()),
        "support_state_change": int((included & (np.any(contact[1:] != contact[:-1], axis=-1))).sum()),
    }
    count = int(mask.sum())
    result = {"substeps": count, "actor_time_s": count*dt, "actual_root_speed_mps": summary(speed[mask]),
              "eligible_transition_pairs": int(included.sum()), "raw_transitions": transitions}
    for name, condition in (("left_support", left), ("right_support", right),
                            ("double_support", left & right), ("no_foot_support", ~left & ~right),
                            ("left_only", left & ~right), ("right_only", ~left & right)):
        amount = int((mask & condition).sum())
        result[name] = {"substeps": amount, "actor_time_s": amount*dt,
                        "fraction": amount/count if count else None}
    return result


def read_run(name, directory, audit_path):
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "passed", "Existing independent native-source audit must pass first"
    result_path = directory/"result.json"
    ground_path, physical_path = directory/"ground_support.npz", directory/"physical_states.npz"
    identities = {"result": fingerprint(result_path), "ground_support": fingerprint(ground_path),
                  "physical_states": fingerprint(physical_path)}
    for key in ("ground_support", "physical_states"):
        assert identities[key] == audit["artifacts"][key], f"Immutable audited artifact mismatch: {key}"
    result = json.loads(result_path.read_text())
    floor = result["ground_support_measurement"]["contract"]["floor_measurement"]
    assert floor["threshold_n"] == POLICY["force_threshold_n"] == result["arguments"]["ground_support_threshold_n"]
    assert floor["substeps_per_control"] == 4 and abs(floor["physics_dt_s"]-1/120) < 1e-12
    with np.load(ground_path, allow_pickle=False) as archive:
        required = ("foot_floor_normal_force_w", "side_contact_substeps", "foot_contact_substeps", "physics_step",
                    "physics_step_before", "episode_ids", "reset_mask", "terminated", "truncated", "valid",
                    "body_link_lin_vel_w", "mass_weighted_world_com", "common_body_names", "foot_names")
        data = {key: archive[key] for key in required}
    assert data["valid"].all(), "Never silently discard invalid raw capture samples"
    force = data["foot_floor_normal_force_w"]
    t, s, n, feet, _, xyz = force.shape
    assert s == feet == n == 4 and xyz == 3 and t == 6000
    assert np.isfinite(force).all()
    foot = np.linalg.norm(force.sum(-2), axis=-1) > 1.
    contact = foot.reshape(t, 4, n, 2, 2).any(-1)
    assert np.array_equal(foot, data["foot_contact_substeps"])
    assert np.array_equal(contact, data["side_contact_substeps"])
    assert data["foot_names"].tolist() == ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
    before = data["physics_step_before"][0]
    assert np.array_equal(data["physics_step"].reshape(-1), np.arange(before+1, before+4*t+1))
    episodes = data["episode_ids"]
    assert np.array_equal(np.diff(episodes, axis=0) != 0, data["reset_mask"][1:])
    assert data["reset_mask"][0].all()
    pelvis = data["common_body_names"].tolist().index("Pelvis")
    speed = np.linalg.norm(data["body_link_lin_vel_w"][..., pelvis, :2], axis=-1).reshape(t*4, n)
    assert np.isfinite(speed).all()
    contact = contact.reshape(t*4, n, 2)
    episode = np.repeat(episodes, 4, axis=0)
    terminal = np.repeat(data["terminated"] | data["truncated"], 4, axis=0)
    com_z = data["mass_weighted_world_com"][..., 2].reshape(t*4, n)
    dt, rows, actors = floor["physics_dt_s"], [], []
    for actor in range(n):
        side = contact[:, actor]
        state = side[:, 0].astype(np.int8)+2*side[:, 1].astype(np.int8)
        for label, values in (("left_contact", side[:, 0]), ("right_contact", side[:, 1]), ("support_state", state)):
            rows.extend(bouts(values.astype(np.int8), episode[:, actor], speed[:, actor], dt,
                              signal=label, actor=actor, run=name))
        masks = {"all": np.ones(t*4, dtype=bool), "slow": speed[:, actor] < .1,
                 "transition": (speed[:, actor] >= .1) & (speed[:, actor] < .5), "moving": speed[:, actor] >= .5,
                 "terminal_control_frames": terminal[:, actor]}
        actors.append({"actor": actor, "episodes": int(len(np.unique(episode[:, actor]))),
                       "groups": {key: exposure(side, speed[:, actor], episode[:, actor], mask, dt) for key, mask in masks.items()},
                       "actual_COM_world_z_m": summary(com_z[:, actor])})
    duration = {}
    for signal in ("left_contact", "right_contact", "support_state"):
        for value in ([1] if signal != "support_state" else range(4)):
            selected = [row for row in rows if row["signal"] == signal and row["value"] == value]
            complete = [row for row in selected if not row["left_censored"] and not row["right_censored"]]
            duration[f"{signal}:{value}"] = {
                "all_raw_bouts_including_censored_s": summary([row["duration_s"] for row in selected]),
                "uncensored_bouts_s": summary([row["duration_s"] for row in complete]),
                "left_censored": sum(row["left_censored"] for row in selected),
                "right_censored": sum(row["right_censored"] for row in selected),
                "duration_by_whole_bout_mean_speed_bin_s": {key: summary([row["duration_s"] for row in selected if row["speed_bin"] == key])
                                                             for key in ("slow", "transition", "moving")}}
    total = {key: sum(actor["groups"]["all"][key]["substeps"] for actor in actors)
             for key in ("left_support", "right_support", "double_support", "no_foot_support", "left_only", "right_only")}
    retained = t*4*n
    for key in identities:
        assert identities[key] == fingerprint(directory/{"result": "result.json", "ground_support": "ground_support.npz", "physical_states": "physical_states.npz"}[key])
    return {"name": name, "artifacts": identities, "existing_independent_native_audit": fingerprint(audit_path),
            "source_binding": "Existing source/archive verification reused; no claim about today's source tree.",
            "execution_status": result["status"], "fall_resets": result["fall_resets"],
            "route_vertices_reached": result["route_vertices_reached"], "controls": t,
            "retained_actor_substeps": retained, "retained_actor_time_s": retained*dt,
            "reset_markers_including_initial": int(data["reset_mask"].sum()),
            "terminal_actor_controls_retained": int((data["terminated"] | data["truncated"]).sum()),
            "raw_support_counts": total, "raw_support_fractions": {key: value/retained for key, value in total.items()},
            "per_actor": actors, "bout_durations": duration,
            "reference_true_COM": None, "reference_true_ground_contacts": None,
            "contact_patch_slip": None, "native_capture_probe_passed_historical": result["ground_support_measurement"]["native_capture_probe_passed"]}, rows


def self_test():
    # Two episodes have identical support at the reset: neither a2s bout nor
    # a transition across the reset is permitted. Both boundaries stay censored.
    values = np.array([1, 1, 1, 1], dtype=np.int8)
    result = bouts(values, np.array([1, 1, 2, 2]), np.ones(4), .1, signal="left_contact", actor=0, run="test")
    assert [row["duration_s"] for row in result] == [.2, .2]
    assert all(row["left_censored"] and row["right_censored"] for row in result)
    contact = np.array([[False, True], [True, False], [False, True], [True, False]])
    stats = exposure(contact, np.ones(4), np.array([1, 1, 2, 2]), np.ones(4, dtype=bool), .1)
    assert stats["eligible_transition_pairs"] == 2 and stats["raw_transitions"]["support_state_change"] == 2
    assert stats["double_support"]["substeps"] == stats["no_foot_support"]["substeps"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=directory=existing_native_audit.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("A fresh output directory is required")
    self_test()
    runs, events = [], []
    for specification in args.run:
        name, path, audit = specification.split("=", 2)
        if name in {row["name"] for row in runs}:
            parser.error("Duplicate run name")
        row, bouts_list = read_run(name, Path(path), Path(audit))
        runs.append(row)
        events.extend(bouts_list)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir/"raw_bouts.csv"
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    report = {"schema": "native_floor_gait_timing_diagnostic.v1", "status": "passed", "CPU_only": True,
              "GPU_used": False, "SDK_started": False, "heldout_outputs_read": False,
              "policy_fixed_before_data_analysis": POLICY, "synthetic_computation_self_tests": "passed",
              "runs": runs, "all_raw_bouts": fingerprint(output), "script": fingerprint(__file__),
              "limitations": ["Raw1N support bouts can contain force-threshold chatter; no debounce was applied.",
                              "Censored bouts retained separately; no transitions or cycles inferred across reset boundaries.",
                              "A terminal control contains4 substeps; the exact substep of30Hz fall detection is unknown.",
                              "Static/slow episodes naturally have longer support; locomotion/style cannot be inferred from pooled support fraction alone.",
                              "AMASS reference trueCOM/forces and measured tangential patch slip are unavailable.",
                              "These diagnostics do not change frozen MMD/style/benchmark scores or select a model."]}
    with (args.output_dir/"report.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
