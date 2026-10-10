"""CPU-only same-phase clearance proxies for all sealed four-actor plans.

Read reference_plans.npz and small existing JSON only, never physical arrays,
weights or a simulator. Negative clearance is a geometric planning proxy;
it is not a measured collision, contact, fall or causal explanation.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import stat
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMPARISON = ROOT / "docs/style_improvement_20261003/default_NEW_samples10_actual_runs_v1/actual_complete_comparison.json"
COMPARISON_SHA = "0eb0951eb9cfb0175eba061ca18040555f067f7684bf32901e065c68bfc14d51"
EXPECTED_AMP = "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
DP = {"OLD63e_S1": "63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac",
      "NEWb563_S1": "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90",
      "NEWb563_S10": "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90"}


def record(path, *, readonly=False):
    path = Path(path)
    info = path.stat()
    if not path.is_file() or path.is_symlink() or info.st_nlink != 1:
        raise ValueError(f"Expected ordinary single-link file: {path}")
    if readonly and info.st_mode & 0o222:
        raise ValueError(f"Sealed input is writable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": info.st_size, "sha256": digest.hexdigest(),
            "mode": oct(stat.S_IMODE(info.st_mode)), "nlink": info.st_nlink}


def admitted(expected):
    actual = record(expected["path"], readonly=True)
    if any(actual[key] != expected[key] for key in ("path", "bytes", "sha256")):
        raise ValueError(f"Sealed input identity mismatch: {actual['path']}")
    return actual


def headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, order, dtype = np.lib.format._read_array_header(stream, version)
            result[name.removesuffix(".npy")] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": order}
    return result


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite scalar summary")
    return {"count": int(values.size), "min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def compare(raw, reference, mask=None):
    # Last dimension is phase; each preceding item is one plan-pair/actor.
    mask = np.ones(raw.shape, bool) if mask is None else np.broadcast_to(mask, raw.shape)
    raw_bad, ref_bad = (raw < 0) & mask, (reference < 0) & mask
    raw_min = np.where(mask, raw, np.inf).min(-1)
    ref_min = np.where(mask, reference, np.inf).min(-1)
    return {
        "item_count": int(raw_min.size), "phase_observation_count": int(mask.sum()),
        "raw_clearance_m": summary(raw[mask]), "reference_clearance_m": summary(reference[mask]),
        "raw_unsafe_items": int((raw_min < 0).sum()), "reference_unsafe_items": int((ref_min < 0).sum()),
        "raw_unsafe_phases": int(raw_bad.sum()), "reference_unsafe_phases": int(ref_bad.sum()),
        "same_phase_raw_safe_to_reference_unsafe": int(((raw >= 0) & ref_bad).sum()),
        "same_phase_raw_unsafe_to_reference_safe": int((raw_bad & (reference >= 0)).sum()),
        "whole_item_raw_safe_to_reference_unsafe": int(((raw_min >= 0) & (ref_min < 0)).sum()),
        "whole_item_raw_unsafe_to_reference_safe": int(((raw_min < 0) & (ref_min >= 0)).sum()),
    }


def aabb_distance(points, centers, sizes):
    # Exact NumPy translation of paper.py signed_aabb_distance, in world metres.
    q = np.abs(points[..., None, :] - centers) - sizes / 2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q, 0).max(-1)


def static_channels(points, scene, radius):
    centers = np.asarray([box["center"][:2] for box in scene["cuboids"]], dtype=np.float64)
    sizes = np.asarray([box["size"][:2] for box in scene["cuboids"]], dtype=np.float64)
    obstacle = aabb_distance(points, centers, sizes) - radius
    bounds = np.asarray(scene["bounds_xy"], dtype=np.float64)
    boundary = -aabb_distance(points, ((bounds[:2] + bounds[2:]) / 2)[None], (bounds[2:] - bounds[:2])[None]) - radius
    return np.concatenate((obstacle, boundary), -1)


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(name, original, pair_rows, actor_rows):
    identities = {key: admitted(original[key]) for key in ("result", "actual_GPU_owner", "actual_CPU_owner", "actual_raw_audit")}
    plan_record = admitted(original["whole_original_idle_diagnostics"]["saved_plans"])
    for key in ("actual_GPU_owner", "actual_CPU_owner"):
        owner = json.loads(Path(identities[key]["path"]).read_text())
        if owner.get("status") != "closed" or owner.get("exit_code") != 0 or owner.get("child_exit_observed") is not True:
            raise ValueError("Original GPU and CPU owners must be genuinely closed0")
    result = json.loads(Path(identities["result"]["path"]).read_text())
    arguments, guide, scene = result["arguments"], result["diffusion_guidance"], result["evaluation_scene"]
    actual_dp = result.get("evaluation_planner_change", {}).get("evaluation_sha256", result["provenance"]["diffusion_sha256"])
    if (result["evaluation_actor"]["checkpoint_sha256"] != EXPECTED_AMP or actual_dp != DP[name]
            or arguments["num_envs"] != 4 or arguments["evaluate_steps"] != 6000
            or result["actual_control_steps"] != 6000 or result["time_limit_resets"] != 0
            or result["resets"] != len(original["fall_events"])
            or guide["agent_collision_radius_m"] != .6 or guide["scene_collision_radius_m"] != .8
            or result["reference_adaptation"]["plan_capture"] != plan_record["path"]
            or not original["whole_original_idle_diagnostics"]["all_saved_plan_target_phase_checks_passed"]):
        raise ValueError("Actual checkpoint, scene, reset, phase or radius contract mismatch")
    metadata = headers(plan_record["path"])
    with np.load(plan_record["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = len(arrays["control_step"])
    if count % 4 or arrays["raw_xy"].shape != (count, 51, 2) or arrays["reference_xy"].shape != (count, 151, 2):
        raise ValueError("Need complete four-actor 51/151 source rows")
    for key in ("raw_xy", "reference_xy", "reference_velocity_xy", "deviation_m"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite source plan: {key}")
    controls = arrays["control_step"].reshape(-1, 4)
    actors = arrays["actor_ids"].reshape(-1, 4)
    samples = arrays["selected_sample"].reshape(-1, 4)
    if (not np.array_equal(actors, np.broadcast_to(np.arange(4), actors.shape))
            or not (controls == controls[:, :1]).all() or not (samples == samples[:, :1]).all()):
        raise ValueError("Source archive is not complete ordered joint-scene rows")
    controls = controls[:, 0]
    if controls[0] != 0 or controls[-1] >= 6000 or not (np.diff(controls) > 0).all():
        raise ValueError("Expected unique chronological replan groups starting at control0")
    durations = np.diff(np.append(controls, 6000))
    if not ((durations > 0) & (durations <= 15)).all() or durations.sum() != 6000:
        raise ValueError("All 6000 saved target controls must belong to a replan prefix")
    raw_knots = arrays["raw_xy"].reshape(-1, 4, 51, 2).astype(np.float64)
    ref = arrays["reference_xy"].reshape(-1, 4, 151, 2).astype(np.float64)
    phase = np.arange(151)
    lower, upper, alpha = phase // 3, np.minimum(phase // 3 + 1, 50), (phase % 3) / 3
    raw = raw_knots[:, :, lower] * (1 - alpha[None, None, :, None]) + raw_knots[:, :, upper] * alpha[None, None, :, None]
    offset = np.linalg.norm(ref[:, :, 0] - raw[:, :, 0], axis=-1)
    deviation = np.linalg.norm(ref - raw, axis=-1)
    recorded_deviation = arrays["deviation_m"].reshape(-1, 4, 151)
    pairs = list(itertools.combinations(range(4), 2))
    peer_raw = np.stack([np.linalg.norm(raw[:, i] - raw[:, j], axis=-1) - 1.2 for i, j in pairs], 1)
    peer_ref = np.stack([np.linalg.norm(ref[:, i] - ref[:, j], axis=-1) - 1.2 for i, j in pairs], 1)
    channel_raw, channel_ref = static_channels(raw, scene, .8), static_channels(ref, scene, .8)
    static_raw, static_ref = channel_raw.min(-1), channel_ref.min(-1)
    prefix = (phase[None, None, :] > 0) & (phase[None, None, :] <= durations[:, None, None])
    epochs = np.ones((len(controls), 4), np.int64)
    # Archive lacks episode/reset arrays: infer only from sealed fall JSON and
    # observed zero time limits; a completed fall f resets before plan control f.
    for fall in original["fall_events"]:
        epochs[controls >= fall["completed_control"], fall["actor"]] += 1
    for group, control in enumerate(controls):
        for pair_index, (i, j) in enumerate(pairs):
            pr, pf = peer_raw[group, pair_index], peer_ref[group, pair_index]
            pair_rows.append({"run": name, "control_step": int(control), "actor_i": i, "actor_j": j,
                              "inferred_episode_i": int(epochs[group, i]), "inferred_episode_j": int(epochs[group, j]),
                              "selected_sample": int(samples[group, 0]), "saved_target_prefix_controls": int(durations[group]),
                              "raw_start_clearance_m": float(pr[0]), "reference_start_clearance_m": float(pf[0]),
                              "raw_min_clearance_m": float(pr.min()), "reference_min_clearance_m": float(pf.min()),
                              "raw_unsafe_phases": int((pr < 0).sum()), "reference_unsafe_phases": int((pf < 0).sum()),
                              "same_phase_raw_safe_ref_unsafe": int(((pr >= 0) & (pf < 0)).sum()),
                              "whole_pair_raw_safe_ref_unsafe": bool(pr.min() >= 0 and pf.min() < 0)})
        for actor in range(4):
            sr, sf = static_raw[group, actor], static_ref[group, actor]
            actor_rows.append({"run": name, "control_step": int(control), "actor": actor,
                               "inferred_episode": int(epochs[group, actor]), "selected_sample": int(samples[group, actor]),
                               "saved_target_prefix_controls": int(durations[group]), "start_raw_reference_offset_m": float(offset[group, actor]),
                               "raw_min_static_clearance_m": float(sr.min()), "reference_min_static_clearance_m": float(sf.min()),
                               "raw_static_unsafe_phases": int((sr < 0).sum()), "reference_static_unsafe_phases": int((sf < 0).sum()),
                               "same_phase_raw_safe_ref_unsafe": int(((sr >= 0) & (sf < 0)).sum()),
                               "whole_actor_raw_safe_ref_unsafe": bool(sr.min() >= 0 and sf.min() < 0)})
    return {
        "input_identities": {**identities, "reference_plans": plan_record}, "NPZ_headers": metadata,
        "AMP_sha256_from_record": EXPECTED_AMP, "DP_sha256_from_record": actual_dp,
        "scene_source": "source-bound result.evaluation_scene; no USD or physical scene opened", "scene": scene,
        "scene_canonical_sha256": hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "plan_rows": count, "joint_replan_groups": len(controls), "all_six_pairs_per_group": pairs,
        "replan_prefix_controls": summary(durations), "prefix_control_coverage": int(durations.sum()),
        "episode_source": "inferred from existing sealed fall summaries and zero time-limit resets; not recorded in plan archive",
        "inferred_final_episodes": epochs[-1].tolist(), "all_original_falls_retained": original["fall_events"],
        "original_physical_contact_counts": original["original_contact_counts"],
        "original_idle_counts": {"idle": original["pooled_valid_idle_intervals"], "legal": original["pooled_legal_intervals"]},
        "start_raw_reference_offset_m": summary(offset), "all_phase_raw_reference_deviation_m": summary(deviation),
        "max_numpy_resampled_vs_recorded_deviation_difference_m": float(np.abs(deviation - recorded_deviation).max()),
        "peer": {"all_151_phases": compare(peer_raw, peer_ref), "future_150_phases": compare(peer_raw[:, :, 1:], peer_ref[:, :, 1:]),
                 "saved_target_prefix": compare(peer_raw, peer_ref, prefix)},
        "static_combined": {"all_151_phases": compare(static_raw, static_ref), "future_150_phases": compare(static_raw[:, :, 1:], static_ref[:, :, 1:]),
                            "saved_target_prefix": compare(static_raw, static_ref, prefix)},
        "static_each_channel": {("cuboid_" + str(index) if index < len(scene["cuboids"]) else "domain_boundary"):
                                compare(channel_raw[..., index], channel_ref[..., index]) for index in range(channel_raw.shape[-1])},
        "all_plans_actors_pairs_failed_idle_retained": True,
    }


def main():
    started = time.perf_counter()
    comparison_record = record(COMPARISON, readonly=True)
    if comparison_record["sha256"] != COMPARISON_SHA:
        raise ValueError("Frozen original summary changed")
    original = json.loads(COMPARISON.read_text())
    pair_rows, actor_rows = [], []
    runs = {name: analyze(name, original["runs"][name], pair_rows, actor_rows) for name in DP}
    if len({run["scene_canonical_sha256"] for run in runs.values()}) != 1:
        raise ValueError("Actual three runs use different scene geometry")
    write_csv(HERE / "all_joint_plan_pair_rows.csv", pair_rows)
    write_csv(HERE / "all_actor_plan_static_rows.csv", actor_rows)
    report = {
        "schema": "causal_reference.same_phase_clearance_proxy.v1", "status": "completed_CPU_diagnostic",
        "execution_source": record(__file__), "original_complete_comparison": comparison_record,
        "definition": {"world_units": "metres", "raw_knots_seconds": "0..5 at .1; actual51 source boundary",
                       "dense_phase_seconds": "0..5 at1/30;151samples; rational index/3 linear interpolation in float64",
                       "peer_clearance": "norm(x_i-x_j)-2*.6", "static_clearance": "min(all signed_cuboid_distances-.8, -signed_domain_distance-.8)",
                       "unsafe": "strict clearance<0; zero is geometrically safe at the declared radius",
                       "same_phase_transition": "raw>=0 and reference<0 at identical relative time",
                       "whole_item_transition": "raw minimum>=0 and reference minimum<0 over the same mask",
                       "saved_target_prefix": "phases1..next_replan_control-current_control, final prefix ends at6000; reference target proxy, not physical position",
                       "guidance_boundary": "producerguidanceused52futureframes(.1..5.2), archivecontainsselected50futureframesplusactualt0; diagnosticcannotrebuilddiscarded5.1/5.2 orallunselectedcandidates"},
        "runs": runs, "all_pair_rows": record(HERE / "all_joint_plan_pair_rows.csv"), "all_actor_rows": record(HERE / "all_actor_plan_static_rows.csv"),
        "physical_collision_or_contact_recomputed": False, "causality_established": False,
        "models_weights_SDK_GPU_sealed34_read": False, "CPU_elapsed_seconds": time.perf_counter() - started,
    }
    with (HERE / "actual_diagnostic.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "CPU_elapsed_seconds": report["CPU_elapsed_seconds"],
                      "runs": {name: {"groups": run["joint_replan_groups"], "peer": run["peer"]["all_151_phases"],
                                      "static": run["static_combined"]["all_151_phases"],
                                      "start_offset": run["start_raw_reference_offset_m"]} for name, run in runs.items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
