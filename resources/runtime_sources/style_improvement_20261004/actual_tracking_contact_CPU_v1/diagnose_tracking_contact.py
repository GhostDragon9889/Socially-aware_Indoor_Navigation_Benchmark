"""NumPy-only diagnostic of three genuine, sealed 4 x 200 s captures.

Targets and root positions share the post-control phase. Contact flags aggregate
four physics substeps, so a root-point clearance is only a geometric proxy.
This reader keeps every frame and never loads a model or starts a simulator.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import stat
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = repository_root(__file__)
COMPARISON = ROOT / "resources/runtime_sources/style_improvement_20261003/default_NEW_samples10_actual_runs_v1/actual_complete_comparison.json"
COMPARISON_SHA = '0917b58d784c2f3737e70f5fa23f250c95c5db72f12d06617a1c889bce7b4192'
PRODUCER = ROOT / "scripts/train_paper_amp.py"
PRODUCER_SHA = "d8b85401ace1da7d0039dcad4e023590577944e487789e742019c363a602e3d3"
AMP_SHA = "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
DP = {"OLD63e_S1": "63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac",
      "NEWb563_S1": "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90",
      "NEWb563_S10": "b56346fbb0c23dc9656a28fa054b45178ef5922a0e52c66b18531b94e296da90"}
PAIRS = list(itertools.combinations(range(4), 2))
FIELDS = ("root_pos", "target_xy", "raw_target_xy", "reference_target_velocity_xy", "reference_deviation_m",
          "episode_ids", "reset_mask", "terminated", "truncated", "static_contact", "peer_contact",
          "static_contact_max_force_n", "peer_contact_max_force_n", "route_vertex_reached")


def record(path, *, readonly=False):
    path = Path(path)
    info = path.stat()
    if not path.is_file() or path.is_symlink() or info.st_nlink != 1 or (readonly and info.st_mode & 0o222):
        raise ValueError(f"Expected ordinary single-link {'sealed' if readonly else ''} file: {path}")
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


def read(path):
    return json.loads(Path(path).read_text())


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scalar diagnostic")
    if not values.size:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(values.size), "min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def aabb_distance(points, centers, sizes):
    # Same signed world-XY AABB distance as the previous clearance diagnostic.
    q = np.abs(points[..., None, :] - centers) - sizes / 2
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q, 0).max(-1)


def geometry(points, scene):
    centers = np.asarray([box["center"][:2] for box in scene["cuboids"]], dtype=np.float64)
    sizes = np.asarray([box["size"][:2] for box in scene["cuboids"]], dtype=np.float64)
    cuboid = aabb_distance(points, centers, sizes) - .8
    bounds = np.asarray(scene["bounds_xy"], dtype=np.float64)
    boundary = -aabb_distance(points, ((bounds[:2]+bounds[2:])/2)[None], (bounds[2:]-bounds[:2])[None]) - .8
    peer = np.stack([np.linalg.norm(points[:, i]-points[:, j], axis=-1)-1.2 for i, j in PAIRS], -1)
    return {"peer": peer, "static_cuboids": cuboid.min(-1),
            "static_combined": np.concatenate((cuboid, boundary), -1).min(-1)}


def proxy_contact_table(clearance, contact):
    unsafe = clearance < 0
    return {"observations": int(unsafe.size), "contact_observations": int(contact.sum()),
            "proxy_unsafe_observations": int(unsafe.sum()),
            "proxy_unsafe_contact": int((unsafe & contact).sum()),
            "proxy_safe_contact": int((~unsafe & contact).sum()),
            "proxy_unsafe_no_contact": int((unsafe & ~contact).sum()),
            "proxy_safe_no_contact": int((~unsafe & ~contact).sum()),
            "clearance_all_m": summary(clearance), "clearance_contact_m": summary(clearance[contact]),
            "clearance_no_contact_m": summary(clearance[~contact])}


def admission(name, original):
    identities = {key: admitted(original[key]) for key in ("result", "actual_GPU_owner", "actual_CPU_owner", "actual_raw_audit")}
    gpu, cpu = (read(identities[key]["path"]) for key in ("actual_GPU_owner", "actual_CPU_owner"))
    for owner in (gpu, cpu):
        if (owner["status"] != "closed" or type(owner["exit_code"]) is not int or owner["exit_code"] != 0
                or owner["child_exit_observed"] is not True or owner["driver_completed"] is not True
                or owner["binding_before"] != owner["binding_after"]):
            raise ValueError("Original GPU/CPU owner must be genuinely closed0 with unchanged binding")
    audit, result = read(identities["actual_raw_audit"]["path"]), read(identities["result"]["path"])
    if (audit["status"] != "passed" or not audit["actual_capture_independently_audited"]
            or cpu["audit_artifact"] != original["actual_raw_audit"]
            or cpu["actual_capture_admission"] != audit["actual_source_admission"]
            or audit["all_failed_contact_idle_reset_samples_retained"] is not True
            or audit["style_score_used_for_admission"] is not False):
        raise ValueError("Original raw CPU audit binding mismatch")
    identities["physical_states"] = admitted(audit["artifacts"]["physical_states"])
    identities["reference_plans"] = admitted(original["whole_original_idle_diagnostics"]["saved_plans"])
    identities["entry"] = admitted(audit["sources"]["entry"])
    artifacts = {row["path"]: row for row in gpu["artifacts"]}
    for key in ("result", "physical_states", "reference_plans", "entry"):
        actual = identities[key]
        if artifacts[actual["path"]] != {k: actual[k] for k in ("path", "bytes", "sha256")}:
            raise ValueError("Original GPU artifact identity mismatch")
    entry = read(identities["entry"]["path"])
    scene_path = result["arguments"]["scene_config"]
    identities["scene_yaml"] = record(scene_path)
    if identities["scene_yaml"]["sha256"] != entry["deployment_contract"]["scene_sha256"]:
        raise ValueError("Actual fixture YAML changed")
    # This fixed 709-byte fixture uses JSON flow arrays/maps for geometry.
    # No generic YAML parser, scene loader, SDK or optional dependency is used.
    scene = result["evaluation_scene"]
    lines = Path(scene_path).read_text().splitlines()
    bounds = [json.loads(line.split(":", 1)[1]) for line in lines if line.startswith("bounds_xy:")]
    cuboids = [json.loads(line.strip()[2:].replace("center:", '"center":').replace("size:", '"size":'))
               for line in lines if line.strip().startswith("- {center:")]
    if bounds != [scene["bounds_xy"]] or cuboids != scene["cuboids"]:
        raise ValueError("Result scene geometry differs from genuine fixture YAML")
    actual_dp = result.get("evaluation_planner_change", {}).get("evaluation_sha256", result["provenance"]["diffusion_sha256"])
    if (result["evaluation_actor"]["checkpoint_sha256"] != AMP_SHA or actual_dp != DP[name]
            or result["arguments"]["num_envs"] != 4 or result["arguments"]["evaluate_steps"] != 6000
            or result["actual_control_steps"] != 6000 or audit["counts"]["control_actor_frames"] != 24000
            or result["state_capture"]["path"] != identities["physical_states"]["path"]
            or result["reference_adaptation"]["plan_capture"] != identities["reference_plans"]["path"]
            or result["diffusion_guidance"]["agent_collision_radius_m"] != .6
            or result["diffusion_guidance"]["scene_collision_radius_m"] != .8
            or result["physical_contact_measurement"]["threshold_n"] != 1.
            or result["physical_contact_measurement"]["physics_substeps_per_control"] != 4
            or result["physical_contact_measurement"]["physics_dt_s"] != 1/120
            or result["physical_contact_measurement"]["control_dt_s"] != 1/30):
        raise ValueError("Actual controller, planner, capture or radius contract mismatch")
    return identities, audit, result, scene


def targets_from_saved_plans(state, plans):
    count = len(plans["actor_ids"])
    if count % 4 or plans["raw_xy"].shape != (count, 51, 2) or plans["reference_xy"].shape != (count, 151, 2):
        raise ValueError("Expected complete source51/reference151 plan rows")
    errors, phase = {}, np.empty((6000, 4), dtype=np.int64)
    for actor in range(4):
        rows = np.flatnonzero(plans["actor_ids"] == actor)
        controls = plans["control_step"][rows]
        if not len(rows) or controls[0] != 0 or controls[-1] >= 6000 or not (np.diff(controls) > 0).all():
            raise ValueError("Missing or unordered source replans")
        selected = rows[np.searchsorted(controls, np.arange(6000), side="right")-1]
        age = np.arange(6000)-plans["control_step"][selected]+1
        if not ((age >= 1) & (age <= 15)).all():
            raise ValueError("Saved target prefix outside declared replan interval")
        phase[:, actor] = age
        lower, alpha = age//3, (age % 3)/3
        raw = (plans["raw_xy"][selected, lower]*(1-alpha[:, None])
               + plans["raw_xy"][selected, lower+1]*alpha[:, None])
        for key, rebuilt in (("raw_target_xy", raw), ("target_xy", plans["reference_xy"][selected, age]),
                             ("reference_target_velocity_xy", plans["reference_velocity_xy"][selected, age])):
            error = float(np.abs(rebuilt-state[key][:, actor]).max())
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError("Post-step target differs from pre-control plan at age+1")
            errors[key] = max(error, errors.get(key, 0.))
    return errors, phase


def events(state, proxies, raw_error, reference_error, idle, valid):
    result = []
    episodes = state["episode_ids"]
    union = state["peer_contact"] | state["peer_contact"].swapaxes(1, 2)
    union_force = np.maximum(state["peer_contact_max_force_n"], state["peer_contact_max_force_n"].swapaxes(1, 2))
    channels = [("static", (actor,), state["static_contact"][:, actor]) for actor in range(4)]
    channels += [("peer", (i, j), union[:, i, j]) for i, j in PAIRS]
    for kind, actors, flags in channels:
        same_episode = np.all(episodes[1:, actors] == episodes[:-1, actors], axis=1)
        onset = flags & ~np.r_[False, flags[:-1] & same_episode]
        for start in np.flatnonzero(onset):
            end = int(start)+1
            while end < 6000 and flags[end] and np.array_equal(episodes[end, actors], episodes[start, actors]):
                end += 1
            window = []
            for frame in range(max(0, int(start)-15), min(6000, int(start)+16)):
                index = PAIRS.index(actors) if kind == "peer" else actors[0]
                key = "peer" if kind == "peer" else "static_cuboids"
                window.append({"source_frame": frame, "completed_control": frame+1, "relative_control": frame-int(start),
                               "contact": bool(flags[frame]), "episode_ids": episodes[frame, actors].tolist(),
                               "reset_mask": state["reset_mask"][frame, actors].tolist(),
                               "terminated": state["terminated"][frame, actors].tolist(),
                               "truncated": state["truncated"][frame, actors].tolist(),
                               "raw_root_error_m": raw_error[frame, actors].tolist(),
                               "reference_root_error_m": reference_error[frame, actors].tolist(),
                               "same_episode_previous_interval": valid[frame, actors].tolist(),
                               "idle_previous_interval": idle[frame, actors].tolist(),
                               "clearance_m": {label: float(proxy[key][frame, index]) for label, proxy in proxies.items()},
                               "static_force_max_n": state["static_contact_max_force_n"][frame, actors].tolist(),
                               "peer_force_max_per_actor_n": union_force[frame, actors].max(-1).tolist(),
                               "contact_pair_max_force_n": float(union_force[frame, actors[0], actors[1]]) if kind == "peer" else None})
            result.append({"kind": kind, "actors": list(actors), "source_frame": int(start), "completed_control": int(start)+1,
                           "derived_time_s": (int(start)+1)/30, "first_frame_left_censored": bool(start == 0),
                           "starts_new_episode": bool(start == 0 or not same_episode[start-1]),
                           "same_episode_contact_controls": end-int(start),
                           "window_source_frames": [window[0]["source_frame"], window[-1]["source_frame"]],
                           "all_window_frames_retained_including_episode_changes": True, "window": window})
    return sorted(result, key=lambda row: (row["source_frame"], row["kind"], row["actors"]))


def analyze(name, original, writer):
    identities, audit, result, scene = admission(name, original)  # Before any NPZ array is read.
    with np.load(identities["physical_states"]["path"], allow_pickle=False) as archive:
        state = {key: archive[key] for key in FIELDS}
    with np.load(identities["reference_plans"]["path"], allow_pickle=False) as archive:
        plans = {key: archive[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    for key, array in state.items():
        shape = (6000, 4, 4) if key in ("peer_contact", "peer_contact_max_force_n") else (6000, 4, 3) if key == "root_pos" else (6000, 4, 2) if key in (
            "target_xy", "raw_target_xy", "reference_target_velocity_xy") else (6000, 4)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid physical array shape/finite: {key}")
        if key in ("reset_mask", "terminated", "truncated", "static_contact", "peer_contact", "route_vertex_reached") and array.dtype != np.bool_:
            raise ValueError(f"Expected genuine boolean field: {key}")
        if key == "episode_ids" and array.dtype.kind not in "iu":
            raise ValueError("Expected recorded integer episode IDs")
    target_errors, phase = targets_from_saved_plans(state, plans)
    root, raw, reference = (state[key][..., :2].astype(np.float64) for key in ("root_pos", "raw_target_xy", "target_xy"))
    raw_error, reference_error = np.linalg.norm(root-raw, axis=-1), np.linalg.norm(root-reference, axis=-1)
    proxy = {label: geometry(points, scene) for label, points in (("raw_target", raw), ("reference_target", reference), ("actual_root", root))}
    peer_union = state["peer_contact"] | state["peer_contact"].swapaxes(1, 2)
    peer_pairs = np.stack([peer_union[:, i, j] for i, j in PAIRS], -1)
    peer_actor = peer_union.any(-1)
    static = state["static_contact"]
    any_contact = peer_actor | static
    valid = np.zeros((6000, 4), dtype=bool)
    valid[1:] = ((state["episode_ids"][1:] == state["episode_ids"][:-1]) & ~state["reset_mask"][1:])
    speeds = np.zeros((6000, 4), dtype=np.float64)
    speeds[1:] = np.linalg.norm(np.diff(root, axis=0), axis=-1)*30
    idle = valid & (speeds < .1)
    falls = [{"source_frame": int(t), "completed_control": int(t)+1, "derived_time_s": (int(t)+1)/30,
              "actor": int(b), "episode_id": int(state["episode_ids"][t, b])} for t, b in np.argwhere(state["terminated"])]
    if (falls != original["fall_events"] or int(static.sum()) != audit["original_obstacle_peer_channels"]["static_contact_actor_frames"]
            or int(peer_pairs.sum()) != audit["original_obstacle_peer_channels"]["peer_contact_unordered_pair_frames"]
            or int(idle.sum()) != original["pooled_valid_idle_intervals"]
            or int(valid.sum()) != original["pooled_legal_intervals"]):
        raise ValueError("Complete terminal, contact or idle counts differ from genuine original CPU audit")
    per_actor = []
    for actor in range(4):
        per_actor.append({"actor": actor, "all_controls": 6000,
                          "raw_root_error_m": summary(raw_error[:, actor]), "reference_root_error_m": summary(reference_error[:, actor]),
                          "contact_controls": int(any_contact[:, actor].sum()), "idle_intervals": int(idle[:, actor].sum())})
    tracking = {}
    masks = {"all": np.ones((6000, 4), bool), "any_contact": any_contact, "no_contact": ~any_contact,
             "peer_contact": peer_actor, "static_contact": static, "terminated": state["terminated"],
             "reset_first_poststep": state["reset_mask"], "idle_previous_interval": idle}
    for label, mask in masks.items():
        tracking[label] = {"raw_root_error_m": summary(raw_error[mask]), "reference_root_error_m": summary(reference_error[mask])}
    for frame in range(6000):
        for actor in range(4):
            row = {"run": name, "source_frame": frame, "completed_control": frame+1, "derived_time_s": (frame+1)/30,
                   "actor": actor, "episode_id": int(state["episode_ids"][frame, actor]), "plan_phase": int(phase[frame, actor]),
                   "reset_first_poststep": bool(state["reset_mask"][frame, actor]), "terminated": bool(state["terminated"][frame, actor]),
                   "truncated": bool(state["truncated"][frame, actor]), "static_contact": bool(static[frame, actor]),
                   "peer_contact": bool(peer_actor[frame, actor]), "raw_root_error_m": float(raw_error[frame, actor]),
                   "reference_root_error_m": float(reference_error[frame, actor]), "same_episode_previous_interval": bool(valid[frame, actor]),
                   "idle_previous_interval": bool(idle[frame, actor]),
                   "previous_interval_displacement_speed_mps": float(speeds[frame, actor]) if valid[frame, actor] else None,
                   "static_force_max_n": float(state["static_contact_max_force_n"][frame, actor]),
                   "peer_force_max_per_actor_n": float(np.maximum(state["peer_contact_max_force_n"][frame, actor],
                                                                   state["peer_contact_max_force_n"][frame, :, actor]).max())}
            for label, points in (("actual_root", root), ("raw_target", raw), ("reference_target", reference)):
                row[label+"_x_m"], row[label+"_y_m"] = (float(x) for x in points[frame, actor])
                row[label+"_static_cuboid_clearance_m"] = float(proxy[label]["static_cuboids"][frame, actor])
                row[label+"_static_combined_clearance_m"] = float(proxy[label]["static_combined"][frame, actor])
            if writer.fieldnames is None:
                writer.fieldnames = list(row)
                writer.writeheader()
            writer.writerow(row)
    return {"input_identities": identities, "all_actor_controls": 24000, "all_unordered_pair_controls": 36000,
            "scene": scene, "all_6000_target_phase_checks_max_abs_error": target_errors,
            "tracking": tracking, "per_actor": per_actor,
            "geometry_vs_contact": {label: {"peer_unordered_pairs": proxy_contact_table(value["peer"], peer_pairs),
                                            "static_cuboids": proxy_contact_table(value["static_cuboids"], static),
                                            "static_with_domain_boundary": proxy_contact_table(value["static_combined"], static)}
                                    for label, value in proxy.items()},
            "all_actual_contact_onsets": events(state, proxy, raw_error, reference_error, idle, valid),
            "original_contact_counts": original["original_contact_counts"], "all_actual_falls": falls,
            "contact_time_contract_from_genuine_result": {key: result["physical_contact_measurement"][key]
                                                          for key in ("source", "threshold_n", "substep_reduction", "physics_substeps_per_control", "physics_dt_s", "control_dt_s")},
            "reset_first_poststep_controls": int(state["reset_mask"].sum()), "truncated_actor_controls": int(state["truncated"].sum()),
            "original_idle_actor_intervals": int(idle.sum()), "same_episode_actor_intervals": int(valid.sum()),
            "all_failed_contact_idle_reset_frames_retained": True,
            "episode_source": "physical_states.episode_ids/reset_mask, never inferred from raw plan t0",
            "reference_target_speed_mps": summary(np.linalg.norm(state["reference_target_velocity_xy"], axis=-1)),
            "route_vertices_reached": state["route_vertex_reached"].sum(0).tolist(),
            "genuine_original_physical_zero_gate_passed": original["physical_zero_gate_passed"],
            "new_success_gate_or_promotion": False}


def main():
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run this NumPy reader with CUDA_VISIBLE_DEVICES='' explicitly")
    source, comparison = record(PRODUCER), record(COMPARISON, readonly=True)
    if source["sha256"] != PRODUCER_SHA or comparison["sha256"] != COMPARISON_SHA:
        raise ValueError("Source-bound producer or actual comparison changed")
    original = read(COMPARISON)
    output_csv, output_json = HERE / "all_actor_control_rows.csv", HERE / "actual_diagnostic.json"
    if output_csv.exists() or output_json.exists():
        raise ValueError("Preserve existing diagnostic outputs; never overwrite")
    with output_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=None)
        runs = {name: analyze(name, original["runs"][name], writer) for name in DP}
    if len({json.dumps(run["scene"], sort_keys=True) for run in runs.values()}) != 1:
        raise ValueError("Three actual runs have different geometry")
    forbidden = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
    if any(module.split(".")[0] in forbidden for module in sys.modules):
        raise ValueError("CPU diagnostic imported a model or simulator module")
    for run in runs.values():
        for expected in run["input_identities"].values():
            actual = record(expected["path"], readonly=expected["mode"] == "0o444")
            if actual != expected:
                raise ValueError("Source input changed during actual CPU observation")
    report = {"schema": "actual_tracking.same_control_contact_diagnostic.v1", "status": "completed_CPU_diagnostic",
              "execution_source": record(__file__), "producer": source, "original_comparison": comparison,
              "definition": {"time": "source frame f is post control f, completed_control=f+1, derived_time=(f+1)/30",
                             "target_phase": "pre-control replan plus phase age+1, then post-step root and both targets saved before reset",
                             "tracking": "root world XY Euclidean distance to same-phase raw/reference target, all24000 actor-controls",
                             "peer_proxy": "root/target point separation minus2*.6m, all6 unordered pairs percontrol",
                             "static_proxy": "signed root/target XY AABB distance minus.8m; cuboids and cuboids+domain separately",
                             "proxy_unsafe": "strict clearance<0; equality0 counted proxy-safe",
                             "contact": "original PhysX normal-force flag: max body-pair magnitude across4 substeps strictly>1N; floor/self excluded",
                             "peer_flags": "upper-triangle(peer_contact OR transpose), each unordered pair counted once",
                             "onset": "recorded true flag after false, atfirstcaptureframe, or across actual episode boundary; not first physical contact; windows±15controls preservechanges",
                             "idle": "same-episode previous root displacement speed<.1m/s; no resetjump treated as speed; frames retained"},
              "runs": runs, "all_actor_control_rows": record(output_csv), "all_actor_control_rows_count": 72000,
              "CPU_elapsed_seconds": time.perf_counter()-started,
              "limits": ["Target/root proxy is sampled at control end; contact summarizes the wholecontrol's4substeps.",
                         "Peer/static per-substep force series and original body-pair force matrices are not independently archived.",
                         "Root points and fixed discs are not articulated body geometry or force reconstruction.",
                         "Simultaneous contacts and every fall/reset/idle frame remain in their original denominators.",
                         "Onset windows may overlap and are boundary-clipped; their rows are not independent whole-run denominators.",
                         "S10 and S1 differ in RNG consumption and closed-loop histories; this is an observed comparison, not paired causality.",
                         "No geometric association establishes a fall/contact cause or equivalence to the paper's unknown reducers."],
              "causality_established": False, "weights_read_or_models_loaded": False, "GPU_used": False,
              "SDK_started": False, "sealed34_outputs_read": False, "default_promotion": False}
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "CPU_elapsed_seconds": report["CPU_elapsed_seconds"],
                      "runs": {name: {"reference_error": run["tracking"]["all"]["reference_root_error_m"],
                                      "contact_onsets": len(run["all_actual_contact_onsets"])} for name, run in runs.items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
