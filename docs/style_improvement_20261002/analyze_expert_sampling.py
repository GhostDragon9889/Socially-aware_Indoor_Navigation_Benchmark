"""Read-only CPU audit of existing AMASS samplers and measured AMP runs."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("stratified_expert_sampling_review.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(source):
    return source.split("/processed/smpl/")[1].split("/")[0]


def yaw(q):
    return np.arctan2(2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1]),
                      1-2*(q[..., 1]**2+q[..., 2]**2))


def reference_distribution(manifest_path, split):
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["splits"][split]
    path = Path(entry["path"])
    if not path.is_absolute():
        path = manifest_path.parent/path
    assert path.stat().st_size == entry["bytes"]
    data = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    rows = []
    for record in entry["clips"]:
        index = record["motion_id"]
        count = int(data["motion_num_frames"][index])
        assert count == record["frames"]
        start = int(data["length_starts"][index])
        velocity = data["gvs"][start:start+count, 0, :2].numpy()
        q = data["grs"][start:start+count, 0].numpy()
        heading = np.unwrap(yaw(q))
        speed = np.linalg.norm(velocity, axis=-1)
        forward = velocity[:, 0]*np.cos(heading)+velocity[:, 1]*np.sin(heading)
        lateral = np.abs(-velocity[:, 0]*np.sin(heading)+velocity[:, 1]*np.cos(heading))
        centers = slice(9, count-1)
        dt = float(data["motion_dt"][index])
        yaw_rate = (heading[9:count-1]-heading[:count-10])/(9*dt)
        hist_speed = np.median(np.lib.stride_tricks.sliding_window_view(speed, 10)[:count-10], -1)
        hist_side = np.median(np.lib.stride_tricks.sliding_window_view(lateral/np.maximum(speed, .05), 10)[:count-10], -1)
        assert len(yaw_rate) == count-10
        stationary = hist_speed < .12
        running = (~stationary) & (hist_speed >= 1.8)
        side = (~stationary) & (~running) & (hist_side > .55)
        turning = (~stationary) & (~running) & (~side) & (np.abs(yaw_rate) >= 2/3)
        walking = ~(stationary | running | side | turning)
        window_mass = {name: float(mask.mean()) for name, mask in (
            ("standing_low_speed", stationary), ("running_high_speed", running),
            ("sideways", side), ("turning_at_least_0_2rad_in_0_3s", turning),
            ("other_moving", walking))}
        values = {"speed_mean_m_s": float(speed[centers].mean()),
                  "abs_yaw_rate_0_3s_mean_rad_s": float(np.abs(yaw_rate).mean()),
                  "negative_forward_fraction": float((forward[centers] < 0).mean()),
                  "backward_moving_fraction": float(((forward[centers] < -.05) & (speed[centers] >= .12)).mean()),
                  "speed_above2_fraction": float((speed[centers] > 2).mean()),
                  "speed_above3_fraction": float((speed[centers] > 3).mean()),
                  "standing_speed_fraction": float((speed[centers] < .12).mean())}
        parent = record.get("parent_source_path", record["source_path"])
        rows.append({"motion_id": index, "source_path": record["source_path"],
                     "parent": parent, "dataset": record.get("dataset", dataset(parent)),
                     "subject": record.get("subject_namespace", "/".join(parent.split("/processed/smpl/")[1].split("/")[:2])),
                     "style_class": record.get("style_class", "not_declared"),
                     "frames": count, "eligible_centers": count-10,
                     "values": values, "window_mass": window_mass})
    parents = Counter(row["parent"] for row in rows)
    results = {}
    for name, weights in (("current_uniform_clip", [1/len(rows)]*len(rows)),
                          ("uniform_parent_then_uniform_segment", [1/(len(parents)*parents[row["parent"]]) for row in rows]),
                          ("uniform_eligible_center", [row["eligible_centers"]/sum(r["eligible_centers"] for r in rows) for row in rows])):
        stats = {k: sum(w*row["values"][k] for w, row in zip(weights, rows, strict=True)) for k in rows[0]["values"]}
        stats["short_history_window_proxy_mass"] = {k: sum(w*row["window_mass"][k] for w, row in zip(weights, rows, strict=True)) for k in rows[0]["window_mass"]}
        for field in ("dataset", "style_class"):
            masses = defaultdict(float)
            for row, weight in zip(rows, weights, strict=True):
                masses[row[field]] += weight
            stats[field+"_probability"] = dict(sorted(masses.items()))
        results[name] = stats
    subjects = Counter(row["subject"] for row in rows)
    return {"manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha(manifest_path),
            "source": str(path), "source_bytes_stat": path.stat().st_size,
            "source_sha256_declared_not_recomputed": entry["sha256"],
            "clips": len(rows), "parent_sources": len(parents), "subject_namespaces": len(subjects),
            "total_control_frames": sum(row["frames"] for row in rows),
            "eligible_expert_centers": sum(row["eligible_centers"] for row in rows),
            "parent_segment_count_histogram": dict(sorted(Counter(parents.values()).items())),
            "top_subjects_by_uniform_clip_mass": [{"subject": subject, "clips": count, "probability": count/len(rows)} for subject, count in subjects.most_common(10)],
            "dataset_clip_counts": dict(Counter(row["dataset"] for row in rows)),
            "style_clip_counts": dict(Counter(row["style_class"] for row in rows)),
            "per_center_probability_ratio_max_min": max(row["eligible_centers"] for row in rows)/min(row["eligible_centers"] for row in rows),
            "distributions": results}


def training_metrics(name):
    path = ROOT/"output/diffusion_amp"/name/"metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result_path = path.with_name("result.json")
    result = json.loads(result_path.read_text())
    keys = ["amp_rewards_mean", "trajectory_rewards_mean", "direction_rewards_mean",
            "energy_rewards_mean", "forward_velocity_mean", "backwards_fraction",
            "disc_expert_probability", "disc_policy_probability", "approximate_kl",
            "optimizer_minibatches", "policy_target_kl_stopped"]
    windows = []
    previous_falls = 0
    previous_transitions = 0
    for start, stop in ((1, 100), (101, 250), (251, 500), (501, 750), (751, 1000)):
        selected = [row for row in rows if start <= row["update_count"] <= stop]
        end = selected[-1]
        delta_falls = end["fall_resets"]-previous_falls
        delta_transitions = end["physical_transitions"]-previous_transitions
        windows.append({"updates": [start, stop], "mean": {k: float(np.mean([row[k] for row in selected])) for k in keys if k in selected[0]},
                        "fall_delta": delta_falls, "physical_transition_delta": delta_transitions,
                        "falls_per100000_transitions": delta_falls/delta_transitions*100000})
        previous_falls, previous_transitions = end["fall_resets"], end["physical_transitions"]
    return {"result": str(result_path.relative_to(ROOT)), "result_sha256": sha(result_path),
            "metrics": str(path.relative_to(ROOT)), "metrics_sha256": sha(path),
            "updates": len(rows), "physical_transitions": result["physical_transition_count"],
            "fall_resets": result["fall_resets"], "time_limit_resets": result["time_limit_resets"],
            "fall_fraction_among_completed_resets": result["fall_resets"]/result["resets"],
            "falls_per100000_transitions": result["fall_resets"]/result["physical_transition_count"]*100000,
            "initialization": result["initialization"], "training_config": result["training_config"],
            "arguments": result["arguments"], "windows": windows, "final": rows[-1]}


def tracking(name):
    result_path = ROOT/"output/diffusion_amp"/name/"result.json"
    result = json.loads(result_path.read_text())
    details = []
    totals = Counter()
    for sequence in result["sequences"]:
        with np.load(sequence["capture"], allow_pickle=False) as capture:
            mask = capture["metric_frame"].astype(bool)
            p, rp = capture["root_pos"][mask, 0], capture["reference_root_pos"][mask, 0]
            q, rq = capture["root_rot"][mask, 0], capture["reference_root_rot"][mask, 0]
            v = capture["root_vel"][mask, 0]
            heading = yaw(q)
            forward = v[:, 0]*np.cos(heading)+v[:, 1]*np.sin(heading)
            yaw_error = np.abs(np.arctan2(np.sin(heading-yaw(rq)), np.cos(heading-yaw(rq))))
            dot = np.clip(np.abs((q*rq).sum(-1)), 0, 1)
            root_rotation_error = 2*np.arccos(dot)
            root_error = np.linalg.norm(p-rp, axis=-1)
            local = capture["rigid_body_pos"][mask, 0]-p[:, None]
            ref_local = capture["reference_rigid_body_pos"][mask, 0]-rp[:, None]
            local_error = np.linalg.norm(local-ref_local, axis=-1).mean(-1)
            values = {"root_position_mean_error_m": float(root_error.mean()),
                      "root_rotation_mean_error_rad": float(root_rotation_error.mean()),
                      "root_yaw_mean_error_rad": float(yaw_error.mean()),
                      "root_yaw_error_above90deg_fraction": float((yaw_error > np.pi/2).mean()),
                      "body_root_translation_removed_mean_error_m": float(local_error.mean()),
                      "backwards_fraction": float((forward < 0).mean()),
                      "forward_velocity_mean_m_s": float(forward.mean())}
            for key, value in values.items():
                totals[key] += value*len(p)
            totals["frames"] += len(p)
        base = Path(sequence["source_path"]).name
        group = ("backwards_named" if re.search("backwards", base, re.I) else
                 "jog_run_named" if re.search("jog|run", base, re.I) else
                 "walk_named" if re.search("walk", base, re.I) else "other_named")
        details.append({"motion_id": sequence["motion_id"], "source_path": sequence["source_path"],
                        "group": group, "observed_steps": sequence["observed_metric_steps"],
                        "tracking_success": sequence["tracking_success"], "had_fall": sequence["sequence_had_fall"],
                        "position_error_m": sequence["mean_body_position_error_m"],
                        "rotation_error_rad": sequence["mean_body_rotation_error_rad"], **values})
    return {"result": str(result_path.relative_to(ROOT)), "result_sha256": sha(result_path),
            "checkpoint_sha256": result["arguments"]["checkpoint_sha256"], "aggregate": result["aggregate"],
            "additional_frame_weighted_metrics": {key: value/totals["frames"] for key, value in totals.items() if key != "frames"},
            "sequences": details}


def main():
    torch.set_num_threads(4)
    old = ROOT/"data/diffusion_amp/amass_manifest.json"
    new = ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    result = {"schema": "navisaaclab.stratified_expert_sampling_review.v1", "date": "2026-10-02",
              "cpu_only": True, "kit_started": False, "gpu_used": False,
              "reference_sampling_semantics": "uniform selected clip; uniform center9..frames-2; newest-first history10; same corpus private RNG shared by physical resets and expert examples",
              "short_window_proxy_rules": {"history_s": .3, "standing_speed_m_s": .12,
                  "running_speed_m_s": 1.8, "lateral_fraction": .55, "turn_net_yaw_delta_rad": .2,
                  "source": "diagnostic proxy thresholds; not author labels or training edits"},
              "references": {"old_train319": reference_distribution(old, "train"),
                             "new_train2848": reference_distribution(new, "train"),
                             "heldout_validation22": reference_distribution(new, "validation")},
              "training": {name: training_metrics(name) for name in ("amp_independent_curriculum_v1", "amp_style_amass2848_child_v1")},
              "tracking": {name: tracking(name) for name in ("motion_tracking_legacy_full_validation_v1", "motion_tracking_amass2848_child_full_validation_v1")},
              "source_files": {p: sha(ROOT/p) for p in ("CrowdSim/humanoid/amass_corpus.py", "CrowdSim/humanoid/amp_world.py", "CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/training/rewards.py", "CrowdSim/humanoid/training/trajectories.py", "configs/diffusion_amp/amp_style_amass_replay.yaml")}}
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(OUT), "status": "measured", "reference_counts": {k: v["clips"] for k, v in result["references"].items()}}, indent=2))


if __name__ == "__main__":
    main()
