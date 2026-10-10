"""Read-only CPU pairing audit; does not replace any benchmark or style metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/style_improvement_20261002"
RUNS = ROOT / "output/diffusion_amp"
NEGATIVE_TRAIN = RUNS / "amp_style_amass2848_backward_penalty_v1"
STRATIFIED_TRAIN = RUNS / "amp_style_amass2848_stratified_experts_v1"
NEGATIVE_TRACK = RUNS / "motion_tracking_amass2848_backward_penalty_full_validation_v1"
STRATIFIED_TRACK = RUNS / "motion_tracking_amass2848_stratified_experts_full_validation_v1"
POSE_FIELDS = (
    "root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
    "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel",
)
PAIR_FIELDS = (
    *("reference_" + field for field in POSE_FIELDS), "reference_frame", "reference_time_s",
    "metric_frame", "native_reset_frame", "trajectory_xy", "waypoint_query_phase_frame",
    "waypoint_held_at_clip_end", "action_valid",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read(path):
    return json.loads(Path(path).read_text())


def differences(a, b):
    missing = {"missing": True}
    return {key: {"negative": a.get(key, missing), "stratified": b.get(key, missing)}
            for key in sorted(set(a) | set(b)) if a.get(key, missing) != b.get(key, missing)}


def archive_verification(path):
    manifest = read(path)
    archive = manifest["archive"]
    if isinstance(archive, str):
        expected = {"path": archive, "bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]}
    else:
        expected = archive
    archive_path = ROOT / expected["path"]
    actual = record(archive_path)
    assert actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"]
    entries = {entry["path"]: entry for entry in manifest["files"]}
    contents = {}
    with tarfile.open(archive_path) as stream:
        members = stream.getmembers()
        assert len(members) == len(entries)
        for member in members:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name in entries and member.name not in contents
            data = stream.extractfile(member).read()
            entry = entries[member.name]
            assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
            contents[member.name] = data
    return {"manifest": record(path), "archive": actual, "verified_regular_members": len(contents)}, entries, contents


def training_summary(directory, report):
    metrics_path = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(rows) == 1000
    assert [row["update_count"] for row in rows] == list(range(1, 1001))
    assert [row["physical_transitions"] for row in rows] == [8192 * n for n in range(1, 1001)]
    assert report["physical_transition_count"] == 8192000 and report["actual_control_steps"] == 32000
    assert rows[-1]["fall_resets"] == report["fall_resets"]
    assert rows[-1]["time_limit_resets"] == report["time_limit_resets"]
    phases = []
    fields = ("reward_mean", "trajectory_rewards_mean", "amp_rewards_mean", "direction_rewards_mean",
              "energy_rewards_mean", "backwards_fraction", "forward_velocity_mean",
              "disc_expert_probability", "disc_live_policy_probability", "value_loss", "approximate_kl")
    previous_falls = previous_timeouts = 0
    for start in (0, 250, 500, 750):
        selected = rows[start:start + 250]
        end = selected[-1]
        phases.append({"updates_inclusive": [start + 1, start + 250],
                       "physical_transitions": 250 * 8192,
                       "fall_resets": end["fall_resets"] - previous_falls,
                       "time_limit_resets": end["time_limit_resets"] - previous_timeouts,
                       "arithmetic_update_means": {key: float(np.mean([r[key] for r in selected])) for key in fields}})
        previous_falls, previous_timeouts = end["fall_resets"], end["time_limit_resets"]
    falls, timeouts = report["fall_resets"], report["time_limit_resets"]
    return {"result": record(directory / "result.json"), "metrics": record(metrics_path),
            "checkpoint": record(directory / "step_1000.pt"), "updates": 1000,
            "physical_transitions": 8192000, "actual_control_steps": 32000,
            "fall_resets": falls, "time_limit_resets": timeouts,
            "fall_resets_per_100000_transitions": falls / 8192000 * 100000,
            "fall_fraction_of_completed_episodes": falls / (falls + timeouts),
            "timeouts_are_not_tracking_successes": True, "phases": phases}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    negative, stratified = (read(p / "result.json") for p in (NEGATIVE_TRAIN, STRATIFIED_TRAIN))
    assert negative["status"] == stratified["status"] == "passed"
    config_diff = differences(negative["training_config"], stratified["training_config"])
    assert set(config_diff) == {"expert_sampling"}
    arg_diff = differences(negative["arguments"], stratified["arguments"])
    assert set(arg_diff) == {"config", "output", "measure_ground_support", "ground_support_threshold_n"}
    assert stratified["arguments"]["measure_ground_support"] is False
    common_keys = ("initialization", "curriculum", "physics", "power_measurement", "runtime_execution",
                   "runtime_numerics", "amp_reward_timestamp", "planning_clock", "manifest_gate")
    assert all(negative[key] == stratified[key] for key in common_keys)
    assert set(differences(negative["provenance"], stratified["provenance"])) == {"expert_sampling"}
    old_summary, old_metrics = training_summary(NEGATIVE_TRAIN, negative)
    new_summary, new_metrics = training_summary(STRATIFIED_TRAIN, stratified)
    assert new_summary["checkpoint"]["sha256"] == "93c25e567d88cc4f4c639b1dea5e14494f0ef51dd304661fb23d36690e55328d"
    assert old_summary["checkpoint"]["sha256"] == "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
    old_archive, old_entries, _ = archive_verification(DOCS / "backward_penalty_training_source.json")
    new_archive, new_entries, new_contents = archive_verification(DOCS / "stratified_expert_training_source_v1.json")
    common_paths = sorted(set(old_entries) & set(new_entries))
    changed = [p for p in common_paths if old_entries[p]["sha256"] != new_entries[p]["sha256"]]
    unchanged = [p for p in common_paths if p not in changed]
    # Core optimizer/networks/curriculum/state conversion are frozen byte-identical.
    asserted_paths = ("CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/amp/network.py",
                      "CrowdSim/humanoid/training/rewards.py", "CrowdSim/humanoid/training/trajectories.py",
                      "CrowdSim/humanoid/amass_corpus.py", "CrowdSim/humanoid/amp/observations.py")
    for path in asserted_paths:
        assert path in unchanged
    first_diff = differences(old_metrics[0], new_metrics[0])
    substantive_first_same = [k for k in old_metrics[0] if k not in first_diff and not k.endswith("wall_s")]
    old_track, new_track = (read(p / "result.json") for p in (NEGATIVE_TRACK, STRATIFIED_TRACK))
    assert old_track["status"] == new_track["status"] == "passed"
    assert len(old_track["sequences"]) == len(new_track["sequences"]) == 22
    track_diff = differences(old_track["arguments"], new_track["arguments"])
    assert set(track_diff) == {"checkpoint", "checkpoint_sha256", "output"}
    for key in ("protocol", "physics", "reference_velocity_contract", "layout"):
        assert old_track[key] == new_track[key]
    source_diff = differences(old_track["selected_source_sha256"], new_track["selected_source_sha256"])
    assert set(source_diff) == {"CrowdSim/humanoid/amp_world.py"}
    assert source_diff["CrowdSim/humanoid/amp_world.py"]["negative"] == old_entries["CrowdSim/humanoid/amp_world.py"]["sha256"]
    assert source_diff["CrowdSim/humanoid/amp_world.py"]["stratified"] == new_entries["CrowdSim/humanoid/amp_world.py"]["sha256"]
    assert b"def configure_expert_sampling" in new_contents["CrowdSim/humanoid/amp_world.py"]
    paired = []
    for old, new in zip(old_track["sequences"], new_track["sequences"]):
        for key in ("motion_id", "source_path", "tracking_window", "observed_metric_steps", "initial_translation_xyz_m"):
            assert old[key] == new[key]
        paths = [Path(row["capture"]) for row in (old, new)]
        for path, row, directory in zip(paths, (old, new), (NEGATIVE_TRACK, STRATIFIED_TRACK)):
            assert path.parent.resolve() == (directory / "captures").resolve()
            assert sha256(path) == row["capture_sha256"]
        with np.load(paths[0], allow_pickle=False) as aa, np.load(paths[1], allow_pickle=False) as bb:
            assert set(aa.files) == set(bb.files)
            for key in PAIR_FIELDS:
                assert aa[key].dtype == bb[key].dtype and np.array_equal(aa[key], bb[key]), (old["motion_id"], key)
            for key in POSE_FIELDS:
                assert np.array_equal(aa[key][0], bb[key][0]), (old["motion_id"], "initial " + key)
            frames = len(aa["reference_frame"])
            assert not aa["metric_frame"][0] and aa["metric_frame"][1:].all()
            assert np.all(np.diff(aa["reference_frame"]) == 1)
            np.testing.assert_allclose(np.diff(aa["reference_time_s"]), 1 / 30, rtol=0, atol=1e-10)
        paired.append({"motion_id": old["motion_id"], "source_path": old["source_path"],
                       "metric_frames": frames - 1, "paired_reference_fields_bitwise_equal": list(PAIR_FIELDS),
                       "paired_initial_actual_fields_bitwise_equal": list(POSE_FIELDS),
                       "negative_capture": record(paths[0]), "stratified_capture": record(paths[1]),
                       "negative_success": old["tracking_success"], "stratified_success": new["tracking_success"],
                       "negative_fall": old["sequence_had_fall"], "stratified_fall": new["sequence_had_fall"]})
    output = {"schema": "navisaaclab.stratified_expert_comparison_audit.v1", "status": "passed",
              "CPU_only": True, "Kit_started": False, "GPU_used": False,
              "business_or_evaluator_modified": False, "script": record(Path(__file__)),
              "training_config_differences": config_diff, "training_argument_differences": arg_diff,
              "identical_training_contracts": list(common_keys), "dataset": negative["provenance"]["dataset"],
              "initialization": negative["initialization"], "curriculum": negative["curriculum"],
              "training": {"negative": old_summary, "stratified": new_summary},
              "first_rollout_equal_logged_metrics": substantive_first_same,
              "first_update_metric_differences": first_diff,
              "source_archives": {"negative": old_archive, "stratified": new_archive,
                                  "unchanged_common_files": unchanged, "changed_common_files": changed,
                                  "added_files": sorted(set(new_entries) - set(old_entries)),
                                  "byte_equal_core_paths_asserted": list(asserted_paths),
                                  "current_tree_identity_asserted": False},
              "tracking_argument_differences": track_diff, "tracking_selected_source_differences": source_diff,
              "tracking_results": {"negative": record(NEGATIVE_TRACK / "result.json"),
                                   "stratified": record(STRATIFIED_TRACK / "result.json")},
              "paired_clip_count": len(paired), "paired_metric_frames": sum(p["metric_frames"] for p in paired),
              "paired_clips": paired,
              "limitations": [
                  "The only effective training configuration change is the declared expert prior. Source archives are not wholly identical: explicit sampler and disabled optional capture code were added.",
                  "Legacy expert batches consume corpus RNG also used by resets. The new sampler has private RNG, so later training reset choices are not bitwise paired despite unchanged reset rules/seed/data.",
                  "Same initial rollout logged rewards/PPO values and exact benchmark reset states do not prove stepwise equivalence of separate training runs.",
                  "One seed and22 heldout clips do not establish statistical significance or reproduce author training/data/scene IDs.",
                  "All benchmark fall frames and complete-sequence failures remain included. Local diagnostics do not replace the primary world-space metric.",
                  "No MMD/style evaluation is recalculated here; the separate frozen-style report is authoritative."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "paired_clip_count": len(paired),
                      "paired_metric_frames": output["paired_metric_frames"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
