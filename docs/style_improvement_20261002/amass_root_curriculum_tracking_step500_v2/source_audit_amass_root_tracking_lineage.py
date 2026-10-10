"""CPU-only500/750 lineage diagnostics; preserves the completed1000 v1 audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
RUNS = ROOT / "output/diffusion_amp"
INITIAL_FIELDS = (
    "root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
    "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "amp_history",
)
CLOCK_FIELDS = (
    "reference_frame", "reference_time_s", "metric_frame", "native_reset_frame", "episode_id",
    "trajectory_xy", "waypoint_query_phase_frame", "waypoint_held_at_clip_end", "action_valid",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read(path):
    return json.loads(Path(path).read_text())


def differences(left, right):
    missing = {"missing": True}
    return {key: {"baseline": left.get(key, missing), "amass_root": right.get(key, missing)}
            for key in sorted(set(left) | set(right)) if left.get(key, missing) != right.get(key, missing)}


def helper():
    path = DOCS / "decompose_motion_tracking.py"
    spec = importlib.util.spec_from_file_location("amass_root_frozen_decomposition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_archive(path):
    manifest = read(path)
    expected = manifest.get("source_archive", manifest.get("archive"))
    if isinstance(expected, str):
        expected = {"path": expected, "bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]}
    require(isinstance(expected, dict), "Frozen source manifest lacks its archive record")
    actual = record(ROOT / expected["path"])
    require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"], "Source archive hash/size mismatch")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    require(len(entries) == len(manifest["files"]), "Duplicate source manifest members")
    observed = set()
    with tarfile.open(actual["path"]) as stream:
        for member in stream.getmembers():
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute() and ".." not in name.parts,
                    "Source archive contains a non-regular or unsafe member")
            require(member.name in entries and member.name not in observed, "Unexpected/duplicate source member")
            expected_file = entries[member.name]
            require(member.size == expected_file["bytes"], "Source member size mismatch")
            body = stream.extractfile(member).read()
            require(hashlib.sha256(body).hexdigest() == expected_file["sha256"], "Source member hash mismatch")
            observed.add(member.name)
    require(observed == set(entries), "Source archive is incomplete")
    return {"manifest": record(path), "archive": actual, "verified_regular_members": len(entries)}, entries


def require_full_validation(result):
    args = result["arguments"]
    require(result["status"] == "passed" and args["split"] == "validation", "Only completed validation captures are accepted; no test split")
    require(all(args[key] is None for key in ("motion_ids", "max_clips", "max_steps")), "Validation subset/step caps are forbidden")
    rows = result["sequences"]
    require(len(rows) == 22 and len({row["motion_id"] for row in rows}) == 22, "Exactly22 distinct validation clips are required")
    require(all(row["complete_clip"] and row["in_full_sequence_success_denominator"]
                and row["observed_metric_steps"] == row["expected_control_steps"] for row in rows),
            "Every full real clip must remain in the denominator")
    return {row["motion_id"]: row for row in rows}


def exact_capture_pair(left, right):
    """Clock/targets and complete native initial state must match, not learned future states."""
    left_keys, right_keys = set(left), set(right)
    require(left_keys == right_keys, "Paired capture schemas differ")
    keys = sorted({key for key in left_keys if key.startswith("reference_")} | set(CLOCK_FIELDS))
    require(set(keys + list(INITIAL_FIELDS)) <= left_keys, "Required pair fields are missing")
    for key in keys:
        require(left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key]),
                f"Paired reference/clock field differs: {key}")
    for key in INITIAL_FIELDS:
        require(left[key].dtype == right[key].dtype and np.array_equal(left[key][0], right[key][0]),
                f"Paired initial state/history differs: {key}")
    return {"reference_and_clock_fields_bitwise_equal": keys,
            "initial_physical_state_and_history_bitwise_equal": list(INITIAL_FIELDS)}


def fall_intervals(capture, sequence):
    frames = len(capture["reference_frame"])
    mask = capture["fall_mask"]
    require(mask.dtype == np.bool_ and mask.shape == (frames,) and not mask[0], "Invalid initial/physical fall mask")
    indices = np.flatnonzero(mask)
    require(indices.tolist() == sequence["fall_capture_frames"], "Recorded fall frames differ from result")
    starts = indices[np.r_[True, np.diff(indices) > 1]] if len(indices) else indices
    stops = indices[np.r_[np.diff(indices) > 1, True]] if len(indices) else indices
    require(len(starts) == sequence["fall_onset_events"] and bool(len(indices)) == sequence["sequence_had_fall"],
            "Fall onset/sequence denominator mismatch")
    t = capture["reference_time_s"]
    return [{"capture_frames_inclusive": [int(start), int(stop)],
             "source_frames_inclusive": [int(capture["reference_frame"][start]), int(capture["reference_frame"][stop])],
             "elapsed_seconds_from_reset_inclusive": [float(t[start] - t[0]), float(t[stop] - t[0])],
             "observed_fall_frames": int(stop - start + 1),
             "root_xyz_at_onset_m": capture["root_pos"][start, 0].tolist()}
            for start, stop in zip(starts, stops, strict=True)]


def capture_diagnostics(path, sequence, decomposition, parents):
    with np.load(path, allow_pickle=False) as archive:
        capture = {key: archive[key] for key in archive.files}
    arrays, _ = decomposition.diagnostic_arrays(capture, parents)
    metric = capture["metric_frame"]
    intervals = fall_intervals(capture, sequence)
    first = intervals[0]["capture_frames_inclusive"][0] if intervals else None
    summaries = {}
    for label, lower, upper in (("one_second_before_first_fall", None if first is None else first - 30, first),
                                ("one_second_from_first_fall", first, None if first is None else first + 30)):
        if first is None:
            summaries[label] = None
            continue
        selected = metric & (np.arange(len(metric)) >= lower) & (np.arange(len(metric)) < upper)
        summaries[label] = decomposition.summarize({key: value[selected] for key, value in arrays.items()}) if selected.any() else None
    return {"fall_intervals": intervals, "first_fall_capture_frame": first,
            "all_fall_frames_retained_in_primary_metric": True, "near_first_fall": summaries}


def verified_checkpoint(result):
    actual = record(result["arguments"]["checkpoint"])
    require(actual["sha256"] == result["arguments"]["checkpoint_sha256"], "Actual controller SHA differs from captured argument")
    return actual


def check_saved_payload(state, training, step):
    args = training["arguments"]
    n, rollout = args["num_envs"], args["rollout_steps"]
    require(state["format"] == "navisaaclab_method_reconstruction_training"
            and state["physical_runner_state_included"] is True, "Incomplete saved training format")
    require(type(state["update_count"]) is int and state["update_count"] == step, "Actual saved update counter differs from selected checkpoint")
    require(type(state["physical_transition_count"]) is int and state["physical_transition_count"] == step * n * rollout,
            "Saved cumulative physical transition budget is inconsistent")
    runner = state["runner_state"]
    require(isinstance(runner, dict) and runner["control_step"] == step * rollout, "Saved cumulative control clock is inconsistent")
    require(all(key in state for key in ("model", "optimizer", "discriminator_optimizer", "normalizers", "rng", "policy_replay")),
            "Checkpoint is missing full training resume sections")
    require(all(key in state["rng"] for key in ("python", "numpy", "torch_cpu", "torch_cuda")), "Checkpoint is missing saved RNG sections")
    actual, expected = dict(state["config"]), dict(training["training_config"])
    for value in (actual, expected):
        value.setdefault("expert_sampling", None)
        # The physical action decoder saves its mask as a tuple; JSON reports
        # necessarily encode the identical elements as an array/list.
        if isinstance(value.get("action_mask"), tuple):
            value["action_mask"] = list(value["action_mask"])
    require(actual == expected, "Checkpoint training config differs from completed run")
    return {"update_count": step, "physical_transition_count": state["physical_transition_count"],
            "control_step": runner["control_step"], "num_envs": n, "rollout_steps": rollout,
            "full_resume_sections_present": True, "saved_config_matches_training": True,
            "serialization_normalization": "action_mask tuple/list only; identical values required",
            "logical_state_only_not_physx_cache_identity": True}


def ledger_entry(ledger, step, checkpoint):
    require(ledger["status"] == "passed", "Immutable checkpoint ledger is incomplete")
    rows = [row for row in ledger["snapshots"] if row["update_count"] == step]
    require(len(rows) == 1, "Selected sealed checkpoint is missing/duplicated")
    row = rows[0]
    require(Path(row["path"]).resolve() == Path(checkpoint["path"]).resolve()
            and row["bytes"] == checkpoint["bytes"] and row["sha256"] == checkpoint["sha256"],
            "Actual checkpoint differs from its independently sealed ledger entry")
    require(row["complete_resume_payload_checked"] and row["active_plans_match_actual_bank"], "Seal lacks curriculum admission evidence")
    return {key: row[key] for key in ("update_count", "physical_transition_count", "control_step", "path", "bytes", "sha256",
                                     "complete_resume_payload_checked", "active_plans_match_actual_bank", "physical_bitwise_equivalence_claim")}


def verify_lineage(checkpoint, training, step, mode, ledger_path):
    import torch

    torch.set_num_threads(4)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Safe checkpoint inspection is CPU-only")
    state = torch.load(checkpoint["path"], weights_only=True, map_location="cpu", mmap=True)
    saved = check_saved_payload(state, training, step)
    del state
    ledger = read(ledger_path)
    args = training["arguments"]
    if mode == "original_snapshot":
        require(not args["resume"] and training["update_count"] == 1000 and args["updates"] == 1000,
                "Original snapshot must belong to the completed fresh1000 trajectory")
        sealed = ledger_entry(ledger, step, checkpoint)
        require(Path(ledger["training_output_read_only"]).resolve() == Path(args["output"]).resolve(),
                "Snapshot ledger belongs to another training run")
        return {"mode": mode, "evaluated": checkpoint, "safe_weights_only_saved_counters": saved,
                "independent_ledger": record(ledger_path), "sealed_entry": sealed,
                "selected_budget_updates": step, "equal_budget_to_baseline1000": step == 1000}
    require(mode == "continued_from500" and step == 750, "Only the explicit500+250 continuation is accepted")
    require(Path(args["output"]).resolve() != Path(ledger["training_output_read_only"]).resolve(),
            "Continuation must have a new independent training output")
    parent = record(args["resume"])
    sealed = ledger_entry(ledger, 500, parent)
    parent_state = torch.load(parent["path"], weights_only=True, map_location="cpu", mmap=True)
    parent_saved = check_saved_payload(parent_state, training, 500)
    del parent_state
    require(args["updates"] == 250 and training["update_count"] == 750
            and training["actual_control_steps"] == 250 * args["rollout_steps"], "Continuation is not250 actual additional updates")
    terminal = record(Path(args["output"]).resolve() / "last.pt")
    require(terminal["sha256"] == checkpoint["sha256"] and terminal["bytes"] == checkpoint["bytes"],
            "Evaluated750 differs from the completed continuation terminal checkpoint")
    metrics_path = Path(args["output"]).resolve() / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    require([row["update_count"] for row in metrics] == list(range(501, 751)), "Continuation update ledger is not contiguous501..750")
    require([row["physical_transitions"] for row in metrics] == [k * args["num_envs"] * args["rollout_steps"] for k in range(501, 751)],
            "Continuation transition ledger is inconsistent")
    return {"mode": mode, "evaluated": checkpoint, "safe_weights_only_saved_counters": saved,
            "independently_sealed_parent500": sealed, "parent_saved_counters": parent_saved,
            "independent_parent_ledger": record(ledger_path), "terminal_checkpoint": terminal,
            "additional_stage_metrics": record(metrics_path), "additional_updates": 250,
            "additional_physical_transitions": 250 * args["num_envs"] * args["rollout_steps"],
            "selected_budget_updates": 750, "equal_budget_to_baseline1000": False,
            "is_original_trajectory750": False, "physical_resume_bitwise_guarantee": False}


def subset_primary(rows, selected_ids):
    selected = [row for mid, row in rows.items() if mid in selected_ids]
    frames = sum(row["observed_metric_steps"] for row in selected)
    if not selected:
        return {"clips": 0, "metric_frames": 0, "motion_ids": [], "per_clip_mean": None, "frame_weighted_mean": None}
    keys = ("mean_body_position_error_m", "mean_body_rotation_error_rad")
    return {"clips": len(selected), "metric_frames": frames, "motion_ids": sorted(selected_ids),
            "sequences_with_fall": sum(row["sequence_had_fall"] for row in selected),
            "per_clip_mean": {key: float(np.mean([row[key] for row in selected])) for key in keys},
            "frame_weighted_mean": {key: sum(row[key] * row["observed_metric_steps"] for row in selected) / frames for key in keys}}


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def report_markdown(output):
    lines = ["# AMASS root 课程：完整22段验证误差分解", "",
             "本报告读取既有真实物理捕获；CPU处理，没有启动SDK/GPU或重算官方style/MMD。",
             "所有22段和跌倒帧均保留在主指标中，未来XY tracking不等于learned Diffusion导航。", "",
             "## 主指标（完整片段、同一相位）", "",
             f"| 指标 | negative-only1000 | AMASS-root{output['candidate_step']} |", "| --- | --- | --- |"]
    labels = {"tracking_successes": "成功片段 /22", "sequences_with_fall": "有跌倒片段 /22",
              "observed_metric_steps": "真实poststep总帧",
              "frame_weighted_mean_body_position_error_m": "全24body位置误差 m",
              "frame_weighted_mean_body_rotation_error_rad": "全24body旋转误差 rad"}
    for key, label in labels.items():
        values = [output["aggregate"][name][key] for name in ("baseline", "amass_root")]
        lines.append(f"| {label} | {values[0]:.6g} | {values[1]:.6g} |")
    lines += ["", "## 同帧诊断（不替换世界坐标主指标）", "",
              f"| 平均值 | negative-only1000 | AMASS-root{output['candidate_step']} |", "| --- | --- | --- |"]
    diagnostic_keys = {"root_xy_drift_m": "root XY m", "root_yaw_abs_wrapped_error_rad": "root yaw rad",
                       "frame_mean_rootcenter_heading_body_position_error_m": "各自root/heading局部body位置 m",
                       "frame_mean_heading_body_orientation_error_rad": "heading局部body旋转 rad",
                       "frame_mean_parent_relative_joint_orientation_error_rad": "parent局部关节旋转 rad"}
    for key, label in diagnostic_keys.items():
        values = [output["same_frame_diagnostics"][name][key]["mean"] for name in ("baseline", "amass_root")]
        lines.append(f"| {label} | {values[0]:.6g} | {values[1]:.6g} |")
    lines += ["", "局部诊断没有时间best-match，不能把heading移除后的下降当作独立因果比例。",
              "速度区间保持reference_root_vel XY范数的原合同；各区间帧数和误差详见comparison.json。", "",
              "## 描述性success子集（明确两种分母）", "",
              "| 子集 / run | clip分母 | frame分母 | clip均值位置/旋转 | frame加权位置/旋转 |",
              "| --- | --- | --- | --- | --- |"]
    subsets = output["secondary_success_subsets"]
    for label in ("each_run_success_only", "common_success_intersection"):
        for name, row in subsets[label].items():
            if row["clips"]:
                clip = row["per_clip_mean"]
                frame = row["frame_weighted_mean"]
                clip_text = f"{clip['mean_body_position_error_m']:.6g}m /{clip['mean_body_rotation_error_rad']:.6g}rad"
                frame_text = f"{frame['mean_body_position_error_m']:.6g}m /{frame['mean_body_rotation_error_rad']:.6g}rad"
            else:
                clip_text = frame_text = "None"
            lines.append(f"| {label} /{name} | {row['clips']} | {row['metric_frames']} | {clip_text} | {frame_text} |")
    lines += ["", "这些按结果选择的子集仅解释分母敏感性，不替代ALL22，不证明论文.218m/.343rad采用相同分母。", "",
              "## 配对、跌倒和来源", "",
              "全部22初始状态/AMP history与所有参考目标/时钟数组通过dtype及bitwise配对。",
              "每段跌倒连续时段、真实source帧和相对reset时间及首fall前后1秒分解在comparison.json；不会重置或移除跌倒帧。",
              "源码tar逐member验证，真实step1000检查点SHA及捕获来源绑定均记录。", ""]
    for name, checkpoint in output["checkpoints"].items():
        lines.append(f"- {name} controller SHA256: `{checkpoint['sha256']}`")
    lines += ["", f"候选预算{output['candidate_step']}updates，baseline1000；半/较少预算的探索诊断不是同预算对照。",
              f"Lineage模式: {output['candidate_lineage']['mode']}；continued_from500是新独立续训，不是原轨迹750，不能承诺PhysX cache bitwise恢复。",
              "该工程课程同时改变固定source命令、合法reset/clip-end和明确clip外线性延拓；不是tail单变量试验。",
              "单seed、反复使用的validation不能证明统计显著性、作者复现或未查看的34条test性能。",
              "训练源码归档与验证捕获source分别保留，不假定今天整个工作树等于历史运行。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, default=RUNS / "motion_tracking_amass2848_backward_penalty_full_validation_v1/result.json")
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--baseline-training-result", type=Path, default=RUNS / "amp_style_amass2848_backward_penalty_v1/result.json")
    parser.add_argument("--candidate-training-result", type=Path, required=True)
    parser.add_argument("--candidate-step", type=int, choices=(500, 750, 1000), required=True)
    parser.add_argument("--lineage-mode", choices=("original_snapshot", "continued_from500"), default="original_snapshot")
    parser.add_argument("--sealed-checkpoint-ledger", type=Path, default=RUNS / "amp_style_amass_root_snapshots_v1/snapshots.json")
    parser.add_argument("--baseline-source", type=Path, default=DOCS / "backward_penalty_training_source.json")
    parser.add_argument("--candidate-source", type=Path, default=DOCS / "amass_root_linear_training_source_v1.json")
    parser.add_argument("--baseline-decomposition", type=Path, default=DOCS / "backward_penalty_tracking_decomposition.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for this CPU-only audit")
    require(not args.output.exists(), "Output must be a fresh directory; historical evidence is immutable")
    results = {"baseline": read(args.baseline_result), "amass_root": read(args.candidate_result)}
    rows = {name: require_full_validation(value) for name, value in results.items()}
    require(rows["baseline"].keys() == rows["amass_root"].keys(), "Validation motion IDs differ")
    protocol_keys = ("protocol", "physics", "layout", "reference_velocity_contract")
    require(all(results["baseline"][key] == results["amass_root"][key] for key in protocol_keys), "Evaluation protocol/physics/reference contract differs")
    track_diff = differences(results["baseline"]["arguments"], results["amass_root"]["arguments"])
    require(set(track_diff) <= {"checkpoint", "checkpoint_sha256", "output"}, "Evaluation flags changed beyond checkpoint/output")
    checkpoints = {name: verified_checkpoint(result) for name, result in results.items()}
    archives, source_entries = {}, {}
    for name, path in (("baseline", args.baseline_source), ("amass_root", args.candidate_source)):
        archives[name], source_entries[name] = verify_archive(path)
    runtime_bindings = {}
    for name, result in results.items():
        runtime_bindings[name] = {}
        for path, digest in result["selected_source_sha256"].items():
            require(path in source_entries[name] and source_entries[name][path]["sha256"] == digest,
                    f"Captured{name} runtime source differs from its immutable archive: {path}")
            runtime_bindings[name][path] = digest
    training = {"baseline": read(args.baseline_training_result), "amass_root": read(args.candidate_training_result)}
    require(all(value["status"] == "passed" for value in training.values()) and training["baseline"]["update_count"] == 1000,
            "Completed training and the fixed baseline1000 are required")
    canonical = {name: dict(value["training_config"]) for name, value in training.items()}
    # Older report predates this explicitly disabled optional field. No other
    # missing setting may be filled implicitly for experiment admission.
    for value in canonical.values():
        value.setdefault("expert_sampling", None)
    require(canonical["baseline"] == canonical["amass_root"] and canonical["amass_root"]["expert_sampling"] is None,
            "Canonical training settings must equal baseline, with expert sampler disabled")
    setup_keys = ("initialization", "physics", "power_measurement", "runtime_numerics", "runtime_execution", "amp_reward_timestamp")
    require(all(training["baseline"][key] == training["amass_root"][key] for key in setup_keys), "Non-curriculum training setup differs")
    require(training["amass_root"]["trajectory_source"] == "amass_root", "Candidate is not the declared AMASS-root curriculum")
    candidate_lineage = verify_lineage(checkpoints["amass_root"], training["amass_root"], args.candidate_step,
                                       args.lineage_mode, args.sealed_checkpoint_ledger)
    seal = read(args.sealed_checkpoint_ledger)
    require(seal["source_manifest"]["sha256"] == archives["amass_root"]["manifest"]["sha256"]
            and seal["source_archive"]["sha256"] == archives["amass_root"]["archive"]["sha256"],
            "Parent/snapshot seal differs from the verified source archive")
    decomposition = helper()
    old_decomposition = read(args.baseline_decomposition)
    require(old_decomposition["source_result"]["sha256"] == sha256(args.baseline_result)
            and old_decomposition["primary_aggregate_unchanged"] == results["baseline"]["aggregate"], "Cached baseline decomposition changed")
    require(old_decomposition["diagnostic_script"]["sha256"] == sha256(DOCS / "decompose_motion_tracking.py"), "Frozen decomposition helper differs")
    new_decomposition = decomposition.decompose(args.candidate_result.resolve().parent)
    parents = decomposition.parent_indices(ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml")
    paired = []
    for mid, old in rows["baseline"].items():
        new = rows["amass_root"][mid]
        keys = ("source_path", "tracking_window", "observed_metric_steps", "initial_translation_xyz_m")
        require(all(old[key] == new[key] for key in keys), f"Clip{mid} source/window/alignment differs")
        paths = {"baseline": Path(old["capture"]), "amass_root": Path(new["capture"])}
        for name, sequence in (("baseline", old), ("amass_root", new)):
            result_path = args.baseline_result if name == "baseline" else args.candidate_result
            require(paths[name].parent.resolve() == (result_path.resolve().parent / "captures"), "Capture path is not bound to its result")
            require(sha256(paths[name]) == sequence["capture_sha256"], "Capture SHA mismatch")
        with np.load(paths["baseline"], allow_pickle=False) as a, np.load(paths["amass_root"], allow_pickle=False) as b:
            exact = exact_capture_pair(a, b)
        paired.append({"motion_id": mid, "source_path": old["source_path"], **exact,
                       "captures": {name: record(path) for name, path in paths.items()},
                       "baseline": {"tracking_success": old["tracking_success"],
                                    **capture_diagnostics(paths["baseline"], old, decomposition, parents)},
                       "amass_root": {"tracking_success": new["tracking_success"],
                                      **capture_diagnostics(paths["amass_root"], new, decomposition, parents)}})
    common = sorted(source_entries["baseline"].keys() & source_entries["amass_root"].keys())
    changed = [key for key in common if source_entries["baseline"][key]["sha256"] != source_entries["amass_root"][key]["sha256"]]
    success_ids = {name: {mid for mid, row in values.items() if row["tracking_success"]} for name, values in rows.items()}
    shared_success = success_ids["baseline"] & success_ids["amass_root"]
    output = {"schema": "navisaaclab.amass_root_tracking_lineage_comparison.v2", "status": "passed",
              "candidate_step": args.candidate_step, "candidate_lineage": candidate_lineage,
              "same_training_budget_as_baseline1000": args.candidate_step == 1000,
              "CPU_only": True, "GPU_used": False, "Kit_started": False, "style_metrics_recomputed": False,
              "heldout_test_outputs_read": False, "script": record(__file__),
              "results": {"baseline": record(args.baseline_result), "amass_root": record(args.candidate_result)},
              "training_results": {"baseline": record(args.baseline_training_result), "amass_root": record(args.candidate_training_result)},
              "checkpoints": checkpoints, "archives": archives,
              "captured_runtime_bound_to_source_archives": runtime_bindings,
              "source_changes": {"changed_common_files": changed,
                                 "unchanged_common_files": [key for key in common if key not in changed],
                                 "added_files": sorted(source_entries["amass_root"].keys() - source_entries["baseline"].keys()),
                                 "captured_runtime_sources": {name: value["selected_source_sha256"] for name, value in results.items()},
                                 "current_tree_equals_historical_runs_claim": False},
              "training_config_differences_raw": differences(training["baseline"]["training_config"], training["amass_root"]["training_config"]),
              "canonical_training_settings_equal": True, "explicit_legacy_missing_default": {"expert_sampling": None},
              "training_argument_differences": differences(training["baseline"]["arguments"], training["amass_root"]["arguments"]),
              "training_provenance_differences": differences(training["baseline"]["provenance"], training["amass_root"]["provenance"]),
              "identical_training_setup_keys": list(setup_keys), "evaluation_argument_differences": track_diff,
              "equal_evaluation_protocol_keys": list(protocol_keys), "paired_clip_count": len(paired), "paired_clips": paired,
              "aggregate": {name: value["aggregate"] for name, value in results.items()},
              "secondary_success_subsets": {
                  "each_run_success_only": {name: subset_primary(values, success_ids[name]) for name, values in rows.items()},
                  "common_success_intersection": {name: subset_primary(values, shared_success) for name, values in rows.items()},
                  "all22_primary_unchanged": True,
                  "interpretation": "Post-outcome descriptive subsets only; not replacement metrics or evidence for unknown author error denominators."},
              "decomposition": {"baseline": record(args.baseline_decomposition), "amass_root_file": "amass_root_decomposition.json"},
              "same_frame_diagnostics": {"baseline": old_decomposition["aggregate_diagnostic"], "amass_root": new_decomposition["aggregate_diagnostic"]},
              "speed_conditioned_diagnostics": {"baseline": old_decomposition["by_reference_speed_bin"], "amass_root": new_decomposition["by_reference_speed_bin"]},
              "limitations": ["All22 real full sequences and all physical fall frames remain included; no best-time/phase fitting.",
                              "Heading-local diagnostics preserve the primary world-space error and are not additive causal shares.",
                              "True futureXY tracking validation does not establish learned Diffusion navigation or author reproduction.",
                              "The new fixed-source/reset/end/linear-tail recipe is a declared engineering curriculum; not an isolated tail-only experiment.",
                              "One seed and iteratively exposed validation set do not establish significance or untouched-test performance.",
                              "No official style/MMD metrics are recalculated; separate resource-agent reports remain authoritative."]}
    args.output.mkdir(parents=True)
    write_new(args.output / "amass_root_decomposition.json", new_decomposition)
    write_new(args.output / "comparison.json", output)
    with (args.output / "report.md").open("x") as stream:
        stream.write(report_markdown(output))
    print(json.dumps({"status": "passed", "paired_clip_count": len(paired), "aggregate": output["aggregate"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
