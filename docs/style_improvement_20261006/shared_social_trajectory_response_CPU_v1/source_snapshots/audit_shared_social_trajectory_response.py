#!/usr/bin/env python3
"""CPU frozen S1/S2 trajectory response on matched real S2 validation/train rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import trajectory_response as kernel  # noqa: E402
from CrowdSim.humanoid.analysis import tracking_commands  # noqa: E402
from CrowdSim.humanoid.runtime.social_popart_inference import FrozenSocialPopArtCheckpointInference  # noqa: E402
from scripts import verify_shared_social_evidence_admission as gate  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402
from scripts.evaluate_shared_social_tracking import admit_training  # noqa: E402

POSTS = {"S1": ROOT / "docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json",
    "S2": ROOT / "docs/style_improvement_20261005/shared_social_full_epochs_v1/post_training_execution.json"}
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
BATCH = 128


def evaluate_batch(actor, raw_self, raw_task, std):
    torch = actor._torch
    before = raw_self.clone(), raw_task.clone()
    tasks = kernel.task_cases(raw_task)
    kernel.require(torch.equal(tasks[0], raw_task) and all(torch.equal(task[:, 20:], raw_task[:, 20:]) for task in tasks),
        "A probe changed original task or raster")
    packed = tasks.reshape(-1, 12308)
    prediction = actor.act(raw_self.repeat((len(kernel.CASES), 1)), packed)
    means = prediction["mean_actions"].reshape(len(kernel.CASES), len(raw_self), 69).numpy()
    applied = prediction["applied_actions"].reshape(len(kernel.CASES), len(raw_self), 69).numpy()
    values = kernel.action_response(means, applied, actor._action_mask.bool().numpy(), std)
    normalized = actor._task_rms.normalize(packed, clamp=False).reshape(len(kernel.CASES), len(raw_self), 12308)
    kernel.require(bool(torch.isfinite(normalized).all()), "Unclipped diagnostic normalization is nonfinite")
    values["trajectory_RMS_strictly_clipped_coordinates"] = (normalized[:, :, :20].abs() > 5).sum(-1).numpy()
    values["trajectory_unclipped_normalized_RMS"] = normalized[:, :, :20].square().mean(-1).sqrt().numpy().astype(np.float64)
    kernel.require(torch.equal(raw_self, before[0]) and torch.equal(raw_task, before[1]), "Probe modified original raw input")
    return values, applied[0]


def concatenate(rows):
    return {key: np.concatenate([row[key] for row in rows], axis=1) for key in rows[0]}


def validation_rows(actor, tracking, std):
    torch = actor._torch
    results, comparisons, lineage, inputs, sequences = [], [], [], [], []
    for row in tracking["sequences"]:
        source = identity(row["capture"])
        kernel.require(source["sha256"] == row["capture_sha256"], "Actual validation capture changed")
        inputs.append(source)
        with np.load(source["path"], allow_pickle=False) as arrays:
            tracking_commands.audit_capture(arrays, row)
            metric = arrays["metric_frame"]
            raw_self = torch.from_numpy(arrays["observation_raw_self"][metric, 0].copy())
            raw_task = torch.from_numpy(arrays["observation_raw_task"][metric, 0].copy())
            captured = arrays["applied_actions"][metric, 0]
            clip_results = []
            for start in range(0, len(raw_self), BATCH):
                values, applied = evaluate_batch(actor, raw_self[start:start+BATCH], raw_task[start:start+BATCH], std)
                clip_results.append(values)
                comparisons.append(np.abs(applied.astype(np.float64)-captured[start:start+BATCH]).max(-1))
            values = concatenate(clip_results)
            results.append(values)
            sequences.append({"motion_id": row["motion_id"], "rows": len(raw_self), "responses": kernel.summarize(values)})
            lineage.append(np.stack((np.full(int(metric.sum()), row["motion_id"]), np.flatnonzero(metric), arrays["fall_mask"][metric]), -1))
    values, labels = concatenate(results), np.concatenate(lineage)
    kernel.require(len(sequences) == 22 and labels.shape == (11982, 3), "Require full22/11982 including all falls/failures")
    return inputs, values, labels, np.concatenate(comparisons), sequences


def training_rows(actor, rollout, std):
    torch = actor._torch
    saved = torch.load(rollout, weights_only=True, map_location="cpu", mmap=True)
    rows = saved["collection"]["rollout"]
    kernel.require(rows["self_obs"].shape == (32, 256, 357) and rows["task_obs"].shape == (32, 256, 12308)
        and rows["self_obs"].dtype == rows["task_obs"].dtype == torch.float32, "Require actual final32x256 raw training rows")
    raw_self, raw_task = rows["self_obs"].flatten(0, 1), rows["task_obs"].flatten(0, 1)
    results = []
    for start in range(0, 8192, BATCH):
        values, _ = evaluate_batch(actor, raw_self[start:start+BATCH].clone(), raw_task[start:start+BATCH].clone(), std)
        results.append(values)
    indices = np.arange(8192, dtype=np.int64)
    return concatenate(results), np.stack((indices//256, indices%256), -1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output_directory.exists(), "Require fresh CPU-only output")
    sources = [identity(p) for p in (Path(__file__), Path(kernel.__file__), Path(tracking_commands.__file__),
        ROOT / "CrowdSim/humanoid/analysis/tracking_error_decomposition.py", ROOT / "CrowdSim/humanoid/motion_tracking.py",
        Path(gate.__file__), Path(gate.original.__file__), ROOT / "scripts/diagnose_shared_social_training.py",
        ROOT / "scripts/evaluate_shared_social_tracking.py", ROOT / "CrowdSim/humanoid/runtime/social_popart_inference.py")]
    bundles, inputs, admissions = {}, [], {}
    for name, path in POSTS.items():
        bundle, measurements = gate.admit_post(path)
        bundles[name] = bundle
        inputs.extend([identity(path), *measurements.values(), *gate.bundle_sources(bundle, measurements)])
        admission = bundle["tracking"]["training_execution_admission"]
        checkpoint, sha, admitted = admit_training(Path(admission["receipt"]["path"]))
        kernel.require(admitted == admission, "Independent training admission differs from completed evaluation")
        result = json.loads(Path(admission["result"]["path"]).read_text())
        kernel.require(result["manifest"]["sha256"] == MANIFEST_SHA, "Model was not trained on fixed2848")
        admissions[name] = checkpoint, sha, admitted
        inputs.extend([identity(admission["receipt"]["path"]), admission["result"], admission["audit"]])
    gate.tracking_protocol(bundles["S1"]["tracking"], bundles["S2"]["tracking"])
    audit = json.loads(Path(admissions["S2"][2]["audit"]["path"]).read_text())
    rollout = audit["rollout"]
    kernel.require(identity(rollout["path"]) == rollout, "Audited S2 terminal rollout changed")
    inputs.append(rollout)
    import torch

    torch.set_num_threads(4)
    kernel.require(not torch.cuda.is_initialized(), "CPU reader initialized CUDA")
    rng = torch.random.get_rng_state().clone()
    std = float(torch.tensor(-2.9, dtype=torch.float32).exp())
    runs, outputs = {}, {}
    for name, (checkpoint, sha, admission) in admissions.items():
        started = time.monotonic()
        actor = FrozenSocialPopArtCheckpointInference.from_checkpoint(checkpoint, expected_sha256=sha, device="cpu")
        rms_before = [{k: v.clone() for k, v in rms.state_dict().items()} for rms in (actor._self_rms, actor._task_rms)]
        bound, validation, labels, native_difference, sequences = validation_rows(actor, bundles["S2"]["tracking"], std)
        inputs.extend(bound)
        training, train_labels = training_rows(actor, rollout["path"], std)
        for rms, before in zip((actor._self_rms, actor._task_rms), rms_before, strict=True):
            kernel.require(all(torch.equal(v, rms.state_dict()[k]) for k, v in before.items()), "Frozen RMS changed")
        outputs[name+"_validation"] = dict(validation, input_motion_capture_frame_fall=labels,
            CPU_original_minus_native_S2_applied_max_abs=native_difference)
        outputs[name+"_training"] = dict(training, input_control_actor=train_labels)
        runs[name] = {"admission": admission, "frozen_actor": actor.describe(), "actual_CPU_forward_executed": True,
            "validation": {"rows": 11982, "sequences": sequences, "responses": kernel.summarize(validation),
                "native_S2_action_difference_is_same_model": name == "S2",
                "CPU_vs_native_S2_original_applied_max_abs": float(native_difference.max()) if name == "S2" else None},
            "training": {"rows": 8192, "responses": kernel.summarize(training), "replays_saved_preupdate_actions": False},
            "elapsed_wall_s": time.monotonic()-started, "RMS_updated": False}
        print(json.dumps({"model": name, "validation_rows": 11982, "training_rows": 8192,
            "forward_validation_applied_delta_RMS_mean": runs[name]["validation"]["responses"]["forward_1m_s"]["applied_delta_active_RMS"]["mean"]}), flush=True)
        del actor
    inputs = gate.unique_artifacts(inputs)
    kernel.require(all(identity(row["path"]) == row for row in inputs+sources), "Actual input/source changed during measurement")
    kernel.require(torch.equal(rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized(), "CPU RNG or CUDA changed")
    kernel.require(not any(n.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for n in sys.modules), "SDK imported")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    archives = {}
    for name, arrays in outputs.items():
        p = args.output_directory / (name+"_all_rows.npz")
        with p.open("xb") as stream:
            np.savez(stream, **arrays)
        archives[name] = identity(p)
    report = {"schema": "shared_social.actual_frozen_S1_S2_matched_trajectory_response_CPU.v1",
        "status": "completed_CPU_policy_response_observation", "sources": sources, "inputs": inputs, "runs": runs,
        "all_row_archives": archives, "cases": list(kernel.CASES), "fixed_active_Gaussian_std_CPU_float32_exp": std,
        "cohorts": {"validation": "All11982 S2 PRE action observations; S1/S2 terminal actors share identical inputs; every fallen/failed frame retained",
            "training": "All8192 S2 final32x256 raw observations collected before the last PPO update; evaluated with terminal actors, not a replay of preupdate actions"},
        "scope": "Only raw trajectory20 changes. Self357/raster12288 and frozen weights/RMS remain fixed; normalized action/KL responses do not prove correct gait or closed-loop tracking",
        "native_action_CPU_difference_is_measured_without_relaxed_admission": True,
        "SDK_started": False, "CUDA_initialized": False, "caller_CPU_RNG_preserved": True,
        "trained_weights_or_metrics_changed": False, "causal_training_failure_proven": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    p = args.output_directory / "result.json"
    with p.open("x") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"result": identity(p), "models": 2, "input_rows_per_model": 20174, "cases": 6, "SDK_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
