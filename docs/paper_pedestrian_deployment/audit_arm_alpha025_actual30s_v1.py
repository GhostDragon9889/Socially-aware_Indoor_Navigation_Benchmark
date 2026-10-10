"""CPU-only admission of actual alpha .25 / baseline 900-control captures.

Reuses the frozen raw normal-force/COM audit without editing its formulas.
Source154 is a prelaunch archive, not a retrospective full GPU import map.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "experimental_arm_alpha025_actual30s_comparison_v1"
READER_SHA = "334bdc58ad1f455c6184763a64adcded72bf5deba1b1990c6180aebd7a57c826"
BASE_SOURCE_SHA = "e924af188b079208148357a7ed8c81e7429629b6121c52cace5ff4a48fee3455"
NEW_SOURCE_SHA = "b250be690ec95dc683613f2679e6c3f5263e2aa55f88a69eac46e4b5ce51d52a"
DONOR_SHA = "28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72"


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or (OUT / "native_ground_comparison.json").exists():
        raise ValueError("CPU-only and fresh output required")
    reader_path = HERE / "compare_physical_prefix.py"
    if hashlib.sha256(reader_path.read_bytes()).hexdigest() != READER_SHA:
        raise ValueError("Frozen source/measurement admission reader changed")
    reader = load_module("alpha025_original_prefix_reader", reader_path)
    helper = reader.load_helper()
    baseline_dir = ROOT / "output/diffusion_amp/pure_inference_baseline_30s_v1"
    candidate_dir = ROOT / "output/diffusion_amp/experimental_arm_alpha025_30s_v1"
    baseline_result = reader.read(baseline_dir / "result.json")
    result = reader.read(candidate_dir / "result.json")
    old_report_path = HERE / "pure_inference_actual30s_prefix_comparison_v1/comparison.json"
    old_report = reader.read(old_report_path)
    baseline_audit = old_report["candidate_actual_native_measurement_audit"]
    audit = helper.Audit()
    for key, filename in (("result", "result.json"), ("ground_support", "ground_support.npz"),
                          ("physical_states", "physical_states.npz")):
        audit.require(reader.record(baseline_dir / filename) == baseline_audit["artifacts"][key],
                      "actual900 baseline unchanged after independent audit:" + key)
    audit.require(old_report["paired_bitwise_equal"] is True, "actual900 baseline was exact old negative prefix")
    old_source, old_raw = reader.verify_archive(HERE / "pure_inference_deployment_source_v1.json", BASE_SOURCE_SHA, 146)
    new_source, new_raw = reader.verify_archive(
        ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v1.json",
        NEW_SOURCE_SHA, 154)
    audit.require(reader.control_loop_sha(old_raw["scripts/train_paper_amp.py"]) ==
                  reader.control_loop_sha(new_raw["scripts/train_paper_amp.py"]),
                  "actual source146/source154 complete physical control-loop AST equal")
    for name in ("CrowdSim/humanoid/ground_support_archive.py", "CrowdSim/humanoid/foot_capture.py",
                 "CrowdSim/humanoid/foot_support.py", "CrowdSim/humanoid/physics_views.py",
                 "CrowdSim/humanoid/runtime/evaluation_actor.py", "CrowdSim/humanoid/runtime/amp_inference.py",
                 "CrowdSim/humanoid/runtime/arm_style.py"):
        audit.require(old_raw[name] == new_raw[name], "exact source146/source154 measurement/inference member:" + name)
    audit.require(result["status"] == "passed" and baseline_result["status"] == "passed" and
                  result["actual_control_steps"] == baseline_result["actual_control_steps"] == 900,
                  "both actual runs completed900 controls")
    old_args, args = baseline_result["arguments"], result["arguments"]
    audit.require(set(args) == set(old_args) and
                  all(args[key] == old_args[key] for key in args if key != "output"),
                  "sole driver argument difference is output")
    for key in ("physics", "runtime_numerics", "evaluation_dataset", "evaluation_scene", "training_config"):
        audit.require(result[key] == baseline_result[key], "same paired physical/evaluation contract:" + key)
    for key in ("optimizer_constructed", "replay_constructed", "critic_evaluated",
                "discriminator_evaluated", "training_transitions_collected"):
        audit.require(result["evaluation_training_components"][key] is False, "no training component:" + key)
    entry = reader.read(candidate_dir / "deployment_entry.json")
    audit.require(entry["parsed_driver_arguments"] == args, "thin entry forwarded actual driver arguments")
    request = entry["evaluation_actor_request"]
    audit.require(request["alpha"] == .25 and result["evaluation_actor"]["experimental_arm_style"]["alpha"] == .25,
                  "explicit arm raw-mean mixing alpha .25")
    resources = reader.resource_identity(result)
    audit.require(resources == reader.resource_identity(baseline_result), "same baseline controller/diffusion/profile/scene resources")
    donor = reader.record(Path(request["style_checkpoint"]["path"]))
    audit.require(donor["sha256"] == request["style_checkpoint"]["sha256"] == DONOR_SHA,
                  "actual donor full SHA256 matches explicit request")
    original_admission = helper.verify_sources
    try:
        def source_admission(_audit, _result, **_kwargs):
            return {"prelaunch_source_archive154": new_source, "baseline_prelaunch_source_archive146": old_source,
                    "physical_control_loop_AST_sha256": reader.control_loop_sha(new_raw["scripts/train_paper_amp.py"]),
                    "resources_full_sha256_recomputed": resources, "style_checkpoint": donor,
                    "actual_full_GPU_import_mapping_verified": False,
                    "source_boundary": "Prelaunch154 archive is exact. Later provider edits are not substituted; native branch disabled. No full runtime import map is claimed."}
        helper.verify_sources = source_admission
        native = helper.audit_capture(candidate_dir, audit)
    finally:
        helper.verify_sources = original_admission
    native.update(status="passed", checks=list(audit.checks), audit_script=reader.record(__file__),
                  actual_native_auditor_sha256=reader.HELPER_SHA)
    changed = {name: {"baseline_sha256": hashlib.sha256(old_raw[name]).hexdigest(),
                      "candidate_sha256": hashlib.sha256(new_raw[name]).hexdigest()}
               for name in old_raw.keys() & new_raw.keys() if old_raw[name] != new_raw[name]}
    report = {"schema": "arm_style.actual_paired900_native_audit.v1", "status": "passed", "CPU_only": True,
              "SDK_started": False, "GPU_used": False, "sealed34_outputs_read": False,
              "baseline_report": reader.record(old_report_path), "baseline": baseline_audit, "candidate": native,
              "source_archive_changed_members": changed, "current_tree_used_as_historical_source": False,
              "normal_force_proxy_formula_changed": False, "controller_trajectory_equivalence_claim": False,
              "limits": ["Both900 controls/30s, not200s stability. No temporal frame filtering.",
                         "Base/donor unchanged raw non-arm coordinates applies only to identical observations; changed physical states alter all future policy outputs.",
                         "Source146 and154 differ even though the complete control-loop AST matches; no same-source or actual full GPU import map claim.",
                         "Contact-conditioned body-origin velocity is a proxy, not measured contact-patch slip."]}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "native_ground_comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(audit.checks), "output": str(OUT)}))


if __name__ == "__main__":
    run()
