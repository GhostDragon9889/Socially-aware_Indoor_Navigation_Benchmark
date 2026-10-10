"""Register the selected62c4 baseline Office3 probe; CPU opaque bytes only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / "docs/style_improvement_20261003/local_Office_gravity_probe_preparation_v1"
MODEL = ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt"
MODEL_SHA = "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
PROFILE = ROOT / "configs/diffusion_amp/amp_style_amass_backward_penalty.yaml"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular bound source/resource required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if (HERE / "plan.json").exists():
        raise ValueError("Preserve existing baseline registration")
    original = json.loads((ORIGINAL / "plan.json").read_text())
    if record(ORIGINAL / "plan.json")["sha256"] != "63b10ff8891e4fe2c4e215a440eb62c47b0cfd6724a8ed936b96c6a569cb5996":
        raise ValueError("Original Office source plan changed")
    guide = ROOT / "FULL_PIPELINE.md"
    guide_bytes = guide.read_bytes()
    guide_record = {"path": str(guide), "bytes": len(guide_bytes),
                    "sha256": hashlib.sha256(guide_bytes).hexdigest()}
    selected_command = guide_bytes.decode().split("## Run the selected physical navigation baseline", 1)[1].split("## Paper pedestrian method", 1)[0]
    if str(MODEL.relative_to(ROOT)) not in selected_command or str(PROFILE.relative_to(ROOT)) not in selected_command:
        raise ValueError("Selected baseline model/profile do not match the current documented command")
    with (HERE / "FULL_PIPELINE_source_snapshot.md").open("xb") as stream:
        stream.write(guide_bytes)
    model = record(MODEL)
    if model["sha256"] != MODEL_SHA or model["bytes"] != 651850375:
        raise ValueError("Actual selected baseline62c4 full bytes differ")
    stable = {}
    for actors, relative in [(4, "native_scene_200s_actual_review_v1"), (15, "native15_200s_independent_review_v1")]:
        path = ROOT / "docs/style_improvement_20261002" / relative / "actual_audit.json"
        audited = json.loads(path.read_text())
        counts = audited["counts"]
        if (audited["status"] != "raw_sensor_source_contracts_passed_bounded_zero_fall_observed"
                or audited["native_200s_artifacts_independently_audited"] is not True
                or counts["controls"] != 6000 or counts["actors"] != actors
                or any(counts[key] != 0 for key in ("falls", "static_actor_controls", "peer_pair_controls"))):
            raise ValueError("Actual historical baseline stable native record required")
        result_row, entry_row = audited["artifacts"]["result.json"], audited["artifacts"]["deployment_entry.json"]
        if record(result_row["path"]) != result_row or record(entry_row["path"]) != entry_row:
            raise ValueError("Historical baseline producer bytes differ")
        result = json.loads(Path(result_row["path"]).read_text())
        entry = json.loads(Path(entry_row["path"]).read_text())
        if (result["evaluation_actor"]["checkpoint_sha256"] != MODEL_SHA
                or result["evaluation_actor"]["checkpoint_bytes"] != model["bytes"]
                or entry["evaluation_actor_request"]["base_checkpoint"] != {"path": model["path"], "sha256": MODEL_SHA}
                or result["arguments"]["config"] != str(PROFILE)):
            raise ValueError("Historical stable record is not the selected baseline controller/profile")
        stable[str(actors)] = {"audit": record(path), "result": result_row, "deployment_entry": entry_row,
                              "counts": counts, "historical_source_epoch": audited["source_epoch"],
                              "same_as_current_production185_execution_source": False}
    sources = dict(original["sources"])
    sources.pop("preparer")
    sources.pop("native_registration")
    sources["preparer"] = record(__file__)
    sources["original_Office_plan"] = record(ORIGINAL / "plan.json")
    sources["original_Office_freeze"] = record(ORIGINAL / "source_freeze_v1.json")
    sources["documented_baseline_command_snapshot"] = record(HERE / "FULL_PIPELINE_source_snapshot.md")
    sources["baseline_profile"] = record(PROFILE)
    sources["runtime_strict_EvaluationActor"] = record(ROOT / "CrowdSim/humanoid/runtime/evaluation_actor.py")
    for actors, evidence in stable.items():
        for kind in ("audit", "result", "deployment_entry"):
            sources[f"historical_native{actors}_{kind}"] = evidence[kind]
    command = list(original["future_deployment_command_without_observer"])
    command[command.index("--amp-checkpoint") + 1] = str(MODEL)
    command[command.index("--config") + 1] = str(PROFILE)
    command[command.index("--output") + 1] = str(ROOT / "output/diffusion_amp/local_Office3_baseline62c4_gravity10s_v1")
    from CrowdSim.humanoid.application.deployment import parse_deployment_args, driver_arguments
    args = parse_deployment_args(command[2:])
    document = {
        "schema": "local_Office.baseline62c4_prepared_probe.v1", "status": "prepared_not_executed",
        "CPU_only_preparation": True, "GPU_used": False, "SDK_started": False,
        "selected_checkpoint": model, "selected_checkpoint_OS_readonly": not bool(MODEL.stat().st_mode & 0o222),
        "checkpoint_hashed_as_opaque_bytes": True, "checkpoint_deserialized_by_preparer": False,
        "profile": record(PROFILE), "documented_command_original_source_at_read": guide_record,
        "sources": sources, "production_source": original["production_source"],
        "historical_baseline_native_stability": stable,
        "new09e_reward_terminal_schema_used_for62c4": False,
        "new09e_reward_terminal_or_checkpoint_consumed": False,
        "runtime_model_load": "existing EvaluationActor.from_checkpoints strict input/action/normalizer/weight SHA and legacy RNG; no custom actor loader",
        "existing_native_scene_request": original["existing_native_scene_request"],
        "scene_resource_evidence_limits": original["scene_resource_evidence_limits"],
        "expected_counts": original["expected_counts"],
        "deployment_command_without_observer": command, "driver_argv": driver_arguments(args),
        "changes_from_original_Office_plan": ["selected baseline62c4 controller", "documented baseline profile", "fresh output alias", "controller admission based on prior actual baseline identity, not new reward2000 schema"],
        "observer": original["sources"]["observation_adapter"],
        "original_world_control_routes_seed_arm_and_guidance_preserved": True,
        "baseline_OS_mode_changed": False, "production185_changed": False, "original_09e_Office_plan_changed": False,
        "actual_Office_gravity_observed": False, "actual_Office_human_stability_observed": False,
        "actual_Office_capture": {"path": command[-1], "sha256": None, "bytes": None, "actual_executed": False},
        "author_eight_scene_equivalence": False, "sealed34_outputs_read": False,
    }
    with (HERE / "plan.json").open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    driver = HERE / "driver_adapter.py"
    code = driver.read_text()
    if code.count("UNFROZEN_BASELINE_PLAN_SHA") != 1:
        raise ValueError("Fresh baseline adapter required")
    driver.write_text(code.replace("UNFROZEN_BASELINE_PLAN_SHA", record(HERE / "plan.json")["sha256"]))
    print(json.dumps({"status": document["status"], "plan": record(HERE / "plan.json"),
                      "checkpoint": model, "GPU_used": False, "SDK_started": False}))


if __name__ == "__main__":
    main()
