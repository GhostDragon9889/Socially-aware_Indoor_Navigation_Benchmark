"""Run fixed 90D/45D style metrics after each native run's own admission."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

from native_metric_sources import CPULedger, ROOT, admit_run, exact, load, record, require, validate_engine

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "commands_native_CPU_metrics_v1.json"
CATALOG_SHA = "f3bed1b88466ad009fa44fa7a08ce2352115b328781753e14b5203fafd00c067"
METRICS = tuple(f"{kind}{actors}_{dimension}" for kind in ("new", "old")
                for actors in (4, 15) for dimension in (90, 45))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def argv_for(row, catalog):
    values = ["--run", row["run_name"] + "=" + row["input"]]
    if row["dimension"] == 90:
        values += ["--reference-artifact", catalog["sources"]["style_reference"]["path"],
                   "--cpu-threads", "4", "--minimum-bin-samples", "30", "--maximum-motion-samples", "256"]
    return values + ["--output", row["output"]]


def validate_catalog(catalog):
    require(catalog.get("schema") == "public_state_prior.prepared_native_CPU_metric_commands.v1"
            and catalog.get("status") == "prepared_not_executed" and catalog.get("future_effect") is False,
            "Only frozen prepared native commands accepted")
    require(set(catalog["metrics"]) == set(METRICS), "All four native runs need separate 90D/45D commands")
    for key, row in catalog["metrics"].items():
        actors = row["actors"]
        require(actors in (4, 15) and row["dimension"] in (90, 45)
                and key == f"{row['kind']}{actors}_{row['dimension']}", "Unknown native metric")
        suffix = "" if actors == 4 else "_15"
        prefix = "native_warehouse_state_prior" if row["kind"] == "new" else "native_warehouse"
        require(row["input"] == str(ROOT / f"output/diffusion_amp/{prefix}{suffix}_clearance08_200s_v1"),
                "Exact native capture aliases required")
        require(row["output"] == str(ROOT / f"docs/style_improvement_20261003/state_prior_native_metrics_v1/{key}"),
                "Fresh independent native metric output required")
        require(row["metric_argv"] == argv_for(row, catalog), "Fixed mathematical parameters changed")
        require(row["source_count"] == (183 if row["kind"] == "new" else 168 if actors == 4 else 174),
                "Native runtime epochs cannot be substituted")
        if row["kind"] == "new":
            require(row["raw_audit"] == str(ROOT / f"docs/style_improvement_20261003/state_prior_native_actual_review_v1/native{actors}_audit.json")
                    and row["registration"] == str(ROOT / f"docs/style_improvement_20261003/state_prior_native_reader_preparation_v1/native{actors}_registration.json"),
                    "Exact new native reader aliases required")
        require(row["command"] == [catalog["python"], str(HERE / "run_native_CPU_metrics.py"), "--metric", key],
                "Exact CPU command required")


def stage_pose_tool(catalog):
    source = catalog["sources"]["pose45_tool"]
    directory = Path(tempfile.mkdtemp(prefix="state_prior_native_pose45_", dir="/tmp"))
    path = directory / "legacy" / "docs" / Path(source["path"]).name
    path.parent.mkdir(parents=True)
    path.write_bytes(Path(source["path"]).read_bytes())
    path.chmod(0o444)
    value = record(path)
    require(value["bytes"] == source["bytes"] and value["sha256"] == source["sha256"], "Original pose45 bytes changed")
    for item in sorted(directory.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if item.is_dir():
            item.chmod(0o555)
    directory.chmod(0o555)
    return load("original_native_pose45_math", path), {"original": source, "staged_execution": value}


def validate_complete_arrays(arrays, result, actors):
    require(result.get("status") == "passed" and result.get("error") is None
            and result.get("actual_control_steps") == 6000
            and result.get("arguments", {}).get("num_envs") == actors
            and arrays["body_pos"].shape[:2] == (6000, actors),
            "Native style needs a complete actual 6000-by-N capture")


def pose45(row, catalog):
    """Generalize only actor-count metadata; call original pose mathematics."""
    import numpy as np
    import torch
    from CrowdSim.humanoid.style_evaluation import StyleSpec, KEYPOINTS, sequence_style_samples
    from scripts.evaluate_pedestrian_style import load_reference, assert_heldout_disjoint
    from scripts.evaluate_paper_pedestrians import read_capture

    tool, stage = stage_pose_tool(catalog)
    torch.set_num_threads(4)
    artifact = json.loads(Path(catalog["sources"]["style_reference"]["path"]).read_text())
    require(artifact["selection"]["split"] == "validation", "Only fixed validation reference allowed")
    spec = StyleSpec()
    require(artifact["metric_spec"] == json.loads(json.dumps(asdict(spec))), "Fixed feature specification changed")
    manifest = Path(artifact["manifest"]["path"])
    exact(artifact["manifest"])
    reference_document = json.loads(manifest.read_text())
    body, rotation, sample_hashes, provenance = load_reference(manifest, artifact["selection"])
    require(sample_hashes == artifact["actual_selected_array_sha256"], "Fixed selected reference samples changed")
    episodes = np.ones(body.shape[:2], dtype=np.int64)
    resets = np.zeros(body.shape[:2], dtype=bool)
    resets[0] = True
    reference = sequence_style_samples(body, rotation, episode_ids=episodes, reset_mask=resets, spec=spec)
    basis_path = Path(catalog["sources"]["pose45_basis"]["path"])
    with np.load(basis_path, allow_pickle=False) as data:
        require(set(data.files) == {"centers_50", "centers_100"}, "Fixed basis keys changed")
        centers = {k: data[f"centers_{k}"].copy() for k in (50, 100)}
    for k, values in centers.items():
        require(values.dtype == np.float64 and values.shape == (k, 45) and np.isfinite(values).all(), "Finite fixed45 centers required")
    original = json.loads(Path(catalog["sources"]["pose45_reference_evaluation"]["path"]).read_text())
    fit = deepcopy(original["method"]["fit"])
    require(len(reference["local_pose"]) == fit["available_reference_poses"], "Fixed reference population changed")
    fit.update(refit_performed_for_this_new_run=False, saved_basis_artifact=record(basis_path))
    directory = Path(row["input"])
    states, result_path = directory / "physical_states.npz", directory / "result.json"
    arrays, result, capture = read_capture(states, result_path)
    validate_complete_arrays(arrays, result, row["actors"])
    declared = Path(result["state_capture"]["path"])
    require((ROOT / declared if not declared.is_absolute() else declared).resolve() == states, "Declared state capture differs")
    training_manifest = Path(result["arguments"]["manifest"])
    training_manifest = ROOT / training_manifest if not training_manifest.is_absolute() else training_manifest
    disjoint = assert_heldout_disjoint(artifact["selection"], json.loads(training_manifest.read_text()), reference_document)
    generated = sequence_style_samples(arrays["body_pos"], arrays["root_rot"], body_names=capture["body_names"],
                episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"], spec=spec)
    item = {"sources": {"states": record(states), "result": record(result_path), "training_manifest": record(training_manifest)},
            "capture_validation": capture, "reference_parent_disjointness": disjoint,
            "raw_source_frames_read_without_failure_filter": 6000, "source_actor_frames": 6000 * row["actors"],
            "included_pose_valid_interval_endpoints": len(generated["local_pose"]),
            "excluded_first_frame_actor_samples": row["actors"],
            "excluded_cross_reset_interval_endpoints": generated["summary"]["excluded_reset_transitions"],
            "poses": tool.compare_pose_only(generated["local_pose"], reference["local_pose"], centers),
            "same_episode_intervals": generated["summary"]["same_episode_transition_count"],
            "cross_reset_intervals_excluded": generated["summary"]["excluded_reset_transitions"],
            "outcomes": {k: result.get(k) for k in ("fall_resets", "route_vertices_reached")},
            "scene_as_published": result.get("evaluation_scene"),
            "native_scene_live_observation_as_published": result.get("native_scene_live_observation"),
            "fall_upright_idle_or_outlier_filter_applied": False}
    report = {"schema": "public_state_prior.native_fixed_pose45_metrics.v1", "status": "passed", "CPU_only": True,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "Kit_started": False, "CUDA_initialized": False,
              "source": {"runner": record(__file__), "original_pose_math": catalog["sources"]["pose45_tool"],
                         "staged_original_tool": stage, "original_style_math": catalog["sources"]["style90_math"]},
              "reference": artifact, "reference_load_provenance": provenance,
              "method": {"feature": "unchanged sequence_style_samples.local_pose only; no velocity coordinates",
                         "dimensions": 45, "dt_s": 1 / 30, "keypoints": list(KEYPOINTS),
                         "fit": fit, "actor_count_metadata_adapter": row["actors"],
                         "author_scale_cluster_settings_or_bandwidth_known": False,
                         "numerical_comparison_to_TableIII_valid": False},
              "reference_self_calibration": tool.compare_pose_only(reference["local_pose"], reference["local_pose"], centers),
              "runs": {row["run_name"]: item}, "default_promotion": False,
              "limits": ["Original 45D pose metric functions and fixed b25 centers; actor count adapted to N=4/15.",
                         "All legal same-episode interval endpoint poses retained; first and reset boundary endpoints excluded.",
                         "Dependent adjacent frames; no paired RNG/bitwise controller comparison or overall style improvement claim.",
                         "Pose metrics do not establish physical stability or navigation success."]}
    output = Path(row["output"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "evaluation.json", report)
    np.savez(output / "reference_basis.npz", **{f"centers_{k}": v for k, v in centers.items()})
    with np.load(output / "reference_basis.npz", allow_pickle=False) as saved:
        require(all(np.array_equal(saved[f"centers_{k}"], v) for k, v in centers.items()), "Output fixed centers changed")
    (output / "heldout_reference.json").write_bytes(Path(catalog["sources"]["style_reference"]["path"]).read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRICS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU metrics require CUDA_VISIBLE_DEVICES=''")
    require(record(CATALOG)["sha256"] == CATALOG_SHA, "Frozen native catalog changed")
    catalog = json.loads(CATALOG.read_text())
    validate_catalog(catalog)
    allowed = validate_engine(catalog)
    if args.prepare_only:
        require(args.metric is None, "Preparation cannot select a capture")
        require(not any(n == "torch" or n == "isaacsim" or n.startswith(("omni.", "isaaclab.")) for n in sys.modules),
                "Preparation imported Torch or SDK")
        print(json.dumps({"status": "prepared_sources_passed_no_actual_capture_read", "future_effect": False,
                          "GPU_used": False, "actual_capture_read": False, "actual_checkpoint_read": False,
                          "actual_metrics_computed": False, "metrics": list(METRICS)}))
        return 0
    require(args.metric is not None, "Choose a frozen metric")
    row = catalog["metrics"][args.metric]
    output = Path(row["output"])
    require(not output.exists() and not output.is_symlink(), "Fresh independent output required")
    source = admit_run(catalog, row)
    require(not any(n == "torch" or n == "isaacsim" or n.startswith(("omni.", "isaaclab.")) for n in sys.modules),
            "Native source admission must not load a model or SDK")
    ledger = CPULedger(output.parent / (args.metric + "_CPU_imports.json"), allowed, record(CATALOG))
    ledger.activate()
    status = "failed"
    try:
        if row["dimension"] == 90:
            tool = importlib.import_module("scripts.evaluate_pedestrian_style")
            exact(catalog["sources"]["style90_entry"])
            tool.main(row["metric_argv"])
            report = json.loads((output / "evaluation.json").read_text())
            require(report["sources"] == {"implementation": catalog["sources"]["style90_entry"],
                                           "module": catalog["sources"]["style90_math"]}, "Fixed90 source identity changed")
        else:
            pose45(row, catalog)
        require(record(output / "heldout_reference.json")["sha256"] == catalog["sources"]["style_reference"]["sha256"],
                "Fixed c274 reference changed")
        import torch
        require(not torch.cuda.is_initialized(), "CPU metrics initialized CUDA")
        require(not any(n == "isaacsim" or n.startswith(("omni.", "isaaclab.")) for n in sys.modules), "CPU metrics started SDK")
        validate_engine(catalog)
        write(output / "actual_native_source_admission.json", {"schema": "public_state_prior.native_style_source_admission.v1",
              "status": "passed", "metric": args.metric, "CPU_only": True, "GPU_used": False, "Kit_started": False,
              "sealed34_outputs_read": False, "runner": record(__file__), "catalog": record(CATALOG),
              "actual_native_source_admission": source, "fixed_metric_argv": row["metric_argv"],
              "native_runtime_epoch_is_math_epoch": row["kind"] == "new", "style_pass_is_quality_pass": False,
              "default_promotion": False, "pose_reference_refit_performed": False})
        status = "passed"
    finally:
        ledger.finish(status)
    for item in output.rglob("*"):
        if item.is_file():
            item.chmod(0o444)
    ledger.path.chmod(0o444)
    output.chmod(0o555)
    print(json.dumps({"status": "passed", "metric": args.metric, "output": str(output), "GPU_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
