"""CPU-only retained native arm-ablation audit; frozen raw gates are reused."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
RUN = ROOT / "output/diffusion_amp/native_warehouse_arm_alpha025_30s_v1"
BASELINE = ROOT / "output/diffusion_amp/native_warehouse_clearance08_200s_v1"
SOURCE = ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v5.json"
OLD_SOURCE = SOURCE.with_name("native_pedestrian_source_v4.json")
POST_SHA = "ee9cf54f8a26f56023b0062e20e87b654ac80dbbd78d3156cf6644c11f773577"
OLD_AUDIT = ROOT / "docs/style_improvement_20261002/native_scene_200s_actual_review_v1/actual_audit.json"


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def load(path):
    spec = importlib.util.spec_from_file_location("frozen_native200_wrapper", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def comparison(left, right):
    same_shape, same_dtype = left.shape == right.shape, left.dtype == right.dtype
    equal = same_shape and same_dtype and left.tobytes(order="C") == right.tobytes(order="C")
    row = {"candidate_shape": list(left.shape), "baseline_prefix_shape": list(right.shape),
           "candidate_dtype": str(left.dtype), "baseline_dtype": str(right.dtype),
           "shape_equal": same_shape, "dtype_equal": same_dtype, "raw_bytes_equal": equal,
           "candidate_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
           "baseline_prefix_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
           "max_absolute_difference": None, "different_scalar_count": None}
    if same_shape and same_dtype and left.dtype.kind in "biuf":
        row["different_scalar_count"] = int(np.count_nonzero(left != right))
        row["max_absolute_difference"] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.
    return row


def main():
    target = DOC / "actual_audit.json"
    if target.exists():
        raise FileExistsError("Preserve completed independent audit")
    post_path = DOC / "source171_actual_post_snapshot.json"
    if record(post_path)["sha256"] != POST_SHA:
        raise ValueError("Actual post-source snapshot changed")
    post = json.loads(post_path.read_text())
    if (post["source_count"] != 171 or post["all_actual_post_current_source171_bytes_exact"] is not True
            or record(SOURCE) != post["manifest"] or record(post["archive"]["path"]) != post["archive"]):
        raise ValueError("Expected exact source171 original prelaunch/post snapshot")
    actual_rows = {str(Path(row["path"]).relative_to(ROOT)): row for row in post["records"]}
    source = json.loads(SOURCE.read_text())
    if len(actual_rows) != 171 or any(any(actual_rows[r["path"]][key] != r[key] for key in ("bytes", "sha256")) for r in source["files"]):
        raise ValueError("Source171 post rows do not match exact prelaunch source bytes")
    wrapper = load(ROOT / "docs/style_improvement_20261002/native_scene_200s_actual_review_v1/audit_native_200s.py")
    native = wrapper.readers()
    report = native.audit(RUN, SOURCE)
    c = report["counts"]
    if any(c[key] != value for key, value in {"controls": 900, "actors": 4, "duration_s": 30.,
            "actor_controls": 3600, "shared_physics_substeps": 3600, "actor_physics_substeps": 14400,
            "first_counter": 0, "last_counter": 3600}.items()):
        raise ValueError("Actual900 controls/four actors/3600 substeps protocol mismatch")
    if report["evidence_limits"]["per_query_counter_rows_retained"] != 132:
        raise ValueError("Native pre-reset132 all-hit floor query proof missing")
    result = json.loads((RUN / "result.json").read_text())
    registration_path = ROOT / "docs/style_improvement_20261002/native_arm_alpha025_30s_registration_v1/registration.json"
    registration = json.loads(registration_path.read_text())
    blend = result["evaluation_actor"]["experimental_arm_style"]
    if (registration["source_manifest_sha256"] != post["manifest"]["sha256"] or registration["alpha"] != .25
            or blend["alpha"] != .25 or blend["blended_active_coordinates"] != 24
            or blend["action_dimensions"] != 69 or blend["projection"] != "blend24 raw means, then clamp[-1,1] times canonical57-active mask"
            or blend["baseline"]["checkpoint_sha256"] != registration["baseline_checkpoint_sha256"]
            or blend["style"]["checkpoint_sha256"] != registration["donor_checkpoint_sha256"]):
        raise ValueError("Actual declared arm-blend contract differs from prior explicit registration")
    for name in ("baseline", "style"):
        actor = blend[name]
        observed = record(actor["checkpoint"])
        if observed["bytes"] != actor["checkpoint_bytes"] or observed["sha256"] != actor["checkpoint_sha256"]:
            raise ValueError("Actual loaded arm actor full checkpoint identity changed")
    if record(OLD_AUDIT)["sha256"] != "882f17cc3c45c5b216d17b0e7bfaf90a9bd41c49bb9e4960b509befdb841b389":
        raise ValueError("Historical Native4 actual200s audit changed")
    old_audit = json.loads(OLD_AUDIT.read_text())
    old_result = json.loads((BASELINE / "result.json").read_text())
    for name, row in old_audit["artifacts"].items():
        if record(BASELINE / name) != row:
            raise ValueError("Native200s baseline artifact changed: " + name)
    if record(OLD_SOURCE) != old_audit["source_epoch"]["manifest"]:
        raise ValueError("Native200s source168 manifest changed")
    comparisons, prefix = {}, {}
    for kind, filename in (("physical", "physical_states.npz"), ("ground", "ground_support.npz"), ("plans", "reference_plans.npz")):
        actual, old = native.BASE.arrays(RUN / filename), native.BASE.arrays(BASELINE / filename)
        selected = old["control_step"] < 900 if kind == "plans" else None
        old_prefix = {key: (value[selected] if kind == "plans" else value[:900] if len(value.shape) and value.shape[0] == 6000 else value)
                      for key, value in old.items()}
        if set(actual) != set(old_prefix):
            raise ValueError("Native baseline/candidate schema differs: " + kind)
        comparisons[kind] = {key: comparison(actual[key], old_prefix[key]) for key in actual}
        prefix[kind] = old_prefix
    old_physical = prefix["physical"]
    arg_diffs = {k: {"candidate": result["arguments"].get(k), "baseline200": old_result["arguments"].get(k)}
                 for k in sorted(set(result["arguments"]) | set(old_result["arguments"]))
                 if result["arguments"].get(k) != old_result["arguments"].get(k)}
    if set(arg_diffs) != {"output", "evaluate_steps", "episode_steps"}:
        raise ValueError("Unexpected driver argument change in the registered ablation")
    equal_contracts = {}
    for key in ("physics", "evaluation_scene", "physical_contact_measurement", "ground_support_measurement", "native_scene_provider"):
        a, b = result[key], old_result[key]
        if key == "ground_support_measurement":
            a, b = a["contract"], b["contract"]
        equal_contracts[key] = a == b
    if not all(equal_contracts.values()):
        raise ValueError("Native rig/scene/measurement/provider contract differs from baseline")
    old_source = json.loads(OLD_SOURCE.read_text())
    old_files = {r["path"]: r for r in old_source["files"]}
    new_files = {r["path"]: r for r in source["files"]}
    source_changes = {"old168": record(OLD_SOURCE), "actual171": record(SOURCE),
                      "common_member_count": len(set(old_files) & set(new_files)),
                      "changed_common_paths": [k for k in sorted(set(old_files) & set(new_files)) if old_files[k] != new_files[k]],
                      "added_paths": sorted(set(new_files) - set(old_files)), "removed_paths": sorted(set(old_files) - set(new_files)),
                      "whole_source_identical": False}
    flat = [v for values in comparisons.values() for v in values.values()]
    report.update(schema="native_arm_alpha025_actual30s_independent_CPU_audit.v1",
                  actual_post_source_snapshot={"record": record(post_path), "snapshot": post},
                  explicit_prior_ablation_registration={"record": record(registration_path), "registration": registration},
                  arm_blend={k: blend[k] for k in ("alpha", "arm_action_names", "arm_action_indices", "blended_active_coordinates", "projection", "observation_normalization")},
                  arm_weights={name: record(blend[name]["checkpoint"]) for name in ("baseline", "style")},
                  baseline={"run": str(BASELINE), "historical_actual200s_audit": record(OLD_AUDIT), "prefix_controls": 900,
                            "prefix_falls": int(old_physical["terminated"].sum()), "prefix_time_limits": int(old_physical["truncated"].sum()),
                            "prefix_route_vertices": old_physical["route_vertex_reached"].sum(0).tolist(),
                            "prefix_static_actor_controls": int(old_physical["static_contact"].sum()),
                            "prefix_peer_unique_pair_controls": int(old_physical["peer_contact"][:, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]].sum())},
                  baseline_contracts_equal=equal_contracts, executed_driver_argument_differences=arg_diffs,
                  source_epoch_comparison=source_changes, raw_prefix_comparisons=comparisons,
                  raw_prefix_counts={"fields": len(flat), "bytes_exact_fields": sum(v["raw_bytes_equal"] for v in flat),
                                     "different_fields": sum(not v["raw_bytes_equal"] for v in flat)},
                  execution_reader=record(__file__))
    report["counts"]["feet_observations"] = 900 * 4 * 4 * 4
    report["evidence_limits"].update(baseline="Real native Warehouse4 first900 controls of immutable200s capture; not cuboid scene.",
                                     whole_code_single_variable_claim=False,
                                     stale_prepared_actor_metadata_note="Nested actor status/physical_validation_performed describe load-time preparation; actual raw900 physics is independently audited here.",
                                     current_checkout_after_post_snapshot_is_not_execution_source=True,
                                     native_capture_probe_passed_historical_value_preserved=True,
                                     raw_prefix_bytes_equal_is_not_required_for_different_controller=True)
    if any(n in sys.modules for n in ("omni", "isaacsim", "isaaclab", "torch")):
        raise RuntimeError("CPU verifier imported prohibited execution provider")
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"], "counts": report["counts"],
                      "raw_prefix": report["raw_prefix_counts"], "source_changes": source_changes["changed_common_paths"]}))


if __name__ == "__main__":
    main()
