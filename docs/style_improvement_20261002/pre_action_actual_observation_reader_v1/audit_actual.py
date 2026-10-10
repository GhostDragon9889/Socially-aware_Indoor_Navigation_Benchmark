"""Independent CPU reader for durable pre-action samples and actual30s prefix.

Reads every sample and retains every rejection; no SDK, model, or CUDA import.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE177 = (
    ROOT
    / "docs/style_improvement_20261002/native_runtime_epoch_v8/native_pedestrian_source_v8.json"
)
SOURCE177_SHA = "42b75f3d8655862721ba271b959ef3287d49581dead258ca0376407dc9d70e00"
SOURCE168 = (
    ROOT
    / "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v4.json"
)
SOURCE168_SHA = "07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f"
ARRAYS = {
    "self_obs",
    "task_obs",
    "task_z_unclamped",
    "task_z_clamped",
    "task_clip_mask",
    "task_mean",
    "task_variance",
    "task_count",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def array_record(array):
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256_C_order": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def read(path):
    return json.loads(Path(path).read_text())


def verify_source(path, expected_sha, expected_count):
    info = record(path)
    require(info["sha256"] == expected_sha, "Selected source manifest changed")
    document = read(path)
    archive = record(document["archive"]["path"])
    require(
        archive["bytes"] == document["archive"]["bytes"]
        and archive["sha256"] == document["archive"]["sha256"],
        "Source archive changed",
    )
    rows = {}
    for row in document["files"]:
        target = Path(row["path"])
        if target.is_absolute():
            target = target.relative_to(ROOT)
        require(
            ".." not in target.parts and str(target) not in rows,
            "Unsafe/duplicate source path",
        )
        rows[str(target)] = row
    require(len(rows) == expected_count, "Source file count differs")
    payloads = {}
    with tarfile.open(archive["path"]) as stream:
        members = stream.getmembers()
        require(
            len(members) == expected_count and {m.name for m in members} == set(rows),
            "Source tar members differ",
        )
        for member in members:
            require(
                member.isfile() and member.size <= 16 * 1024 * 1024,
                "Source member unsafe",
            )
            data = stream.extractfile(member).read()
            require(
                len(data) == rows[member.name]["bytes"]
                and hashlib.sha256(data).hexdigest() == rows[member.name]["sha256"],
                "Source member bytes differ",
            )
            payloads[member.name] = data
    return (
        {
            "manifest": info,
            "archive": archive,
            "selected_files": expected_count,
            "full_import_ledger": False,
        },
        rows,
        payloads,
    )


def source_order(payloads):
    tree = ast.parse(payloads["CrowdSim/humanoid/runtime/pre_action_observation.py"])
    proxy = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PreActionObservedActor"
    )
    act = next(
        n for n in proxy.body if isinstance(n, ast.FunctionDef) and n.name == "act"
    )
    original = [
        n
        for n in ast.walk(act)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "act"
    ]
    publications = [
        n
        for n in ast.walk(act)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "publish"
    ]
    raw_clones = [
        n
        for n in ast.walk(act)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"raw_self", "raw_task"}
            for t in n.targets
        )
    ]
    normalizes = [
        n
        for n in ast.walk(act)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "normalize"
    ]
    driver = ast.parse(payloads["scripts/train_paper_amp.py"])
    loops = [
        n
        for n in ast.walk(driver)
        if isinstance(n, ast.For)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "act"
            for c in ast.walk(n)
        )
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "step"
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "world"
            for c in ast.walk(n)
        )
    ]
    loop = min(loops, key=lambda n: n.end_lineno - n.lineno)
    actor_calls = [
        n
        for n in ast.walk(loop)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "act"
    ]
    steps = [
        n
        for n in ast.walk(loop)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "step"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "world"
    ]
    require(
        len(original) == len(publications) == len(actor_calls) == len(steps) == 1
        and len(raw_clones) == len(normalizes) == 2
        and max(n.lineno for n in raw_clones + normalizes)
        < original[0].lineno
        < publications[0].lineno
        and actor_calls[0].lineno < steps[0].lineno,
        "Frozen source call-order contract differs",
    )
    return {
        "raw_clones_before_original_act": True,
        "publish_after_act_before_driver_step": True,
        "additional_normalizer_forwards_per_sample": 2,
        "additional_model_forward_in_proxy_source": 0,
        "runtime_model_forward_counter_measured": False,
        "scope": "Frozen source AST + CPU contracts; no runtime native timestamps for observer",
    }


def validate_sample(directory, row, actors, stride):
    require(
        row["schema"] == "paper_pre_action_observation_sample.v1",
        "Sample schema differs",
    )
    index = row["call_index"]
    require(
        type(index) is int
        and index >= 0
        and index % stride == 0
        and row["stride"] == stride,
        "Sample index/stride differs",
    )
    require(
        row["call_index_is_native_counter"] is False
        and row["native_counter"] is None
        and row["native_engine_time"] is None
        and row["runtime_completion_claim"] is False,
        "Observer counter/completion scope differs",
    )
    require(
        row["normalization_clip"] == 5.0 and row["normalization_epsilon"] == 1e-5,
        "Normalizer clip/epsilon differs",
    )
    path = directory / f"call_{index:08d}.npz"
    require(record(path) == row["file"], "NPZ file path/bytes/SHA differ")
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(archive.files) == set(row["arrays"]) == ARRAYS,
            "Sample array keys differ",
        )
        arrays = {key: archive[key].copy() for key in archive.files}
    expected = {
        "self_obs": ((actors, 357), "<f4"),
        "task_obs": ((actors, 1044), "<f4"),
        "task_z_unclamped": ((actors, 1044), "<f4"),
        "task_z_clamped": ((actors, 1044), "<f4"),
        "task_clip_mask": ((actors, 1044), "|b1"),
        "task_mean": ((1044,), "<f8"),
        "task_variance": ((1044,), "<f8"),
        "task_count": ((), "<f8"),
    }
    for key, array in arrays.items():
        require(
            array.shape == expected[key][0] and array.dtype.str == expected[key][1],
            "Array shape/dtype differs:" + key,
        )
        require(
            np.isfinite(array).all() and array_record(array) == row["arrays"][key],
            "Array finite/SHA differs:" + key,
        )
    require(
        (arrays["task_variance"] >= 0).all() and arrays["task_count"] >= 0,
        "Invalid RMS moments",
    )
    moments = {
        key: array_record(arrays[key])
        for key in ("task_mean", "task_variance", "task_count")
    }
    require(
        hashlib.sha256(
            json.dumps(moments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == row["moments_SHA256"],
        "Moments composite SHA differs",
    )
    # Independent NumPy recomputation: GPU sqrt/divide can differ in final FP32 bits.
    expected_z = (
        arrays["task_obs"] - arrays["task_mean"].astype(np.float32)
    ) / np.sqrt(arrays["task_variance"].astype(np.float32) + np.float32(1e-5))
    error = float(np.max(np.abs(expected_z - arrays["task_z_unclamped"])))
    require(
        np.allclose(expected_z, arrays["task_z_unclamped"], rtol=3e-6, atol=2e-5),
        "Independent normalization formula mismatch",
    )
    clipped = np.clip(arrays["task_z_unclamped"], -5.0, 5.0)
    require(
        np.array_equal(clipped, arrays["task_z_clamped"]),
        "Stored clipping does not match z",
    )
    require(
        np.array_equal(arrays["task_clip_mask"], arrays["task_z_unclamped"] != clipped),
        "Stored clipping mask differs",
    )
    return arrays, {
        "call_index": index,
        "status": "sample_verified",
        "NPZ": record(path),
        "moments_SHA256": row["moments_SHA256"],
        "normalization_numpy_max_error": error,
        "normalization_comparison_scope": "FP32 numeric tolerance; stored array identity separately exact",
    }


def statistics(arrays, column_slice):
    raw = np.stack([a["task_obs"][:, column_slice] for a in arrays])
    z = np.stack([a["task_z_unclamped"][:, column_slice] for a in arrays])
    mask = np.stack([a["task_clip_mask"][:, column_slice] for a in arrays])

    def distribution(x):
        return {
            "minimum": float(x.min()),
            "maximum": float(x.max()),
            "mean": float(x.mean(dtype=np.float64)),
            "abs_quantiles_50_95_99_100": np.quantile(
                np.abs(x), [0.5, 0.95, 0.99, 1]
            ).tolist(),
        }

    return {
        "sampled_actorframes": int(raw.shape[0] * raw.shape[1]),
        "elements": int(raw.size),
        "clipped_elements": int(mask.sum()),
        "clipped_fraction": float(mask.mean()),
        "positive_clipped_elements": int(((z > 5) & mask).sum()),
        "negative_clipped_elements": int(((z < -5) & mask).sum()),
        "sampled_actorframes_with_any_clip": int(mask.any(-1).sum()),
        "clipped_fraction_per_actor": mask.mean(axis=(0, 2)).tolist(),
        "raw": distribution(raw),
        "unclamped_z": distribution(z),
    }


def audit_samples(directory, *, steps, actors, stride):
    directory = Path(directory).resolve()
    require(
        type(steps) is int
        and steps > 0
        and type(actors) is int
        and actors > 0
        and type(stride) is int
        and stride > 0,
        "Invalid expected sample contract",
    )
    issues, checks, accepted, indices = [], [], [], []
    path = directory / "samples.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    for number, text in enumerate(lines, 1):
        try:
            row = json.loads(text)
            indices.append(row.get("call_index"))
            arrays, result = validate_sample(directory, row, actors, stride)
            checks.append(result)
            accepted.append(arrays)
        except Exception as error:
            issues.append(
                {
                    "line": number,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "raw_JSONL_text": text,
                }
            )
    expected = list(range(0, steps, stride))
    if indices != expected:
        issues.append(
            {
                "scope": "sample_index_completeness",
                "expected_indices": expected,
                "actual_indices": indices,
            }
        )
    npz_names = sorted(p.name for p in directory.glob("*.npz"))
    if npz_names != [f"call_{i:08d}.npz" for i in expected]:
        issues.append(
            {
                "scope": "sample_file_completeness",
                "expected_count": len(expected),
                "actual_names": npz_names,
            }
        )
    hashes = [r["moments_SHA256"] for r in checks]
    if len(set(hashes)) > 1:
        issues.append(
            {
                "scope": "evaluation_RMS_frozen",
                "distinct_moments_SHA256": sorted(set(hashes)),
            }
        )
    result = {
        "status": "passed" if not issues else "failed",
        "expected_samples": len(expected),
        "verified_samples": len(checks),
        "sample_checks": checks,
        "issues": issues,
        "evaluation_RMS_moments_unchanged_across_recorded_samples": len(set(hashes))
        == 1,
        "unrecorded_calls_not_directly_observed": True,
        "all_failed_samples_retained": True,
    }
    if accepted:
        result["verified_sample_statistics"] = {
            "trajectory_prefix20": statistics(accepted, slice(0, 20)),
            "terrain_suffix1024": statistics(accepted, slice(20, 1044)),
        }
    return result


def compare_archive_fields(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(
        right, allow_pickle=False
    ) as b:
        result = {"keys_equal": set(a.files) == set(b.files), "fields": {}}
        for name in sorted(set(a.files) | set(b.files)):
            if name not in a.files or name not in b.files:
                result["fields"][name] = {
                    "missing_side": "baseline" if name not in a.files else "candidate",
                    "bitwise_equal": False,
                }
                continue
            x, y = a[name], b[name]
            shape_dtype = x.shape == y.shape and x.dtype == y.dtype
            row = {
                "baseline": array_record(x),
                "candidate": array_record(y),
                "shape_dtype_equal": shape_dtype,
                "bitwise_equal": shape_dtype and x.tobytes() == y.tobytes(),
            }
            if shape_dtype and x.dtype.kind in "biuf":
                unequal = x != y
                row["different_elements"] = int(unequal.sum())
                row["maximum_absolute_difference"] = (
                    float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                    if x.size
                    else 0.0
                )
                if x.ndim and unequal.any():
                    row["first_different_axis0_index"] = int(
                        np.flatnonzero(unequal.reshape(x.shape[0], -1).any(-1))[0]
                    )
            result["fields"][name] = row
        result["bitwise_equal"] = result["keys_equal"] and all(
            r["bitwise_equal"] for r in result["fields"].values()
        )
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "output/diffusion_amp/pre_action_baseline_cuboid_30s_v1",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "output/diffusion_amp/pure_actor_only_baseline_30s_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args(argv)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(),
        "CPU-only and new evidence output required",
    )
    new_source, new_rows, payloads = verify_source(SOURCE177, SOURCE177_SHA, 177)
    old_source, old_rows, _ = verify_source(SOURCE168, SOURCE168_SHA, 168)
    changed = [
        name
        for name in sorted(set(old_rows) | set(new_rows))
        if name not in old_rows
        or name not in new_rows
        or old_rows[name]["sha256"] != new_rows[name]["sha256"]
    ]
    report = {
        "schema": "actual_pre_action_observation_independent_review.v1",
        "CPU_only": True,
        "SDK_started": False,
        "GPU_used": False,
        "candidate_execution_completed_observed": False,
        "candidate_sample_audit": None,
        "physical_pair_bitwise_equal": None,
        "new_source177": new_source,
        "historical_source168": old_source,
        "changed_selected_source_paths": changed,
        "source_call_order": source_order(payloads),
        "reader": record(__file__),
        "limitations": [
            "Pre-action raw clone before act; durable publication after act before step is frozen source evidence, not engine timestamp.",
            "Additional normalizer forwards are not model forwards; no per-call runtime model counter was instrumented.",
            "30s samples/prefix do not replace full200s stability or guarantee PhysX/Warp bitwise.",
            "Clipping measurement does not prove idle/failure causality.",
            "Historical/current source differ; selected source archives are not full import ledgers.",
            "All physical failed/reset frames are retained; initial capture is post-control, not a pre-step initial GT/history oracle.",
        ],
    }
    if args.preparation_only:
        report["status"] = "prepared_pending_actual"
    else:
        candidate, baseline = args.candidate.resolve(), args.baseline.resolve()
        result, old = read(candidate / "result.json"), read(baseline / "result.json")
        report["actual_result_artifact"] = record(candidate / "result.json")
        report["actual_result_status"] = result.get("status")
        steps = result["actual_control_steps"]
        actors = result["arguments"]["num_envs"]
        report["candidate_execution_completed_observed"] = (
            result["status"] == "passed" and steps == 900 and actors == 4
        )
        metadata = read(candidate / "pre_action_observer.json")
        require(
            metadata["schema"] == "paper_pre_action_observer_delegation.v1"
            and metadata["stride"] == 30,
            "Actual observer schema/stride differs",
        )
        for item in metadata["observer_loaded_source_files"]:
            name = str(Path(item["path"]).relative_to(ROOT))
            require(
                name
                in {
                    "scripts/probe_paper_pre_action.py",
                    "CrowdSim/humanoid/runtime/pre_action_observation.py",
                }
                and item["bytes"] == new_rows[name]["bytes"]
                and item["sha256"] == new_rows[name]["sha256"],
                "Actually loaded observer source differs from source177",
            )
        require(
            len(metadata["observer_loaded_source_files"]) == 2,
            "Observer source scope differs",
        )
        report["actual_observer_metadata"] = {
            "artifact": record(candidate / "pre_action_observer.json"),
            "content": metadata,
        }
        report["candidate_sample_audit"] = audit_samples(
            candidate / "pre_action_observations",
            steps=steps,
            actors=actors,
            stride=metadata["stride"],
        )
        report["status"] = (
            "passed"
            if report["candidate_execution_completed_observed"]
            and report["candidate_sample_audit"]["status"] == "passed"
            else "failed"
        )
        left, right = old["arguments"], result["arguments"]
        report["driver_argument_differences"] = {
            k: {"baseline": left.get(k), "candidate": right.get(k)}
            for k in sorted(set(left) | set(right))
            if left.get(k) != right.get(k)
        }
        entry = read(candidate / "deployment_entry.json")
        require(
            entry["parsed_driver_arguments"] == right,
            "Parsed/actual driver arguments differ",
        )
        report["evaluation_contract_equalities"] = {
            key: old[key] == result[key]
            for key in (
                "runtime_numerics",
                "physics",
                "evaluation_dataset",
                "evaluation_scene",
                "training_config",
            )
        }
        comparisons = {
            name: compare_archive_fields(
                baseline / (name + ".npz"), candidate / (name + ".npz")
            )
            for name in ("physical_states", "reference_plans", "ground_support")
        }
        report["all_raw_field_comparisons"] = comparisons
        report["physical_pair_bitwise_equal"] = all(
            block["bitwise_equal"] for block in comparisons.values()
        )
        report["actual_artifacts"] = [
            record(candidate / (n + ".npz")) for n in comparisons
        ]
        report["historical_artifacts"] = [
            record(baseline / (n + ".npz")) for n in comparisons
        ]
        spec = importlib.util.spec_from_file_location(
            "frozen_prefix_resource_reader",
            ROOT / "docs/paper_pedestrian_deployment/compare_physical_prefix.py",
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        require(
            record(helper.__file__)["sha256"]
            == "334bdc58ad1f455c6184763a64adcded72bf5deba1b1990c6180aebd7a57c826",
            "Frozen resource reader changed",
        )
        old_resources, resources = (
            helper.resource_identity(old),
            helper.resource_identity(result),
        )
        report["resources"] = {
            "baseline": old_resources,
            "candidate": resources,
            "equal": old_resources == resources,
        }
        require(
            old_resources == resources,
            "Actual selected checkpoint/manifest/fixture resources differ",
        )
        report["candidate_outcomes"] = {
            k: result.get(k)
            for k in (
                "actual_control_steps",
                "fall_resets",
                "time_limit_resets",
                "route_vertices_reached",
                "elapsed_s",
            )
        }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    text = (
        "# 实际策略调用前观测独立审计\n\n"
        + report["status"]
        + "。CPU-only，无模型/SDK/GPU。\n\n"
    )
    if report["candidate_sample_audit"]:
        stats = report["candidate_sample_audit"].get("verified_sample_statistics", {})
        text += (
            "完整采样："
            + str(report["candidate_sample_audit"]["verified_samples"])
            + "条。\n\n"
        )
        for name, values in stats.items():
            text += f"{name}: {values['sampled_actorframes']} sampled actorframes / {values['elements']} elements; clipped {values['clipped_elements']} ({values['clipped_fraction']:.6%})。\n\n"
        text += (
            "历史 source168 与实际 source177 的 physical/plan/ground 全字段 bitwise equal="
            + str(report["physical_pair_bitwise_equal"])
            + "；具体差异见JSON，不用容差替代原数组比较。\n\n"
        )
    text += "\n\n".join(report["limitations"]) + "\n"
    (args.output / "report.md").write_text(text)
    print(
        json.dumps(
            {
                "status": report["status"],
                "physical_pair_bitwise_equal": report["physical_pair_bitwise_equal"],
                "samples": report["candidate_sample_audit"]["verified_samples"]
                if report["candidate_sample_audit"]
                else None,
            }
        )
    )
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
