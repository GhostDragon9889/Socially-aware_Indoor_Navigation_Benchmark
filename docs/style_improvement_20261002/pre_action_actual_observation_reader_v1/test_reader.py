"""Corrupted durable sample evidence must fail without hiding other rows."""

from pathlib import Path
import importlib.util
import json
import hashlib

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "independent_pre_action_reader", HERE / "audit_actual.py"
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def sample(directory, index, *, corrupt_mask=False):
    mean = np.zeros(1044, dtype=np.float64)
    variance = np.ones(1044, dtype=np.float64)
    raw = np.zeros((2, 1044), dtype=np.float32)
    raw[0, 20] = 10.0
    raw[1, 0] = -10.0
    z = raw / np.sqrt(variance.astype(np.float32) + np.float32(1e-5))
    clipped = np.clip(z, -5.0, 5.0)
    mask = z != clipped
    if corrupt_mask:
        mask[0, 20] = False
    arrays = {
        "self_obs": np.zeros((2, 357), dtype=np.float32),
        "task_obs": raw,
        "task_z_unclamped": z,
        "task_z_clamped": clipped,
        "task_clip_mask": mask,
        "task_mean": mean,
        "task_variance": variance,
        "task_count": np.asarray(10.0, dtype=np.float64),
    }
    path = directory / f"call_{index:08d}.npz"
    np.savez(path, **arrays)
    fingerprints = {k: reader.array_record(v) for k, v in arrays.items()}
    moments = {k: fingerprints[k] for k in ("task_mean", "task_variance", "task_count")}
    row = {
        "schema": "paper_pre_action_observation_sample.v1",
        "call_index": index,
        "stride": 30,
        "call_index_is_native_counter": False,
        "native_counter": None,
        "native_engine_time": None,
        "runtime_completion_claim": False,
        "normalization_clip": 5.0,
        "normalization_epsilon": 1e-5,
        "moments_SHA256": hashlib.sha256(
            json.dumps(moments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "file": reader.record(path),
        "arrays": fingerprints,
    }
    with (directory / "samples.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def test_known_clip_counts_keep_prefix_and_terrain_denominators_separate(tmp_path):
    sample(tmp_path, 0)
    sample(tmp_path, 30)
    report = reader.audit_samples(tmp_path, steps=60, actors=2, stride=30)
    assert report["status"] == "passed" and report["verified_samples"] == 2
    a, b = report["verified_sample_statistics"].values()
    assert (
        a["elements"] == 80
        and a["clipped_elements"] == 2
        and a["negative_clipped_elements"] == 2
    )
    assert (
        b["elements"] == 4096
        and b["clipped_elements"] == 2
        and b["positive_clipped_elements"] == 2
    )
    assert b["clipped_fraction_per_actor"] == [1 / 1024, 0.0]


def test_rehashed_wrong_mask_rejected_and_valid_second_row_retained(tmp_path):
    sample(tmp_path, 0, corrupt_mask=True)
    sample(tmp_path, 30)
    report = reader.audit_samples(tmp_path, steps=60, actors=2, stride=30)
    assert report["status"] == "failed" and report["verified_samples"] == 1
    assert report["sample_checks"][0]["call_index"] == 30
    assert (
        report["issues"][0]["line"] == 1
        and "clipping mask" in report["issues"][0]["reason"]
    )
    assert report["all_failed_samples_retained"]


def test_incomplete_final_sample_cannot_report_complete(tmp_path):
    sample(tmp_path, 0)
    report = reader.audit_samples(tmp_path, steps=60, actors=2, stride=30)
    assert report["status"] == "failed" and report["verified_samples"] == 1
    assert {r.get("scope") for r in report["issues"]} == {
        "sample_index_completeness",
        "sample_file_completeness",
    }


def test_dtype_or_byte_difference_is_preserved_without_tolerance(tmp_path):
    np.savez(tmp_path / "a.npz", x=np.asarray([1.0, 2.0], dtype=np.float32))
    np.savez(tmp_path / "b.npz", x=np.asarray([1.0, 2.0 + 1e-6], dtype=np.float32))
    report = reader.compare_archive_fields(tmp_path / "a.npz", tmp_path / "b.npz")
    assert (
        report["bitwise_equal"] is False
        and report["fields"]["x"]["different_elements"] == 1
    )
    assert report["fields"]["x"]["first_different_axis0_index"] == 1
    np.savez(tmp_path / "c.npz", x=np.asarray([1.0, 2.0], dtype=np.float64))
    assert (
        reader.compare_archive_fields(tmp_path / "a.npz", tmp_path / "c.npz")["fields"][
            "x"
        ]["shape_dtype_equal"]
        is False
    )
