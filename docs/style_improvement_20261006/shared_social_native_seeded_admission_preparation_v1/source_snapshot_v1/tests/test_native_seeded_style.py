"""Same-seed CPU contracts; synthetic declarations are never SDK evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.analysis import native_navigation_seeded_style as style
from scripts import evaluate_shared_social_native_seeded_style as cli
from test_native_navigation_style import declared_audit, synthetic_capture


def seeded_audit(actors=3, seed=19):
    owner, audit, _ = declared_audit(actors)
    kernels = style.raw.prepare_kernels(actors=actors, controls=6000, seed=seed)
    owner.update(schema=style.OWNER_SCHEMA, seed=seed)
    audit.update(schema=style.raw.AUDIT_SCHEMA, seed=seed, reader=style.metric.record(style.RAW_CLI),
                 kernel_sources=kernels.sources, kernel_derivation=kernels.derivation)
    audit["execution_admission"].update(registered_seed=seed, receipt_driver_initial_and_entry_seed_equal=True)
    return owner, audit, kernels


@pytest.mark.parametrize("actors", (3, 15))
@pytest.mark.parametrize("seed", (7, 19, 42))
def test_explicit_same_seed_native_style_contract_keeps_original_checks(actors, seed):
    owner, audit, kernels = seeded_audit(actors, seed)
    assert style.validate_audit_contract(owner, audit, kernels, planner="diffusion", seed=seed) is None
    assert style.frozen.OWNER_SCHEMA != style.OWNER_SCHEMA
    assert style.frozen.raw.AUDIT_SCHEMA != style.raw.AUDIT_SCHEMA


@pytest.mark.parametrize("role", ("owner", "audit", "kernels", "admission"))
@pytest.mark.parametrize("value", (None, True, 19., 7, 42))
def test_another_or_coerced_seed_cannot_join_the_pair(role, value):
    owner, audit, kernels = seeded_audit()
    if role == "owner":
        owner["seed"] = value
    elif role == "audit":
        audit["seed"] = value
    elif role == "kernels":
        kernels.derivation["registered_seed"] = value
    else:
        audit["execution_admission"]["registered_seed"] = value
    with pytest.raises(ValueError, match="Registered native seed"):
        style.validate_audit_contract(owner, audit, kernels, planner="diffusion", seed=19)


@pytest.mark.parametrize("defect", ("old_owner", "old_audit", "old_reader", "smoke", "exit", "seed_not_admitted"))
def test_same_seed_does_not_relax_existing_complete_capture_and_source_checks(defect):
    owner, audit, kernels = seeded_audit()
    if defect == "old_owner":
        owner["schema"] = style.frozen.OWNER_SCHEMA
    elif defect == "old_audit":
        audit["schema"] = style.frozen.raw.AUDIT_SCHEMA
    elif defect == "old_reader":
        audit["reader"] = style.metric.record(style.ROOT / style.frozen.SOURCES[2][0])
    elif defect == "smoke":
        audit["actual_controls"] = 30
    elif defect == "exit":
        owner["actual_exit_code"] = False
    else:
        audit["execution_admission"]["receipt_driver_initial_and_entry_seed_equal"] = False
    with pytest.raises(ValueError):
        style.validate_audit_contract(owner, audit, kernels, planner="diffusion", seed=19)


@pytest.mark.parametrize("seed", (None, True, 19., 7, 42))
def test_mismatched_raw_owner_seed_is_refused_before_audit_scene_or_output(seed, tmp_path):
    path = tmp_path / "CPU_FIXTURE_ONLY_owner.json"
    path.write_text(json.dumps({"schema": style.OWNER_SCHEMA, "actual_exit_code": 0,
                               "terminal_tool_chunk": "CPU_FIXTURE_NOT_REAL_EXIT", "seed": seed}))
    with pytest.raises(ValueError, match="Registered native seed"):
        style.read_completed_run(path, actors=3, planner="diffusion", seed=19,
                                  scene_config=tmp_path / "MISSING", scene_sha256="0" * 64)
    child = subprocess.run([sys.executable, str(cli.__file__), "--actors", "3", "--seed", "19",
        "--scene-config", str(tmp_path / "MISSING"), "--scene-sha256", "0" * 64,
        "--diffusion-audit-execution", str(path), "--sfm-audit-execution", str(tmp_path / "MISSING_SFM"),
        "--output", str(tmp_path / "MUST_NOT_BE_WRITTEN.json")], cwd=style.ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    assert child.returncode != 0 and "Registered native seed" in child.stderr
    assert not (tmp_path / "MUST_NOT_BE_WRITTEN.json").exists()


@pytest.mark.parametrize("actors", (3, 15))
def test_pose_extraction_and_failure_intervals_are_the_exact_frozen_function(actors):
    assert style.capture_samples is style.frozen.capture_samples
    physical, ground, result, base = synthetic_capture(actors)
    samples, capture = style.capture_samples(physical, ground, result, base, actors=actors)
    old_samples, old_capture = style.frozen.capture_samples(physical, ground, result, base, actors=actors)
    for key in ("local_pose", "local_pose_velocity", "speed_m_s", "end_frame_indices", "actor_indices"):
        np.testing.assert_array_equal(samples[key], old_samples[key])
    assert samples["summary"] == old_samples["summary"] and capture == old_capture
    assert capture["fall_termination_frames"] == 2 and not capture["failed_episodes_filtered"]


@pytest.mark.parametrize("actors,seed", ((3, 19), (15, 42)))
def test_fresh_seeded_style_CLI_source_checks_do_not_load_Torch_or_SDK(actors, seed):
    child = subprocess.run([sys.executable, str(cli.__file__), "--actors", str(actors),
        "--seed", str(seed), "--check-sources"], cwd=style.ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    doc = json.loads(child.stdout)
    assert doc["seed"] == seed and not doc["Torch_imported"] and not doc["SDK_started"]
    assert not doc["frozen_metric_math_changed"] and not doc["actual_native_capture_or_quality_measured"]


def test_seeded_style_CLI_has_no_default_seed_and_keeps_all_metric_calls():
    namespace, proof = cli.prepared_CLI()
    assert proof["literal_seams"] == 5 and proof["complete_source_inverse"]
    assert namespace.style.metric is style.frozen.metric
    child = subprocess.run([sys.executable, str(cli.__file__), "--actors", "3", "--check-sources"],
        cwd=style.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert child.returncode == 2 and "--seed" in child.stderr
