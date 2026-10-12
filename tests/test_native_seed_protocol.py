"""CPU seed forwarding/resource registration; no native World or quality proof."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts import evaluate_shared_social_native_seeded as entry


@pytest.mark.parametrize("seed", (True, False, None, -1, 2**32, 7., "7"))
def test_seed_requires_a_genuine_bounded_integer(seed):
    with pytest.raises(ValueError, match="explicit integer"):
        entry.validate_seed(seed)


@pytest.mark.parametrize("seed", (0, 7, 19, 42, 2**32 - 1))
def test_seed_boundaries_are_explicit(seed):
    entry.validate_seed(seed)


@pytest.mark.parametrize("actors,planner", ((3, "diffusion"), (15, "diffusion"), (3, "sfm"), (15, "sfm")))
@pytest.mark.parametrize("seed", (7, 19, 42))
def test_seed_variant_retains_the_exact_paired_physical_driver(actors, planner, seed):
    before, original_binding = entry.paired.prepare_entry(actors=actors, planner=planner)
    namespace, binding = entry.prepare_entry(actors=actors, planner=planner, seed=seed)
    old_driver, old_proof = before._derive(before.DRIVER.read_text())
    driver, proof = namespace._derive(namespace.DRIVER.read_text())
    assert driver == old_driver and proof == old_proof
    assert namespace._native_registered_seed == seed
    assert "_native_registered_seed" not in before.__dict__
    assert original_binding["entry"] == entry.record(entry.PAIRED)
    assert binding["seed7_paired_entry"] == original_binding["entry"]
    assert binding["entry"] == entry.record(entry.__file__)
    migration = binding["registered_native_seed"]
    assert migration["seed"] == seed and migration["seed_forwarded_to_original_driver"] is True
    assert migration["main_derivation"]["complete_inverse"] is True
    assert migration["main_derivation"]["control_planner_capture_or_metric_math_changed"] is False
    assert migration["extra_reseed_in_capture_hook"] is False
    assert migration["actual_native_SDK_execution_verified"] is False


def test_forwarded_seed_cannot_change_after_namespace_registration(monkeypatch):
    namespace, _ = entry.prepare_entry(actors=3, planner="diffusion", seed=19)
    # Complete argparse declarations only. No resource exists or is read: the
    # changed seed must fail before the actual resource/SDK admission begins.
    arguments = ["CPU_FIXTURE", "--amp-checkpoint", "DOES_NOT_EXIST.pt", "--amp-sha256", "0" * 64,
        "--training-result", "DOES_NOT_EXIST.json", "--training-result-sha256", "0" * 64,
        "--diffusion-checkpoint", "DOES_NOT_EXIST.ckpt", "--diffusion-sha256", "0" * 64,
        "--actors", "3", "--seed", "42", "--scene-config", "DOES_NOT_EXIST.json", "--scene-sha256", "0" * 64,
        "--control-steps", "30", "--output", "DOES_NOT_EXIST_output", "--preflight-only"]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="differs from the registered"):
        namespace.main()


def test_original_source_and_outer_CLI_seams_are_exactly_guarded():
    source, proof = entry.outer_CLI_source()
    assert proof["literal_seams"] == 3 and proof["complete_inverse"] is True
    assert proof["unchanged_helpers_AST_equal"] is True
    assert '"--seed", str(options.seed)' in source
    with pytest.raises(ValueError, match="Ambiguous native seed seam"):
        entry.derive_seed_interface("def main():\n    pass\n", [("MISSING", "NEW")], role="CPU_TEST_ONLY")


@pytest.mark.parametrize("planner,actors,seed", (("diffusion", 3, 19), ("sfm", 15, 42)))
def test_explicit_seed_source_CLI_is_fresh_and_does_not_import_Torch_or_SDK(planner, actors, seed):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(entry.ROOT / "scripts/evaluate_shared_social_native_seeded.py"),
        "--actors", str(actors), "--planner", planner, "--seed", str(seed), "--check-sources"],
        cwd=entry.ROOT, env=env, capture_output=True, text=True, check=True)
    doc = json.loads(result.stdout)
    assert doc["Torch_imported"] is False and doc["SDK_started"] is False
    assert doc["migration"]["registered_native_seed"]["seed"] == seed
    assert doc["migration"]["registered_native_seed"]["outer_CLI_derivation"]["complete_inverse"] is True


def test_seed_cannot_silently_default_to_seven_in_the_new_CLI():
    result = subprocess.run([sys.executable, str(entry.ROOT / "scripts/evaluate_shared_social_native_seeded.py"),
        "--actors", "3", "--planner", "diffusion", "--check-sources"],
        cwd=entry.ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--seed" in result.stderr
