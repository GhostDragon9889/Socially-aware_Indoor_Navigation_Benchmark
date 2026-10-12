"""The frozen reference may move checkout roots, never change content or math."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from CrowdSim.humanoid.analysis import cpu_dependency_migration as migration
from scripts import evaluate_shared_social_navigation_style as metric
from social_navigation.artifact_identity import artifact_path


HISTORICAL_SHA = "394ce99e1eae7d89c886adc5cbcd6cb292b58bee3758609522a7b72270e6c10a"


def frozen_manifest_identity():
    return json.loads(metric.frozen.REFERENCE.read_text())["manifest"]


def test_real_manifest_retains_frozen_bytes_and_only_changes_checkout_spelling():
    old = frozen_manifest_identity()
    current = metric.record(artifact_path(old["path"]))
    assert current["path"] != old["path"]
    assert current["bytes"] == old["bytes"] == 143512
    assert current["sha256"] == old["sha256"] == "4eb6018e83c76408da05c2ca2358b518d4a520ef37d858ed87dc97dca9e7f1e3"
    assert metric.bound(old) == json.loads(Path(current["path"]).read_text())


@pytest.mark.parametrize("field,value", [
    ("bytes", True), ("bytes", 143513), ("sha256", "0" * 64),
    ("path", "data/diffusion_amp/amass_manifest.json"), ("path", 42),
    ("path", "project:data/../amass_manifest.json"),
])
def test_invalid_or_changed_frozen_identity_is_rejected(field, value):
    identity = frozen_manifest_identity()
    identity[field] = value
    with pytest.raises(ValueError):
        metric.bound(identity)


def test_identical_foreign_file_cannot_stand_in_for_registered_path(monkeypatch, tmp_path):
    expected = frozen_manifest_identity()
    other = tmp_path / "manifest.json"
    other.write_bytes(artifact_path(expected["path"]).read_bytes())
    foreign = metric.record(other)
    monkeypatch.setattr(metric, "record", lambda path: foreign)
    with pytest.raises(ValueError, match="Recorded input changed"):
        metric.bound(expected)


def test_actual_content_change_rejected_even_with_same_length(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"value": 1}')
    before = metric.record(path)
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="Recorded input changed"):
        metric.bound(before)


def test_reader_registration_proves_complete_historical_source_inverse():
    proof = migration.verify_reader(Path(metric.__file__), HISTORICAL_SHA)
    assert proof["whole_source_location_inverse"]
    assert proof["all_metric_function_bytes_unchanged"]
    assert proof["reference_manifest_rechecked_after_measurement"]
    assert proof["SDK_execution_verified"] is False


@pytest.mark.parametrize("mutation", ["metric", "adapter", "duplicate"])
def test_reader_inverse_rejects_unregistered_changes(monkeypatch, mutation):
    path = Path(metric.__file__).resolve()
    original = Path.read_text
    source = original(path)
    if mutation == "metric":
        changed = source.replace("np.mean([", "np.max([", 1)
    elif mutation == "adapter":
        changed = source.replace("require(same_artifact(record(path), identity, root=ROOT)", "require(True", 1)
    else:
        changed = source + '\n    body, rotation, hashes, provenance = tool.load_reference(manifest, artifact["selection"])\n'
    assert changed != source
    monkeypatch.setattr(Path, "read_text", lambda p, *args, **kwargs: changed if p == path else original(p, *args, **kwargs))
    with pytest.raises(ValueError, match="Frozen style"):
        migration.verify_reader(path, HISTORICAL_SHA)


def test_reader_inverse_rejects_foreign_source_and_unregistered_parent(tmp_path):
    other = tmp_path / Path(metric.__file__).name
    other.write_bytes(Path(metric.__file__).read_bytes())
    for path, parent in ((other, HISTORICAL_SHA), (Path(metric.__file__), "0" * 64)):
        with pytest.raises(ValueError, match="Unregistered"):
            migration.verify_reader(path, parent)


def test_real_frozen_reference_load_preserves_arrays_basis_and_all_samples():
    # A fresh CPU process avoids contaminating cold admission tests with Torch.
    code = '''
import json
import numpy as np
from scripts import evaluate_shared_social_navigation_style as metric
from CrowdSim.humanoid.analysis.native_navigation_seeded_style import source_bindings
from CrowdSim.humanoid.style_evaluation import sequence_style_samples, StyleSpec
from social_navigation.artifact_identity import artifact_path
before = [metric.record(p) for p in (metric.frozen.REFERENCE, metric.frozen.BASIS, metric.MANIFEST)]
source_bindings()
samples, pose, centers, admission, identities = metric.load_frozen_reference()
artifact = json.loads(metric.frozen.REFERENCE.read_text())
assert admission["actual_array_hashes"] == artifact["actual_selected_array_sha256"]
assert admission["parent_disjoint"]["original_source_intersection"] == 0
assert metric.record(artifact_path(artifact["manifest"]["path"])) in identities
# Execute the unchanged loader with the same resolved frozen resource.
import importlib.util
spec = importlib.util.spec_from_file_location("_original_frozen_tool", metric.frozen.TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
body, rotation, hashes, _ = tool.load_reference(artifact_path(artifact["manifest"]["path"]), artifact["selection"])
assert hashes == artifact["actual_selected_array_sha256"]
resets = np.zeros(body.shape[:2], dtype=bool)
resets[0] = True
expected = sequence_style_samples(body, rotation, episode_ids=np.ones(body.shape[:2], dtype=np.int64), reset_mask=resets, spec=StyleSpec())
def equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: equal(left[key], right[key])
    elif isinstance(left, np.ndarray): np.testing.assert_array_equal(left, right)
    else: assert left == right
equal(samples, expected)
assert set(centers) == {50, 100}
assert before == [metric.record(p) for p in (metric.frozen.REFERENCE, metric.frozen.BASIS, metric.MANIFEST)]
import torch
assert not torch.cuda.is_initialized()
print("frozen arrays, all derived samples and centers preserved")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=metric.ROOT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "frozen arrays, all derived samples and centers preserved" in result.stdout


@pytest.mark.parametrize('mutation', ['metric', 'adapter', 'foreign_parent'])
def test_style_resource_inverse_rejects_unregistered_changes(monkeypatch, mutation):
    path = Path(metric.frozen.__file__).resolve()
    historical = 'd3296011b0e08286ae109daea6a518c95f5f9fa9f24a474a20555cd002baa47c'
    proof = migration.verify_reader(path, historical)
    assert proof['whole_source_location_inverse']
    assert proof['all_metric_function_bytes_unchanged']
    original = Path.read_text
    source = original(path)
    if mutation == 'metric':
        changed = source.replace('resets[0] = True', 'resets[0] = False', 1)
    elif mutation == 'adapter':
        changed = source.replace('artifact_path(artifact["manifest"]["path"])', 'Path(artifact["manifest"]["path"])', 1)
    else:
        changed = source
        historical = '0' * 64
    assert changed != source or mutation == 'foreign_parent'
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **kw: changed if p == path else original(p, *a, **kw))
    with pytest.raises(ValueError, match='Frozen style|Unregistered'):
        migration.verify_reader(path, historical)
