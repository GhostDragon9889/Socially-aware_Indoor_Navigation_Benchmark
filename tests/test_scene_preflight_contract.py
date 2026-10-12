"""Verify current scene admission with real source-bound maps and bad inputs."""

import hashlib
from pathlib import Path

import pytest
import yaml

from social_navigation import preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.needs_lfs
@pytest.mark.parametrize("scene", ["YOHO", "NewMetro", "SecondFloor", "ThirdFloor"])
def test_current_scene_native_map_and_route_admission(scene):
    config = ROOT / "configs/reproduction/four_scene" / (scene + "_scene.yaml")
    configured = yaml.safe_load(config.read_text())
    result = preflight.verify_native_scene_config(config)
    assert result["status"] == "passed"
    assert result["actor_count"] == len(configured["origins_xy"])
    assert result["source_bound_native_map"] and result["route_configuration_verified"]
    assert not result["sdk_started"] and not result["live_physx_execution_verified"]


def test_non_native_scene_is_rejected_before_provider_admission(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("schema: unrelated_scene\n")
    with pytest.raises(ValueError, match="explicit native pedestrian scene"):
        preflight.verify_native_scene_config(path)


def test_changed_probe_snapshot_is_rejected_before_map_admission(tmp_path):
    snapshot = tmp_path / "probe.py"
    snapshot.write_text("value = 1\n")
    scene = {
        "schema": "crowdsim.native_pedestrian_scene.v1",
        "probe_script_snapshot": {
            "path": str(snapshot),
            "sha256": hashlib.sha256(b"original bytes").hexdigest(),
        },
    }
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene))
    with pytest.raises(ValueError, match="Native probe snapshot changed"):
        preflight.verify_native_scene_config(path)
