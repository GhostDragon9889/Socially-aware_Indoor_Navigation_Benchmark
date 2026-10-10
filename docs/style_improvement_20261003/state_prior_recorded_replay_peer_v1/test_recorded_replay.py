"""Real preflight closure tests; original media helper tests use saved old PNGs."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

import audit_recorded_replay as reader


@pytest.fixture
def registered():
    return reader.native_bridge().read(reader.HERE / "registration.json")


def test_real_new_source_preparation_remains_pending(registered):
    assert registered["evidence"] == reader.prepare_evidence()
    assert registered["actual_render_audited"] is False
    assert registered["actual_video_success_verified"] is False
    assert registered["evidence"]["expected_preflight"]["preflight"]["frames"] == 6000
    assert registered["evidence"]["native_source_admission"]["checkpoint"]["sha256"] == "19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f"


@pytest.mark.parametrize("mutation", ["old_states", "old_result", "wrong_camera", "short_source", "only449", "wrong_source_code"])
def test_independently_changed_real_preflight_render_closure_rejected(registered, mutation):
    evidence = registered["evidence"]
    changed = deepcopy(evidence["expected_preflight"])
    changed["status"] = "passed"
    changed["arguments"]["preflight_only"] = False
    if mutation == "old_states":
        changed["sources"]["states"]["sha256"] = "0" * 64
    elif mutation == "old_result":
        changed["sources"]["result"]["path"] = str(reader.ROOT / "output/diffusion_amp/native_warehouse_clearance08_200s_v1/result.json")
    elif mutation == "wrong_camera":
        changed["arguments"]["camera_eye"][2] = 26.
    elif mutation == "short_source":
        changed["preflight"]["frames"] = 900
    elif mutation == "only449":
        changed["preflight"]["captured_step_indices"].pop()
    elif mutation == "wrong_source_code":
        changed["source_code"]["scripts/replay_paper_pedestrians.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="closure differs"):
        reader.validate_render_metadata(changed, evidence)


def test_registration_rewrite_rejected_before_any_future_render_read(registered, monkeypatch):
    changed = deepcopy(registered)
    changed["evidence"]["expected_preflight"]["arguments"]["camera_eye"][2] = 26.
    original_load = reader.load
    visited = []

    def observed_load(name, path):
        module = original_load(name, path)
        if name == "recorded_state_prior_native_evidence":
            original_read = module.read

            def read(path):
                visited.append(str(path))
                assert not Path(path).is_relative_to(reader.OUTPUT)
                return original_read(path)

            module.read = read
        return module

    monkeypatch.setattr(reader, "load", observed_load)
    with pytest.raises(ValueError, match="Editable registration"):
        reader.audit(changed)
    assert not any(str(reader.OUTPUT) in path for path in visited)


def test_running_renderer_rejected_before_result_read(registered, monkeypatch):
    original_load = reader.load

    def observed_load(name, path):
        module = original_load(name, path)
        if name == "recorded_state_prior_native_evidence":
            original_read = module.read

            def read(path):
                path = Path(path)
                if path.name == "recorded_replay_actual_launch.json":
                    return {"schema": "state_prior.actual_recorded_replay_launch.v1", "status": "running"}
                assert not path.is_relative_to(reader.OUTPUT)
                return original_read(path)

            module.read = read
        return module

    monkeypatch.setattr(reader, "load", observed_load)
    with pytest.raises(ValueError, match="closed recorded-only launch"):
        reader.audit(registered)


def test_old_media_reader_is_byte_exact(registered):
    bridge = reader.native_bridge()
    media = reader.media_reader(bridge)
    identity = bridge.read(reader.HERE / "reader_identity.json")
    assert Path(media.__file__).read_bytes() == Path(identity["original"]).read_bytes()
    assert registered["media_mathematics_or_encoding_checks_modified"] is False


def test_original_timestamp_gate_on_actual_new_source_frame(registered):
    bridge = reader.native_bridge()
    media = reader.media_reader(bridge)
    with np.load(registered["evidence"]["source_capture"]["physical_states.npz"]["path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("reset_mask", "episode_ids")}
    frame = {"frame": 0, "source_step_index": 0, "source_time_s": 1 / 30, "source_reset_mask": arrays["reset_mask"][0].tolist(), "source_episode_ids": arrays["episode_ids"][0].tolist(), "timeline_before_s": 0., "timeline_after_s": 0., "app_updates": 48}
    media.check_frame_metadata(frame, 0, 0, arrays)
    frame["source_time_s"] = 0.
    with pytest.raises(ValueError, match="timestamp"):
        media.check_frame_metadata(frame, 0, 0, arrays)


def test_original_banner_gate_on_saved_old_actual_indoor_PNG():
    bridge = reader.native_bridge()
    media = reader.media_reader(bridge)
    # Use the saved historical media path from its immutable actual audit.
    report = bridge.read(reader.ROOT / "docs/style_improvement_20261002/native_recorded_replay_reader_v1/actual_indoor_replay30_v1.json")
    path = Path(report["all_PNG_records"][0]["path"])
    producer = reader.load("recorded_banner_original_CPU", reader.ROOT / "scripts/replay_paper_pedestrians.py")
    media.verify_png(path, {"source_time_s": 1 / 30}, {"width": 1280, "height": 960}, producer)
    with pytest.raises(ValueError, match="banner"):
        media.verify_png(path, {"source_time_s": 9.}, {"width": 1280, "height": 960}, producer)


def test_no_model_or_SDK_imports():
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
