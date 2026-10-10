"""SDK-free seam checks; fake payloads are not rendered RGB-D evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from CrowdSim.humanoid.application import capture_rgbd


def fake_capture(monkeypatch, payload, *, frame_error=None, detach_error=None):
    events = []
    product, rgb = object(), object()

    class Depth:
        def attach(self, selected):
            assert selected is product
            events.append("attach_same_product")

        def get_data(self):
            events.append("depth_read")
            return payload

        def detach(self, selected):
            assert selected is product
            events.append("detach_same_product")
            if detach_error:
                raise detach_error

    depth = Depth()

    def get_annotator(name):
        assert name == "distance_to_image_plane"
        return depth

    rep = SimpleNamespace(AnnotatorRegistry=SimpleNamespace(get_annotator=get_annotator))

    def setup(env, viewer, width=960, height=720):
        events.append(("original_setup", viewer, width, height))
        return rep, product, rgb

    pixels = np.zeros((2, 3, 3), dtype=np.uint8)

    def frame(env, annotator, warmup):
        assert annotator is rgb
        events.append(("original_bounded_frame", warmup))
        if frame_error:
            raise frame_error
        return pixels

    module = SimpleNamespace(_capture_setup=setup, _capture_frame=frame)
    monkeypatch.setattr(capture_rgbd, "import_module", lambda name: module)
    return module, setup, frame, pixels, events


def test_same_product_raw_depth_order_and_control_index(monkeypatch, tmp_path):
    payload = np.array([[0, np.inf, 2], [3, 4, 5]], dtype=np.float32)
    module, setup, frame, pixels, events = fake_capture(monkeypatch, payload)
    viewer = {"camera_eye": [1, 2, 3], "camera_target": [0, 0, 0]}
    with capture_rgbd.capture_helpers(tmp_path, interval=3, depth=True,
                                     camera_eye=(4, 5, 6), camera_target=(1, 2, 3)):
        _, product, rgb = module._capture_setup(object(), viewer, 3, 2)
        assert module._capture_frame(object(), rgb, True) is pixels
        assert module._capture_frame(object(), rgb, False) is pixels
        with pytest.raises(RuntimeError, match="exactly one"):
            module._capture_setup(object(), viewer)
    assert module._capture_setup is setup and module._capture_frame is frame
    assert viewer["camera_eye"] == [1, 2, 3]
    assert events == [
        ("original_setup", {"camera_eye": (4, 5, 6), "camera_target": (1, 2, 3)}, 3, 2),
        "attach_same_product", ("original_bounded_frame", True), "depth_read",
        ("original_bounded_frame", False), "depth_read", "detach_same_product",
    ]
    rows = [json.loads(line) for line in (tmp_path / "depth/index.jsonl").read_text().splitlines()]
    assert [row["control_step"] for row in rows] == [1, 4]
    assert [row["file"] for row in rows] == ["depth/frame_000000.npy", "depth/frame_000003.npy"]
    for row in rows:
        saved = np.load(tmp_path / row["file"], allow_pickle=False)
        assert saved.dtype == np.float32
        np.testing.assert_array_equal(saved, payload)
        assert row["valid_fraction"] == 4 / 6
        assert row["valid_min_m"] == 2 and row["valid_max_m"] == 5
        assert row["actual_synchronization_verified"] is False


@pytest.mark.parametrize("bad", [np.nan, -np.inf, -1.0, "shape"])
def test_reject_invalid_depth_without_writing_frame(monkeypatch, tmp_path, bad):
    payload = np.zeros((1, 3) if bad == "shape" else (2, 3), dtype=np.float32)
    if bad != "shape":
        payload[0, 0] = bad
    module, setup, frame, _, events = fake_capture(monkeypatch, payload)
    with pytest.raises(RuntimeError):
        with capture_rgbd.capture_helpers(tmp_path, depth=True):
            _, _, rgb = module._capture_setup(None, {})
            module._capture_frame(None, rgb, True)
    assert module._capture_setup is setup and module._capture_frame is frame
    assert not list((tmp_path / "depth").glob("*.npy"))
    assert events[-1] == "detach_same_product"


def test_all_invalid_pixels_are_preserved_with_null_range(monkeypatch, tmp_path):
    payload = np.array([[0, np.inf, 0], [np.inf, 0, np.inf]], dtype=np.float32)
    module, *_ = fake_capture(monkeypatch, payload)
    with capture_rgbd.capture_helpers(tmp_path, depth=True):
        _, _, rgb = module._capture_setup(None, {})
        module._capture_frame(None, rgb, True)
    row = json.loads((tmp_path / "depth/index.jsonl").read_text())
    assert row["valid_fraction"] == 0
    assert row["valid_min_m"] is None and row["valid_max_m"] is None


def test_restore_on_rgb_failure_and_keep_original_error(monkeypatch, tmp_path):
    module, setup, frame, _, events = fake_capture(
        monkeypatch, np.ones((2, 3)), frame_error=ValueError("RGB failure"),
        detach_error=RuntimeError("detach failure"))
    with pytest.raises(ValueError, match="RGB failure") as error:
        with capture_rgbd.capture_helpers(tmp_path, depth=True):
            _, _, rgb = module._capture_setup(None, {})
            module._capture_frame(None, rgb, True)
    assert module._capture_setup is setup and module._capture_frame is frame
    assert "depth_read" not in events
    assert "detach failure" in error.value.__notes__[0]


def test_exclusive_files_and_first_warmup(monkeypatch, tmp_path):
    module, setup, frame, _, _ = fake_capture(monkeypatch, np.ones((2, 3)))
    with capture_rgbd.capture_helpers(tmp_path, depth=True):
        _, _, rgb = module._capture_setup(None, {})
        with pytest.raises(RuntimeError, match="first captured"):
            module._capture_frame(None, rgb, False)
        existing = tmp_path / "depth/frame_000000.npy"
        existing.write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            module._capture_frame(None, rgb, True)
        assert existing.read_bytes() == b"keep"
    assert module._capture_setup is setup and module._capture_frame is frame
    with pytest.raises(FileExistsError):
        with capture_rgbd.capture_helpers(tmp_path, depth=True):
            module._capture_setup(None, {})
    assert module._capture_setup is setup and module._capture_frame is frame


def test_rgb_only_restores_and_does_not_create_depth(monkeypatch, tmp_path):
    module, setup, frame, pixels, events = fake_capture(monkeypatch, None)
    with pytest.raises(LookupError, match="caller"):
        with capture_rgbd.capture_helpers(tmp_path):
            _, _, rgb = module._capture_setup(None, {})
            assert module._capture_frame(None, rgb, True) is pixels
            raise LookupError("caller")
    assert module._capture_setup is setup and module._capture_frame is frame
    assert not (tmp_path / "depth").exists()
    assert events == [("original_setup", {}, 960, 720), ("original_bounded_frame", True)]
