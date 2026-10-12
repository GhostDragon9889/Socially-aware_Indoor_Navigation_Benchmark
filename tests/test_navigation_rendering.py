"""Decode real offline navigation renders and verify episode reset boundaries."""

from copy import deepcopy
import io
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest

from CrowdSim.tools import render_navigation_fast as render


@pytest.fixture
def recording(tmp_path):
    map_file = tmp_path / "map.png"
    Image.new("L", (128, 128), 255).save(map_file)
    metadata = {
        "type": "metadata",
        "num_humanoids": 1,
        "num_cars": 1,
        "map_path": str(map_file),
        "map_resolution": 0.1,
        "map_origin_xy": [0.0, 0.0],
    }
    positions = [[2.0, 2.0], [3.0, 2.0], [8.0, 8.0], [9.0, 8.0], [4.0, 4.0], [5.0, 4.0]]
    frames = []
    for index, xy in enumerate(positions):
        frames.append(
            {
                "type": "frame",
                "step": index,
                "time": index / 10,
                "positions_xy": [[1.0 + index / 10, 1.0], xy],
                "goals_xy": [
                    [2.0, 1.0],
                    [3.0, 2.0]
                    if index < 2
                    else [9.0, 8.0]
                    if index < 4
                    else [7.0, 4.0],
                ],
                "reached": [False, index in (1, 3)],
                "velocities_xy": [[0.5, 0.0], [1.0, 0.0]],
                "humanoid_yaw_source": ["sfm_target"],
                "sfm_desired_velocities_xy": [[0.5, 0.1]],
                "sfm_interact_forces_xy": [[0.3, 0.1]],
                "sfm_repulsive_forces_xy": [[-0.1, 0.1]],
                "sfm_d_vel_xy": [[0.0, 0.0]],
                "sfm_ttc_forces_xy": [[0.0, 0.0]],
                "humanoid_future_first_targets_xy": [[2.0, 1.0]],
            }
        )
    frames[1]["path_updates"] = [
        {"agent_id": 1, "goal_xy": [9.0, 8.0], "path_xy": [[8.0, 8.0], [9.0, 8.0]]}
    ]
    agents = [
        {"agent_id": 0, "goal_xy": [2.0, 1.0], "path_xy": []},
        {"agent_id": 1, "goal_xy": [3.0, 2.0], "path_xy": [[2.0, 2.0], [3.0, 2.0]]},
    ]
    path_data = {**metadata, "num_agents": 2, "agents": agents}
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(json.dumps(row) for row in [metadata, *frames]) + "\n"
    )
    (tmp_path / "paths_latest.json").write_text(json.dumps(path_data))
    return trajectory, metadata, path_data, frames


def test_reset_boundaries_never_connect_separate_episodes(recording):
    _, _, _, frames = recording
    positions = np.asarray([row["positions_xy"][1] for row in frames])
    reached = np.asarray([row["reached"][1] for row in frames])
    breaks = render._find_reset_boundaries_frame(reached, positions)
    assert breaks == [0, 2, 4, 6]
    np.testing.assert_equal(
        render._trail_segments_fast(positions, breaks, 0, 3), [positions[2:4]]
    )
    np.testing.assert_equal(
        render._trail_segments(positions, reached, 0, 5), [positions[4:6]]
    )
    np.testing.assert_equal(render._trail_segments(positions, None, 0, 5), [positions])
    assert render._trail_segments_fast(positions, breaks, 6, 5) == []
    assert render._trail_segments_fast(positions, breaks, 0, 6) == []
    assert render._find_reset_boundaries_frame([], np.zeros((0, 2))) == [0]


def test_drawing_primitives_and_nonfinite_force_arrows_stay_decodable(recording):
    _, metadata, path_data, frames = recording
    canvas = render.make_canvas(path_data, metadata, 0)
    image = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    for index, primitive in enumerate(
        (
            render._draw_circle,
            render._draw_square,
            render._draw_diamond,
            render._draw_x,
            render._draw_cross,
        )
    ):
        primitive(draw, (10 + index * 20, 20), 4, (255, 0, 0, 255), (0, 0, 0, 255))
    frame = deepcopy(frames[0])
    frame["sfm_ttc_forces_xy"] = [[float("nan"), 0.0]]
    render._draw_sfm_arrows_for_frame(draw, frame, canvas, 1)
    render._draw_sfm_arrows_for_frame(draw, {}, canvas, 1)
    assert len(image.getcolors(10000)) > 2
    assert render._humanoid_yaw_sources({}, 1) == []
    assert render._format_yaw_source_counts([]) == "none"
    assert (
        render._format_yaw_source_counts(["unknown", "sfm_target"]) == "SFM=1 unknown=1"
    )


def test_gif_writer_caps_frames_and_closes_empty_writer_without_output(tmp_path):
    empty = tmp_path / "empty.gif"
    render._GifWriter(empty, (8, 8), 5.0, 1.0).close()
    assert not empty.exists()
    output = tmp_path / "bounded.gif"
    writer = render._GifWriter(output, (8, 8), 5.0, 2.0, max_frames=2)
    for color in ("red", "green", "blue"):
        writer.write(Image.new("RGB", (8, 8), color))
    assert len(writer.frames) == 2
    writer.close()
    with Image.open(output) as image:
        assert image.size == (16, 16) and image.n_frames == 2
        assert image.convert("RGB").getpixel((0, 0)) == (0, 128, 0)


def test_short_path_registry_retains_only_valid_updates():
    records = render._current_path_records(
        {"agents": [{"agent_id": 0, "path_xy": []}]}, 2
    )
    assert len(records) == 2 and records[1]["goal_xy"] == [0.0, 0.0]
    render._apply_path_updates(records, [])
    render._apply_path_updates(
        records,
        [
            None,
            {"agent_id": -1},
            {"agent_id": 99},
            {"agent_id": 1, "goal_xy": [3.0, 4.0]},
        ],
    )
    assert records[1]["goal_xy"] == [3.0, 4.0] and records[0]["agent_id"] == 0


def test_jsonl_loader_ignores_partial_lines_and_tail_keeps_metadata(recording):
    path, metadata, _, frames = recording
    path.write_text("noise\n{broken\n\n" + path.read_text() + "not json\n")
    assert render.load_trajectory(path) == (metadata, frames)
    assert render.load_trajectory_tail(path, 2) == (metadata, frames[-2:])
    padding = json.dumps({"type": "ignored", "padding": "x" * 90000}) + "\n"
    path.write_text(
        json.dumps(metadata)
        + "\n"
        + padding
        + "\n".join(json.dumps(row) for row in frames)
        + "\n{truncated\n"
    )
    assert render.load_trajectory_tail(path, 2) == (metadata, frames[-2:])


def test_paths_use_current_directory_or_recording_directory(
    recording, monkeypatch, tmp_path
):
    path, metadata, path_data, _ = recording
    assert render.load_path_data(metadata, path, None) == path_data
    monkeypatch.chdir(tmp_path)
    assert render.load_path_data(metadata, path, "paths_latest.json") == path_data
    subdir = tmp_path / "recording"
    subdir.mkdir()
    (subdir / "other.json").write_text(json.dumps({"alternate": 42}))
    assert render.load_path_data(metadata, subdir / path.name, "other.json") == {
        "alternate": 42
    }


@pytest.mark.parametrize("fallback", [False, True])
def test_training_gif_is_decodable_and_has_expected_scaled_size(recording, fallback):
    path, _, _, _ = recording
    payload = render.render_nav_gif_bytes(
        np.zeros((128, 128), dtype=np.uint8),
        0.1,
        (0.0, 0.0),
        1,
        1,
        [np.zeros((0, 2)), np.asarray([[2.0, 2.0], [3.0, 2.0]])],
        np.asarray([[2.0, 1.0], [3.0, 2.0]]),
        trajectory_log_path=path.parent / "missing" if fallback else path,
        n_frames=6,
        scale=0.5,
        fps=5.0,
    )
    image = Image.open(io.BytesIO(payload))
    assert image.format == "GIF" and image.size == (64, 64)
    assert image.n_frames == (1 if fallback else 6)
    assert len(image.convert("RGB").getcolors(10000)) > 2


def test_video_gif_and_static_grid_write_actual_images(recording, tmp_path):
    _, metadata, path_data, frames = recording
    canvas = render.make_canvas(path_data, metadata, 96)
    assert canvas.size == (96, 96) and render.world_to_crop_pixel(
        (0.0, 0.0), 128, 128, 0.1, (0.0, 0.0), (0, 0, 128, 128)
    ) == (0, 127)
    video = tmp_path / "navigation.gif"
    render.render_video(
        canvas,
        metadata,
        path_data,
        frames,
        video,
        5.0,
        20,
        0.5,
        2,
        2,
        1,
        True,
        True,
        True,
        True,
    )
    image = Image.open(video)
    assert image.format == "GIF" and image.size == (48, 48) and image.n_frames == 6
    static = tmp_path / "static.png"
    render.render_static_grid(
        frames, metadata, 0, metadata["map_path"], (0.0, 0.0), 0.1, 1, 2, static
    )
    with Image.open(static) as grid:
        assert grid.format == "PNG" and grid.width > 200 and grid.height > 200


@pytest.mark.parametrize("count", [0, 1])
def test_empty_episode_grid_has_clear_error_before_output_creation(
    recording, tmp_path, count
):
    _, metadata, _, frames = recording
    output = tmp_path / "empty.png"
    with pytest.raises(ValueError, match="No renderable episodes"):
        render.render_static_grid(
            frames[:count],
            metadata,
            0,
            metadata["map_path"],
            (0.0, 0.0),
            0.1,
            1,
            2,
            output,
        )
    assert not output.exists()


def test_static_output_io_failure_closes_its_matplotlib_figure(recording, tmp_path):
    import matplotlib.pyplot as plt

    _, metadata, _, frames = recording
    directory = tmp_path / "directory.png"
    directory.mkdir()
    before = plt.get_fignums()
    with pytest.raises(OSError):
        render.render_static_grid(
            frames, metadata, 0, metadata["map_path"], (0.0, 0.0), 0.1, 1, 2, directory
        )
    assert plt.get_fignums() == before


@pytest.mark.parametrize("stride,cols", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_grid_settings_fail_before_output_creation(
    recording, tmp_path, stride, cols
):
    _, metadata, _, frames = recording
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="positive"):
        render.render_static_grid(
            frames,
            metadata,
            0,
            metadata["map_path"],
            (0.0, 0.0),
            0.1,
            stride,
            cols,
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize("mode", ["video", "static"])
def test_published_renderer_cli_creates_images(recording, tmp_path, mode):
    path, _, _, _ = recording
    output = tmp_path / ("cli.gif" if mode == "video" else "cli.png")
    args = [
        sys.executable,
        "-m",
        "CrowdSim.tools.render_navigation_fast",
        str(path),
        "--stride",
        "1",
        "--tail-frames",
        "0",
        "--output",
        str(output),
    ]
    if mode == "static":
        args.extend(["--static", "--cars", "0", "--cols", "2"])
    child = subprocess.run(args, check=True, text=True, capture_output=True)
    assert str(output) in child.stdout
    with Image.open(output) as image:
        assert image.format == ("GIF" if mode == "video" else "PNG")
