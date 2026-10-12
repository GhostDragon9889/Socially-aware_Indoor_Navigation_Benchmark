"""Run the SFM diagnostics on actual images and Matplotlib's Agg canvas."""

import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from CrowdSim.world import sfm


def test_gradient_diagnostic_does_not_need_a_global_map_and_matches_world_axes():
    field = np.arange(48, dtype=float).reshape(6, 8)
    gradients = sfm.Social_Force.compute_gradient(field, 0.1)
    fig = plt.figure()
    try:
        artist = sfm.visualize_gradient(field, gradients, stride=2)
        fig.canvas.draw()
        np.testing.assert_allclose(artist.U, gradients[0, ::2, ::2].ravel())
        np.testing.assert_allclose(artist.V, -gradients[1, ::2, ::2].ravel())
        assert np.asarray(fig.canvas.buffer_rgba()).shape[2] == 4
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fault", ["shape", "stride"])
def test_gradient_diagnostic_rejects_invalid_shape_or_stride(fault):
    with pytest.raises(ValueError):
        sfm.visualize_gradient(
            np.zeros((6, 8)),
            np.zeros((2, 6, 7) if fault == "shape" else (2, 6, 8)),
            stride=0 if fault == "stride" else 2,
        )


def test_binary_image_diagnostic_and_simulation_plot_are_actual_images(
    tmp_path, monkeypatch
):
    source = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 255], [128, 127]], dtype=np.uint8)).save(source)
    np.testing.assert_array_equal(sfm.load_image(source, scale=1), [[0, 1], [1, 0]])
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    fig = plt.figure()
    try:
        sfm.visualize_simulation(
            np.ones((6, 8)), [[2, 2], [3, 3], [4, 4]], [2, 2], [4, 4]
        )
        fig.canvas.draw()
        assert shown == [True] and len(fig.axes[0].lines) == 3
    finally:
        plt.close(fig)


def test_manual_sfm_entry_uses_the_current_controller_api_and_writes_png(tmp_path):
    image = np.full((128, 128), 255, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    source = tmp_path / "map.png"
    Image.fromarray(image).save(source)
    output = tmp_path / "sfm.png"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "CrowdSim.world.sfm",
            "--map",
            str(source),
            "--headless",
            "--steps",
            "200",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    receipt = json.loads(child.stdout.splitlines()[-1])
    assert 0 < receipt["simulated_steps"] <= 200 and receipt["reached"]
    with Image.open(output) as result:
        assert result.format == "PNG" and result.width > 200


def test_manual_sfm_entry_rejects_nonpositive_budget_before_output(tmp_path):
    output = tmp_path / "absent.png"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "CrowdSim.world.sfm",
            "--steps",
            "0",
            "--headless",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert child.returncode == 2 and "steps must be positive" in child.stderr
    assert not output.exists()


@pytest.mark.parametrize("resolution", ["nan", "0", "-1"])
def test_manual_sfm_entry_rejects_invalid_resolution(resolution, tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "CrowdSim.world.sfm",
            "--resolution",
            resolution,
            "--headless",
            "--output",
            str(tmp_path / "absent.png"),
        ],
        capture_output=True,
        text=True,
    )
    assert (
        child.returncode == 2
        and "resolution must be positive and finite" in child.stderr
    )
    assert not (tmp_path / "absent.png").exists()


def test_short_interactive_diagnostic_closes_its_figure_and_reports_unreached(
    tmp_path, monkeypatch, capsys
):
    image = np.full((128, 128), 255, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    source = tmp_path / "map.png"
    Image.fromarray(image).save(source)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    before = plt.get_fignums()
    assert sfm.main(["--map", str(source), "--steps", "1"]) == 0
    assert shown == [True] and plt.get_fignums() == before
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result == {"simulated_steps": 1, "reached": False, "physics_executed": False}
