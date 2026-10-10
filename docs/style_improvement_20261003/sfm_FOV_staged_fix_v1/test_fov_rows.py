"""One behavior regression: aligned FOV rows, including capped neighbor sets."""
import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_fov_neighbor_rows_and_default_force_path():
    source = Path(os.environ.get("SFM_STAGED_SOURCE", str(HERE / "after/sfm.py")))
    candidate = module(source, "candidate_sfm")
    before = module(HERE / "before/sfm.py", "original_sfm")
    cfg = {"env": {"dt": 1 / 30, "safe_distance": .9},
           "agent": {"radius": .35, "max_vel": 1.5}, "map": {"resolution": .05}}
    field = np.full((16, 16), 10., dtype=np.float32)
    current = np.array([1., 0.], dtype=np.float32)
    cases = (
        np.array([[1., 0.], [2., 1.]], dtype=np.float32),
        np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., 1.]], dtype=np.float32),
        np.array([[1., 0.], [-6., 0.], [-5., 0.], [2., 0.], [-4., 0.],
                  [-3., 0.], [-2., 0.], [-1., 0.], [-7., 0.]], dtype=np.float32),
    )
    for relative in cases:
        ids = np.arange(len(relative), dtype=np.int64) + 100
        distance = np.linalg.norm(relative, axis=-1)
        relative_velocity = np.stack((ids / 100., -ids / 50.), axis=-1).astype(np.float32)
        observation = (ids, distance, relative, relative_velocity)
        # Preserve the existing filter's sign convention; this patch only
        # aligns corresponding rows and does not redefine its field of view.
        selected = np.flatnonzero(np.dot(-relative, current) >= 0)
        if len(selected) > 5:
            selected = selected[np.argsort(distance[selected])[:5]]
        actual = candidate.Social_Force.fov_filter(current, observation)
        for value, original in zip(actual, observation, strict=True):
            assert value.shape == original[selected].shape
            np.testing.assert_array_equal(value, original[selected])
        state = (np.array([0., 0.], dtype=np.float32), np.array([8, 8]), current,
                 np.array([10., 0.], dtype=np.float32))
        original_sfm, new_sfm = before.Social_Force(field, cfg), candidate.Social_Force(field, cfg)
        assert original_sfm.fov is new_sfm.fov is False
        old_force, old_terms = original_sfm.get_action(state, observation)
        new_force, new_terms = new_sfm.get_action(state, observation)
        np.testing.assert_array_equal(new_force, old_force)
        np.testing.assert_array_equal(new_terms, old_terms)
        new_sfm.fov = True
        filtered_force, terms = new_sfm.get_action(state, observation)
        assert np.isfinite(filtered_force).all() and np.isfinite(terms).all()
