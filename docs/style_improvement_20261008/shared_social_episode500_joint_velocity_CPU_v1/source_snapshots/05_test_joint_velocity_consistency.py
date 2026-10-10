"""Independent SciPy frame oracle; fixtures do not validate policy quality."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from CrowdSim.humanoid.analysis import joint_velocity_consistency as audit


def fixture():
    rng = np.random.default_rng(37)
    q = Rotation.from_rotvec(rng.normal(size=(48, 3))).as_quat().reshape(2, 24, 4)
    native = rng.normal(size=(2, 23, 3))
    world = np.zeros((2, 24, 3))
    world[:, 0] = (1.2, -.6, .7)
    for child in range(1, 24):
        world[:, child] = world[:, child - 1] + Rotation.from_quat(q[:, child]).apply(native[:, child - 1])
    return q, world, native.reshape(2, 69), list(range(23))


class JointVelocityConsistencyTests(unittest.TestCase):
    def test_compound_child_basis_matches_independent_rotation_oracle(self):
        q, angular, native, parents = fixture()
        before = [value.copy() for value in (q, angular, native)]
        values, child, parent = audit.reconstruct(q, angular, native, parents)
        np.testing.assert_allclose(child, values, rtol=0, atol=4e-15)
        self.assertGreater(audit.error_statistics(values - parent)["rms_rad_s"], .1)
        for value, saved in zip((q, angular, native), before, strict=True):
            np.testing.assert_array_equal(value, saved)

    def test_world_rotation_does_not_change_joint_frame_vectors(self):
        q, angular, native, parents = fixture()
        rotation = Rotation.from_rotvec((.6, -.7, 1.3))
        rotated_q = (rotation * Rotation.from_quat(q.reshape(-1, 4))).as_quat().reshape(q.shape)
        rotated_angular = rotation.apply(angular.reshape(-1, 3)).reshape(angular.shape)
        _, child, parent = audit.reconstruct(q, angular, native, parents)
        _, transformed_child, transformed_parent = audit.reconstruct(rotated_q, rotated_angular, native, parents)
        np.testing.assert_allclose(child, transformed_child, rtol=0, atol=5e-15)
        np.testing.assert_allclose(parent, transformed_parent, rtol=0, atol=5e-15)

    def test_parent_velocity_mislabel_is_detectable(self):
        q, angular, native, parents = fixture()
        _, _, parent = audit.reconstruct(q, angular, native, parents)
        values, child, actual_parent = audit.reconstruct(q, angular, parent.reshape(2, 69), parents)
        self.assertEqual(audit.error_statistics(values - actual_parent)["max_abs_rad_s"], 0.)
        self.assertGreater(audit.error_statistics(values - child)["rms_rad_s"], .1)

    def test_invalid_states_and_cyclic_parent_graph_rejected(self):
        q, angular, native, parents = fixture()
        for invalid in (q * 2, np.full(q.shape, np.nan), q[:, :23], q.astype(np.int64)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    audit.reconstruct(invalid, angular, native, parents)
        bad = parents.copy()
        bad[0] = 2
        with self.assertRaises(ValueError):
            audit.reconstruct(q, angular, native, bad)
        bad[0] = True
        with self.assertRaises(ValueError):
            audit.reconstruct(q, angular, native, bad)


if __name__ == "__main__":
    unittest.main()
