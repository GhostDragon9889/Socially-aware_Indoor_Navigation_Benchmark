"""Kinematic oracle checks; fixtures do not establish real simulation quality."""
from types import SimpleNamespace
import unittest

import torch

from CrowdSim.humanoid.analysis import amp_kinematic_controls as controls
from CrowdSim.humanoid.analysis import amp_temporal_diagnostic as diagnostic
from CrowdSim.humanoid.amp.observations import build_amp_observation
from tests import test_amp_temporal_diagnostic as fixtures


def state():
    positions = torch.arange(2 * 24 * 3, dtype=torch.float32).reshape(2, 24, 3) / 100
    rotations = torch.zeros(2, 24, 4)
    rotations[..., 3] = 1
    return SimpleNamespace(rigid_body_pos=positions, rigid_body_rot=rotations,
        rigid_body_vel=torch.full((2, 24, 3), .25), rigid_body_ang_vel=torch.full((2, 24, 3), .7),
        dof_pos=torch.full((2, 69), .12), dof_vel=torch.full((2, 69), 1.3))


class KinematicControlsTests(unittest.TestCase):
    def test_static_and_translation_match_independent_full_state_encoder(self):
        fixture = state()
        history = build_amp_observation(fixture)[:, None].expand(-1, 10, -1)
        for translating, name in ((False, "stationary_pose"), (True, "rigid_translation_pose")):
            with self.subTest(translating=translating):
                current, oracle = controls.encode_pose_sequence(fixture, translating=translating)
                torch.testing.assert_close(current, history[:, 0], rtol=0, atol=0)
                # Float32 translated position-minus-root subtraction can round.
                torch.testing.assert_close(controls.controls(history)[name], oracle, rtol=0, atol=1e-6)

    def test_only_declared_velocity_blocks_change_and_inputs_are_owned(self):
        history = torch.arange(2 * 10 * 195, dtype=torch.float32).reshape(2, 10, 195)
        saved = history.clone()
        result = controls.controls(history)
        for name in ("stationary_pose", "rigid_translation_pose"):
            value = result[name]
            for start, end in ((0, 6), (12, 126), (183, 195)):
                self.assertTrue(torch.equal(value[..., start:end], history[:, :1, start:end].expand(-1, 10, -1)))
            self.assertFalse(value[..., 9:12].any())
            self.assertFalse(value[..., 126:183].any())
        self.assertFalse(result["stationary_pose"][..., 6:9].any())
        self.assertTrue(torch.equal(result["rigid_translation_pose"][..., 6:9], history[:, :1, 6:9].expand(-1, 10, -1)))
        result["stationary_pose"][0, 0, 0] = -100
        self.assertTrue(torch.equal(history, saved))

    def test_invalid_history_rejected(self):
        for invalid in (torch.zeros(0, 10, 195), torch.zeros(1, 9, 195), torch.zeros(1, 10, 195, dtype=torch.float64),
                        torch.full((1, 10, 195), float("nan"))):
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    controls.controls(invalid)

    def test_invalid_oracle_clock_rejected(self):
        with self.assertRaises(ValueError):
            controls.encode_pose_sequence(state(), translating=False, dt=True)
        with self.assertRaises(ValueError):
            controls.encode_pose_sequence(state(), translating=1)

    def test_scores_match_original_and_leave_all_inputs_and_moments_frozen(self):
        fixtures.TemporalDiagnosticTests.setUpClass()
        fixture = fixtures.TemporalDiagnosticTests
        proxy = diagnostic.load_discriminator(fixture.state)
        history, actions = fixture.history.clone(), fixture.actions.clone()
        moments = {name: value.clone() for name, value in proxy.amp_normalizer.state_dict().items()}
        _, scores = controls.score(proxy, history, actions, batch_size=8)
        _, original = diagnostic.measure(proxy, history, actions, batch_size=8)
        for name in ("identity", "repeat_current"):
            for kind in ("logits", "rewards"):
                self.assertTrue(torch.equal(scores[name][kind], original[name][kind]))
        self.assertTrue(torch.equal(history, fixture.history))
        self.assertTrue(torch.equal(actions, fixture.actions))
        self.assertTrue(all(torch.equal(value, moments[name]) for name, value in proxy.amp_normalizer.state_dict().items()))

    def test_scoring_rejects_surplus_actions_and_boolean_batch_size(self):
        history = torch.zeros(2, 10, 195)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            controls.score(None, history, torch.zeros(3, 69))
        with self.assertRaisesRegex(ValueError, "integer CPU"):
            controls.score(None, history, torch.zeros(2, 69), batch_size=True)


if __name__ == "__main__":
    unittest.main()
