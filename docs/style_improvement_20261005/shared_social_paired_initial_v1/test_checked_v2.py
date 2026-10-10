"""Synthetic CPU contracts; none of these tests claim native execution."""
import ast
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from CrowdSim.humanoid.analysis import paired_initial as initial
from scripts import evaluate_shared_social_navigation as canonical
from scripts import evaluate_shared_social_high_level as entry


def fixture():
    state = SimpleNamespace(state_conversion=SimpleNamespace(value="common"))
    for name, shape in initial.COMMON_FIELDS.items():
        value = torch.zeros(shape)
        if name == "rigid_body_rot":
            value[..., 3] = 1.
        setattr(state, name, value)
    world = SimpleNamespace(num_envs=4, shared_world=True, scene_provider=None, state=lambda: state)
    for name, shape in initial.WORLD_FIELDS.items():
        dtype = torch.bool if name == "history_available" else (
            torch.int64 if name in {"age", "episode_ids", "reference_ids", "amp_history_episode_ids"} else torch.float32)
        setattr(world, name, torch.zeros(shape, dtype=dtype))
    world.episode_ids.fill_(1)
    world.history_available[:, -1] = True
    world.validation_scene = SimpleNamespace(routes=torch.zeros(4, 4, 2))
    native = {key: torch.zeros(shape) for key, shape in initial.NATIVE_FIELDS.items()}
    for common, key in (("rigid_body_pos", "body_link_pos_w"), ("rigid_body_rot", "body_link_quat_w"),
                        ("rigid_body_vel", "body_link_lin_vel_w"), ("rigid_body_ang_vel", "body_link_ang_vel_w")):
        native[key] = getattr(state, common).clone()
    ground = SimpleNamespace(world=world, counter=lambda: 17, _state_snapshot=lambda: native,
                             native_names=["Pelvis"] + [str(i) for i in range(23)],
                             common_names=["Pelvis"] + [str(i) for i in range(23)])
    return world, ground


def complete_snapshot():
    world, ground = fixture()
    arrays, clock = initial.collect_snapshot(world, ground)
    rng, metadata = initial.rng_snapshot(torch.Generator().manual_seed(7))
    arrays.update(rng)
    contract = {"capture_kind": "CPU_contract_fixture", "clock": clock, "rng": metadata,
                "boundary": "synthetic; no SDK or actual-exit evidence", "actor": {"checkpoint": "fixture"}}
    return arrays, contract


class PairedInitialTests(unittest.TestCase):
    def test_collect_has_full_detached_channels_without_clock_change(self):
        world, ground = fixture()
        before, _ = initial.rng_snapshot(torch.Generator().manual_seed(7))
        arrays, clock = initial.collect_snapshot(world, ground)
        after, _ = initial.rng_snapshot(torch.Generator().manual_seed(7))
        self.assertTrue(initial._equal_arrays(before, after))
        self.assertEqual(clock["physics_step_before"], clock["physics_step_after"])
        world.state().rigid_body_pos.fill_(5.)
        self.assertFalse(np.any(arrays["common_rigid_body_pos"]))
        self.assertEqual(arrays["world_amp_history"].shape, (4, 10, 195))

    def test_native_phase_mask_and_missing_channels_fail(self):
        world, ground = fixture()
        ground.counter = Mock(side_effect=[17, 18])
        with self.assertRaisesRegex(ValueError, "advanced physics"):
            initial.collect_snapshot(world, ground)
        world, ground = fixture()
        world.age[0] = 1
        with self.assertRaisesRegex(ValueError, "age0"):
            initial.collect_snapshot(world, ground)
        world, ground = fixture()
        ground._state_snapshot()["body_link_pos_w"][0, 0, 0] = .01
        with self.assertRaisesRegex(ValueError, "disagree"):
            initial.collect_snapshot(world, ground)
        world, ground = fixture()
        world.state().dof_vel[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            initial.collect_snapshot(world, ground)

    def test_rng_getter_does_not_initialize_CUDA_or_advance_global_or_private_rng(self):
        generator = torch.Generator().manual_seed(7)
        first, meta = initial.rng_snapshot(generator)
        second, other = initial.rng_snapshot(generator)
        self.assertTrue(initial._equal_arrays(first, second))
        self.assertEqual(meta, other)
        self.assertFalse(meta["CUDA_initialized"])
        self.assertFalse(meta["CUDA_getter_called"])
        self.assertFalse(torch.cuda.is_initialized())
        torch.rand(1, generator=generator)
        changed, _ = initial.rng_snapshot(generator)
        self.assertFalse(initial._equal_arrays(first, changed))

    def test_archive_same_payload_matches_and_changed_private_rng_is_rejected(self):
        arrays, contract = complete_snapshot()
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary)/"left", Path(temporary)/"right"
            left.mkdir()
            right.mkdir()
            initial.publish_snapshot(left, arrays, contract)
            initial.publish_snapshot(right, arrays, contract)
            report = initial.compare_initial_conditions(left/"paired_initial_condition.json", right/"paired_initial_condition.json")
            self.assertEqual(report["status"], "initial_conditions_matched")
            self.assertEqual(report["capture_kind"], "CPU_contract_fixture")
            self.assertFalse(report["actual_process_exit_verified"] or report["navigation_quality_established"])
            with self.assertRaises(FileExistsError):
                initial.publish_snapshot(left, arrays, contract)
            changed = Path(temporary)/"changed"
            changed.mkdir()
            copy = {key: value.copy() for key, value in arrays.items()}
            copy["rng_corpus_CPU"][0] ^= 1
            initial.publish_snapshot(changed, copy, contract)
            with self.assertRaisesRegex(ValueError, "do not share exact"):
                initial.compare_initial_conditions(left/"paired_initial_condition.json", changed/"paired_initial_condition.json")

    def test_partial_payload_dtype_or_archive_identity_cannot_match(self):
        arrays, contract = complete_snapshot()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "partial"):
                initial.publish_snapshot(root, {"actor_ids": arrays["actor_ids"]}, contract)
            self.assertFalse(list(root.iterdir()))
            copy = {key: value.copy() for key, value in arrays.items()}
            copy["world_episode_ids"] = copy["world_episode_ids"].astype(np.float32)
            with self.assertRaisesRegex(ValueError, "integers"):
                initial.publish_snapshot(root, copy, contract)
            initial.publish_snapshot(root, arrays, contract)
            receipt = root/"paired_initial_condition.json"
            value = json.loads(receipt.read_text())
            value["arrays"]["sha256"] = "0"*64
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                initial.read_snapshot(receipt)

    def test_both_driver_hooks_before_planner_keep_original_control_and_helpers(self):
        source, _ = canonical.derive_entry(canonical.PARENT.read_text())
        parent = ModuleType("_CPU_contract_parent")
        parent.__file__ = str(canonical.PARENT)
        exec(compile(source, "_CPU_contract_parent", "exec"), parent.__dict__)
        original = parent.DRIVER.read_text()
        for planner in ("diffusion", "sfm"):
            derived, proof = entry.derive_driver(original, parent._derive, planner=planner)
            self.assertTrue(proof["complete_inverse"] and proof["existing_helpers_AST_equal"])
            self.assertFalse(proof["existing_control_or_metric_math_changed"])
            self.assertEqual(derived.count('_paired_initial(world, ground_capture, output, report, args)'), 1)
            capture = derived.index('report["paired_initial_condition"]')
            construction = derived.index("        planner = ")
            self.assertLess(capture, construction)
            for call in ("world.step(", "world.reset("):
                self.assertEqual(original.count(call), derived.count(call))
            migrated, entry_proof = entry.derive_entry(canonical.PARENT.read_text(), canonical.derive_entry, planner=planner)
            self.assertTrue(entry_proof["paired_entry_inverse"])
            tree = ast.parse(migrated)
            hook = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_derive")
            self.assertIn(planner, ast.unparse(hook))
        with self.assertRaises(ValueError):
            entry.derive_driver(original, parent._derive, planner="unsupported")
        with self.assertRaises(ValueError):
            entry.derive_driver(original.replace("PaperDiffusionPlanner", "ChangedPlanner"), parent._derive, planner="diffusion")


if __name__ == "__main__":
    unittest.main()
