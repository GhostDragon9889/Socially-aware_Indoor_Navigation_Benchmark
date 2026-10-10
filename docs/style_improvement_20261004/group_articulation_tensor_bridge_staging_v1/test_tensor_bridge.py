"""Five bounded synthetic CPU tensor tests; no SDK, policy or capture use."""
import os
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU tests require explicitly hidden CUDA")

import torch

import tensor_bridge as bridge


class BridgeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    def test_mixed256_partition_and_fixed_slots(self):
        rows = bridge.mixed_group_rows(256, device="cpu")
        self.assertEqual(bridge.ASSET_SLOTS, ("Robot_0", "Robot_1", "Robot_2", "Robot_3"))
        self.assertEqual(rows.shape, (64, 4))
        self.assertEqual(rows.dtype, torch.int64)
        self.assertTrue(torch.equal(rows[0], torch.tensor([0, 1, 128, 129])))
        self.assertTrue(torch.equal(rows[-1], torch.tensor([126, 127, 254, 255])))
        self.assertTrue(torch.equal(rows.sort(dim=0).values[:, :2].flatten().sort().values, torch.arange(128)))
        self.assertTrue(torch.equal(rows.flatten().sort().values, torch.arange(256)))
        for count in (0, 3, 6, True, 256.0):
            with self.assertRaises(ValueError):
                bridge.mixed_group_rows(count, device="cpu")

    def test_roundtrip_field_shapes_no_mutation_or_rng(self):
        rows = bridge.mixed_group_rows(256, device="cpu")
        before_rows = rows.clone()
        rng = torch.random.get_rng_state().clone()
        for tail in ((3,), (4,), (69,), (24, 3), (24, 4)):
            width = 1
            for size in tail:
                width *= size
            canonical = torch.arange(256*width, dtype=torch.float64).reshape(256, *tail)
            before = canonical.clone()
            assets = bridge.canonical_to_assets(canonical, rows, device="cpu")
            self.assertTrue(torch.equal(bridge.assets_to_canonical(assets, rows, device="cpu"), canonical))
            for slot, value in enumerate(assets):
                self.assertEqual(value.dtype, canonical.dtype)
                self.assertTrue(torch.equal(value, canonical.index_select(0, rows[:, slot])))
                self.assertTrue(torch.equal(bridge.gather_asset(canonical, rows, slot=slot, device="cpu"), value))
                self.assertNotEqual(value.data_ptr(), canonical.data_ptr())
            self.assertTrue(torch.equal(canonical, before))
        self.assertTrue(torch.equal(rows, before_rows))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_asset_slot_group_tags_preserve_world_values(self):
        rows = bridge.mixed_group_rows(256, device="cpu")
        groups = torch.arange(64, dtype=torch.float32)
        assets = tuple(torch.stack((1000*slot+groups, -groups, -50-slot+groups/100), -1) for slot in range(4))
        originals = tuple(value.clone() for value in assets)
        canonical = bridge.assets_to_canonical(assets, rows, device="cpu")
        self.assertEqual(canonical.dtype, torch.float32)
        for slot in range(4):
            for group in (0, 19, 63):
                self.assertTrue(torch.equal(canonical[rows[group, slot]], originals[slot][group]))
            self.assertTrue(torch.equal(assets[slot], originals[slot]))
        # An explicit different group-row order is accepted, with the same order in its assets.
        permutation = torch.arange(63, -1, -1)
        reordered = bridge.assets_to_canonical(tuple(value[permutation] for value in assets), rows[permutation], device="cpu")
        self.assertTrue(torch.equal(reordered, canonical))

    def test_all69_actions_are_split_without_joint_conversion(self):
        rows = bridge.mixed_group_rows(256, device="cpu")
        actions = torch.arange(256*69, dtype=torch.float32).reshape(256, 69)
        before = actions.clone()
        slots = bridge.actions_to_assets(actions, rows, device="cpu")
        for slot in range(4):
            self.assertEqual(slots[slot].shape, (64, 69))
            self.assertTrue(torch.equal(slots[slot], actions[rows[:, slot]]))
        self.assertTrue(torch.equal(bridge.assets_to_canonical(slots, rows, device="cpu"), actions))
        self.assertTrue(torch.equal(actions, before))
        with self.assertRaises(ValueError):
            bridge.actions_to_assets(actions[:, :68], rows, device="cpu")

    def test_whole_resets_and_input_rejections(self):
        rows = bridge.mixed_group_rows(256, device="cpu")
        reset = rows[torch.tensor([7, 1])].flip((1,)).flatten()
        before = reset.clone()
        result = bridge.reset_group_env_ids(reset, rows, device="cpu")
        for group_ids in result:
            self.assertTrue(torch.equal(group_ids, torch.tensor([1, 7])))
        self.assertEqual(len({value.data_ptr() for value in result}), 4)
        self.assertTrue(torch.equal(reset, before))
        for empty in bridge.reset_group_env_ids(torch.empty(0, dtype=torch.int64), rows, device="cpu"):
            self.assertEqual(empty.shape, (0,))
        for bad_ids in (rows[0, :1], rows[0, :3], torch.tensor([-1]), torch.tensor([256]), torch.tensor([0, 0]), torch.tensor([0.])):
            with self.assertRaises(ValueError):
                bridge.reset_group_env_ids(bad_ids, rows, device="cpu")
        canonical = torch.zeros((256, 3))
        duplicate = rows.clone()
        duplicate[0, 0] = 1
        outside = rows.clone()
        outside[0, 0] = 256
        for invalid in (duplicate, outside, rows.float(), rows[:, :3], rows.to(device="meta")):
            with self.assertRaises(ValueError):
                bridge.canonical_to_assets(canonical, invalid, device="cpu")
        for invalid in (canonical[:255], canonical.half(), canonical.bool(), torch.full_like(canonical, float("nan")), canonical.to(device="meta")):
            with self.assertRaises(ValueError):
                bridge.canonical_to_assets(invalid, rows, device="cpu")
        for slot in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError):
                bridge.gather_asset(canonical, rows, slot=slot, device="cpu")
        for device in ("cuda", "cuda:0", "meta"):
            with self.assertRaises(ValueError):
                bridge.canonical_to_assets(canonical, rows, device=device)
        assets = bridge.canonical_to_assets(canonical, rows, device="cpu")
        with self.assertRaises(ValueError):
            bridge.assets_to_canonical(assets[:3], rows, device="cpu")
        with self.assertRaises(ValueError):
            bridge.assets_to_canonical((*assets[:3], assets[3].double()), rows, device="cpu")
        with self.assertRaises(ValueError):
            bridge.assets_to_canonical((*assets[:3], assets[3][:, :2]), rows, device="cpu")


if __name__ == "__main__":
    unittest.main()
