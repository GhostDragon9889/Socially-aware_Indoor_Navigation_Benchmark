"""Five CPU fake-asset contract tests, not physical/SDK evidence."""
import os
import sys
from types import SimpleNamespace
import unittest

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Tests require explicitly hidden CUDA")

import torch

import state_adapter as adapter

BODIES = ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand")
DOFS = tuple(f"{body}_{axis}" for body in BODIES[1:] for axis in "xyz")


def kinematics():
    return SimpleNamespace(body_names=BODIES, dof_names=DOFS, num_bodies=24,
                           hinge_axes_map={index: ((1, 0, 0), (0, 1, 0), (0, 0, 1)) for index in range(1, 24)})


class Proxy:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    @property
    def torch(self):
        self.owner.view_reads[self.name] = self.owner.view_reads.get(self.name, 0)+1
        return self.owner.buffers[self.name]  # Zero-copy view, deliberately not cloned by fake.


class Data:
    def __init__(self, buffers):
        self.buffers, self.property_reads, self.view_reads = buffers, {}, {}

    def __getattr__(self, name):
        if name not in self.buffers:
            raise AssertionError("Only explicit link/joint fields are allowed: " + name)
        self.property_reads[name] = self.property_reads.get(name, 0)+1
        return Proxy(self, name)


class FakeAsset:
    def __init__(self, slot, groups):
        self.num_instances, self.device, self.writes = groups, "cpu", []
        self.fail_kind = None
        _, _, resolver = adapter._dependencies()
        names = resolver(kinematics())
        self.body_order = torch.arange(24).roll(slot*5)
        self.dof_order = torch.arange(69).roll(slot*7).flip((0,))
        self.body_names = tuple(BODIES[index] for index in self.body_order)
        self.joint_names = tuple(names.semantic_to_backend[DOFS[index]] for index in self.dof_order)
        group = torch.arange(groups, dtype=torch.float32)[:, None].expand(-1, 24)
        body = torch.arange(24, dtype=torch.float32)[None].expand(groups, -1)
        pos = torch.stack((1000*slot+10*group+body, -group, .9+.01*body), -1)
        vel = torch.stack((group+.1*body, group-1, .01*body), -1)
        angles = body*.01+slot*.1
        quat = torch.stack((torch.zeros_like(body), torch.zeros_like(body), (angles/2).sin(), (angles/2).cos()), -1)
        dofs = 1000*slot+100*torch.arange(groups, dtype=torch.float32)[:, None]+torch.arange(69, dtype=torch.float32)[None]
        self.common = dict(rigid_body_pos=pos, rigid_body_rot=quat, rigid_body_vel=vel, rigid_body_ang_vel=-vel, dof_pos=dofs, dof_vel=-dofs,
                           root_pos=pos[:, 0].clone(), root_rot=quat[:, 0].clone(), root_vel=vel[:, 0].clone(), root_ang_vel=-vel[:, 0].clone())
        buffers = {}
        for key, (prop, _) in adapter.BODY_FIELDS.items():
            order = self.body_order if key.startswith("rigid_body") else self.dof_order
            buffers[prop] = self.common[key].index_select(1, order).clone()
        for key, (prop, _) in adapter.ROOT_FIELDS.items():
            buffers[prop] = self.common[key].clone()
        self.data = Data(buffers)

    def _record(self, kind, **kwargs):
        self.writes.append((kind, kwargs))  # Keep references, so caller aliasing would be observable.
        if self.fail_kind == kind:
            raise RuntimeError("Synthetic SDK writer failure")

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids=None, full_data=False, skip_forward=False):
        self._record("pose", root_pose=root_pose, env_ids=env_ids, full_data=full_data)

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids=None, full_data=False, skip_forward=False):
        self._record("velocity", root_velocity=root_velocity, env_ids=env_ids, full_data=full_data)

    def write_joint_state_to_sim_index(self, *, position, velocity, joint_ids=None, env_ids=None, full_data=False, skip_forward=False):
        self._record("joints", position=position, velocity=velocity, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data)

    def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False):
        self._record("targets", target=target, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data)


def fixture(groups=2):
    bridge, _, _ = adapter._dependencies()
    rows = bridge.mixed_group_rows(groups*4, device="cpu")
    assets = tuple(FakeAsset(slot, groups) for slot in range(4))
    return adapter.FourArticulationAdapter(assets, rows, kinematics(), device="cpu"), assets, rows


def reset_state(count):
    _, states, _ = adapter._dependencies()
    root = torch.arange(count*3, dtype=torch.float32).reshape(count, 3)/10
    quaternion = torch.zeros((count, 4))
    quaternion[:, 3] = 1
    dofs = torch.arange(count*69, dtype=torch.float32).reshape(count, 69)/100
    return states.ResetState(state_conversion=states.StateConversion.COMMON, root_pos=root, root_rot=quaternion,
                             root_vel=-root, root_ang_vel=root/10, dof_pos=dofs, dof_vel=-dofs)


class AdapterCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA became initialized")

    def test_named_snapshot64groups_and_fresh_owned_views(self):
        obj, assets, rows = fixture(64)
        rng = torch.random.get_rng_state().clone()
        sample = obj.snapshot()
        _, states, _ = adapter._dependencies()
        self.assertEqual(sample.robot_state.state_conversion, states.StateConversion.COMMON)
        before = {key: getattr(sample.robot_state, key).clone() for key in adapter.BODY_FIELDS}
        before_root = {key: getattr(sample.root_state, key).clone() for key in adapter.ROOT_FIELDS}
        for slot, asset in enumerate(assets):
            for key in adapter.BODY_FIELDS:
                self.assertTrue(torch.equal(getattr(sample.robot_state, key)[rows[:, slot]], asset.common[key]), key)
            for key in adapter.ROOT_FIELDS:
                self.assertTrue(torch.equal(getattr(sample.root_state, key)[rows[:, slot]], asset.common[key]), key)
            self.assertTrue(all(count == 1 for count in asset.data.property_reads.values()))
            self.assertTrue(all(count == 1 for count in asset.data.view_reads.values()))
            self.assertEqual(asset.writes, [])
            for name, buffer in asset.data.buffers.items():
                if "quat" not in name:
                    buffer.add_(10)  # Simulate the next refresh overwriting the original zero-copy buffers.
        next_sample = obj.snapshot()
        for key, value in before.items():
            self.assertTrue(torch.equal(getattr(sample.robot_state, key), value))
        for key, value in before_root.items():
            self.assertTrue(torch.equal(getattr(sample.root_state, key), value))
        self.assertFalse(torch.equal(next_sample.root_state.root_pos, sample.root_state.root_pos))
        self.assertTrue(all(count == 2 for asset in assets for count in asset.data.property_reads.values()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_invalid_names_and_kinematic_triples_rejected(self):
        bridge, _, _ = adapter._dependencies()
        rows = bridge.mixed_group_rows(8, device="cpu")
        for field in ("body_names", "joint_names"):
            assets = [FakeAsset(slot, 2) for slot in range(4)]
            names = list(getattr(assets[3], field))
            names[0] = "Unknown"
            setattr(assets[3], field, names)
            with self.assertRaises(ValueError):
                adapter.FourArticulationAdapter(assets, rows, kinematics(), device="cpu")
            self.assertTrue(all(not asset.writes for asset in assets))
        kin = kinematics()
        kin.hinge_axes_map[1] = ((1, 0, 0),)
        with self.assertRaises(ValueError):
            adapter.FourArticulationAdapter([FakeAsset(slot, 2) for slot in range(4)], rows, kin, device="cpu")

    def test_common_pd_targets_named_order_and_no_flush(self):
        obj, assets, rows = fixture()
        targets = torch.arange(8*69, dtype=torch.float32).reshape(8, 69)
        before = targets.clone()
        obj.set_common_position_targets(targets)
        for slot, asset in enumerate(assets):
            self.assertEqual(len(asset.writes), 1)
            kind, kwargs = asset.writes[0]
            self.assertEqual(kind, "targets")
            self.assertTrue(torch.equal(kwargs["target"], targets[rows[:, slot]][:, asset.dof_order]))
            self.assertTrue(torch.equal(kwargs["env_ids"], torch.arange(2)))
            self.assertFalse(kwargs["full_data"])
            self.assertIsNone(kwargs["joint_ids"])
        self.assertTrue(torch.equal(targets, before))
        targets.add_(100)
        self.assertTrue(torch.equal(assets[0].writes[0][1]["target"], before[rows[:, 0]][:, assets[0].dof_order]))

    def test_whole_partial_empty_reset_and_prevalidation_before_writes(self):
        obj, assets, rows = fixture()
        source = reset_state(8)
        before = {key: getattr(source, key).clone() for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")}
        empty = obj.write_common_reset(source, torch.empty(0, dtype=torch.int64))
        self.assertTrue(all(not len(ids) for ids in empty))
        for bad in (rows[0, :1], rows[0, :3]):
            with self.assertRaises(ValueError):
                obj.write_common_reset(source, bad)
        source.dof_vel[-1, -1] = float("nan")
        with self.assertRaises(ValueError):
            obj.write_common_reset(source, rows[1])
        source.dof_vel.copy_(before["dof_vel"])
        with self.assertRaises(ValueError):
            obj.set_common_position_targets(torch.zeros((8, 69), dtype=torch.float64))
        self.assertTrue(all(not asset.writes for asset in assets))
        request = rows[1].flip((0,)).clone()
        obj.write_common_reset(source, request)
        for slot, asset in enumerate(assets):
            self.assertEqual([kind for kind, _ in asset.writes], ["pose", "velocity", "joints", "targets"])
            for _, kwargs in asset.writes:
                self.assertTrue(torch.equal(kwargs["env_ids"], torch.tensor([1])))
                self.assertFalse(kwargs["full_data"])
            canonical_row = rows[1, slot]
            self.assertTrue(torch.equal(asset.writes[0][1]["root_pose"], torch.cat((source.root_pos[canonical_row], source.root_rot[canonical_row]))[None]))
            self.assertEqual(asset.writes[1][1]["root_velocity"].shape, (1, 6))
            self.assertTrue(torch.equal(asset.writes[2][1]["position"], source.dof_pos[canonical_row, asset.dof_order][None]))
            self.assertTrue(torch.equal(asset.writes[2][1]["velocity"], source.dof_vel[canonical_row, asset.dof_order][None]))
        for key, value in before.items():
            self.assertTrue(torch.equal(getattr(source, key), value))
        self.assertTrue(torch.equal(request, rows[1].flip((0,))))

    def test_writer_failure_is_transparent_and_no_SDK_import(self):
        obj, assets, rows = fixture()
        assets[0].fail_kind = "velocity"
        with self.assertRaisesRegex(RuntimeError, "Synthetic SDK writer failure"):
            obj.write_common_reset(reset_state(8), rows[0])
        self.assertEqual([kind for kind, _ in assets[0].writes], ["pose", "velocity"])
        self.assertTrue(all(not asset.writes for asset in assets[1:]))
        self.assertFalse(any(name in sys.modules for name in ("isaaclab", "isaaclab_physx", "warp", "mujoco", "dm_control")))


if __name__ == "__main__":
    unittest.main()
