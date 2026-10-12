"""CPU API-boundary contracts; actual grouped physics is checked by the GPU CLI."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.runtime import grouped_source_migration as migration
from protomotions.utils.rotations import quat_apply


@pytest.mark.parametrize('route', ['direct_load', 'world_child'])
def test_grouped_reset_writes_com_velocity_for_link_state_and_preserves_other_groups(route):
    path = migration.ROOT / 'resources/runtime_sources/style_improvement_20261004/group_articulation_state_adapter_staging_v1/state_adapter.py'
    name = '_grouped_velocity_' + route
    if route == 'direct_load':
        module = migration.load(path, name, migration.record(path))
    else:
        module = ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        migration.adapt(module)
    try:
        bridge, states, _ = module._dependencies()
        rows = bridge.mixed_group_rows(12, device='cpu')
        selected = rows[torch.tensor([2, 0])].flatten().flip(0)
        local_com = torch.tensor([-.0055, 0., -.0121])

        class Asset:
            def __init__(self):
                self.data = SimpleNamespace(body_com_pos_b=local_com.expand(3, 24, 3).clone(),
                    root_link_quat_w=torch.tensor([[0., 0., 0., 1.]]).repeat(3, 1))
                self.native_com_velocity = torch.full((3, 6), 123.)
                self.calls = []

            def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids, **unused):
                self.data.root_link_quat_w[env_ids] = root_pose[:, 3:]

            def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids, **unused):
                # This is the observed SDK 6.1 boundary: the link buffer is
                # passed unchanged to the native root COM velocity API.
                self.calls.append('link')
                self.native_com_velocity[env_ids] = root_velocity

            def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids, **unused):
                self.calls.append('com')
                self.native_com_velocity[env_ids] = root_velocity

            def write_joint_state_to_sim_index(self, **unused):
                pass

            def set_joint_position_target_index(self, **unused):
                pass

        adapter = module.FourArticulationAdapter.__new__(module.FourArticulationAdapter)
        adapter.device, adapter.rows, adapter.count = torch.device('cpu'), rows, rows.numel()
        adapter.assets = tuple(Asset() for _ in range(4))
        mapping = states.DataConversionMapping(body_convert_to_common=torch.arange(24),
            body_convert_to_sim=torch.arange(24), dof_convert_to_common=torch.arange(69),
            dof_convert_to_sim=torch.arange(69), sim_w_last=True)
        adapter.mappings = (mapping,) * 4
        velocity = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3) / 10
        angular = torch.tensor([[1., 2., 3.]]).repeat(12, 1)
        fields = dict(root_pos=torch.zeros(12, 3),
            root_rot=torch.tensor([[0., 0., 2 ** -.5, 2 ** -.5]]).repeat(12, 1),
            root_vel=velocity, root_ang_vel=angular, dof_pos=torch.zeros(12, 69), dof_vel=torch.zeros(12, 69))
        originals = {key: value.clone() for key, value in fields.items()}
        state = states.ResetState(state_conversion=states.StateConversion.COMMON, **fields)
        groups = adapter.write_common_reset(state, selected)
        for slot, (asset, ids) in enumerate(zip(adapter.assets, groups)):
            expected = torch.cat((velocity[rows[ids, slot]], angular[rows[ids, slot]]), dim=-1)
            actual = asset.native_com_velocity[ids].clone()
            offset = quat_apply(asset.data.root_link_quat_w[ids], local_com.expand(len(ids), 3), w_last=True)
            actual[:, :3] -= torch.cross(actual[:, 3:], offset, dim=-1)
            torch.testing.assert_close(actual, expected)
            assert asset.calls == ['com']
            assert torch.equal(asset.native_com_velocity[1], torch.full((6,), 123.))
        for key, value in fields.items():
            assert torch.equal(value, originals[key]), key
        before_calls = [asset.calls.copy() for asset in adapter.assets]
        adapter.write_common_reset(state, selected[:0])
        assert [asset.calls for asset in adapter.assets] == before_calls
    finally:
        sys.modules.pop(name, None)


def test_grouped_velocity_derivation_rejects_changed_helper(monkeypatch):
    path = migration.ROOT / 'resources/runtime_sources/style_improvement_20261004/group_articulation_state_adapter_staging_v1/state_adapter.py'
    helper = migration.ROOT / 'src/protomotions/simulator/isaaclab/utils/runtime_compat.py'
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'unreviewed helper' if p == helper else original_read(p))
    name = '_grouped_velocity_unreviewed_helper'
    with pytest.raises(ValueError, match='Grouped Sim61 dependency changed'):
        migration.load(path, name, migration.record(path))
    assert name not in sys.modules


@pytest.mark.parametrize('defect', [None, 'velocity', 'foreign_group', 'input'])
def test_gpu_reset_probe_checks_state_and_restores_after_failure(defect):
    from scripts.validate_grouped_social import validate_grouped_reset
    rows = torch.tensor([[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]])
    state = dict(root_pos=torch.zeros(12, 3), root_rot=torch.tensor([[0., 0., 0., 1.]]).repeat(12, 1),
                 root_vel=torch.zeros(12, 3), root_ang_vel=torch.zeros(12, 3),
                 dof_pos=torch.zeros(12, 69), dof_vel=torch.zeros(12, 69))
    baseline = {key: value.clone() for key, value in state.items()}
    assets = tuple(SimpleNamespace(data=SimpleNamespace(
        body_com_pos_b=torch.tensor([-.0055, 0., -.0121]).expand(3, 24, 3).clone(),
        root_link_quat_w=state['root_rot'][rows[:, slot]].clone())) for slot in range(4))
    class Adapter:
        writes = 0

        def snapshot(self):
            values = {key: value.clone() for key, value in state.items()}
            return SimpleNamespace(root_state=SimpleNamespace(**values), robot_state=SimpleNamespace(**values))

        def write_common_reset(self, reset, ids):
            self.writes += 1
            for key, value in state.items():
                value[ids] = getattr(reset, key)[ids]
            if self.writes == 1:
                if defect == 'velocity':
                    state['root_vel'][ids] += .01
                elif defect == 'foreign_group':
                    state['root_vel'][rows[1]] += .01
                elif defect == 'input':
                    reset.root_vel[ids] += .01
            for slot, asset in enumerate(assets):
                asset.data.root_link_quat_w.copy_(state['root_rot'][rows[:, slot]])
    world = SimpleNamespace(adapter=Adapter(), assets=assets, rows=rows, num_envs=12, device=torch.device('cpu'),
        control_step=0, _counter=lambda: 4, sim=SimpleNamespace(forward=lambda: None),
        age=torch.zeros(12, dtype=torch.int64), episode_ids=torch.ones(12, dtype=torch.int64),
        amp_history=torch.zeros(12, 10, 195), amp_history_episode_ids=torch.ones(12, 10, dtype=torch.int64),
        plan_age=torch.zeros(12, dtype=torch.int64), plans=torch.zeros(12, 5, 2), plan_valid=torch.zeros(12, dtype=torch.bool))
    if defect is None:
        report = validate_grouped_reset(world)
        assert report['passed'] is True and report['selected_group_ids'] == [2, 0]
        assert report['physical_steps_advanced'] == 0
        assert all(value > 1e-5 for value in report['max_COM_velocity_corrections_m_s'])
    else:
        with pytest.raises(AssertionError):
            validate_grouped_reset(world)
    assert world.adapter.writes == 2
    for key in state:
        assert torch.equal(state[key], baseline[key]), key
