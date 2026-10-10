"""CPU contracts and actual source execution; no Isaac SDK/Kit import.

The simulator class is extracted unchanged from amp_world.py and executed
against an instrumented CPU response source. This validates its call order,
pairing, aggregation and native-to-common conversion, not PhysX numerics.
"""

import ast
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.power import LEGACY_POWER, PROJECTED_POWER, power_contract, power_stage
from CrowdSim.humanoid.training.rewards import paper_task_reward
from protomotions.simulator.base_simulator.simulator_state import (
    DataConversionMapping,
    RobotState,
    StateConversion,
)


ROOT = Path(__file__).resolve().parents[1]
WORLD_SOURCE = ROOT / "CrowdSim/humanoid/amp_world.py"


def test_power_identities_are_distinct_and_leave_motor_channel_unknown():
    legacy, projected = power_contract(LEGACY_POWER), power_contract(PROJECTED_POWER)
    assert legacy["timestamp_aligned"] is False
    assert projected["timestamp_aligned"] is True
    assert legacy["units"] == projected["units"] == "W"
    assert legacy["isolated_motor_drive_torque"] is False
    assert projected["isolated_motor_drive_torque"] is False
    assert "substep" in projected["quadrature"]
    with pytest.raises(ValueError, match="Unknown"):
        power_contract("motor_torque")


@pytest.mark.parametrize("source", [LEGACY_POWER, PROJECTED_POWER])
def test_same_stage_resume_preserves_exact_contract(source):
    saved = power_contract(source)
    before = copy.deepcopy(saved)
    old, current, changed = power_stage(saved, source)
    assert old == current == before and changed is False
    assert saved == before


def test_old_unannotated_bundle_requires_explicit_reward_transition():
    old, current, changed = power_stage(None, LEGACY_POWER)
    assert old == current == power_contract(LEGACY_POWER) and changed is False
    with pytest.raises(ValueError, match="explicit"):
        power_stage(None, PROJECTED_POWER)
    old, current, changed = power_stage(None, PROJECTED_POWER, explicit_stage=True)
    assert old == power_contract(LEGACY_POWER)
    assert current == power_contract(PROJECTED_POWER) and changed is True


@pytest.mark.parametrize("saved", [[], 7, "legacy", {}, {"source": "unknown"},
                                    {**power_contract(LEGACY_POWER), "extra": 1},
                                    {**power_contract(PROJECTED_POWER), "units": "J"}])
def test_invalid_or_extended_saved_contracts_fail_closed(saved):
    with pytest.raises(ValueError):
        power_stage(saved, PROJECTED_POWER, explicit_stage=True)


@pytest.mark.parametrize("requested", ["unknown", None, ""])
def test_unknown_requested_sources_are_rejected(requested):
    with pytest.raises(ValueError):
        power_stage(power_contract(LEGACY_POWER), requested, explicit_stage=True)


@pytest.mark.parametrize("old,new,evaluation", [
    (LEGACY_POWER, LEGACY_POWER, False),
    (PROJECTED_POWER, PROJECTED_POWER, False),
    (PROJECTED_POWER, LEGACY_POWER, False),
    (LEGACY_POWER, PROJECTED_POWER, True),
])
def test_new_stage_only_allows_actual_forward_training_transition(old, new, evaluation):
    with pytest.raises(ValueError, match="legacy-to-projected"):
        power_stage(power_contract(old), new, explicit_stage=True, evaluation=evaluation)


@pytest.mark.parametrize("old,new", [(LEGACY_POWER, PROJECTED_POWER),
                                     (PROJECTED_POWER, LEGACY_POWER)])
def test_evaluation_may_change_signal_without_mutating_training_contract(old, new):
    saved = power_contract(old)
    before = copy.deepcopy(saved)
    previous, current, changed = power_stage(saved, new, evaluation=True)
    assert previous == before and current == power_contract(new) and changed is True
    assert saved == before


@pytest.mark.parametrize("explicit,evaluation", [("yes", False), (False, "yes"),
                                                 (1, False), (False, 1)])
def test_flags_must_be_explicit_booleans(explicit, evaluation):
    with pytest.raises(ValueError):
        power_stage(None, PROJECTED_POWER, explicit_stage=explicit, evaluation=evaluation)


def _power_admission_fragment():
    tree = ast.parse((ROOT / "scripts/train_paper_amp.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    resume = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "args.resume")
    # Skip trainer.load; stop before physical-runner admission. Execute the
    # actual power contract/provenance statements, without copying their logic.
    stop = next(index for index, node in enumerate(resume.body)
                if isinstance(node, ast.If) and "isinstance(saved, dict)" in ast.unparse(node.test))
    start = next(index for index, node in enumerate(resume.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                 and [item.id for item in node.targets[0].elts] == ["old_power", "new_power", "changed_power"])
    module = ast.fix_missing_locations(ast.Module(body=resume.body[start:stop], type_ignores=[]))
    return compile(module, "runner_power_admission", "exec")


@pytest.mark.parametrize("evaluation", [False, True])
def test_actual_runner_records_stage_or_evaluation_without_rewriting_old_provenance(evaluation):
    original = {"initialization": {"author_weights": False}}
    trainer = SimpleNamespace(provenance=copy.deepcopy(original), update_count=1000)
    env = {"trainer": trainer, "report": {}, "power_stage": power_stage,
           "file_sha256": lambda path: "f" * 64,
           "args": SimpleNamespace(power_source=PROJECTED_POWER, new_power_stage=not evaluation,
                                   evaluate_steps=6000 if evaluation else 0, resume="own_1000.pt")}
    exec(_power_admission_fragment(), env)
    if evaluation:
        assert trainer.provenance == original
        assert env["report"]["evaluation_power_measurement_changed"] is True
    else:
        assert trainer.provenance["initialization"] == original["initialization"]
        stage = trainer.provenance["power_measurement_stages"]
        assert len(stage) == 1 and stage[0]["at_update"] == 1000
        assert stage[0]["source_checkpoint_sha256"] == "f" * 64
        assert stage[0]["reward_measurement_changed"] is True
        assert stage[0]["paper_signed_energy_coefficient_preserved"] is True
        assert trainer.provenance["power_measurement"] == power_contract(PROJECTED_POWER)
    assert env["report"]["training_power_measurement"] == power_contract(LEGACY_POWER)


class _LegacyParent:
    """CPU sink for the two explicitly delegated methods only."""

    def _physics_step(self):
        self.events.append(("legacy_step",))
        return "legacy-step-result"

    def _get_simulator_dof_forces(self, env_ids=None):
        self.events.append(("legacy_force",))
        force = torch.full((self.num_envs, 69), 7.)
        if env_ids is not None:
            force = force[env_ids]
        return RobotState(dof_forces=force, state_conversion=StateConversion.SIMULATOR)


@pytest.fixture
def source_class(monkeypatch):
    warp = ModuleType("warp")
    warp.to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "warp", warp)
    tree = ast.parse(WORLD_SOURCE.read_text())
    actual = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "PaperPowerSimulator")
    # get_dof_forces is the unchanged base method; it calls the subclass getter
    # then the real RobotState conversion with a nontrivial permutation below.
    base_tree = ast.parse((ROOT / "protomotions/simulator/base_simulator/simulator.py").read_text())
    base = next(node for node in base_tree.body if isinstance(node, ast.ClassDef) and node.name == "Simulator")
    getter = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "get_dof_forces")
    env = {"torch": torch, "RobotState": RobotState, "StateConversion": StateConversion,
           "IsaacLabSimulator": _LegacyParent, "Optional": __import__("typing").Optional}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[getter], type_ignores=[])), "base_force_getter", "exec"), env)
    monkeypatch.setattr(_LegacyParent, "get_dof_forces", env["get_dof_forces"], raising=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[actual], type_ignores=[])), str(WORLD_SOURCE), "exec"), env)
    return env["PaperPowerSimulator"]


def _cpu_response_world(cls, *, headless=True):
    # Independent analytic result: sum(1..69)=2415. Four scalar products
    # average to -2.5 for actor0 and +4 for actor1, giving [-6037.5,9660] W.
    weights = torch.arange(1, 70, dtype=torch.float32)
    efforts = torch.arange(1, 5, dtype=torch.float32)[:, None, None] * weights[None, None]
    efforts = efforts.expand(4, 2, 69).clone()
    velocity = torch.tensor([[1., 2.], [-2., 3.], [3., -4.], [-4., 5.]])[:, :, None].expand(4, 2, 69).clone()
    obj = cls.__new__(cls)
    obj.events = []
    obj.num_envs, obj.device, obj.decimation, obj.headless = 2, "cpu", 4, headless
    obj.power_source = PROJECTED_POWER
    obj.index = -1
    obj.updated_index = -1
    obj.buffer = torch.empty(2, 69)

    def apply():
        obj.events.append(("apply", obj.index + 1))

    def write():
        obj.events.append(("write", obj.index + 1))

    def step(*, render):
        assert render is False
        obj.index += 1
        obj.events.append(("step", obj.index))

    def update(*, dt):
        assert dt == 1 / 120
        obj.updated_index = obj.index
        obj.events.append(("update", obj.index))

    def measured():
        assert obj.updated_index == obj.index >= 0, "measurement preceded post-step update"
        obj.events.append(("measured_force", obj.index))
        obj.buffer.copy_(efforts[obj.index % 4])
        return obj.buffer

    class Velocity:
        @property
        def joint_vel(self):
            assert obj.updated_index == obj.index >= 0, "velocity preceded post-step update"
            obj.events.append(("post_velocity", obj.index))
            return velocity[obj.index % 4]

    obj._apply_control = apply
    obj._scene = SimpleNamespace(write_data_to_sim=write, update=update)
    obj._sim = SimpleNamespace(step=step, get_physics_dt=lambda: 1 / 120,
                               render=lambda: obj.events.append(("render", obj.index)))
    obj._robot = SimpleNamespace(root_view=SimpleNamespace(get_dof_projected_joint_forces=measured), data=Velocity())
    permutation = torch.arange(69).roll(17)
    obj.data_conversion = DataConversionMapping(body_convert_to_common=torch.arange(24),
        body_convert_to_sim=torch.arange(24), dof_convert_to_common=permutation,
        dof_convert_to_sim=torch.argsort(permutation), sim_w_last=True)
    return obj, efforts, velocity


@pytest.mark.parametrize("headless", [True, False])
def test_actual_class_pairs_each_substep_before_mean_and_preserves_watts(source_class, headless):
    obj, _, _ = _cpu_response_world(source_class, headless=headless)
    obj._physics_step()
    expected_events = []
    for index in range(4):
        expected_events.extend((name, index) for name in ("apply", "write", "step"))
        if not headless and index == 3:
            expected_events.append(("render", index))
        expected_events.extend((name, index) for name in ("update", "measured_force", "post_velocity"))
    assert obj.events == expected_events
    torch.testing.assert_close(obj.mean_projected_joint_power_w, torch.tensor([-6037.5, 9660.]), rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_next_control_step_overwrites_average_instead_of_accumulating_it(source_class):
    obj, _, _ = _cpu_response_world(source_class)
    obj._physics_step()
    original = obj.mean_projected_joint_power_w.clone()
    obj._physics_step()
    assert torch.equal(obj.mean_projected_joint_power_w, original)


def test_force_read_is_cloned_subset_and_actual_common_permutation(source_class):
    obj, efforts, velocity = _cpu_response_world(source_class)
    obj._physics_step()
    native = obj._get_simulator_dof_forces()
    assert native.state_conversion == StateConversion.SIMULATOR
    assert torch.equal(native.dof_forces, efforts[3])
    obj.buffer.fill_(-999.)
    assert torch.equal(native.dof_forces, efforts[3]), "view aliases a refreshed SDK buffer"
    subset = obj.get_dof_forces(torch.tensor([1, 0]))
    assert subset.state_conversion == StateConversion.COMMON
    permutation = obj.data_conversion.dof_convert_to_common
    assert torch.equal(subset.dof_forces, efforts[3, [1, 0]][:, permutation])
    # Dot product is invariant when both generalized quantities use that order.
    assert torch.equal((efforts[3] * velocity[3]).sum(-1),
                       (efforts[3, :, permutation] * velocity[3, :, permutation]).sum(-1))


def test_legacy_methods_delegate_without_new_measurement_or_power(source_class):
    obj, _, _ = _cpu_response_world(source_class)
    obj.power_source = LEGACY_POWER
    assert obj._physics_step() == "legacy-step-result"
    assert obj.mean_projected_joint_power_w is None
    assert obj.events == [("legacy_step",)]
    state = obj._get_simulator_dof_forces(torch.tensor([1]))
    assert torch.equal(state.dof_forces, torch.full((1, 69), 7.))
    assert obj.events == [("legacy_step",), ("legacy_force",)]


def test_getter_failure_does_not_fall_back_to_proxy(source_class):
    obj, _, _ = _cpu_response_world(source_class)

    def missing():
        raise RuntimeError("measured getter unavailable")

    obj._robot.root_view.get_dof_projected_joint_forces = missing
    with pytest.raises(RuntimeError, match="unavailable"):
        obj._physics_step()
    assert obj.mean_projected_joint_power_w is None
    assert not any(event[0].startswith("legacy") for event in obj.events)


def test_reward_consumes_substep_average_and_retains_paper_coefficient(source_class):
    obj, efforts, velocities = _cpu_response_world(source_class)
    obj._physics_step()
    result = paper_task_reward(torch.zeros(2), efforts[-1], velocities[-1], torch.zeros(2),
                               signed_power_w=obj.mean_projected_joint_power_w)
    assert torch.equal(result["signed_power_w"], torch.tensor([-6037.5, 9660.]))
    torch.testing.assert_close(result["energy"], torch.tensor([3.01875, -4.83]))
    assert not torch.equal(result["signed_power_w"], (efforts[-1] * velocities[-1]).sum(-1))
