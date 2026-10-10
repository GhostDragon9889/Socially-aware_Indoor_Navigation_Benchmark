"""Inference configuration, frozen policy and failed Kit startup contracts."""

from protomotions.repository import repository_root

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from CrowdSim import protomotions_runtime as runtime
from CrowdSim.world import builder


@pytest.mark.needs_lfs
@pytest.mark.parametrize("hz", [25.0, 30.0])
def test_local_checkpoint_uses_smpl_amass_clock_and_no_projectiles(hz):
    root = repository_root(__file__)
    checkpoint = root / "data/pretrained_models/masked_mimic/smpl/last.ckpt"
    if not checkpoint.is_file():
        pytest.skip("Local SMPL inference checkpoint is not installed")
    inference = checkpoint.parent / "resolved_configs_inference.pt"
    if not inference.is_file():
        pytest.skip("Local SMPL inference configuration is not installed")
    with inference.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            pytest.skip("SMPL inference configuration is an unresolved LFS pointer")
    motion = root.parent / "Assets/motion/amass_smpl_validation.pt"
    robot, sim, _, _, lib, _, _ = runtime.configure_from_checkpoint(
        checkpoint, motion, num_envs=2, headless=True, control_hz=hz,
        compile_components=False, shared_world=True,
    )
    assert robot.kinematic_info.num_bodies == 24
    assert robot.kinematic_info.num_dofs == 69
    assert sim.w_last and sim.num_envs == 2 and sim.headless
    assert not sim.filter_collisions
    assert sim.sim.fps / sim.sim.decimation == hz
    assert sim.projectile.num_projectiles == 0
    assert lib.motion_file == str(motion)
    assert lib.motion_file_shard_indices is None


def test_build_runtime_loads_only_policy_and_freezes_it(monkeypatch, tmp_path):
    calls = []
    config = SimpleNamespace(_target_="env", save_dir=None)
    simulator = SimpleNamespace(native_quaternion_order="xyzw")
    model = torch.nn.Linear(2, 1)

    class Agent:
        def __init__(self, **kwargs):
            self.model = model
        def setup(self):
            pass
        def load(self, path, **kwargs):
            calls.append(kwargs)
        def eval(self):
            self.model.eval()

    monkeypatch.setattr(runtime, "configure_from_checkpoint", lambda *a, **k: (
        object(), SimpleNamespace(), None, object(), object(), config,
        SimpleNamespace(_target_="agent"),
    ))
    import protomotions.simulator.base_simulator.utils as sim_utils
    import protomotions.utils.component_builder as components
    import protomotions.utils.hydra_replacement as hydra
    monkeypatch.setattr(sim_utils, "convert_friction_for_simulator", lambda terrain, sim: (terrain, sim))
    monkeypatch.setattr(components, "build_all_components", lambda **kwargs: dict(
        terrain=None, scene_lib=None, motion_lib=None, simulator=simulator
    ))
    monkeypatch.setattr(hydra, "get_class", lambda target: Agent if target == "agent" else lambda **k: SimpleNamespace(simulator=simulator))
    result = runtime.build_runtime(tmp_path / "last.ckpt", tmp_path / "motion.pt", 2, True, None, SimpleNamespace(device="cpu"))
    assert calls == [{"load_env": False, "load_training_state": False}]
    assert not result.agent.model.training
    assert not any(p.requires_grad for p in result.agent.model.parameters())
    assert result.env._crowdsim_articulation_quaternion_order == "xyzw"


def test_failed_environment_construction_closes_kit(monkeypatch):
    closed = []
    def fail(config, *, lifecycle, **kwargs):
        lifecycle["app"] = SimpleNamespace(close=lambda **kwargs: closed.append(kwargs))
        raise RuntimeError("mesh initialization failed")
    monkeypatch.setattr(builder, "_build_env_impl", fail)
    with pytest.raises(RuntimeError, match="mesh initialization failed"):
        builder.build_env({})
    assert closed == [{"exit_code": 1}]
