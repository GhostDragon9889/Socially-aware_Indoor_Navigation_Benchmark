"""CPU geometry and visibility regressions for the SMPL visual overlay."""

from protomotions.repository import repository_root
from pathlib import Path
import pickle
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.tools.smpl_mesh_visualizer import (
    HumanMeshConfig,
    MJCF_JOINT_NAMES,
    ProtoMotionsHumanMeshAdapter,
    SMPLMeshVisualizer,
    SMPL_JOINT_NAMES,
    SMPLRobotPoseMapper,
    inspect_smpl_pickle,
    resolve_smpl_model_path,
)

ROOT = repository_root(__file__)


@pytest.fixture(scope="module")
def real_smpl_joint_names():
    from protomotions.components.pose_lib import extract_kinematic_info
    from protomotions.simulator.isaaclab.utils.actuator_groups import build_isaaclab_joint_name_map

    kinematics = extract_kinematic_info(str(ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"))
    mapping = build_isaaclab_joint_name_map(kinematics)
    semantic = list(kinematics.dof_names)
    backend = [mapping.semantic_to_backend[name] for name in semantic]
    assert mapping.semantic_to_backend["L_Hip_y"] == "L_Hip_x:1"
    return semantic, backend


@pytest.mark.parametrize("seed", [3, 17, 91])
def test_real_d6_names_map_identically_under_backend_permutation(real_smpl_joint_names, seed):
    semantic, backend = real_smpl_joint_names
    mapper = SMPLRobotPoseMapper()
    generator = torch.Generator().manual_seed(seed)
    joints = torch.randn(2, 69, generator=generator) * 0.15
    expected = mapper(joints, semantic)
    permutation = torch.randperm(69, generator=generator)
    permuted_names = [backend[index] for index in permutation.tolist()]
    actual = mapper(joints[:, permutation], permuted_names)
    torch.testing.assert_close(actual, expected)


def test_d6_hip_motion_does_not_drive_arm(real_smpl_joint_names):
    semantic, backend = real_smpl_joint_names
    joints = torch.zeros(1, 69)
    joints[:, semantic.index("L_Hip_y")] = 0.4
    pose = SMPLRobotPoseMapper()(joints.flip(1), list(reversed(backend))).reshape(1, 23, 3)
    assert torch.linalg.vector_norm(pose[:, SMPL_JOINT_NAMES[1:].index("L_Hip")]) > 0.39
    for name in ("L_Shoulder", "L_Elbow", "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"):
        assert torch.count_nonzero(pose[:, SMPL_JOINT_NAMES[1:].index(name)]) == 0


@pytest.mark.parametrize("bad_case", ["unknown", "duplicate", "missing", "mixed_axes"])
def test_mapper_rejects_unknown_or_ambiguous_names(real_smpl_joint_names, bad_case):
    _, backend = real_smpl_joint_names
    names = list(backend)
    if bad_case == "unknown":
        names[1] = "unknown_joint"
    elif bad_case == "duplicate":
        names[1] = names[0]
    elif bad_case == "missing":
        names.pop()
    else:
        names[1] = "L_Hip_y"
    with pytest.raises(ValueError):
        SMPLRobotPoseMapper()(torch.zeros(1, 69), names)


@pytest.fixture(scope="module")
def visualizer():
    model = ROOT / "data/smpl/SMPL_NEUTRAL.pkl"
    if not model.is_file():
        pytest.skip("A licensed local SMPL neutral model is required for real geometry tests")
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    result = SMPLMeshVisualizer.from_config(HumanMeshConfig(
        model_dir=str(model.parent), num_envs=2, device="cpu", shape_std=0.0,
        hide_humanoid=True, num_active_humanoids=1,
    ))
    yield result
    torch.set_num_threads(threads)


def test_missing_model_and_lfs_pointer_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="SMPL neutral model missing"):
        resolve_smpl_model_path(str(tmp_path))
    pointer = tmp_path / "SMPL_NEUTRAL.pkl"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        inspect_smpl_pickle(pointer)


def test_non_numeric_pickle_global_is_rejected_before_loading(tmp_path):
    path = tmp_path / "untrusted.pkl"
    path.write_bytes(pickle.dumps(os.system, protocol=2))
    with pytest.raises(ValueError, match="Unsupported SMPL pickle global"):
        inspect_smpl_pickle(path)


def test_real_neutral_model_skins_poses_with_correct_world_rotation(visualizer):
    joint_names = [f"{name}_{axis}" for name in MJCF_JOINT_NAMES[1:] for axis in "xyz"]
    mapper = SMPLRobotPoseMapper()
    joints = torch.zeros(2, 69)
    body_pose = mapper(joints, joint_names)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    vertices = visualizer.compute_world_vertices(body_pose, pos, identity)
    assert vertices.shape == (2, 6890, 3)
    assert torch.isfinite(vertices).all()
    assert visualizer.faces.shape == (13776, 3)
    assert not visualizer.has_texture
    torch.testing.assert_close(vertices[1], vertices[0] + pos[1], atol=1e-6, rtol=1e-6)

    yaw90 = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]]).repeat(2, 1)
    rotated = visualizer.compute_world_vertices(body_pose, pos, yaw90)
    expected = vertices[0][:, [1, 0, 2]].clone()
    expected[:, 0] *= -1
    torch.testing.assert_close(rotated[0], expected, atol=1e-6, rtol=1e-6)

    for angle in (0.2, 0.5):
        joints[:, joint_names.index("L_Knee_x")] = angle
        posed = visualizer.compute_world_vertices(mapper(joints, joint_names), pos, identity)
        assert torch.isfinite(posed).all()
        assert not torch.allclose(posed, vertices)


def test_invalid_uv_data_uses_solid_material(visualizer, tmp_path):
    texture = tmp_path / "body.png"
    texture.write_bytes(b"unused by solid-color fallback")
    cfg = HumanMeshConfig(
        model_dir=visualizer.cfg.model_dir, num_envs=2, device="cpu",
        shape_std=0.0, texture_paths=(str(texture),),
    )
    for uv, indices in [
        (np.zeros((3, 2)), None),
        (np.zeros((3, 2)), np.zeros_like(visualizer.faces, dtype=float)),
        (np.full((3, 2), np.nan), visualizer.faces),
        (np.array([["invalid", "uv"]]), visualizer.faces),
    ]:
        result = SMPLMeshVisualizer(cfg, visualizer.smpl_model, visualizer.faces, uv, indices)
        assert not result.has_texture
        assert result.uvs is None and result.uv_faces is None


@pytest.mark.parametrize("proxy_views", [False, True])
def test_adapter_converts_xyzw_and_restores_skeleton_on_failure(visualizer, monkeypatch, proxy_views):
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    for i in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{i}/Robot")
    omni = ModuleType("omni")
    omni.__path__ = []
    usd = ModuleType("omni.usd")
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni.usd = usd
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)

    visualizer.stage = None
    visualizer._created = False
    visualizer._points_attrs = []
    data = SimpleNamespace(
        joint_pos=torch.zeros(2, 69),
        joint_names=[f"{name}_{axis}" for name in MJCF_JOINT_NAMES[1:] for axis in "xyz"],
        root_pos_w=torch.zeros(2, 3),
        root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    )
    quaternion_tensor = data.root_quat_w
    if proxy_views:
        # Warp vec/quat arrays report shape (N,), unlike their Torch view.
        for name in ("joint_pos", "root_pos_w", "root_quat_w"):
            value = getattr(data, name)
            setattr(data, name, SimpleNamespace(torch=value, shape=(2,)))
    simulator = SimpleNamespace(_robot=SimpleNamespace(data=data), native_quaternion_order="xyzw")
    adapter = ProtoMotionsHumanMeshAdapter(simulator, visualizer)
    adapter.create()
    colors = []
    for env_id in range(2):
        mesh_prim = stage.GetPrimAtPath(f"{visualizer.cfg.prim_root}/env_{env_id}/Body")
        assert mesh_prim.HasAPI(UsdShade.MaterialBindingAPI)
        bound_material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        assert bound_material.GetPath() == f"{visualizer.cfg.prim_root}/Materials/person_{env_id}"
        shader = UsdShade.Shader.Get(stage, f"{bound_material.GetPath()}/Shader")
        assert shader.GetOutput("surface").GetTypeName() == Sdf.ValueTypeNames.Token
        assert bound_material.GetSurfaceOutput().GetConnectedSource()[0].GetPrim() == shader.GetPrim()
        assert shader.GetInput("roughness").Get() == pytest.approx(0.85)
        color = np.asarray(shader.GetInput("diffuseColor").Get())
        assert color.max() <= 0.3
        colors.append(color)
        np.testing.assert_allclose(UsdGeom.Mesh(mesh_prim).GetDisplayColorAttr().Get()[0], color)
    assert not np.allclose(colors[0], colors[1])
    points = np.asarray(visualizer._points_attrs[0].Get())
    expected = visualizer.compute_world_vertices(
        torch.zeros(2, 69), torch.zeros(2, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
    )
    np.testing.assert_allclose(points, expected[0].numpy(), atol=1e-6)
    active = UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
    filler = UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_1/Robot"))
    assert active.ComputeVisibility() == UsdGeom.Tokens.invisible
    assert filler.ComputeVisibility() == UsdGeom.Tokens.inherited

    quaternion_tensor.zero_()
    with pytest.raises(ValueError, match="nonzero norm"):
        adapter.update()
    assert active.ComputeVisibility() == UsdGeom.Tokens.inherited
    overlay = UsdGeom.Imageable(stage.GetPrimAtPath(visualizer.cfg.prim_root))
    assert overlay.ComputeVisibility() == UsdGeom.Tokens.invisible
