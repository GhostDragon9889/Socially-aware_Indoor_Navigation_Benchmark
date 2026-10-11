"""Real USD schema contracts; CPU tests do not simulate or certify PhysX."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from protomotions.simulator.isaaclab.utils.usd_body_paths import (
    _select_physx_variant,
    physx_usd_variant_selection,
    resolve_robot_prim_paths,
)
from protomotions.simulator.isaaclab.utils.actuator_groups import (
    build_isaaclab_joint_name_map,
    semantic_dof_names_for_articulation,
)


def authored_robot(path, *, include_physx=True):
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    variant = root.GetVariantSets().AddVariantSet("Physics")
    for selection in ("physics", "physx") if include_physx else ("physics",):
        variant.AddVariant(selection)
        variant.SetVariantSelection(selection)
        with variant.GetVariantEditContext():
            body = UsdGeom.Xform.Define(stage, "/robot/Pelvis").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            UsdPhysics.ArticulationRootAPI.Apply(body)
            if selection == "physics":
                # Generic payload models three parallel MJCF hinges.
                for axis in "XYZ":
                    hinge = UsdPhysics.RevoluteJoint.Define(stage, f"/robot/Hip_{axis.lower()}")
                    hinge.CreateAxisAttr(axis)
            else:
                # PhysX payload models a single D6 with all three rotations.
                joint = UsdPhysics.Joint.Define(stage, "/robot/Hip_x")
                for axis in "XYZ":
                    limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), f"rot{axis}")
                    limit.CreateLowAttr(-180.)
                    limit.CreateHighAttr(180.)
    variant.SetVariantSelection("physics")
    stage.GetRootLayer().Save()
    return stage


def test_inspection_and_spawn_select_physx_without_rewriting_source(tmp_path):
    path = tmp_path / "with spaces.usda"
    stage = authored_robot(path)
    original = path.read_bytes()
    assert sum(p.IsA(UsdPhysics.RevoluteJoint) for p in stage.Traverse()) == 3
    assert _select_physx_variant(stage) == {"Physics": "physx"}
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert len(joints) == 1
    assert joints[0].GetTypeName() == "PhysicsJoint"
    assert all(joints[0].HasAPI(UsdPhysics.LimitAPI, f"rot{a}") for a in "XYZ")
    assert path.read_bytes() == original
    assert physx_usd_variant_selection(str(path)) == {"Physics": "physx"}
    assert resolve_robot_prim_paths(str(path), ["Pelvis"]) == ("/Pelvis", {"Pelvis": "Pelvis"})
    assert path.read_bytes() == original
    # A fresh session still observes the authored importer default.
    fresh = Usd.Stage.Open(str(path))
    assert fresh.GetDefaultPrim().GetVariantSet("Physics").GetVariantSelection() == "physics"


def test_multi_physics_asset_without_physx_is_rejected(tmp_path):
    path = tmp_path / "missing.usda"
    authored_robot(path, include_physx=False)
    with pytest.raises(ValueError, match="without a physx payload"):
        physx_usd_variant_selection(str(path))


def test_single_payload_asset_needs_no_variant_override(tmp_path):
    path = tmp_path / "single.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/robot").GetPrim())
    stage.GetRootLayer().Save()
    assert physx_usd_variant_selection(str(path)) is None


@pytest.fixture
def smpl_kinematics():
    from protomotions.robot_configs.smpl import SmplRobotConfig
    return SmplRobotConfig().kinematic_info


def test_all_69_smpl_axes_map_in_live_order(smpl_kinematics):
    mapping = build_isaaclab_joint_name_map(smpl_kinematics)
    backend_names = list(reversed(mapping.backend_to_semantic))
    articulation = SimpleNamespace(num_joints=69, joint_names=backend_names)
    actual = semantic_dof_names_for_articulation(articulation, smpl_kinematics)
    assert len(actual) == 69
    assert set(actual) == set(smpl_kinematics.dof_names)
    assert actual == [mapping.backend_to_semantic[name] for name in backend_names]


def test_lost_physical_dofs_cannot_be_fixed_by_labels(smpl_kinematics):
    mapping = build_isaaclab_joint_name_map(smpl_kinematics)
    articulation = SimpleNamespace(num_joints=23, joint_names=list(mapping.backend_to_semantic))
    with pytest.raises(ValueError, match="requires 69 DOFs"):
        semantic_dof_names_for_articulation(articulation, smpl_kinematics)


def test_duplicate_or_unknown_live_axes_are_rejected(smpl_kinematics):
    names = list(build_isaaclab_joint_name_map(smpl_kinematics).backend_to_semantic)
    names[-1] = names[0]
    with pytest.raises(ValueError, match="DOF names do not match"):
        semantic_dof_names_for_articulation(SimpleNamespace(num_joints=69, joint_names=names), smpl_kinematics)
