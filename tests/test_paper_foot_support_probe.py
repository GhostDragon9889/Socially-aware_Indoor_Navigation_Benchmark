"""Standalone probe CPU seams; mock contacts do not validate native PhysX."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.foot_support import FOOT_NAMES, FootSupportMonitor
from CrowdSim.humanoid.motion_tracking import SMPL_BODY_NAMES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_paper_foot_support.py"
spec = importlib.util.spec_from_file_location("foot_support_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ContactView:
    def __init__(self, sources, filter_path, force):
        self.sensor_paths = list(reversed(sources))
        self.sensor_count, self.filter_count = len(sources), 1
        self.filter_paths = [[filter_path] for _ in sources]
        self.force, self.calls = force.clone(), []

    def get_contact_force_matrix(self, *, dt):
        self.calls.append(dt)
        return self.force


def adapter_fixture(*, floor_force=20., wall_force=0., foot_height=3.):
    common = SMPL_BODY_NAMES
    native = tuple(reversed(common))
    pos = torch.zeros(1, 24, 3)
    pos[..., 2] = foot_height
    # All rotations are 90deg aroundZ; local offset has an independently known world offset.
    quat = torch.tensor([0., 0., 2 ** -.5, 2 ** -.5]).expand(1, 24, 4).clone()
    local_com = torch.tensor([.02, .03, .01]).expand(1, 24, 3).clone()
    world_com = pos + torch.tensor([-.03, .02, .01])
    native_masses = torch.arange(1, 25, dtype=torch.float32).reshape(1, 24)
    clock = SimpleNamespace(value=100)

    def get_physics_step_count():
        return clock.value

    data = SimpleNamespace(body_link_pos_w=pos.clone(), body_link_quat_w=quat,
                           body_link_lin_vel_w=torch.zeros(1, 24, 3), body_link_ang_vel_w=torch.zeros(1, 24, 3),
                           body_com_pos_w=world_com.clone())
    root_view = SimpleNamespace(shared_metatype=SimpleNamespace(link_names=native),
                                get_coms=lambda: torch.cat((local_com, quat), -1), get_masses=lambda: native_masses)
    robot = SimpleNamespace(body_names=native, data=data, root_view=root_view)
    world = SimpleNamespace(num_envs=1, device=torch.device("cpu"), episode_ids=torch.tensor([3]),
                            mapping=SimpleNamespace(body_names=common),
                            simulator=SimpleNamespace(_robot=robot, _sim=SimpleNamespace(get_physics_step_count=get_physics_step_count)))
    paths = [{name: f"/World/envs/env_0/Robot/{name}" for name in FOOT_NAMES}]
    sources = [paths[0][name] for name in FOOT_NAMES]
    fforce = torch.zeros(4, 1, 3)
    fforce[..., 2] = floor_force
    wforce = torch.zeros(4, 1, 3)
    wforce[..., 0] = wall_force
    floor_view = ContactView(sources, "/World/ground/mesh", fforce)
    floor = FootSupportMonitor(floor_view, paths, ["/World/ground/mesh"], contact_reporting_enabled=True)
    wall_view = ContactView(sources, "/World/wall", wforce)
    adapter = probe.ProbeSubstepAdapter(world, floor, wall_view, "/World/wall", com_tolerance_m=1e-4)
    return adapter, clock, data, floor_view, wall_view


def finish_mock_control(adapter, clock):
    adapter.begin_control_step()
    for _ in range(4):
        clock.value += 1  # explicit fake physics event, not an SDK call
        adapter.sample(1 / 120)
    result = adapter.measurements()
    result.update(common_body_names=SMPL_BODY_NAMES, fall_mask=False, time_limit_mask=False)
    return result


def test_existing_zero_argument_power_hook_adapter_reads_four_real_clock_values_without_advance():
    adapter, clock, data, floor_view, wall_view = adapter_fixture()
    result = finish_mock_control(adapter, clock)
    assert clock.value == 104
    assert result["physics_step_before_read"].tolist() == [101, 102, 103, 104]
    assert result["physics_step_after_read"].tolist() == [101, 102, 103, 104]
    assert floor_view.calls == wall_view.calls == [1 / 120] * 4
    assert result["body_masses_kg"][0, 0].tolist() == list(range(24, 0, -1))
    assert result["com_composition_max_error_m"].max() < 1e-6
    assert result["foot_floor_normal_force_w"].shape == (4, 1, 4, 1, 3)
    data.body_link_pos_w.fill_(999)
    floor_view.force.fill_(999)
    assert result["body_link_pos_w"][0, 0, 0, 2] == 3
    assert result["foot_floor_normal_force_w"][0, 0, 0, 0, 2] == 20
    assert adapter.manifest["native_engine_time_observed"] is False


def test_missing_or_noninteger_clock_cannot_fall_back_to_a_synthetic_substep():
    adapter, _, _, _, _ = adapter_fixture()
    adapter.counter_getter = lambda: 3.5
    with pytest.raises(RuntimeError, match="actual Lab"):
        adapter.begin_control_step()
    world = adapter.world
    world.simulator._sim.get_physics_step_count = None
    with pytest.raises(RuntimeError, match="no synthetic"):
        probe.ProbeSubstepAdapter(world, adapter.floor, adapter.wall_view, "/World/wall", com_tolerance_m=1e-4)


def test_repeated_physics_tick_and_episode_reset_are_caught_in_original_hook():
    adapter, clock, _, _, _ = adapter_fixture()
    adapter.begin_control_step()
    with pytest.raises(RuntimeError, match="physics substep"):
        adapter.sample(1 / 120)
    assert len(adapter.pending) == 1
    with pytest.raises(RuntimeError, match="four archived"):
        adapter.measurements()
    adapter, clock, _, _, _ = adapter_fixture()
    adapter.begin_control_step()
    clock.value += 1
    adapter.world.episode_ids.add_(1)
    with pytest.raises(RuntimeError, match="reset inside"):
        adapter.sample(1 / 120)


def test_invalid_physical_channel_preserves_partial_native_evidence(tmp_path):
    adapter, clock, data, _, _ = adapter_fixture()
    adapter.begin_control_step()
    clock.value += 1
    data.body_link_lin_vel_w[0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="body_link_lin_vel_w"):
        adapter.sample(1 / 120)
    assert "invalid_native_body_link_lin_vel_w" in adapter.pending[0]
    assert adapter.last_partial_error.startswith("FloatingPointError:")
    with pytest.raises(RuntimeError, match="Archive interrupted"):
        adapter.reset_phase()
    path = tmp_path / "partial.npz"
    record = probe.save_capture(adapter.pending, path)
    assert record["records"] == 1
    with np.load(path, allow_pickle=False) as capture:
        assert capture["physics_step_before_read"].tolist() == [101]
        assert np.isinf(capture["invalid_native_body_link_lin_vel_w"]).any()
        assert "native_floor_normal_force_w" not in capture.files  # unavailable, never fabricated0


def test_partial_force_failure_keeps_native_force_snapshot(tmp_path):
    adapter, clock, _, floor_view, _ = adapter_fixture()
    adapter.begin_control_step()
    clock.value += 1
    floor_view.force[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="floor normal forces"):
        adapter.sample(1 / 120)
    saved = probe.save_capture(adapter.pending, tmp_path / "bad_force.npz")
    with np.load(saved["path"], allow_pickle=False) as capture:
        assert np.isnan(capture["native_floor_normal_force_w"]).any()


def test_live_COM_failure_does_not_pass_with_a_body_origin_proxy():
    adapter, clock, data, _, _ = adapter_fixture()
    adapter.begin_control_step()
    data.body_com_pos_w.copy_(data.body_link_pos_w)
    clock.value += 1
    with pytest.raises(RuntimeError, match="Live world COM"):
        adapter.sample(1 / 120)


def test_wall_and_floor_partner_paths_are_independently_gated():
    adapter, _, _, _, view = adapter_fixture()
    view.filter_paths[2][0] = "/World/ground/mesh"
    with pytest.raises(RuntimeError, match="wall collider"):
        probe.ProbeSubstepAdapter(adapter.world, adapter.floor, view, "/World/wall", com_tolerance_m=1e-4)


def test_native_contact_lookup_uses_live_Lab_manager_and_exact_existing_factory():
    calls = []
    sources = [f"/World/envs/env_0/Robot/{name}" for name in FOOT_NAMES]

    class ExistingWarpView:
        def create_rigid_contact_view(self, pattern, filter_patterns, max_contact_data_count):
            calls.append((pattern, filter_patterns, max_contact_data_count))
            return ContactView(pattern, filter_patterns[0][0], torch.zeros(4, 1, 3))

    view = ExistingWarpView()

    class ActualLabPhysxManager:
        @classmethod
        def get_physics_sim_view(cls):
            return view

    class LabContext:
        physics_manager = ActualLabPhysxManager

        @property
        def physics_sim_view(self):
            return self.physics_manager.get_physics_sim_view()

        def __getattr__(self, name):
            if name == "_physics_sim_view":
                raise AssertionError("A guessed private SDK attribute must never be read")
            raise AttributeError(name)

    factory, contract = probe.resolve_native_contact_factory(LabContext())
    assert factory is view
    assert contract["active_manager_getter_same_view"] is True
    assert contract["legacy_isaacsim_SimulationManager_used"] is False
    assert contract["new_simulation_view_created"] is False
    paths = [{name: sources[i] for i, name in enumerate(FOOT_NAMES)}]
    FootSupportMonitor.from_factory(factory, paths, ["/World/ground/mesh"], contact_reporting_enabled=True)
    assert calls == [(sources, [["/World/ground/mesh"]] * 4, 512)]


@pytest.mark.parametrize("problem", ["no_manager", "no_live_view", "different_owner_view", "no_contact_factory"])
def test_native_lookup_fails_closed_for_unavailable_wrong_owner_or_backend(problem):
    existing = SimpleNamespace(create_rigid_contact_view=lambda *args, **kwargs: None)
    manager = SimpleNamespace(get_physics_sim_view=lambda: existing)
    context = SimpleNamespace(physics_manager=manager, physics_sim_view=existing)
    if problem == "no_manager":
        context.physics_manager = None
    elif problem == "no_live_view":
        context.physics_sim_view = None
    elif problem == "different_owner_view":
        context.physics_sim_view = SimpleNamespace(create_rigid_contact_view=lambda *args, **kwargs: None)
    else:
        existing.create_rigid_contact_view = None
    with pytest.raises(RuntimeError):
        probe.resolve_native_contact_factory(context)


def test_standing_airborne_and_wall_phase_criteria_use_distinct_actual_inputs():
    args = SimpleNamespace(stable_window_steps=2, stable_support_fraction=.9, threshold_n=1.,
                           zero_force_tolerance_n=1e-5, com_tolerance_m=1e-4)
    adapter, clock, _, _, _ = adapter_fixture()
    standing = [finish_mock_control(adapter, clock) for _ in range(4)]
    assert probe.summarize_phase("standing", standing, args)["criteria_passed"]
    standing[2]["fall_mask"] = True
    assert not probe.summarize_phase("standing", standing, args)["criteria_passed"]
    adapter, clock, _, _, _ = adapter_fixture(floor_force=0., wall_force=0.)
    airborne = [finish_mock_control(adapter, clock) for _ in range(4)]
    assert probe.summarize_phase("airborne", airborne, args)["criteria_passed"]
    assert not probe.summarize_phase("airborne_wall", airborne, args)["criteria_passed"]
    adapter, clock, _, _, _ = adapter_fixture(floor_force=0., wall_force=40.)
    wall = [finish_mock_control(adapter, clock) for _ in range(4)]
    assert probe.summarize_phase("airborne_wall", wall, args)["criteria_passed"]
    # A wall force must never rescue nonzero measured floor forces in airborne test.
    wall[0]["foot_floor_normal_force_w"].fill_(2.)
    assert not probe.summarize_phase("airborne_wall", wall, args)["criteria_passed"]


def test_completed_stance_failure_does_not_abort_independent_sensor_phases():
    args = SimpleNamespace(stable_window_steps=2, stable_support_fraction=.9, threshold_n=1.,
                           zero_force_tolerance_n=1e-5, com_tolerance_m=1e-4)
    observed = []

    def execute(name, steps):
        observed.append(name)
        adapter, clock, _, _, _ = adapter_fixture(floor_force=20. if name == "standing" else 0.,
                                                  wall_force=40. if name == "airborne_wall" else 0.)
        rows = [finish_mock_control(adapter, clock) for _ in range(steps)]
        if name == "standing":
            rows[-1]["fall_mask"] = True
        return {"phase": name, "summary": probe.summarize_phase(name, rows, args)}

    result = probe.execute_independent_phases((("standing", 4), ("airborne", 4), ("airborne_wall", 4)), execute)
    assert observed == ["standing", "airborne", "airborne_wall"]
    assert result["all_independent_phases_completed"]
    assert result["sensor_channel_criteria_passed"]
    assert result["standing_no_fall_criterion_passed"] is False
    assert result["failed_criteria_phases"] == ["standing"]
    assert result["native_physical_probe_passed"] is False


def test_nonfinite_acquisition_still_aborts_and_missing_phase_cannot_pass():
    observed = []

    def execute(name, steps):
        observed.append(name)
        raise FloatingPointError("actual invalid native force")

    with pytest.raises(FloatingPointError, match="invalid native force"):
        probe.execute_independent_phases((("standing", 60), ("airborne", 4), ("airborne_wall", 4)), execute)
    assert observed == ["standing"]
    with pytest.raises(RuntimeError, match="All three"):
        probe.aggregate_phase_results([])


def synthetic_neutral_geometry():
    names = ("Pelvis", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe", "Torso")
    position = [[0, 0, 0], [0, .1, -.9], [.1, .1, -.9], [0, -.1, -.9], [.1, -.1, -.9], [0, 0, .3]]
    feet = {}
    for name, y, z in (("L_Ankle", .1, -.95), ("L_Toe", .1, -.95),
                       ("R_Ankle", -.1, -.9518), ("R_Toe", -.1, -.9518)):
        feet[name] = [[x, yy, zz] for x in (-.5, .5) for yy in (y - .04, y + .04) for zz in (z, z + .03)]
    return {"body_names": names, "body_pos_relative_pelvis_m": position,
            "parent_body_names": [None, "Pelvis", "L_Ankle", "Pelvis", "R_Ankle", "Pelvis"],
            "foot_collision_vertices_relative_pelvis_m": feet}


def test_grounded_neutral_uses_lowest_transformed_collision_point_not_body_origin():
    geometry = synthetic_neutral_geometry()
    masses = torch.ones(6, dtype=torch.float64)
    local = torch.zeros(6, 3, dtype=torch.float64)
    estimate = probe.neutral_support_estimate(geometry, masses, local, sole_clearance_m=.001)
    assert estimate["root_height_m"] == pytest.approx(.9528)
    assert estimate["sole_bottoms_m"]["R_Ankle"] == pytest.approx(.001)
    assert estimate["sole_bottoms_m"]["L_Ankle"] == pytest.approx(.0028)
    # Origin-only placement would use .901m and penetrate the collision sole.
    assert estimate["root_height_m"] != pytest.approx(.901)
    assert estimate["support_forces"][:, 2].sum() == pytest.approx(6 * 9.81)
    assert estimate["footprint_limitation"].endswith("not observed active contact-patch polygon/COP")
    local[:, 0] = 2
    with pytest.raises(ValueError, match="outside an ankle"):
        probe.neutral_support_estimate(geometry, masses, local, sole_clearance_m=.001)


def test_static_balance_torque_known_upper_body_lever_and_floating_base_wrench():
    geometry = synthetic_neutral_geometry()
    masses = torch.ones(6, dtype=torch.float64)
    masses[-1] = 2
    local = torch.zeros(6, 3, dtype=torch.float64)
    local[-1, 0] = .2
    estimate = probe.neutral_support_estimate(geometry, masses, local, sole_clearance_m=.001)
    torques, report = probe.neutral_static_joint_torques(geometry, masses, estimate)
    # A 2kg upper body has COM20cm forward of its own joint: gravity's +Y
    # moment is 3.924Nm, so its child-axis drive must apply -Y3.924Nm.
    assert torques[-1].tolist() == pytest.approx([0., -3.924, 0.])
    assert report["floating_base_force_residual_n"] == pytest.approx([0., 0., 0.], abs=1e-12)
    assert report["floating_base_moment_residual_nm"] == pytest.approx([0., 0., 0.], abs=1e-12)
    estimate["support_forces"][0, 2] += 10
    with pytest.raises(ValueError, match="floating-base wrench"):
        probe.neutral_static_joint_torques(geometry, masses, estimate)


def test_exact_cube_transform_and_nested_rigid_owner_determine_grounded_geometry(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "neutral.usda"
    stage = Usd.Stage.CreateNew(str(path))
    names = ("Pelvis", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
    bodies = {"Pelvis": "/Rig/Pelvis", "L_Ankle": "/Rig/Pelvis/L_Ankle", "L_Toe": "/Rig/Pelvis/L_Ankle/Wrapper/L_Toe",
              "R_Ankle": "/Rig/Pelvis/R_Ankle", "R_Toe": "/Rig/Pelvis/R_Ankle/Wrapper/R_Toe"}
    for name in names:
        body = UsdGeom.Xform.Define(stage, bodies[name])
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        if "Ankle" in name:
            body.AddTranslateOp().Set(Gf.Vec3d(0, .1 if name.startswith("L_") else -.1, -.9))
        elif "Toe" in name:
            body.AddTranslateOp().Set(Gf.Vec3d(.2, 0, -.05))
        if name != "Pelvis":
            cube = UsdGeom.Cube.Define(stage, bodies[name] + "/Geometry/Box")
            cube.GetSizeAttr().Set(2)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(.04, .03, .02))
            UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(.5, 0, -.1))
    stage.GetRootLayer().Save()
    geometry = probe.load_neutral_collision_geometry(path, names)
    ankle = np.asarray(geometry["foot_collision_vertices_relative_pelvis_m"]["L_Ankle"])
    toe = np.asarray(geometry["foot_collision_vertices_relative_pelvis_m"]["L_Toe"])
    # Gf's actual op composition scales the translated local coordinates too;
    # this checks true USD transforms rather than assumed size-only placement.
    assert ankle[:, 2].min() == pytest.approx(-.922, abs=1e-7)
    assert toe[:, 2].min() == pytest.approx(-.972, abs=1e-7)
    assert len(ankle) == len(toe) == 8  # Toe geometry is not counted again by its ankle ancestor.
    assert geometry["parent_body_names"][names.index("L_Toe")] == "L_Ankle"


def test_gravity_bias_fixed_target_keeps_initial_neutral_pose_and_uses_real_reordered_native_gains(monkeypatch, tmp_path):
    from CrowdSim.humanoid.amp.observations import PacerSMPLMapping

    mapping = PacerSMPLMapping()
    names = mapping.body_names
    base = synthetic_neutral_geometry()
    geometry = {**base, "body_names": list(names),
                "body_pos_relative_pelvis_m": [base["body_pos_relative_pelvis_m"][base["body_names"].index(name)]
                                               if name in base["body_names"] else [0, 0, .2] for name in names],
                "parent_body_names": [None if name == "Pelvis" else name[:2] + "Ankle" if name.endswith("Toe") else "Pelvis" for name in names]}
    native_dofs = tuple(reversed(mapping.dof_names))
    stiff = torch.tensor([[1000. if name.startswith("Torso") else 800. for name in native_dofs]])
    local = torch.zeros(1, 24, 7)
    local[0, names.index("Torso"), 0] = .05
    usd = tmp_path / "declared_test_geometry.usda"
    usd.write_text("mock: geometry provider injected, no SDK")
    root_view = SimpleNamespace(shared_metatype=SimpleNamespace(link_names=names), get_masses=lambda: torch.ones(1, 24),
                                get_coms=lambda: local, get_dof_stiffnesses=lambda: stiff, get_dof_max_forces=lambda: torch.full((1, 69), 500.))
    robot = SimpleNamespace(root_view=root_view, cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(usd))))
    world = SimpleNamespace(device=torch.device("cpu"), mapping=mapping, lower=torch.full((69,), -torch.pi), upper=torch.full((69,), torch.pi),
                            simulator=SimpleNamespace(_robot=robot, _get_sim_body_ordering=lambda: SimpleNamespace(dof_names=native_dofs)))
    monkeypatch.setattr(probe, "load_neutral_collision_geometry", lambda *args: geometry)
    args = probe.parse_args(["--output", str(tmp_path / "out"), "--pose-anchor", "gravity_balanced_neutral",
                             "--standing-placement", "collision_grounded"])
    anchor = probe.build_anchor(world, None, args)
    assert anchor["reset_dof_pos"].count_nonzero() == 0
    assert anchor["dof_target"].count_nonzero() > 0
    assert float(anchor["dof_target"][0, mapping.dof_names.index("Torso_y")]) == pytest.approx(-.05 * 9.81 / 1000, abs=1e-8)
    assert anchor["stance_geometry"]["stability_proven"] is False
    assert anchor["stance_geometry"]["external_support_force_applied"] is False
    for index, name in enumerate(mapping.dof_names):
        if name.rsplit("_", 1)[0] in ("L_Toe", "R_Toe", "L_Hand", "R_Hand"):
            assert float(anchor["dof_target"][0, index]) == 0


def create_manifest(tmp_path):
    package = tmp_path / "package.pt"
    package.write_bytes(b"PK\x03\x04" + b"mock signature only")
    manifest = {"schema": "navisaaclab.amass_reconstruction.v1", "splits": {}}
    for split in ("train", "validation", "test"):
        manifest["splits"][split] = {"path": str(package), "bytes": package.stat().st_size, "sha256": "0" * 64,
                                     "clips": [{"motion_id": 0, "frames": 50, "dt": 1 / 30, "source_path": f"/{split}/Walking.motion"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_source_and_CLI_preflight_are_sdk_torch_free_and_do_not_create_output(tmp_path):
    manifest = create_manifest(tmp_path)
    out = tmp_path / "not_created"
    code = f"""import importlib.util,sys
spec=importlib.util.spec_from_file_location('probe',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert 'torch' not in sys.modules
assert m.main(['--manifest',{str(manifest)!r},'--output',{str(out)!r},'--preflight-only'])==0
assert 'torch' not in sys.modules
assert not any(k.startswith(('omni.','isaaclab','isaacsim','warp')) for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report["sdk_started"] is False
    assert report["torch_imported"] is False
    assert report["author_gait_success_test"] is False
    assert not out.exists()


def test_preflight_rejects_pointer_clip_phase_and_overwrite_without_native_calls(tmp_path):
    manifest = create_manifest(tmp_path)
    args = probe.parse_args(["--manifest", str(manifest), "--output", str(tmp_path / "output")])
    args.output.mkdir()
    (args.output / "existing.json").write_text("{}")
    with pytest.raises(ValueError, match="new empty"):
        probe.preflight(args)
    (args.output / "existing.json").unlink()
    args.anchor_frame = 80
    with pytest.raises(ValueError, match="inside"):
        probe.preflight(args)
    args.anchor_frame = 9
    package = tmp_path / "package.pt"
    package.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    doc = json.loads(manifest.read_text())
    doc["splits"]["validation"]["bytes"] = package.stat().st_size
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="LFS pointer"):
        probe.preflight(args)


@pytest.mark.parametrize("option,value", [("--threshold-n", "nan"), ("--standing-steps", "2"),
                                         ("--wall-penetration-m", ".5"), ("--zero-force-tolerance-n", "2"),
                                         ("--stable-support-fraction", "0")])
def test_bad_probe_parameters_fail_before_simulator_launch(tmp_path, option, value):
    with pytest.raises(SystemExit) as error:
        probe.parse_args(["--output", str(tmp_path), option, value])
    assert error.value.code == 2


def test_new_stance_options_require_explicit_compatible_placement_and_preserve_defaults(tmp_path):
    args = probe.parse_args(["--output", str(tmp_path / "out")])
    assert args.pose_anchor == "neutral"
    assert args.standing_placement == "fixed_height"
    assert args.standing_root_height_m == 1.
    for options in (("--pose-anchor", "gravity_balanced_neutral"),
                    ("--pose-anchor", "walking_frame", "--standing-placement", "collision_grounded"),
                    ("--initial-sole-clearance-m", "-.001"),
                    ("--initial-sole-clearance-m", "nan")):
        with pytest.raises(SystemExit) as error:
            probe.parse_args(["--output", str(tmp_path / "out"), *options])
        assert error.value.code == 2
