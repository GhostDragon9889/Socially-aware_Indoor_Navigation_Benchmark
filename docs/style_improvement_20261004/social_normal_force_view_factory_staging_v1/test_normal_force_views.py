"""Three prepared stdlib fake-native contract cases, not SDK/force evidence.

These cases have not been executed. Structural group rows and the already
loaded public stage module are mocked explicitly; no Torch/SDK is imported.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("normal_force_views.py")
data = SOURCE.read_bytes()
if len(data) != 11810 or hashlib.sha256(data).hexdigest() != "517afb01605e5ce90a417f14c663c233d443630e4cec29f54d6169e94358132b":
    raise RuntimeError("Reviewed factory whole-byte source changed")
import normal_force_views as factory  # noqa: E402


class Rows:
    dtype, device = "torch.int64", "cpu"  # Structural seam, not a Torch Tensor.

    def __init__(self, groups):
        self.shape = (groups, 4)
        self.values = [[2*g, 2*g+1, 2*groups+2*g, 2*groups+2*g+1] for g in range(groups)]

    def tolist(self):
        return [row.copy() for row in self.values]


class Stage:
    bad_path = None

    def GetPrimAtPath(self, path):
        return SimpleNamespace(IsValid=lambda: True,
            GetAppliedSchemas=lambda: [] if path == self.bad_path else ["PhysxContactReportAPI"])


class Native:
    filter_count, filter_paths = 0, ()

    def __init__(self, paths):
        self.sensor_count, self.sensor_paths = 24, paths.copy()
        self.valid, self.gets = True, []
        self.payload = object()  # Opaque native buffer, not asserted to be force data.

    def check(self):
        return self.valid

    def get_net_contact_forces(self, *, dt):
        self.gets.append(dt)
        return self.payload


class PhysicsView:
    is_valid = True

    def __init__(self):
        self.creations, self.bad_native = [], None

    def create_rigid_contact_view(self, paths, *, filter_patterns, max_contact_data_count):
        self.creations.append((paths.copy(), filter_patterns.copy(), max_contact_data_count))
        result = Native(paths)
        if self.bad_native == "count":
            result.sensor_count = 23
        elif self.bad_native == "order":
            result.sensor_paths.reverse()
        return result


def fixture(groups=2):
    names, _ = factory._dependencies()  # Byte/AST/std-lib helper only, no SDK/Torch.
    stage, view, rows = Stage(), PhysicsView(), Rows(groups)
    sim = SimpleNamespace(device="cpu", stage=stage, physics_sim_view=view,
        get_physics_step_count=lambda: 137, get_physics_dt=lambda: 1/120, is_playing=lambda: True)
    sim.physics_manager = SimpleNamespace(get_physics_sim_view=lambda: sim.physics_sim_view)
    assets = []
    for slot in range(4):
        ordered = names[slot:] + names[:slot]
        paths = [[f"/World/envs/env_{g}/Robot_{slot}/Nested/{name}" for name in ordered] for g in range(groups)]
        root = SimpleNamespace(link_paths=paths, prim_paths=[f"/World/envs/env_{g}/Robot_{slot}" for g in range(groups)], check=lambda: True)
        assets.append(SimpleNamespace(root_view=root, body_names=ordered, num_instances=groups, device="cpu"))
    scene = SimpleNamespace(stage=stage, env_prim_paths=[f"/World/envs/env_{g}" for g in range(groups)],
        cfg=SimpleNamespace(num_envs=groups, filter_collisions=True, replicate_physics=True), physics_backend="physx",
        articulations={f"robot_{slot}": asset for slot, asset in enumerate(assets)})
    current = SimpleNamespace(stage=stage)
    stage_module = SimpleNamespace(__file__=str(factory.PINS["stage"][0]), get_current_stage=lambda: current.stage)
    return sim, scene, assets, rows, current, stage_module


class NormalViewFactoryStdlib(unittest.TestCase):
    def tearDown(self):
        forbidden = ("torch", "warp", "isaaclab", "isaaclab_physx", "isaacsim", "omni", "pxr", "mujoco", "CrowdSim.humanoid.amp", "CrowdSim.humanoid.training")
        self.assertFalse(any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden))

    def test_exact_per_actor_paths_no_filters_numeric_env10_and_delegation(self):
        sim, scene, assets, rows, _, stage_module = fixture(12)
        before = rows.tolist()
        with patch.dict(sys.modules, {"isaaclab.sim.utils.stage": stage_module}):
            views, layouts, names = factory.create_normal_force_bundle(sim, scene, assets, rows)
            self.assertEqual(len(views), 48)
            self.assertEqual(len(sim.physics_sim_view.creations), 48)
            for group, row in enumerate(rows.values):
                for slot, actor in enumerate(row):
                    self.assertEqual(layouts[actor], {"actor": actor, "sources": assets[slot].root_view.link_paths[group]})
                    self.assertEqual(names[actor], assets[slot].body_names)
                    self.assertEqual(sim.physics_sim_view.creations[actor], (layouts[actor]["sources"], [], 0))
            self.assertIn("/env_10/", layouts[rows.values[10][0]]["sources"][0])
            self.assertIs(views[0].get_net_contact_forces(1/120), views[0].native.payload)
            self.assertEqual(views[0].native.gets, [1/120])
        self.assertEqual(rows.tolist(), before)
        self.assertEqual(sim.get_physics_step_count(), 137)

    def test_all_path_schema_partition_checks_precede_native_creation(self):
        for defect in ("report", "group_order", "partition"):
            with self.subTest(defect=defect):
                sim, scene, assets, rows, _, stage_module = fixture()
                if defect == "report":
                    scene.stage.bad_path = assets[-1].root_view.link_paths[-1][-1]
                elif defect == "group_order":
                    assets[-1].root_view.link_paths.reverse()  # Native lexical/order assumption must fail.
                else:
                    rows.values[-1][-1] = rows.values[0][0]
                with patch.dict(sys.modules, {"isaaclab.sim.utils.stage": stage_module}), self.assertRaises(ValueError):
                    factory.create_normal_force_bundle(sim, scene, assets, rows)
                self.assertEqual(sim.physics_sim_view.creations, [])

    def test_native_mismatch_and_stale_lifecycle_reject_without_rebuild(self):
        for mismatch in ("count", "order"):
            with self.subTest(mismatch=mismatch):
                sim, scene, assets, rows, _, stage_module = fixture()
                sim.physics_sim_view.bad_native = mismatch
                with patch.dict(sys.modules, {"isaaclab.sim.utils.stage": stage_module}), self.assertRaises(RuntimeError):
                    factory.create_normal_force_bundle(sim, scene, assets, rows)
                self.assertEqual(len(sim.physics_sim_view.creations), 1)  # Earlier allocation is not rolled back.
        for defect in ("dt", "contact_invalid", "stage", "simulation_invalid", "root_replaced", "owner_replaced", "stop"):
            with self.subTest(lifecycle=defect):
                sim, scene, assets, rows, current, stage_module = fixture()
                with patch.dict(sys.modules, {"isaaclab.sim.utils.stage": stage_module}):
                    views, _, _ = factory.create_normal_force_bundle(sim, scene, assets, rows)
                    if defect == "dt":
                        with self.assertRaises(ValueError):
                            views[0].get_net_contact_forces(1/30)
                    else:
                        if defect == "contact_invalid":
                            views[0].native.valid = False
                        elif defect == "stage":
                            current.stage = Stage()
                        elif defect == "simulation_invalid":
                            sim.physics_sim_view.is_valid = False
                        elif defect == "root_replaced":
                            assets[0].root_view = object()
                        elif defect == "owner_replaced":
                            sim.physics_sim_view = PhysicsView()
                        else:
                            sim.is_playing = lambda: False
                        with self.assertRaises(RuntimeError):
                            views[0].get_net_contact_forces(1/120)
                    self.assertEqual(views[0].native.gets, [])
                    self.assertEqual(len(views[0].binding.view.creations), 8)
                    self.assertEqual(sim.get_physics_step_count(), 137)


if __name__ == "__main__":
    unittest.main()
