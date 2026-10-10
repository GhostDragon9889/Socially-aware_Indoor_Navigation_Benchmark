"""Stdlib config-interface mocks; these do not import or exercise IsaacLab."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import group_scene as module


@dataclass
class USD:
    usd_path: str = "reviewed_smpl.usd"
    spawn_path: str | None = "/World/envs/env_0/Robot"


class MultiUSD(USD):
    pass


class Light:
    pass


@dataclass
class Asset:
    prim_path: str = "/World/Ground"
    collision_group: int = -1
    spawn: object = None


@dataclass
class Articulation(Asset):
    prim_path: str = "/World/envs/env_.*/Robot"
    collision_group: int = 0
    spawn: object = field(default_factory=USD)
    init_state: object = field(default_factory=lambda: SimpleNamespace(
        pos=(0., 0., .95), rot=(0., 0., 0., 1.), joint_pos={".*": 0.}, joint_vel={".*": 0.}))
    actuators: dict = field(default_factory=lambda: {"legs": {"stiffness": [800.]}})


@dataclass
class Terrain:
    prim_path: str = "/World/Ground"
    collision_group: int = -1
    num_envs: int = 256
    env_spacing: float = 2.


@dataclass
class Scene:
    num_envs: int = 256
    env_spacing: float = 2.
    replicate_physics: bool = True
    filter_collisions: bool = True


SDK = SimpleNamespace(articulation=Articulation, asset=Asset, scene=Scene,
                      usd=USD, multi_usd=MultiUSD, light=Light, terrain=Terrain)
OFFSETS = ((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5))


class GroupSceneContract(unittest.TestCase):
    def setUp(self):
        self.template = Articulation()
        self.base = Scene()
        self.base.robot = self.template
        self.base.terrain = Terrain()
        self.base.light = Asset(prim_path="/World/Light", collision_group=0, spawn=Light())

    def make(self, **kwargs):
        with patch.object(module, "_sdk_types", return_value=SDK):
            return module.create_group_scene_cfg(self.template, self.base,
                num_actors=kwargs.get("num_actors", 256),
                actor_offsets_xy=kwargs.get("offsets", OFFSETS), env_spacing=kwargs.get("spacing", 8.))

    def test_four_copied_slots_and_global_configs(self):
        before = deepcopy(self.template)
        cfg = self.make()
        self.assertEqual((cfg.num_envs, cfg.terrain.num_envs, cfg.env_spacing, cfg.terrain.env_spacing), (64, 64, 8., 8.))
        self.assertTrue(cfg.filter_collisions and cfg.replicate_physics)
        self.assertFalse(hasattr(cfg, "robot"))
        self.assertEqual(cfg.light.collision_group, 0)  # A light is not a collider.
        self.assertEqual(cfg.terrain.collision_group, -1)
        actors = [getattr(cfg, name) for name in module.ACTOR_NAMES]
        self.assertEqual(tuple(a.prim_path for a in actors), module.ACTOR_PRIM_PATHS)
        self.assertEqual(tuple(a.init_state.pos for a in actors), tuple((*xy, .95) for xy in OFFSETS))
        for actor in actors:
            self.assertEqual(actor.collision_group, 0)
            self.assertIsNone(actor.spawn.spawn_path)
            self.assertIsNot(actor.spawn, self.template.spawn)
            self.assertIsNot(actor.actuators["legs"]["stiffness"], self.template.actuators["legs"]["stiffness"])
        actors[0].actuators["legs"]["stiffness"][0] = 1.
        actors[0].init_state.joint_pos[".*"] = 2.
        actors[0].spawn.spawn_path = "cloner_mutation"
        self.assertEqual(actors[1].actuators["legs"]["stiffness"], [800.])
        self.assertEqual(self.template, before)
        self.assertEqual(self.base.terrain.num_envs, 256)

    def test_reject_unsupported_bindings_and_bad_layout(self):
        for kwargs in ({"num_actors": 6}, {"num_actors": True}, {"offsets": ((0., 0.),) * 4},
                       {"spacing": 2.}, {"spacing": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)
        self.template.spawn = MultiUSD()
        with self.assertRaises(TypeError):
            self.make()
        self.template.spawn = USD()
        self.base.sensor = SimpleNamespace(prim_path="/World/envs/env_.*/Robot/Head")
        with self.assertRaises(TypeError):
            self.make()
        del self.base.sensor
        self.base.other_robot = Articulation()
        with self.assertRaises(TypeError):
            self.make()
        del self.base.other_robot
        self.base.ground = Asset(collision_group=0)
        with self.assertRaises(ValueError):
            self.make()


if __name__ == "__main__":
    unittest.main()
