"""Four local SMPL articulations per IsaacLab env; configuration factory only.

Importing this module uses stdlib only. Calling the factory imports SDK config
types and creates configs, so that call belongs after the future SDK launch.
No scene, simulation context, stage, asset, tensor or controller is constructed.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from types import SimpleNamespace


ACTOR_NAMES = tuple(f"robot_{slot}" for slot in range(4))
ACTOR_PRIM_PATHS = tuple(f"/World/envs/env_.*/Robot_{slot}" for slot in range(4))


def _sdk_types():
    """Resolve only config classes lazily, never instantiate a scene."""
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.sim.spawners.lights.lights_cfg import LightCfg
    from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiUsdFileCfg
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

    return SimpleNamespace(
        articulation=ArticulationCfg, asset=AssetBaseCfg, scene=InteractiveSceneCfg,
        usd=UsdFileCfg, multi_usd=MultiUsdFileCfg, light=LightCfg,
        terrain=TerrainImporterCfg,
    )


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def _offsets(values, initial_pos, spacing):
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("actor_offsets_xy must contain exactly four XY pairs")
    offsets = []
    for row in values:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("each actor offset must be an XY pair")
        offsets.append(tuple(_number(value, "actor offset") for value in row))
    # Conservative coordinate margins, not a measured SMPL body AABB guarantee.
    if any(math.dist(a, b) < 2.0 for i, a in enumerate(offsets) for b in offsets[i + 1:]):
        raise ValueError("initial actor root centers must be at least 2 metres apart")
    positions = tuple((initial_pos[0] + x, initial_pos[1] + y, initial_pos[2]) for x, y in offsets)
    if any(abs(value) + 1.0 > spacing / 2 for position in positions for value in position[:2]):
        raise ValueError("actor root centers require a 1 metre margin inside the env spacing cell")
    return positions


def _global_copy(entity, sdk, name, groups, spacing):
    if not isinstance(entity, (sdk.asset, sdk.terrain)) or isinstance(entity, sdk.articulation):
        raise TypeError(f"unsupported base entity {name}: sensors/extra articulations need explicit wiring")
    path = entity.prim_path
    if (not isinstance(path, str) or not path.startswith("/World/")
            or path.startswith("/World/envs") or "{" in path or ".*" in path):
        raise ValueError(f"base entity {name} must have a constant global /World path")
    # Lights have no collider. Ground/terrain/other allowed global assets must
    # explicitly join IsaacLab's global collision group, including USD colliders.
    is_light = isinstance(entity, sdk.asset) and isinstance(entity.spawn, sdk.light)
    if not is_light and (type(entity.collision_group) is not int or entity.collision_group != -1):
        raise ValueError(f"global non-light entity {name} must use collision_group=-1")
    copied = deepcopy(entity)
    if isinstance(copied, sdk.terrain):
        copied.num_envs = groups
        copied.env_spacing = spacing
    return copied


def create_group_scene_cfg(single_smpl_template, base_scene_cfg, *, num_actors,
                           actor_offsets_xy, env_spacing):
    """Return a new ``InteractiveSceneCfg`` with four named articulations per env.

    ``num_actors`` is the flat policy actor count N; the scene has G=N/4 envs.
    The explicit offsets are added to the reviewed template's initial XY only.
    Base entities may contain the same ``robot`` template (removed), global
    terrain/ground/light configs and None. Extra actors/sensors/local entities
    are rejected, rather than silently retaining old single-robot bindings.
    All config objects are copied independently; inputs remain caller-owned.
    """
    if type(num_actors) is not int or num_actors < 4 or num_actors % 4:
        raise ValueError("num_actors must be a positive multiple of four")
    spacing = _number(env_spacing, "env_spacing")
    if spacing <= 0:
        raise ValueError("env_spacing must be positive")
    sdk = _sdk_types()
    if not isinstance(single_smpl_template, sdk.articulation):
        raise TypeError("single_smpl_template must be an ArticulationCfg")
    if not isinstance(base_scene_cfg, sdk.scene):
        raise TypeError("base_scene_cfg must be an InteractiveSceneCfg")
    spawn = single_smpl_template.spawn
    if not isinstance(spawn, sdk.usd) or isinstance(spawn, sdk.multi_usd):
        raise TypeError("the reviewed articulation must use a single UsdFileCfg, not URDF/MultiUsd")
    if not isinstance(spawn.usd_path, str) or not spawn.usd_path:
        raise ValueError("the articulation must name one USD file")
    if type(single_smpl_template.collision_group) is not int or single_smpl_template.collision_group != 0:
        raise ValueError("the articulation must use local collision_group=0")
    if not isinstance(single_smpl_template.actuators, dict) or not single_smpl_template.actuators:
        raise ValueError("the reviewed articulation must have explicit actuators")
    pos = single_smpl_template.init_state.pos
    if not isinstance(pos, (tuple, list)) or len(pos) != 3:
        raise ValueError("template init_state.pos must contain XYZ")
    initial_pos = tuple(_number(value, "template initial position") for value in pos)
    positions = _offsets(actor_offsets_xy, initial_pos, spacing)
    groups = num_actors // 4
    fields = sdk.scene.__dataclass_fields__
    options = {name: deepcopy(getattr(base_scene_cfg, name)) for name in fields}
    options.update(num_envs=groups, env_spacing=spacing, filter_collisions=True, replicate_physics=True)
    cfg = sdk.scene(**options)
    terrain_count = 0
    for name, entity in vars(base_scene_cfg).items():
        if name in fields or entity is None:
            continue
        if name == "robot" and entity is single_smpl_template:
            continue
        if name in ACTOR_NAMES:
            raise ValueError(f"base entity name {name} conflicts with a group actor slot")
        copied = _global_copy(entity, sdk, name, groups, spacing)
        terrain_count += isinstance(copied, sdk.terrain)
        if terrain_count > 1:
            raise ValueError("only one terrain importer is supported")
        setattr(cfg, name, copied)
    for name, path, position in zip(ACTOR_NAMES, ACTOR_PRIM_PATHS, positions, strict=True):
        actor = deepcopy(single_smpl_template)
        actor.prim_path = path
        actor.spawn.spawn_path = None  # Actual IsaacLab 3 field; assigned by make_clone_plan.
        actor.init_state.pos = position
        setattr(cfg, name, actor)
    return cfg
