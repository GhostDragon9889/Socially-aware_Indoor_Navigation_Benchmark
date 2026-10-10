# Four SMPL articulations per cloned environment — independent config stage

`group_scene.py` is a lazy config factory, not an installed world or simulator.
Its module imports only stdlib. The factory resolves IsaacLab config classes
when called; that call is pending a future SDK launch. It never constructs an
InteractiveScene, simulation context, asset, stage, tensor, controller or App.
Current production185, robot assets, formal training and defaults are unchanged.

Root verified source88e5a8d2/test02b01f48/CPUrecordc16c1213 after full reads and sealed all three readonly in actual tool`f55b13`. The original CPU record preserves its mock-only scope; no SDK or physics evidence is added by this freeze.

## Interface and geometry

```python
cfg = create_group_scene_cfg(
    single_smpl_template, base_scene_cfg,
    num_actors=256,
    actor_offsets_xy=((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)),
    env_spacing=8.0,
)
```

The input is a reviewed `ArticulationCfg` with one `UsdFileCfg`, explicit
actuators, local `collision_group=0`, and a reviewed base `InteractiveSceneCfg`.
URDF/MJCF spawners and `MultiUsdFileCfg` are rejected. ProtoMotions already
converts its SMPL MJCF into one USD before creating that articulation template;
this factory does not convert assets or guess an articulation root/body layout.

For flat policy actor count N (positive, divisible by4), the scene has G=N/4
environments. N256 means64 environment origins and four articulation instances
per origin, **not**256 independently filtered environments. Asset names are
`robot_0`, `robot_1`, `robot_2`, `robot_3`; their prim expressions are
`/World/envs/env_.*/Robot_0` through `Robot_3`. No `robot` alias is added.

Slot order matches the separate mixed layout contract:
`group_actor_rows[g] = [2*g, 2*g+1, N/2+2*g, N/2+2*g+1]`.
Each asset exposes G rows; the external bridge maps these to the original
canonical N rows. Scene IDs and stable actor IDs are not invented in the config.

Only XY offsets are added to the template's initial position. Initial Z, XYZW
quaternion, root/joint velocities, joints, actuator groups, stiffness/damping,
effort limits, spawn properties, articulation-root relative path and body/joint
ordering remain copied. The root centers must be at least2m apart and stay1m
inside each spacing cell. These are conservative **coordinate** checks; they
are not SMPL body AABB, upright-state, terrain feasibility or no-contact proof.
The factory does not sample a spawn, run collision queries or guarantee future
reset poses. Caller must choose reviewed positions on its terrain.

## Copying, entities and collision filtering

The returned base config is newly constructed, and all inherited config fields,
global entities and each complete articulation template are independently
deep-copied. All four spawners, init-state dictionaries and actuator dictionaries
are separate. The original input config/template remain owned by the caller.
`spawn.spawn_path` is cleared on each copied articulation: that is the installed
IsaacLab3 single-USD temporary field assigned by its clone planner. There is no
single-USD `spawn_paths` field; `MultiUsdFileCfg`, which has that field and is a
subclass of `UsdFileCfg`, is explicitly rejected.

The original base `robot` is omitted only if it is the exact passed template
object. Other articulations, sensors, non-global paths and unknown entities are
rejected; old single-robot sensor bindings cannot silently survive. Supported
base entities are global `AssetBaseCfg` and `TerrainImporterCfg` (including
subclasses such as ProtoMotions' mesh importer), or None. Only one terrain
importer is supported. Its real `num_envs` and `env_spacing` fields are copied
and synchronized to G and the chosen spacing. Global light configs may retain
collision_group0 because `LightCfg` has no collider. Every other allowed global
entity must explicitly use collision_group−1; ground/terrain then enter the
SDK's global collision paths instead of becoming isolated from the actors.

The output enables `filter_collisions=True` and `replicate_physics=True`.
All four actors remain local collision_group0. Installed PhysX filtering groups
the complete `/World/envs/env_g` subtree: four actors within one env remain
eligible to collide, other envs are filtered, global ground stays eligible.
This does not override a reviewed USD's self-collision settings or other
pre-existing USD collision relationships. No filtering function is called here.
Actual PhysX collision behavior still requires future SDK execution.

## Installed primary source supporting this path

Root: `/project/EXP/SangforAndSaut/IsaacLab-Protomotions`, `VERSION` =3.0.0.

- `source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:44–48` contains the
  author's two-articulation example; `:109–132` documents clone replication and
  collision filtering. Multiple named articulation cfgs are supported directly.
- `scene/interactive_scene.py:164–195` creates env Xforms, then uses the current
  **queued ReplicateSession** over `_collect_asset_cfgs`; it does not use the old
  `clone_environments` API. `:215` resolves/mutates paths, and `:766–774` requires
  assigned prototype spawn paths. `:783–784` registers each articulation by name.
- `cloner/cloner_utils.py:285–297` plans all single-variant assets together:
  each spawns at `env_0/Robot_slot`, followed by cloning the complete env root.
  Four independently copied single-USD cfgs therefore need no manual spawn loop.
- `assets/asset_base_cfg.py:86–93` defines collision0 as env-local and−1 as
  global. `scene/interactive_scene.py:865–867` gathers global prim paths.
  `cloner/cloner_utils.py:345–449` creates one inverted collision group per env,
  permits that group with itself and the global group, and excludes other groups.
  This filtering implementation is PhysX-specific; no Newton claim is made.
- `assets/asset_base.py:93` spawns using init-state position; the PhysX
  articulation `_process_cfg` (`isaaclab_physx/assets/articulation/articulation.py:3937`)
  stores the template pose. Future state writes must explicitly combine each
  group origin with local slot/reset positions, without adding origins twice.
- Current `protomotions/simulator/isaaclab/utils/scene.py:156–203` builds the
  reviewed single SMPL USD articulation, root path, actuators and spawn settings.
  Its mesh terrain uses collision_group−1 (`:244`). Those inputs are preserved.

## Integration that remains absent

Current `protomotions/simulator/isaaclab/simulator.py:145` hard-selects
`scene["robot"]`; PD writes (`:661`), state writes (`:690–694`) and robot state
reads are all single-articulation operations. This factory cannot be dropped
into that simulator without the separate multi-asset bridge and caller wiring.
Grouped resets must convert canonical actor IDs to whole group env IDs, then
write all four assets; group-induced truncated masks alone do not do that.

Current AMPWorld builds terrain/SceneLib and origins for N actors, and its
native contact paths still use `/env_actor/Robot`. Future world wiring must
change physical origins to G, map canonical actors through group/slot IDs,
adapt contact paths/body mapping and decide terrain/spawn/reset integration.
It must also update curriculum, episode/history state and social observations.
None of physics initialization, contact sensor views, body/DOF canonicalization,
real inter-person contacts, trajectory planning, training or shared reset has
been executed or installed by this stage. This is a new independent layout.

## Source checks

Compile-only tool `e54f32` exited0 for source and tests without executing them.
Ruff `39c2e8` exited0, all checks passed. Root whole-read (`19650f`/`d7e359`) and
independent source peer found no Required defect. After root authorization, the
two stdlib mock tests ran once with SDK Python, hidden CUDA and bytecode writes
disabled: actual tool `d54009` exited0 without a yielded session; unittest
reported2 passed in0.001s. `actual_CPU_checks.json` binds the source/test bytes
and command. These are copies/slot-config/rejection checks with mocked cfg
types, not actual SDK construction or physics proof (`mocknotSDKphysics`).
