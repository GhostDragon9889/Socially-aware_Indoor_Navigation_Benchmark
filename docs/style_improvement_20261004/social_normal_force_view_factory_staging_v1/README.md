# Prepared native net-normal view factory

This is source preparation for the frozen `SharedGroupWorld` and net-normal
sampler. No native SDK, Torch, NN, data, checkpoint, CUDA or physical scene has
been imported or executed for this stage. The three stdlib mock tests now pass in one actual execution; no real native view or force was measured. No existing source/report, production185 or simulation
defaults were changed. This factory is not a physics or training result.

## Interface and caller ownership

Pass `create_normal_force_bundle` directly as the World's
`normal_force_bundle_factory`. Its four positional parameters are
`(sim, scene, assets, group_actor_rows)`, and it returns exactly
`(tuple_of_views, tuple_of_layouts, tuple_of_body_name_tuples)`.
Each layout contains the frozen sampler's `actor` and ordered `sources` keys.
Rows come from the existing canonical actor partition; this factory neither
invents stable actor IDs nor creates scene/group IDs. The accepted structural
row interface exposes int64 `[G,4]`, the simulation device, and `tolist()`;
the frozen World/bridge remains responsible for actual Torch class admission.
The factory checks all integer values and complete partition itself. The one
small ID transfer to host is explicit; no force payload is transferred.

The caller owns its one live App, initialized simulation, four Articulations
and scene. Invoke only after its `sim.reset()` has created native views.
The factory reads `sim.physics_sim_view` and the active Lab manager getter,
using the existing reviewed ownership helper. It creates neither an App nor
a SimulationView; it does not change subspace roots, import SDK/Torch, start,
step, reset, stop, close, seed or alter USD/collision filters. The already loaded
public Lab stage module must have its exact installed source origin. Its
`get_current_stage()` must return the same caller-owned stage as sim/scene.

## Exact sensors and force semantics

Four slot assets each hold `G` native Articulations; the scene holds `G` group
environments. This factory creates **N=4G separate per-actor views**, each with
24 exact body paths. It does not create four `[G×24]` grouped views, and does
not assert that 256 native views have been validated or profiled.

Paths come from actual public `asset.root_view.link_paths[G][24]`; names come
from `asset.body_names`, which reads the same root view's ordered metatype
link names. All native instance paths must match the numerical group and slot
prefix `/World/envs/env_{g}/Robot_{slot}/`, including nested body paths. The
Articulation root may be the Robot prim itself or a descendant. Every link
basename must match the corresponding native body name, and names must form
the exact canonical SMPL24 bijection. Native serialization shape/order is
validated; lexical `env_10` ordering is never assumed or silently remapped.
Every actual prim must already carry `PhysxContactReportAPI`. A spawn flag
alone is insufficient, and this factory never authors the missing schema.
All paths, body names, whole partition and actual reporting schemas are
checked before the first contact view is created.

The native call is
`create_rigid_contact_view(exact24paths, filter_patterns=[], max_contact_data_count=0)`.
The installed API and Lab ContactSensor explicitly support the flat empty
filter list. No peer filter columns or detailed contact-data allocation are
requested. Created views must return `sensor_count=24`, exact ordered
`sensor_paths`, zero filters and valid native objects. Any mismatch is a
failure, not an inferred successful path binding. Earlier view allocations
may remain if a later native call fails; no transaction or rollback is claimed.

The returned wrapper exposes only `sensor_count`, `sensor_paths` and
`get_net_contact_forces(dt)`, delegating the native buffer without converting
it. The frozen sampler immediately detaches/clones and validates native
float32 `[24,3]`, then performs semantic body and canonical actor mapping.
The getter's net force covers **all actual contact partners, including floor**,
independently of filters. This stage uses the Lab normal-force channel, not
friction data or a filtered matrix sum. Actual `dt=1/120 s` converts solver
impulses to Newtons; `1/30 s` is rejected. The World samples once after the
fourth physics substep. The factory does not turn this into an impulse metric,
four-substep maximum, peer-pair quantity or PACER/Gym force equivalence.

The scene must already declare four assets, group collision filtering and
PhysX replication. This declaration does not independently prove cross-group
collision exclusion or real floor/peer contact. The factory changes no filters.

## Reset and invalidation boundary

Each wrapper preserves its original manager-owned SimulationView, four native
Articulation views, stage and creation counter. Before/after each net getter,
it checks the active manager view, validity, current public stage identity,
root-view identity, playing status, counter not below its creation baseline and actual
physics dt. Strict four-substep and per-control clock checks belong to the World. Contact-object `check()` is called before the getter. Native API
`check()` documents a boolean return; `is_valid` forwards a backend validity
flag. The code uses their documented truth predicates without assuming that
the backend returns the singleton Python `True`. `sim.is_playing()` is a
source-declared Python bool flag. No native return type was observed here.

The World's indexed state writes, `scene.reset(group_ids)` and `sim.forward()`
preserve topology and do not ask this factory to rebuild views. Raw net-force
buffers can retain earlier solver results immediately after a pose reset;
they must only be consumed at the later real control end. This factory does
not certify force freshness at reset time or a four-substep sampling clock.
The World owns actual control/counter checks, and the sampler's stamps remain
caller declarations. Native reset validity and synchronization are pending SDK
verification.

On timeline stop, stage/topology change, replacement root views or invalid
contact objects, reads fail transparently. There is no automatic view/World
rebuild or reuse after restart. The caller must discard the old driver and
bindings and deliberately initialize a new world. Keeping references here
does not authorize this factory to close or invalidate caller-owned objects.

## Installed primary source evidence

Paths below are read-only source evidence, not execution receipts. Runtime
dependency admission hashes complete bytes before the helper or names are
loaded. SDK modules themselves are never source-executed by this factory.

- Installed Tensor API: `IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py`,
  382927 B, SHA `b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903`.
  `SimulationView.is_valid`137–146; no-filter creation393–398; native
  `prim_paths`685–689, `link_paths`700–704; Articulation `check()`3215–3230;
  contact sensor paths4891; all-partner net getter4918–4953 and contact
  `check()`5246–5261. These support native path lists and boolean validity
  semantics; real returned list shape/order must still be checked.
- Installed Lab3 Articulation:224858 B, SHA
  `2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87`,
  `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`184–195:
  `body_names` comes from `root_view.shared_metatype.link_names`.
- Installed Lab3 articulation test:131239 B, SHA
  `048013a2cda0c0edeb760b122966673ccb99f515f41d0cf3bee66fe00c65c107`,
  `source/isaaclab_physx/test/assets/test_articulation.py`354–355 actually
  asserts native link-path basename order equals `body_names`. This source
  was read; the upstream test was not run.
- Installed ContactSensor:30148 B, SHA
  `d8a7bb388fd1f6d2aae80a81c357f0c9ebaf6124f1cbd9f9564188860c3c3b37`,
  `source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py`:
  normal world-force declaration46–52; actual ContactReportAPI315–334;
  flat no-filter initialization347–353; getter with actual physics dt486.
- Installed PhysX manager:34974 B, SHA
  `d83ef1b3a375dbbdbd538f7e5f552064ee57528ae7b35b9ba58719fd8a4cfda5`,
  `source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py`317–339:
  sim reset creates views when missing;795–806 manager creates its views;
  813–831 invalidation on stop;835–852 stage-open lifecycle.
- Installed InteractiveScene:39939 B, SHA
  `5d509530070ffb8f4616f769a6608299a386800d7e2582833181ba28dfee32d1`,
  `source/isaaclab/isaaclab/scene/interactive_scene.py`161 exact numerical
  env paths;194–195 declared collision filtering;453–473 reset buffers.
- Installed SimulationContext:42425 B, SHA
  `8c5d73af45f82b0d5c36b05a1c293094319c5d17a75f75a2fe679c382334ea89`,
  `source/isaaclab/isaaclab/sim/simulation_context.py`266–268 manager-owned
  view;625–640 reset/play;772–782 stop and Python playing flag.
- Installed stage helper:20480 B, SHA
  `7443a822773b7d8152757854c75cd8104f34d3fe68bb0c9da1d6e0bb56ba6f89`,
  `source/isaaclab/isaaclab/sim/utils/stage.py`522–552 current public USD stage.
- Existing ownership helper `CrowdSim/humanoid/physics_views.py`:1845 B,
  SHA `1b13d982864a3a2dae6eaebccd15f7ecfdda666c1e1a8dbce4996786a59a0c21`.
  Its complete unchanged stdlib source is loaded, not a legacy manager fallback.
- Existing eight-actor smoke `group_articulation_physics_smoke_staging_v1/smoke_group_physics.py`:
  21236 B, SHA `d000200abf55545d0583f896fcfe48f6ebba673afd78b8820782a6a25b6c063d`,
  read in full. Its168-peer filters/8192 detailed pair-data capacity address
  another measurement; this factory does not copy those settings or execute it.
- Frozen World35097 B / `b608cae6cd8249e2c66d841813b4291b8bd229ca7beae45af625c33daf28fceb`;
  sampler7398 B / `04d21596a8f1fd54d9d9a43977c8a76137840a574fb4f04a73ed4ce94beb5d09`;
  canonical tuple source10438 B / `a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba`.

## Prepared checks and remaining work

The three unexecuted stdlib cases cover exact actor/slot/numeric env10
binding and no-filter delegation; all pre-creation path/schema/partition
admission; native count/order mismatches and stale ownership/stage/contact/dt
rejection without automatic recreation. Their structural rows, stage module
and native views are explicit fakes, not actual SDK/force buffers. Compilation
and Ruff inspect source without importing it. Root review and separate
authorization are required before the single bounded mock invocation.

Actual SDK creation, native force shape/nonzero floor/peer probes, body
reporting readiness across all clones, scene/reset timing, collision filtering,
stop/restart rejection and 64×4 performance remain unverified. A future
exclusive single-App probe must establish these; mock success cannot do so.
No success threshold, promotion gate, policy quality claim or training hook is
introduced by this factory.

## Root actual CPU checks and seal

Root whole source3133f1 and whole-test29681f reads plus independent state_prior_training_prepare review returned Required=[]. Author source-only compile29b9b9/Ruff34fdcd had passed. One root-authorized hidden-CUDA stdlib execution **69328e directly closed0; 3 tests/.012s/OK**. [Actual receipt](actual_CPU_checks.json), factory11810B/517afb01605e and test8397B/711a55f8c1c0 are readonly ordinary single-link files.

Tests use synthetic native views/paths/prim ReportAPI and structural rows, including12 groups/numericalenv10. They cover exact no-filter allocation/delegation, pre-allocation schema/path/partition rejection and stale lifecycle/native mismatch rejection. Teardown asserts no Torch/SDK/NN-bearing package imports. Opaque payloads are not force evidence; neither the current native SDK return types, reset synchronization, collision filtering, actual24×3 normal data nor N256 performance has been executed.
