# Raw social observations for existing evaluation worlds

**Source only.** `observation_producer.py` defines a callable producer; no import or execution of it, Torch, NN, SDK, data, weights or tests has occurred. Production185 and all sealed training/evaluation stages remain unchanged. Source compilation and Ruff are the only executed checks. Root whole-source reads aa6680/77ffd6 and final Required delta/README read3b8fec are complete. Independent paper peer reported Required=[] / Optional=[] after the two fixes. Source and README are now ready for readonly sealing.

## Public interface

```python
producer = RawSocialEvaluationObservations.from_world(
    world, actor_ids, scene_ids,
    terrain_sampler=explicit_static_callback,
    terrain_sampler_semantics="caller description of absolute static policy-height feature",
    grid_center="root", grid_heading="root", height_reference="static_ground",
    include_ego=False,
)
producer.require_policy_contract(frozen_social_actor.describe())
producer.begin_episode()                # AFTER the world's actual initial reset
packet = producer.observe(trajectory_world_xy)  # float32[N,10,2], world metres

# Original lifecycle, no grouped-training replacement:
world.reset(reset_rows)
producer.begin_episode(reset_rows)      # AFTER every actual full or partial reset
```

The optional `policy_description=` keyword in `from_world` performs the same explicit binding. `require_policy_contract` accepts exactly the two reviewed descriptor schemas `local.neutral_smpl.frozen_social_actor_snapshot.v1` and `local.neutral_smpl.frozen_social_checkpoint_inference.v1`, with self357/task12308 and exact grid/include-ego choices; no default model/config is inferred. This is an observation contract check, not proof of weights, training quality or checkpoint admission. A producer cannot change its bound contract.

Supported worlds are the actual original `AMPWorld`, **N1 and shared_world=False** for22-motion tracking, or **N4 and shared_world=True** for fixed four-person navigation. Four stable unique nonnegative actor IDs must refer to all four current rows; scene IDs explicitly name the same scene. A one-person world has no additional invented peers. This stage rejects the new b608 grouped-training World and does not change old per-actor reset, truncation, termination or routing behavior.

This first adapter deliberately supports only root-centered/root-heading/static-ground and include_ego=False. API names `static_ground` and `static_center_ground_z` map to the same canonical training declaration `height_reference="static_center_ground_z"`. Other/head/root-Z experiments need explicit compatible training and producer changes; matching dimensions alone is insufficient. The supplied actor snapshot must describe these exact choices.

## One state, phase and ownership

The original `AMPWorld.state()` calls `simulator.get_robot_state()`, whose getter merges COMMON-converted body/DOF/contact/effort fields. `RobotState` inherits a source-verified clone that copies each materialized Tensor field; the producer calls **one** state getter and then this clone. It checks the actual RobotState/COMMON identity and uses the original named PACER mapping and self357 function. No pelvis quaternion, rotation metric, heading equation or original self observation is rewritten.

`observe` returns `SimpleNamespace(state, raw_self, raw_task, owners, diagnostics)`. `state` is the full owned COMMON snapshot for the caller's same-phase task/self/action path. The raw tensors have shapes `[N,357]` and `[N,12308]`; owners are `[N,64,64]` stable IDs or−1. The frozen6975 task assembly and234e raster handle the20 future XY prefix (+0.5…+5s, local metres, no×10), HWC three-channel grid, root-relative velocities, ownership, clip/scaling and static height plus1.7m occupant addition. These source conventions are local neutral-SMPL/PACER adaptations, not proven private Nav equivalence.

The phase is **current COMMON snapshot before the caller action**, not the previous or next physics pose. An initial no-action frame may establish defined heading with action_valid=False; a caller pairs later pre-action observations with any separately captured post-physics body state using explicit phase labels. Do not relabel the original post-step root_rot archive as this observation quaternion. `world.age`, `episode_ids` and the actual SDK physics-step count are read before/after queries and must remain unchanged. The counter is monotonic across calls. No invented global control counter is reported: original AMPWorld owns age while its existing driver owns the global control count.

## Terrain and fallen heading

The explicit terrain sampler receives world XY `[N,64,64,2]` and must return finite absolute static/policy-feature Z there. The task helper checks query immutability, shape and device. Height reference is the original world's `ground_height_at` at each current COMMON Pelvis/root XY, not rootZ, a nine-point old1044 mean, an implicit floor, or an added1.8m band. For a flat unobstructed tracking world, explicitly passing `world.height_at` uses the existing static ground feature. For reconstructed navigation, `world.height_at` includes static cuboid tops. Native provider `height_at` may itself use a floor+1.8m capped occupancy policy feature; callers must declare that proxy and cannot call it a measured full surface. The new producer adds no such cap, outside-map clipping or implicit height callback. Terrain semantics are a caller declaration, not independently measured geometry proof.

Physical fall is computed from this current COMMON Pelvis snapshot and the original ground query: height<0.45m **or** quaternion upZ<0.1, exactly the old AMPWorld physical-fall predicate. It is not timeout, clip end, contact flag or any broadened grouped-training termination. All current Pelvis XYZW quaternions must first pass float32/finite/shape and unit-norm tolerance1e-3, including singular fallen rows; no quaternion is renormalized. Projected-X heading is defined only when XY direction norm>1e-12. A singular task heading can continue a previous actually defined heading **only** for this physically fallen actor in the same notified episode. Healthy singulars and first-frame singulars without a prior reject. Actual unit XYZW remains unchanged and is returned; self357 retains its own original heading behavior.

`begin_episode(rows)` requires age0 and positive actual episode IDs. It clears only the notified rows, captures actual IDs, requires a strictly newer ID for already-active rows and rejects any unnotified changed active row. `observe` requires all rows active, exact episode-ID equality and nondecreasing age. It never infers a reset merely from age0, auto-clears caches or calls reset. Exceptions fault the producer without pretending physical rollback or resume.

## Numeric diagnostics and source boundaries

| Field | Type/shape |
| --- | --- |
| physics_step | Actual nonnegative Python integer |
| actor_ids, scene_ids, age, episode_ids | int64[N] on the physical device |
| actual_pelvis_quat_xyzw | float32[N,4] |
| actual_ground_height_world_z, actual_pelvis_height_above_ground_m, actual_up_z, used_task_heading | float32[N] |
| physical_fall, task_heading_defined, singular_heading_fallback | bool[N] |
| metadata | Small serializable `describe()` declaration, sources, grid, sampler semantics and phase |

No Tensor stringification or full-array JSON is used. `describe()` contains source/current-world metadata, not model or state tensors. Exact whole-byte pins cover original AMPWorld, observation/mapping, RobotState clone, COMMON simulator getter, rotation functions, frozen task and raster. Actual preconstructed class/function module origins are checked; the adapter does not import/construct AMPWorld or a new SDK. Its initial execution source record is locked, and dependency identities are checked before/after observation. Checks that physics/age/episode did not advance do not independently prove that an arbitrary external callback could not mutate other simulator fields; the callback is explicitly required to be a static query.

No seed, policy, normalizer, App, stepping, routing, physics reset, training or checkpoint loading is introduced. Real SDK phase, data/numerical parity and complete22/N4 evaluation remain future runtime work after the scheduled serial runtime gates. No tests or receipts are added here.
