# Source-bound native DynBench navigation companion

Status: **CPU artifact admission and navigation contracts verified; new scene physics/navigation execution not verified by this report.** The parent-owned world/driver hook is a separate implementation and experiment. No App, Physics world, policy checkpoint, existing control loop, or physical geometry is created by this companion.

## Interfaces

```python
from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene

provider = DynBenchSceneProvider.from_preflight(
    preflight_json,
    expected_sha256=preflight_sha,
    probe_script_snapshot=verified_v5_script_copy,  # Optional explicit historical epoch.
)
provider.admit_native_map(native_result_json, expected_sha256=native_result_sha)
scene = NativeGridScene.from_provider(
    provider,
    origins_xy=explicit_actor_origins,
    routes_xy=explicit_actor_routes,  # [B,V>=2,2], all conservative known-free.
    clearance_m=0.35,
    arrival_radius_m=0.75,
    goal_lookahead_m=5.0,
    device="cpu",  # Explicit device string or already imported torch.device.
)

scene.origins  # FP32 [B,2], owned tensor copy.
scene.routes   # FP32 [B,V,2], owned tensor copy.
scene.bounds   # FP32 [xmin,ymin,xmax,ymax], owned tensor copy.
scene.world_to_pixel(world_xy)  # NumPy ROS row/column, floor, no clipping.
scene.pixel_to_world(integer_pixels)  # Source-world cell centers.
scene.nearest_spawn(world_hint, max_distance_m=5.0)  # Explicit bounded query, not a reset.
scene.astar_path(start_world_xy, goal_world_xy)  # NumPy path or None; no fallback.

new_indices, reached = scene.advance_route(actual_nonterminal_poststep_xy, route_indices, actor_ids=ids)
goal, indices, reached = scene.route_goal(xy, route_indices, actor_ids=ids, advance=False)
waypoints, indices, reached = scene.route_waypoints(xy, route_indices, actor_ids=ids, advance=False)
# route_waypoints defaults to .5,1,...,5s, using lookahead*time/5 arclength.

scene.raster(world_states_B5)  # Source-bound TRACE map [B,2,224,224].
scene.collision_cost(world_xy_BST2, radius=0.5)  # Differentiable [B,S] cost, Torch imported lazily.
scene.describe()
```

Route calls return tensors on the declared device and preserve subset actor order. `advance_route` accepts actual nonterminal post-control positions only; the caller must reset route indices after a physical reset and must never count reset positions as progress. Arrival requires the distance threshold **and** a conservative direct free connector, preventing apparent arrival across a thin wall. The companion itself has no physical reset or accumulated success counter.

The cost callback must be called within the future planner's clean-output autograd graph on **world-frame** predictions. It averages the soft clearance deficit `max(radius-signed_distance,0)` over trajectory time, returning independent per-candidate costs `[B,S]`; summing candidates retains batch-independent gradients. It creates neither boxes nor decoded poses. The parent's separately owned Diffusion callback hook is outside this source archive.

## Admission and coordinate contracts

Only a concrete admitted `DynBenchSceneProvider` is accepted. Preflight, CPU projection, limited-height/floor sampling, stale artifact bytes, and the historical v4 extra-floor result cannot admit a navigation map.

The provider now requires v5 source-side inventory of every enabled collider type and the full union of original selected floors plus all near-ground flat Plane/Mesh paths to be session-disabled. Its public `floor_roots` is the one finite reference floor; `static_collision_paths` excludes **all** disabled floors/Plane/decals. Native floor admission retains every ray hit, verifies CSR offsets/counts, geometry, exact paths/classes and row masks, and rejects foreign paths or any undeclared upward near-ground support even when the nearest selected-floor hit is valid. Origin all-hit rays and actual enabled flat-ground inventory must agree on the single owner.

Full native known5 rows require exact floor support and explicit overlap boxes over `.15..1.8 m` above the selected floor. CSR overlap paths, classifications, all row masks/counts and unchanged observed counter/timeline are checked. NativeGridScene additionally reconstructs fixed SDK raw4/5/6 labels from the original buffer/world-position artifacts and compares canonical rows, images and export metadata. No layout is fitted dynamically. Native raw6 unknown is never promoted to known free. CPU projection unknown remains forbidden, while an actual native obstacle may retain occupied0 rather than being erased to unknown205.

The pinned official native buffer is row-major X fastest, row0=minY and column0=maxX. Its ROS export is row0=maxY and column0=minX. Source negative XY coordinates remain unchanged; map origins are lower-left cell corners, conversions use floor and cell centers, and the upper map boundary is excluded. Resolution must be finite and exactly `.1 m`.

Navigation forbids occupied0, unknown205 and all outside cells. EDT is padded with a blocked exterior. Planning subtracts **two cell half-diagonals** from center distance before applying `clearance_m`, covering the blocked-cell volume and every point of a candidate cell conservatively. A* is 8-connected with no diagonal corner cutting, no smoothing and no blocked-start recovery. A failed route query holds the actual finite root and increments `global_path_failures`; this is an explicit failure, not a movement or navigation success.

SDF sampling is a **soft grid proxy**, bilinear within the source grid plus an analytic signed map-domain boundary. Unknown/outside are forbidden; coordinates are never clipped. It does not certify full-body swept volume, exact contact clearance, or contact-point geometry. The measured occupancy is never presented as invented PhysX cuboids.

The immutable SDF table is sampled by four-corner indexed constants and bilinear world-coordinate weights. This avoids `grid_sample`'s CUDA backward, which strict deterministic Torch rejects. Independent CPU comparisons against the former real `grid_sample` operation cover FP32/FP64 values and world-coordinate Jacobians, including border and exterior points (tolerances `2e-6`/`1e-12`). The floor/domain formula and border-padding semantics are retained; the replacement is not claimed byte-identical to `grid_sample`, and its actual CUDA execution remains a separate parent-owned gate. No global deterministic setting is disabled.

## Single stage/floor and height features

Scene injection still requires the explicit pre-SimulationContext scene-query hook and one shared collision configuration. The provider spawner uses the pinned public `isaaclab.sim.get_current_stage()` getter, honoring the SDK thread-local `use_stage(active_stage)` context; it has no global `omni.usd` fallback. Only the native XYZW identity `(0,0,0,1)` and zero translation are accepted for the one global source USD. Existing authored floors/Plane/decals are disabled only in the session layer; original USD layers are not saved.

After a real native query binding, `ground_height_at` remains a strict exact-floor downward query. Physical roots outside the finite floor are rejected; no support is invented. Optional `dynamic_robot_roots` are exact absolute Robot prim roots with slash-boundary matching. They cannot exclude source floor/static paths; other undeclared collider hits fail closed. The completed physics counter must remain unchanged through queries.

`height_at` is now a deliberately **clipped policy feature**, not a true obstacle top. Known-free cells contribute selected-floor height; measured blocked, unknown and map/floor-exterior crop points contribute floor+1.8 m. This detects tall walls/racks whose tops lie above the old ray origin and permits the PACER head-centered 32×32 crop to extend beyond the finite floor without inventing physical support. It uses the admitted full-height grid and issues **zero per-pixel native callbacks**. Invalid/nonfinite/extreme coordinates and counter advance still fail. `native_surface_height_at` is a separately named bounded-ray diagnostic; it cannot discover tops above its floor+1.8 m origin.

## Preserving the probe source epoch

By default the provider requires the live recorded probe-script bytes to match. A caller may explicitly pass an archived script copy via `probe_script_snapshot`; its bytes and full SHA must match the native preflight's recorded script epoch. That copy is rehashed on source verification and is never executed. This allows later independent v6 probe development without invalidating an already measured v5 artifact. Source USD/layers, maps, raw hit artifacts, floor recipe, preflight and native result gates remain unchanged. Wrong or changed snapshots fail.

The actual CPU reader recovered v5 script `16f845af4d315cd9328d7e1ebdf68317b64192947b5f49df07b3942b4f2f9c0d` from the formally sealed probe archive; see [actual_warehouse_verification.json](actual_warehouse_verification.json) and [probe_source_v5.py](probe_source_v5.py).

## Executed verification

- Final provider/grid CPU union: **70 passed**; [JUnit evidence](tests.xml). Tests cover winding walls, conservative narrow passages, unknown barriers, exterior/negative coordinates, bad resolution/layout/CSR/classification/clock, false near-wall route arrivals, subset/reset-safe route APIs, finite-difference and batch-invariant SDF gradients, independent former-operator values/Jacobians, public floor/static paths, coplanar hidden ray hits, historical source identity, stage ownership, XYZW identity and exterior policy feature versus strict physical support.
- Ruff target Python3.12: passed for the owned source/test/reader files.
- Actual already executed Warehouse v5 artifact admission: **56,239 canonical known5 cells**, complete height/floor raw evidence, single reference-floor owner; **51,157** conservative cells at `.35 m`. An explicitly chosen CPU smoke path in the largest four-connected free region has **292 vertices / 36.804372 m**; every vertex and dense segment sample is walkable. The actual-grid SDF gradient is finite and nonzero. These points are mathematical smoke-test choices, not author routes or physically spawned pedestrians.
- Final export labels: occupied0=3,863, unknown205=33,288, free254=55,719. The raw known5 count can exceed final free count because measured height obstacles and retained unknown support remove candidates.
- No CUDA initialization, SDK start, new physical step, policy training, weights/dataset change, heldout34 read, or A* replacement of diffusion output occurred in this verification.

Reproduce the narrow CPU gates:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest \
  tests/test_dynbench_scene_provider.py tests/test_native_grid_scene.py -q

env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/native_grid_scene_v1/verify_actual_warehouse.py
```

The prior provider v1 bytes are preserved under `before/` and its historical archive remains intact. This source bundle owns only the provider, new grid companion, their tests and evidence reader. Parent-owned world/driver/core hooks, previous source135/nav138 experiments, weights, and old thin entry are outside this implementation. Actual new provider cooking, live humanoid contacts, stability and navigation outcomes require the parent's separate SDK execution; CPU proof does not supply them.
