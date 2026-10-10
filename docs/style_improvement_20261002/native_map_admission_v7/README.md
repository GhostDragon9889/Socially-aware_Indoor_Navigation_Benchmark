# Explicit v7 native-map consumers: CPU ready

The provider and grid companion now select v7 only by the frozen probe SHA `a538bf92ffb64b536a7005329f38c080bd1b8818114df58d9880d5506f0665d5`. The producer keeps its existing v1 JSON schemas, so a schema name or newly added fields cannot select v7. The public `from_preflight` / `admit_native_map` / `NativeGridScene.from_provider` APIs remain the same. A historical probe copy may bind the exact original probe bytes.

## Contracts

| Evidence | v7 consumer requirement |
| --- | --- |
| Actual result | Completed matching native resource gate, full direct-cell validation, source/preflight/probe SHA identity, no fallback, unchanged source/static snapshot |
| Raw grid | Fixed official native layout; occupied API positions exactly raw4; API free positions only raw5/raw6; canonical positions cover every raw5 once |
| Missing API rows | Explicit `official_buffer_centers`; row `-1`, source string, native index and derived center all agree. The `known5_API_complete=false` declaration remains visible |
| World crop | Only zero/one lower cell per axis; .1m alignment, exact original projection bounds/dimensions, fixed native-to-ROS window. Raw and full-height images stay native-sized |
| Floor | Every native5 has exactly one actual selected-floor support hit. CSR geometry, per-cell location, selected hit fields, masks/counts and origin ownership agree; no foreign or uncertified coplanar support |
| Low static obstacle | Exact enabled static Mesh/Plane with the frozen negative-floor certificate, identical to the source-bound preflight inventory. Its near-floor hit makes the entire cell occupied even below the .15m band |
| Height / export | Complete native5 height CSR, exact classes and row masks. Export is the verified crop of raw labels refined by actual height and low-obstacle hits, then intersected with projection unknown. Raw6/unknown/exterior never become free |
| Clocks | Persisted initial observation and full-height clock agree; actual nonnegative integer submitted counter and frozen timeline0; no query-time advance |

The v5 branches retain their original validators and output fields. Before copies preserve provider `cc8b59f1…`, grid `b954dc99…`, and live verifier `3d9db0f3…`; the live verifier was not edited in this task.

## Evidence and remaining work

- **116 focused CPU tests passed**, Ruff passed: old provider/grid/live tests plus23 new protocol/corruption/regression cases. Success fixtures use synthetic rays, overlaps and clocks, with official CPU layout/crop formulas. They do not represent v7 native execution.
- Actual preserved Warehouse v5 provider/grid descriptions match before exactly. Four actual fixture A* routes match dtype/shape/bytes, and FP32/FP64 SDF values/Jacobians match bytes.
- Preserved actual v6 raw archives from Office, Warehouse, Full_Warehouse and Hospital satisfy the consumer's fixed indices, complete canonical raw5 set and the official world crop. Office's missing3,532 API known5 rows remain explicit. Original v6 statuses and artifacts were not changed or promoted to v7 success. This proof admits no new runtime map.
- The dependency owner independently compared3,629/745/3,331/1,981 real static nonfloor certificate rows from those respective scenes: runtime and source-preflight dictionaries have zero mismatches. The extra certificate identity gate has real source evidence.

Artifacts: [source freeze](source_freeze.json), [actual v5 / four-scene v6 raw proof](actual_v5_and_four_scene_v6_raw_proof.json), [CPU verifier](verify_preserved_actual_sources.py), [tests](tests.log), [before hashes](before_source.json).

**Fresh v7 four-scene native results are still pending.** Their actual success, complete floor/height data and recorded clocks must pass this validator before deployment. This change starts no App/Kit/GPU/world, does not alter maps/weights/datasets/physics, and does not establish body clearance, navigation success or controller stability.

## Focused command

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
../IsaacLab-Protomotions/.venv/bin/python -m pytest \
  tests/test_native_map_v7.py tests/test_dynbench_scene_provider.py \
  tests/test_native_grid_scene.py tests/test_dynbench_live_runtime.py -q
```
