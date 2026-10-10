# Prepared actual V8 map reader

`registration_v1/report.json` is a CPU-only registration: zero scenes audited, native execution unobserved, map admission and route validation pending. At registration time the proposed production consumers were not installed. This record is not a map or navigation success.

The reader binds the frozen V2 proposal and exact producer SHA `20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a`. Before actual audit it requires the approved four consumer/producer payloads to be installed. It imports `native_map_v8` only when reading actual evidence, then recomputes the full raw/canonical/native-world-crop/floor/height/unknown/query contract and the additional original/session range evidence. The original v7 failure results remain untouched. Synthetic fixture markers, failed statuses and absent native results cannot be promoted.

The actual reader is pure CPU and does not create an App/SDK/Context or advance physics. Its array inventory retains dtype, shape and C-order byte digests. Complete measured query clocks, session range metadata, floor ownership and source SHA records are retained. Optional `--scene-config` can check new cold configurations bound to this exact scene/map/preflight and every initial/route-wrap A* path; this is a geometric grid test, not physical pedestrian success.

## Planned native execution (root only, after approved merge and new freeze)

```sh
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/scripts/probe_paper_scene_maps.py --projection-dir /project/EXP/SangforAndSaut/ProtoMotions/data/diffusion_amp/maps_resource_complete --output /project/EXP/SangforAndSaut/ProtoMotions/output/scene_maps/native_reference_four_scenes_session_clock_v8 --scenes Office Warehouse Full_Warehouse Hospital --floor-owner reference_flat_terrain --known5-position-source official_buffer_centers --floor-probe-limit 0 --comparison-points 256 --snapshot-steps 2 --execute-physx
```

The arguments equal the actual v7 protocol except the new producer and fresh output directory. Physics queries remain CPU at120Hz, with exactly two submitted snapshot steps; one Kit process handles all four scenes. This command has not been executed by the reader author.

## CPU read after actual results exist

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src ../IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261002/native_map_clock_v8_actual_reader_preparation/audit_actual_four_scenes.py --native-output output/scene_maps/native_reference_four_scenes_session_clock_v8 --output docs/style_improvement_20261002/native_map_clock_v8_actual_reader_preparation/actual_four_scenes_v1
```

All outputs must be new; existing evidence is never overwritten. An audit failure preserves the successful earlier scene rows, the error and pending route status. An all-four PASS still cannot establish eight paper scenes or physical navigation quality.
