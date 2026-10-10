# v8 native-map clock proposal: reviewable, not applied

This directory is a bounded repair proposal. No production file has been changed by its author. The parent will request automatic approval review of the exact patch using the user's existing authorization to complete the reproduction. The earlier rejected production action was not executed. Current native Warehouse evaluation uses its previously frozen source epoch and remains unchanged.

## Actual v7 evidence

The four-scene native run completed with two resource successes (Warehouse, Full_Warehouse) and two failures (Office, Hospital). Both failed scenes have authored start/end/TCPS = −1/0/60 and an actually observed submitted step counter 2 with timeline −1/60 seconds. Their original results and clock artifacts are retained. No failed result is admitted or promoted.

The successful two scenes were independently admitted on CPU using their original v7 source identity and complete raw maps, canonical coordinates, selected-floor CSR, height-band CSR, unknown masks, conservative low-obstacle masks and query clocks. See [partial actual audit](../native_map_admission_v7/actual_four_scenes_partial_v1/report.json).

## Minimal repair

The proposed producer writes start=0 and end=max(original_end_time_code, original_TCPS) **only to its already-owned anonymous session layer**, then calls the documented Timeline range setters before reset. It retains original authored/composed source range metadata, source-root identity and time-codes-per-second. Immediately before map generation it records the actual Timeline range, composed stage range, current session EditTarget and source-root RAM digest again.

Root RAM equality is scoped from helper entry to queries. It does not retrospectively prove absence of RAM edits during earlier SimulationContext initialization. Full source-layer file digests and real USD transform comparisons are independent evidence. No source USD layer is saved or edited on disk; no SDK patch, App, Context or extra physics advance is introduced.

The existing query gate still requires an actual integer nonnegative submitted counter and an actual finite **zero** timeline; negative times are never clamped. The session range setup is not a claimed clock observation. v8 admission is dispatched by the exact new producer SHA `20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a` and independently checks the exported session header, source identity/RAM/range evidence and all original v7 map/CSR/query contracts. v5 and v7 implementations and output semantics remain intact; failed v6/v7 results cannot enter the v8 branch.

The existing historical v7 evidence reader is proposed to verify its frozen archive's member bytes and record current checkout bytes separately. It continues to require the exact original v7 probe snapshot; its default now names that immutable snapshot. This records later intentional dispatch changes without inventing a historical source match.

## Evidence and limits

- `tests.log` / `tests.xml`: 37 CPU tests passed, including synthetic v8 corruption rejections and exact original v5/v7 provider/grid descriptions, A* paths and FP32/FP64 SDF values/Jacobians. Synthetic fixture success is a contract test, not a native resource result.
- `actual_four_scene_USD_session_proof.json`: real standalone OpenUSD26.08 opened all four actual source USDs. Original root RAM, all source-layer full hashes and default/time-zero Xform paths/matrix bytes remained unchanged after the session edit. Timeline in this proof is an explicitly fake documented interface; no Kit/SDK or GPU was started.
- `preserved_actual_v5_v7_branches.json` (when present): staged v8 dispatch independently compared against actual historical v5 Warehouse and successful v7 Warehouse/Full_Warehouse data. No new native execution.
- `ruff.log`: all staged source and helper checks passed.

**Native v8 has not run.** Root must execute the four-scene native probe after approved merge and a new source freeze. Only each scene's complete actual passed result may subsequently be admitted. This proposal does not establish navigation or pedestrian quality.

## Review and bounded merge

`proposed.patch` contains nine exact targets; `merge_manifest.json` records every before/after SHA and three protected unchanged sources. The total proposed payload is 245,852 bytes with a hard 250,000-byte per-file and 500,000-byte total limit. No `.git`, SDK, original USD, checkpoint, dataset, AMP world/runner/trainer or original capture path is a target.

The author executed only this read-only check:

```sh
python docs/style_improvement_20261002/native_map_clock_v8_proposal/merge_v8_proposal.py
```

It defaults to no writes, refuses changed before hashes, existing new targets, symlink redirects, unknown paths and oversized payloads. `--apply` is reserved for the parent **after automatic approval review**, after the current source168 native evaluation completes. Every approved target is atomically published from a same-directory temporary file; it is not a transaction spanning nine files. The full before snapshot is retained for review/recovery.

After approval/merge, run the existing map/provider/grid tests plus the two new v8 files, Ruff, historical v5/v7 regression and source freezing before native execution. A suitable CPU test command is:

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src:tests ../IsaacLab-Protomotions/.venv/bin/python -m pytest tests/test_paper_scene_maps_probe.py tests/test_paper_scene_maps_clock_v8.py tests/test_dynbench_scene_provider.py tests/test_native_grid_scene.py tests/test_native_map_v7.py tests/test_native_map_v8.py tests/test_native_map_audit_reader.py -q
```

## Primary interface evidence

OpenUSD stage time setters author the current root/session EditTarget: [UsdStage API](https://openusd.org/dev/api/class_usd_stage.html). The installed Timeline1.1.0 interface states its range setters operate on the current opened stage, consistent with [ITimeline API](https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/omni.timeline/omni.timeline.ITimeline.html). The default Timeline copies the stage range: [Multiple timelines](https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/MULTIPLE_TIMELINES.html). These support the repair mechanism; only the future actual native run can establish the fixed runtime clock.
