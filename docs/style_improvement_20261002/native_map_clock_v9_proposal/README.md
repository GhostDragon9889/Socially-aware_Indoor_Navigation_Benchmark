# V9 anonymous runtime root proposal

## Scope and evidence

This reviewable proposal contains **8 repository target paths, 299,801 proposed bytes**. It does not alter the current production tree, source USD files, SDK, weights, or AMP world/training loops. The author executed the merge helper only in its default dry-run mode. Root must review and apply the exact before/after manifest through the existing approval process.

The producer remains frozen at `e47cd51d9f40cd0ead27595283fefab95871ea229c7029c5622b57391a01ad82`. Root actually executed its Warehouse candidate in `output/scene_maps/native_warehouse_anonymous_root_v9_candidate_v1/Warehouse`. Its native resource gate passed. The CPU consumer recomputed all **56,239 known5 cells**, exact floor/all-height CSR, world crop, unknown preservation, original source/layer SHA, saved clock and ownership evidence: `actual_Warehouse_candidate_admission.json`. This is one scene's candidate resource validation; four-scene V9 and pedestrian deployment are pending.

Final docs-staged focused gate: **269 passed, 0 skipped**, 7.80 seconds; `full_artifact_tests.log` and JUnit XML. Ruff passed. The initial copied-fixture path errors are retained in `full_artifact_tests_initial_path_errors.log`; the corrected harness points those copied tests to immutable real-artifact roots, without changing their assertions. Nine standalone OpenUSD helper tests passed previously with explicitly fake App/Timeline interfaces, and are included in the final focused union.

## Concrete change

Public IsaacLab `create_new_stage()` creates one anonymous runtime root and registers the thread-local stage/cache. Its absolute sublayer is the original scene USD. Up axis, meters/unit, TCPS, **actual source FPS** and defaultPrim are copied explicitly because OpenUSD does not inherit these root metadata through sublayers. Runtime range is 0 through `max(original_end, TCPS)`, in the owned runtime root/session. The original authored range remains separately recorded. Warehouse's actual FPS is 24, not an assumed 60.

The existing SimulationContext performs its normal asynchronous Omni context attachment. No manual PhysX/context attach is added. A bounded observer verifies active context root/session and StageCache ID, with submitted counter unchanged; it does not issue physics steps or create another App/Context. Actual Warehouse observed cache ID 9223002, zero waiting App updates, submitted counter 0→0, original source RAM `d588ee…` unchanged and actual query timeline 0.

V9 admission requires this exact producer SHA, separate original/owned RAM hashes, actual context/cache ownership, unchanged original full-file/layer hashes, and observed query clocks. The saved owned-root artifact must match the actual query-time RAM SHA and contain no prim definitions. Standalone OpenUSD recomputes original source RAM and rejects source session geometry/xform/property changes beyond exact declared collision disables, deinstance roots, and inactive authored PhysicsScene; declared reference-floor and probe-PhysicsScene roots remain permitted. Raw4/5/6, full floor/all-height, unknown, foreign-path and crop mathematics are retained. Native engine time remains explicitly unobserved.

## Legacy preservation

Original `native_map_v7.py` and `native_map_v8.py` are protected byte-for-byte. V5, V7 and V8 dispatch still use exact historic identities. Synthetic legacy equality tests compare descriptions, A*, signed-distance values and FP32/FP64 Jacobians. V8's synthetic fixture now copies the historical 201731… producer rather than whichever script is current. The old setup test exercises the actual new range/reset block with a real independent USD stage and a declared fake Timeline; post-reset-zero and original-source RAM assertions remain.

The failed actual V7 and V8 artifacts remain unchanged. V8's Office/Warehouse/Hospital failures proved original source RAM range metadata was modified; Full_Warehouse alone passed. None of those failures is promoted or hidden by V9.

## Review / merge / validation

- `proposed.patch`: complete human-readable diff.
- `merge_manifest.json`: exact 8 targets, before and after bytes/SHA, protected files.
- `merge_v9_proposal.py`: bounded allowlist, no symlinks, per-file <250 KB and total <500 KB; dry run by default, same-directory atomic publication only with explicit root-authorized `--apply`.
- `run_staged_focus.py`: CPU-only proposal testing before a merge.
- `verify_actual_warehouse.py`: independent actual candidate resource recomputation, never launches SDK/GPU.

After root's approved merge, run the same full focused gate/Ruff, freeze a new source epoch, then execute fresh complete V9 probes for all four scenes. Keep actual statuses and all partial failures. Only each scene's completed resource gate admits its map; this gate is separate from closed-loop motion and rendering quality.
