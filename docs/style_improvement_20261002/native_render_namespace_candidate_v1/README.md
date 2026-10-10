# Native recorded visual `/Render` overlap candidate

**Docs-only review candidate. Production has not been changed; no new Kit app, GPU, physics step, policy call, or rendered frame was executed by this audit.**

The preserved first renderer failed before frame 1 (`native_warehouse_recorded_replay_30s_v1/result.json`). The actual SDK probe disproves the initial empty `/World` theory: no `/World` exists. The overlapping root is `/Render`, containing SDK viewport metadata/products; Warehouse contributes only an empty `over Render (hide_in_stage_window = true)`.

## Proposed narrow change

[production_patch.diff](production_patch.diff) targets only `CrowdSim/humanoid/runtime/native_visual_scene.py`; [native_visual_scene_candidate.py](native_visual_scene_candidate.py) is the reviewable full candidate. Original source USD is appended as the **weakest renderer root sublayer**, below existing root/session opinions. Physics disabling remains authored into the stronger session. Source prim paths and original global coordinates stay unchanged.

Overlap admission requires anonymous root/session ownership, the exact observed seven-path SDK viewport namespace, active nonspatial prims, and no foreign layers, xforms, boundable geometry, physics, references, payloads, instances, unknown metadata, or unknown properties. Existing API names must start `OmniRtx`; ordinary properties are `camera`, `products`, `orderedVars`, `resolution`, `overrideClipRange`, `sourceName`, `sourceType`, `viewPickingId`, `viewportHandle`, or `omni:rtx:*`. Literal authored SDK `hide_in_stage_window` and `no_delete` must be true. Standalone USD represents unregistered Kit metadata as `Sdf.UnregisteredValue("true")`; the candidate checks that exact token, not truthiness or a textual substring.

The source `/Render` must be the metadata-only empty over. Source render products, children, properties, APIs or arcs are **unsupported and rejected**, so no extra source capture products are introduced. Any collision with the other source roots `/Root`, `/PhysicsScene`, `/NavMesh`, or the reference-floor path is rejected. An existing active PhysicsScene elsewhere is also rejected. No roots are deleted or reanchored.

## CPU evidence

- [tests_final.log](tests_final.log): **22 passed**, including explicit false markers, source render products, inactive/foreign/spatial/API opinions, other-root/floor conflicts and pre-existing physics. Ruff passes.
- [actual_warehouse_CPU_proof.json](actual_warehouse_CPU_proof.json): real Warehouse USD and all **53 recorded source-layer hashes** unchanged; **1,442 non-Render Xformable paths × default/t0 = 2,884 exact matrix comparisons**, zero differences. **42 SDK authored render properties** remain exactly unchanged. All source physics scenes are inactive, collision/body flags false, visual reference floor has no CollisionAPI. Source occurs once in root sublayers and zero times in session sublayers.
- Probe evidence: [actual_empty_stage.json](../native_visual_placeholder_sdk_probe_v1/actual_empty_stage.json). The CPU proof imports its actual root-layer text; the probe did not serialize session-layer text. It therefore does **not** claim a complete live SDK session reconstruction or render success. Production additionally validates both root/session ownership at runtime.
- The existing in-memory `prepare → compose` contract is reused. Saved JSON bounds become lists, whereas the provider returns tuples; the CPU proof reconstructs the contract through `prepare`, verifies serialized equality and leaves the existing strict verification unchanged.
- Initial standalone metadata failures remain in `initial_standalone_metadata_tests.log`; final evidence is separately recorded.

## Acceptance still pending

Root must review/apply the patch and run the sole actual renderer. A successful CPU composition is not a rendered-video or physics/navigation acceptance result. The original failed artifact is retained. The cuboid replay branch is outside this patch.
