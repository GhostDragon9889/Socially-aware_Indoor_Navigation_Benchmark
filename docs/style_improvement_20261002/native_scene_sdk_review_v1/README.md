# Native scene SDK ownership review

**Read-only conclusion: no first-run blocking defect found in the frozen provider's global AssetBase spawn path.** This is source review plus an extracted CPU copy contract, not SDK scene creation or navigation validation. Actual native evaluation had not been executed by this reviewer. No frozen provider/world/driver source was edited.

## Actual pinned call path

- `ISAACLABSimulator.__init__` creates its `SimulationContext` after the provider's pre-create query hook, then gets `SceneCfg` and constructs `InteractiveScene` once ([simulator.py](../../../protomotions/simulator/isaaclab/simulator.py)). The native subclass calls `provider.prepare_scene_cfg` once from `_get_scene_cfg`.
- `InteractiveScene.__init__` stores `self.cfg = cfg`; it does not deep-copy the supplied scene config. Its clone planner explicitly skips global paths outside `/World/envs/`. The provider marker `/__PaperDynBenchSceneProvider` therefore is not a per-actor clone source.
- `InteractiveScene._add_entities_from_cfg` visits the one `dynbench_source` AssetBase entry once. For a global path it sets `spawn.spawn_path = prim_path`, invokes `spawn.func` once, then creates an extras `FrameView` of that marker. It does not construct an `AssetBase` object for this plain config, so the separate `AssetBase.__init__` spawn path does not cause a second call.
- Native `AssetBaseCfg.InitialStateCfg` defaults are position `(0,0,0)` and quaternion **XYZW `(0,0,0,1)`**. Those exact values are passed to the frozen provider callback and satisfy its identity checks. The callback is deliberately not a regex/cloning-decorated USD spawner; it authors one untransformed global source, not one USD per actor.
- `SimulationContext` installs its actual stage in the SDK thread-local `_context.stage` before `InteractiveScene` is created. Both `InteractiveScene` and the provider's public `isaaclab.sim.get_current_stage()` getter read that stage. The getter has no global `omni.usd.get_context().get_stage()` fallback. `use_stage` temporarily replaces/restores the same thread-local binding. The SDK may attach that stage to the Kit viewport context, but this is not how the provider chooses its target stage.

The exact files, hashes and line locations reviewed are recorded in [source_review.json](source_review.json). The SDK is the installed pinned source, not assumed latest API documentation.

## Bound-method copy ownership

The outer `AssetBaseCfg` constructor's configclass post-init deep-copies its noncallable `spawn` config. Python deep-copying that config's bound `func=provider._spawn` also deep-copies the bound provider owner. Consequently the callback owner **can differ from `world.scene_provider`**. In contrast, configclass post-init does not copy a directly callable `func` field by itself.

[config_copy_verification.json](config_copy_verification.json) executes the unchanged pinned `_custom_post_init` body in isolation against actual source-bound provider metadata, with string-resolution wrappers stubbed. Seven ownership checks pass. No SDK config instance, USD stage, callback invocation, App, simulation or physical measurement is represented by that test.

This does not block the current first scene construction:

- The copied provider keeps identical source records and the already-true pre-create query-hook flag. Its spawn owner has no live query binding yet, so it does not carry an already-active native query interface through the copy.
- The callback's source-root conflict, existing floor/marker and session-sublayer checks operate on the actual stage. A new provider copy still cannot stack the same source/floor in that stage. `_spawned` supplies an additional local check rather than the only duplicate protection.
- The world's original provider subsequently receives the live query binding and supplies map/floor/height calls. Its `_spawned` remains false after the copied callback runs. **Do not use that original flag or prepared-status metadata as evidence of actual spawning.** A live stage inventory/identity record is the correct proof.

No provider change is necessary for the current one-world path. A future second-scene/reuse feature would need an explicit new lifecycle contract; this review does not authorize such reuse or promise shared callback state.

## First actual-run audit contract

[audit_schema.json](audit_schema.json) is a **pending evidence specification**, not a passed result. It requires raw arrays and source identity for every requested control (150 for 5 seconds, 6,000 for 200 seconds), including every four completed 120 Hz substeps. It separates:

1. Rehashed native map/USD/probe-epoch admission from **fresh runtime** stage/floor/contact observations.
2. Code-level single App/context call paths from measured lifecycle identity; unavailable measured fields remain `null`, not inferred success.
3. Exact feet-to-selected-floor normal force and same-instant live COM from static/peer forces, proxy slip and planned trajectories.
4. Actual falls, clip/time-limit resets, reached vertices and within-episode cycles; reset teleports never become distance/progress.
5. Observed completed-step counters and unavailable native engine-time APIs; no invented native timestamps or PhysX/Warp bitwise guarantee.

The frozen provider's public `floor_roots` and `static_collision_paths` must be the exact filters. The map's v5 probe measured one reference floor, but its evidence does not itself prove the newly constructed runtime has no extra plane. Any missing fresh stage/query ownership evidence is reported separately. PhysX stability/navigation outcomes remain empirical and must preserve all raw frames, including failures. A passing 5-second probe cannot substitute for a 200-second outcome.

The provider/grid remain frozen at SHA `558b4e8ff8016c900c7a571db9466ba1b0b4b8be4303a9463bc39fbeba9b4730` / `b954dc9949cac33eb361dbda2ee1510cfe2bac8bd2e9e0fff7670fc0376b3f5f`. The parent's later hooks and their live source epoch are reviewed independently; older source135/nav138 archives are preserved rather than asserted equal to the whole current tree.
