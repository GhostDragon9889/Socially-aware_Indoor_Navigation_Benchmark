# Social World lifetime dependency migration

This independent source stage changes one complete `PINS["world"]` literal in each copied collector/expert module. The selected World is the ROOT-sealed owned lifetime revision (`fbfb01`), 39943 bytes, SHA256 `e76182d2be5a3fa4dad19a04d71275d91929ce3ea2d63a1c23f081aa1a59880b`.

| New module | Unchanged parent source | Parent SHA256 |
| --- | --- | --- |
| `social_rollout_collector.py` | `social_rollout_collector_staging_v1/social_rollout_collector.py` | `f7d2727051d92da5efc0fc4eae37338c88152aca3904971745d4daa7389556fd` |
| `expert_batch.py` | `social_amass_expert_staging_v1/expert_batch.py` | `48bc49764f54bbc6afdf2671f0ed6635f2884534359af1aa1b9bf113696d2afb` |

Each new module retains the original filename and stage depth. Its class-origin check now requires the actual World module to come from `shared_group_world_owned_lifetime_revision_v2/shared_group_world.py`; no fake module/source identity is supplied. Every other source pin, manifest identity, function and method is unchanged.

The source derivation records the exact before/after literals and complete reverse comparison. All 12 collector functions/methods and 11 expert functions/methods retain identical source bytes and ASTs. The collector's three `prepare_control`, `step` and `complete_transition` AST pins still match the new World.

## Scope and status

The static construction/compile check `2f11b7` exited 0. Ruff `8f85c3` exited 0 with all checks passed. Both returned directly without a yielded session. No candidate module was imported, and no Torch, SDK, NN, dataset, checkpoint or array operation was executed. These checks establish the metadata migration only; existing fake CPU checks belong to their original sources and are not reported as executions of these new modules.

This stage does not add a release call to any driver or change the collector's reward, Gaussian/applied action separation, frozen RMS, terminal-next-value, callback failure handling, replay/expert sampling or reset ordering. The caller must still wire the reviewed native release callback and manage stop/GC/context/App cleanup. A weak-reference release receipt is not proof that native objects have died. Real SDK execution and training remain unverified by this stage.

Sources remain writable pending ROOT review and sealing. The original frozen collector, expert and World sources are untouched.
