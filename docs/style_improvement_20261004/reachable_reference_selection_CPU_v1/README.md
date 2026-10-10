# Reachable-reference candidate ranking: CPU preparation

This is an experimental selection helper. It is not connected to production185 or the default planner. No model, checkpoint, simulator, GPU or sealed34 data was used for its tests.

## One variable, two matched arms

`select_reachable_candidates` accepts world XY `[B,S,51,2]` at 0..5s, the existing `CausalReferenceAdapter`, the previous executed reference XY/velocity, scene IDs and world guide points/times. It adapts all candidates in batch-major order, retaining every raw/resampled/adapted candidate and cost. The adapter's speed/acceleration limits are inherited without modification.

- `matched_raw50` scores the resampled raw XY.
- `reachable50` scores the adapted reference XY.

Both arms use the same 50 future nodes (.1..5s; dense indices3,6,..150), target times, original geometry functions, weights and shared selection within each scene. The original planner scores 52 future nodes through5.2s; neither new arm is identical to that historical selector. Generation/guidance remains a separate unchanged52-node operation in the proposed isolated runner.

The score is the sum of actor goal/static costs plus each peer pair once. Existing defaults are goal20, peer5, static10, peer radius.6m (1.2m center separation), and scene radius.8m. Goal distance uses the original mean reducer or explicitly supplied original waypoint weights. The static term uses the existing box SDF or supplied native callback; the helper creates no simulator resources. Both arms use the same scorer and candidate set.

## Returned data and control boundary

`adapted_xy`, `adapted_velocity`, `raw_dense_xy`, adapter flags/deviations, `goal_cost`, `static_cost`, `independent_cost`, `scene_sample_cost`, and `selected_sample` retain all candidates, including unsafe ones. `peer_cost` is the original upper triangle `[B,B,S]`: `sum(dim=1)` gives per-actor bookkeeping `[B,S]`, whose sum counts each pair once. This attribution is not a symmetric conflict measure.

The downstream prediction must still contain the selected **raw51** trajectory, followed by the original adapter call. It must not contain151 adapted points in a raw51 field. The existing driver's previous-reference handoff can be reused immediately before its original `reference_adapter.build`; initialization/reset alone uses the physical root. The helper does not mutate inputs, RNG, adapter state or physical state, and introduces no fallback.

## Actual bounded CPU verification

Eight tests passed with CUDA hidden using real Torch CPU geometry and the original adapter/cost functions. No NN class was instantiated. Tests cover joint same-scene sample selection, both domains sharing candidates and clocks, exact single-candidate adapter output, continuity-state effects, .1/.5/5s indices3/15/150, speed/acceleration limits, original static/peer costs, callback shape/single invocation, scene partitioning, and Python/NumPy/Torch RNG preservation including the first lazy import. Synthetic tests are not evidence of physical safety or navigation quality.

Final invocation: session40551, launche6ba15 → completion8ffcb0, observed exit0, **8passed/1.48s**. The prior invocation20117 also exited0 (8passed/1.54s); the final invocation adds first-call import coverage. Final Ruff18fd8e exited0. Exact commands and source identities are in [actual_CPU_tool_receipt.json](actual_CPU_tool_receipt.json).

The earlier [actual recorded clearance diagnostic](../causal_reference_clearance_CPU_v1/README.md) motivates testing selection domain; it does not establish collision causality. Runtime integration, actual contacts/falls and quality remain untested. Root will review the isolated runner and schedule any GPU experiment serially.
