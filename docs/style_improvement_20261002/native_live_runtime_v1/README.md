# Separate observed native runtime verification gate

**Ready for the parent's next actual GPU run; CPU fake-interface contracts only.** Prior provider/grid/source154 bytes and the two real first5s experiments remain separate historical epochs. No App/SimulationContext, native view, physical step, actor reset or source USD save occurs in the new verifier.

```python
record = world.scene_provider.verify_live_runtime(
    origins_xy=world.origins,
    simulation_context=world.simulator._sim,
    free_cell_limit=128,  # Optional deterministic0..256 known-free cells.
)
report["native_scene_live_observation"] = record
```

Call after the sole world initializes and binds its existing native query, after origins are chosen, and before initial actor resets. The method checks the actual public SDK singleton identity, SDK thread-local stage identity, enabled query config, exact bound counter owner and one active PhysicsScene. It creates no context or query view. Historical App/context constructor counts are unobserved/null; current singleton identity is a distinct measured assertion.

The live stage must compose the recorded source exactly once, contain its authored roots and provider marker, have every authored source PhysicsScene inactive, and have exactly the declared enabled static/floor collision paths when exact Robot roots are excluded. Proxy-instance traversal follows the probe's pinned USD API. Extra Terrain/Plane/foreign collider, enabled original flat ground, missing static geometry, wrong floor vertices/transform, or wrong actual bound physics material fails closed. Source USD/layer/resource records are fully rehashed before and after observations.

Every requested origin must remain a genuine admitted free map cell; no relocation/clipping occurs. Native downward all-hit rays are collected at each reset origin plus the optional deterministic row-major sample. All hit geometry must be finite with unit normals and valid distances. Exact slash-boundary Robot roots are classified as excluded; similarly named Robot2 and unknown colliders fail. Any upward near-ground nonfloor support fails even if a selected floor hit is present. Every ray must include the declared floor, and its boolean result must match callback evidence. Per-ray and batch **submitted-step** counters remain unchanged. No native engine timestamp is invented.

The returned record is separate from static/prepared component descriptions and preserves all ray-hit positions/normals/distances/path classifications and observed stage/material identities. It is sampled current support ownership, not a fresh full-grid native map rebuild, full-body clearance guarantee, or stable navigation result. The currently admitted Warehouse v5 protocol is supported; future v6/v7 scene resources need their own admission integration and actual execution.

CPU verification: **23 new live contracts, 93 total provider/grid/live contracts PASS**, Ruff Python3.12 PASS. Tests use explicitly fake context/USD/query interfaces and assert no clock advance. They cover duplicate source, original floor, foreign Terrain, two PhysicsScenes, stage/singleton/counter owner mismatches, wrong material/geometry, hidden coplanar support, Robot2 path mismatch, absent floor, nonfinite/unit-normal failures, advancing queries and bounded input validation. These are not labeled native SDK PASS.

See [source_freeze.json](source_freeze.json), [tests.log](tests.log), [tests.xml](tests.xml). The prior provider SHA558b4e… is preserved under `before/` and remains in the historical source154 archive. This new provider epoch adds only the opt-in verification method; the previously executed control loop/height/map behavior is unchanged.
