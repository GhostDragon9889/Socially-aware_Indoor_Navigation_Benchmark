# Warehouse native30: independent CPU verification

All **865 checks passed** against the retained actual GPU capture. The reader launched no SDK/GPU. The old five-second reports and frozen v1 reader remain unchanged.

## Actual observations

| Quantity | Observed |
| --- | --- |
| Controls / actors / duration | 900 / 4 / 30 s |
| Shared / actor physics substeps | 3600 / 14400 |
| Falls / time-limit resets | 0 / 0 |
| Actual arrivals / complete four-vertex cycles | [2, 2, 2, 2] / [0, 0, 0, 0] |
| Static contact actor-controls, threshold >1 N | 0 / 3600 |
| Peer contact unique-pair controls, threshold >1 N | 0 / 5400 |
| Maximum static / peer force in retained reductions | 0.0 / 0.0 N |
| Minimum pelvis height above actual declared floor | 0.825075 m |
| True body COM / mass-weighted COM recomposition max error | 9.53674316e-07 / 3.81469727e-06 m |

All eight arrival events match the actual route vertex and lie within 0.75 m. Each actor reached two vertices within its one observed episode; no actor completed a four-vertex cycle. Distance excludes reset crossings. All ground controls retain four valid native substeps and exact final link-body positions matching the physical capture.

## Measured live-stage evidence

The saved live observation reports the same actual singleton/context stage, one source USD occurrence, one active `/physicsScene`, the authored source `/PhysicsScene` inactive, one selected reference floor and745 exact static colliders. Its actual floor geometry/transform/material match the bound recipe; the original Plane and11 decals are disabled. All53 source layers were independently fully rehashed and match the original preflight. The current floor-filter force view uses the same active native physics view and excludes all static/peer filters.

The four actual FP32 reset origins plus128 deterministic admitted-free samples retain132 all-hit ray rows. Every row has exactly one selected-floor support hit, valid finite geometry and unchanged submitted-step counter0→0. No foreign or undeclared coplanar support appears in those rows. The verifier creates no App/context and advances no physics. Historical App/context construction counts are unmeasured (`null`), and native engine time is unavailable; submitted-step counters must not be called an observed native engine timestamp.

Actual source158 archive SHA `e5b7e5b811ca19b415fcc113e78156f71f398bb64787f4c081fc57271d6101f9` and manifest SHA `f6f38348e2085624403838a595b8bc72748c35fdeaf4763fbfe8ecbc8559cccd` were fully checked. The report lists 1 current-tree source change after launch (the separately authorized recorded replay extension); the archived captured source remains intact. It does not substitute current files for launch provenance.

## Support and limits

Per-actor double-support fractions over native substeps: [0.15416666666666667, 0.15666666666666668, 0.17027777777777778, 0.14805555555555555]. No-support fractions: [0.0011111111111111111, 0.0, 0.0008333333333333334, 0.0002777777777777778]. Support is exact foot-to-selected-floor normal-force thresholding. Any saved slip quantity remains a body-origin tangential velocity proxy, not contact-point slip.

This is a bounded30-second observation. It does not establish200-second stability, author benchmark equivalence, PhysX/Warp bitwise determinism, or causality relative to prior failures. Clearance and spawn/route geometry changed together. The renderer does not validate these physical channels. The separately prepared native replay has CPU preflight only; actual rendering remains pending.

Artifacts: [actual audit](actual_clearance08_30s_audit.json), [reader](audit_actual_capture.py), [six negative controls](negative_controls.json), [source freeze](source_freeze.json). Raw array shape/dtype/SHA and full actual measured live rows remain in the audit JSON.
