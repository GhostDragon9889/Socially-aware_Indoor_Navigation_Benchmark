# IsaacLab 3.0 EA / IsaacSim 6.1 migration audit

Dates: 2026-10-10–11. Changes remain local; **no commits or pushes**. [Detailed Chinese report](ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md) / [evidence index](migration_sim61_lab3ea_20261010/README.md).

## Target and changes

The default target is IsaacLab product 3.0 EA / Python package17.0.2 at commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`, IsaacSim6.1.0.0, Python3.12.13, Torch2.11.0+cu128 and Warp1.16.0. A separate project overlay preserves the SDK providers and locks. The old SDK/runtime remain available; neither SDK source was modified.

Fixed ProxyArray bridges, XYZW boundaries, body/joint/contact ordering, SMPL69-DOF D6 selection, N4 USD EditTarget leakage, owned terminal SimulationView/Timeline cleanup, RTX extension startup, motion clocks, path corner/spline collision checks, Newton1.5.2 controls/FK/contacts and optional baseline API boundaries. Public reference adaptations are preserved as [pinned source patches](../configs/compatibility/reference_patches/README.md); their40 files were applied and hash-checked against clean local commit trees, with idempotence and foreign-change rejection verified.

## Evidence

- Static scan: 1,687 Python files / 362,615 lines; all AST parsing passed, zero F821/F822/F823/F901/E9 findings. Configuration syntax and exact source hashes are recorded. This does not establish exhaustive dynamic execution.
- Final current CPU suite: **4,286 passed /7 skipped**, plus61 passing subtests. Five skipped CUDA nodes were separately run on real GPU and passed. Exact-node merged results: **4,291 passed /2 legacy IsaacGym skips**.
- Collection:4,647 nodes =4,293 current +the original354 historical guards. The original complete second CPU run,4,129 passed/294 failed/102 errors/14 skipped, is preserved with failure stacks. Historical source and SDK seals were not weakened or newly excluded.
- Statement coverage is retained with source-version matching and uncovered lines; full branch coverage is not claimed.
- Eight-env AMP smoke:12 updates,3,072 actual physical transitions, all2,848 training clips sampled, four network groups updated, zero falls, clean real process exit0. Checkpoint/optimizer/replay/normalizer CPU roundtrip and inference equivalence passed. GPU mid-episode resume was not performed; this smoke model was not promoted to the production weights.
- PointGoal training/resume/evaluation, wheel/sdist builds, isolated five-module/six-CLI loading and strict offline Sphinx HTML rendering passed. This does not prove a fresh full SDK install or all heavy autodoc/external inventory behavior.
- RTX smoke:3 genuine rendered H264 frames,960×720,30fps,0.1s; full decode and real parent exit0. It faithfully replays accepted physical states, not a second fresh physics run. Existing60s videos retain their historical Sim6.0 identity.

## Native scene acceptance

Each completed run uses1,800 controls and7,200 real PhysX substeps, full-frame state/contact archives and unchanged declared source/input hashes.11 local reliability gates include actual falls, route progress, body-envelope clearance, measured support, peer/static normal forces and unfiltered normal-force residuals.

| Scene | Actors | Simulated duration | Local result |
|---|---:|---|---|
| YOHO | 4 | 60 s | passed_local_proxy_gates |
| NewMetro | 4 | 60 s | passed_local_proxy_gates |
| SecondFloor | 1 | 60 s | passed_local_proxy_gates |
| ThirdFloor | 1 | 60 s | passed_local_proxy_gates |

The original profiles use4/4/1/1 actors. The first ThirdFloor run exited normally but failed two contact gates: eight controls had left-hand static contacts, peaking at26.8N. Its unchanged failure receipts are retained. A new run uses the already tested conservative AMASS model and larger measured-body forecasts; scene, route and quality thresholds remain unchanged. Single-actor acceptance does not establish crowd interaction or full-scene route coverage. Speed limits do not guarantee every physical velocity. SecondFloor averages0.393 m/s along its path and has an18.1-second route-vertex gap; this remains below the earlier0.5m/s lower speed target and requires later model/turning optimization. Gates and recorded states were not changed to hide it. ThirdFloor contact-safe retry averages0.084m/s, reaches only one vertex and has a49.6-second vertex gap: over-conservative braking remains an unresolved navigation regression, and its profile remains quality-pending. Geometric envelopes, foot-origin speed and normal-force residuals are proxies; tangential friction, contact-patch slip, visual gait quality and author eight-scene results remain separate validation work.

## Reproduction boundaries

Use the [installation](guides/installation.en.md), [configuration](guides/configuration.en.md) and [clone guide](../CLONE_REPRODUCTION.en.md). Core native execution requires the checkout, fixed SDK, licensed AMASS/SMPL and model/scene resources. Wheels alone do not provide all native assets. Final tests also used an optional real ORCA binary; its provenance is recorded. Optional legacy baselines still need their external services/checkpoints/platform dependencies and have not received full native GPU acceptance.

No finite suite can guarantee every line, branch, input and hardware is bug-free. The current tests, preserved failures, coverage gaps and native evidence define the actual verification scope.
