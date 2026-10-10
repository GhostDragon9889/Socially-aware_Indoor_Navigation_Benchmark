# Actual native style metrics: 4-person comparison and saved 15-person baseline

Six CPU commands and the 4-person comparison completed on their first attempts with exit code 0. CUDA was disabled. The [sealed completion receipt](native4_and_old15_CPU_completion_v1.json) binds all 42 result, source, ledger, log and launch artifacts (SHA256 `9d502615b1f526a9c6f206d461b2aa6e2a17282fcf3668d6ec7acb95e6331cea`). The frozen preparation passed [independent source review](../state_prior_native_metrics_peer_review_v1/peer_review.json) (`17efd688…`).

| Measurement | Selected old native4 | New state-prior native4 |
| --- | ---: | ---: |
| Runtime selected source files | 168 | 183 (V11) |
| Controls / duration / actors | 6000 / 200 s / 4 | 6000 / 200 s / 4 |
| Falls / static-contact controls / peer-pair controls | 0 / 0 / 0 | 0 / 0 / 0 |
| Route vertices | 57 | 44 |
| Legal style intervals retained | 23996 | 23996 |
| Inclusive primary 90D MMD² | 0.07683825264771788 | 0.06992829554896746 |
| Supported speed bins | 4 | 4 |
| Secondary mean over the 3 common supported bins | 0.06870244769117906 | 0.05244885272245936 |
| Fixed 45D pose MMD² | 0.038296637853313165 | 0.025512020968780025 |
| K=50 coverage / normalized entropy | 0.34 / 0.5756056295705451 | 0.58 / 0.6910975121084226 |
| K=100 coverage / normalized entropy | 0.31 / 0.5675963452182238 | 0.47 / 0.6851285917166475 |

The [complete native4 comparison](../state_prior_native_metrics_v1/comparison4/comparison.json) retains both primary aggregates and every original speed bucket. The fourth supported bucket differs: old native4 supports 1.5–2.0 m/s; new native4 supports 0–0.1 m/s. The common 0.1–0.5, 0.5–1.0 and 1.0–1.5 m/s bins all have lower new-run MMD². Unsupported bins remain unknown. Route progress is lower in the new run.

The saved old native15 baseline was independently admitted with its original 174-file source archive. Its CPU metrics retain all 89985 legal intervals: primary 90D MMD² **0.09741569561683164** (5 supported bins), fixed 45D MMD² **0.028319794140056143**, K=50/100 coverage **0.76 / 0.61**, and normalized entropy **0.634283036770646 / 0.6074829195318023**. No new native15 capture was read or scored by this completion.

Each new-run calculation admitted the actual closed native owner receipt, production V11 source, deployment entry, actor-only true state-prior provenance, selected terminal evidence and complete independent raw audit before processing poses. Historical runs use their own frozen raw audit, source archive and saved artifact hashes; their old runtime source is not replaced by current V11 evidence. Fixture 934 admission is not used. Actual mathematics imports finalized with 30 owned executions for each 90D calculation and 20 for each 45D calculation.

All failed, contact and idle observations remain in the original legal same-episode populations. Only the unchanged first-frame and reset-boundary exclusions apply. The exact `c274348…` reference and `b25dffa…` saved centers were reused, with no fit. No checkpoint was deserialized or GPU/Kit started by these computations. Network shapes, initialization RNG consumption and runtime source epochs differ; these numerical comparisons do not establish a causal single-factor gain or an author-equivalent score. Default promotion remains false.
