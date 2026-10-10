# Actual native15 style comparison

New native15 90D, fixed 45D and comparison commands completed on their first style attempts with exit code 0 and CUDA disabled. The [sealed completion receipt](native15_CPU_completion_v2.json) binds all final style, source, ledger, log and launch artifacts. The [complete comparison](../state_prior_native_metrics_v2/comparison15/comparison.json) has SHA256 `2ce702fc936542a980df93575d95ba394bdd90508973864f2d68b500ec59e75e`.

| Measurement | Selected old native15 | New state-prior native15 |
| --- | ---: | ---: |
| Runtime selected source files | 174 | 183 (V11) |
| Controls / duration / actors | 6000 / 200 s / 15 | 6000 / 200 s / 15 |
| Falls / static-contact controls / peer-pair controls | 0 / 0 / 0 | 0 / 0 / 0 |
| Route vertices / complete cycles | 211 / 49 | 160 / 32 |
| Legal style intervals retained | 89985 | 89985 |
| Inclusive primary 90D MMD² | 0.09741569561683164 | 0.06987130237860084 |
| Supported speed bins | 5 | 4 |
| Secondary mean over the 4 common supported bins | 0.09969935635241661 | 0.06987130237860084 |
| Fixed 45D pose MMD² | 0.028319794140056143 | 0.022732354539911404 |
| K=50 coverage / normalized entropy | 0.76 / 0.634283036770646 | 0.70 / 0.7123737071252526 |
| K=100 coverage / normalized entropy | 0.61 / 0.6074829195318023 | 0.61 / 0.7095008984600282 |

The new run has lower MMD² in each common supported speed bin. All seven original speed buckets remain in both reports. New-run 1.5–2.0 m/s has only 21 samples, so its MMD² remains unknown; the old run supports that bucket. The primary aggregates are not replaced by the common-bin comparison. Pose occupancy coverage falls at K=50 and stays equal at K=100; normalized entropy rises. Route progress is lower.

The first native15 raw CPU audit failed because four proof tuples became lists in registered JSON. That original failure remains sealed. A separate, approved JSON representation revision produced the complete raw audit (`66f16e6…`, 1241 checks), with unchanged compiled raw mathematics. This style revision requires the exact representation adapter, outer registration, immutable v1 mathematics freeze, first-failure record and canonical proof values. It also retains the original actual native source/closed-owner/entry/true state-prior admission. Old metrics and frozen v1 preparation were not changed.

Both style metrics retain all 89985 legal same-episode intervals, including failed/contact/idle observations. The unchanged helper excludes the first 15 actor samples; there are no cross-reset exclusions. The exact c274 validation reference and b25 saved centers are reused without fitting. CPU import ledgers finalized with 30 owned executions for 90D and 20 for 45D. No checkpoint was deserialized, GPU/Kit started or sealed 34-test output read by these computations.

Runtime epochs, network shapes and initialization RNG consumption differ. The reported numerical comparisons do not establish a causal single-factor gain, an author-equivalent score or automatic promotion. Default promotion remains false.
