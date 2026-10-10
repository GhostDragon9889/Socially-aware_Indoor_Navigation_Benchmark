# Actual q navigation CPU metrics

All three original commands exited 0. The complete capture is 6,000 controls / 200 s / 4 actors: zero falls, time limits, static contacts and peer contacts; routes [8, 8, 8, 8] total 32 arrivals. All 23,996 legal intervals are retained. Idle intervals are [3, 20, 187, 38], 248 total / 8.266666667 actor-seconds; no idle/failure/upright filter was added.

The registered full-6000 physical zero-fall/static/peer condition is observed. MMD was not used for this condition; root owns any native launch decision.

| Saved run | Primary 90D MMD² | Supported bins | Common-three secondary MMD² | 45D MMD² | K50 / K100 coverage |
|---|---:|---:|---:|---:|---:|
| Selected default 62c4 | 0.072380603153 | 5 | 0.040982561029 | 0.020156771502 | 0.86 / 0.72 |
| Prior 19f | 0.068116951491 | 3 | 0.068116951491 | 0.046967459851 | 0.58 / 0.52 |
| Reward 0.5 / 09e | 0.090404587220 | 5 | 0.063998614692 | 0.030503855383 | 0.82 / 0.67 |
| Stratified q / 98a | 0.079921039798 | 4 | 0.067512188131 | 0.034606695778 | 0.58 / 0.54 |

Primary supported populations differ (5 / 3 / 5 / 4); these whole aggregates are not directly ranked. The common-three secondary preserves the original equal-bin arithmetic and uses only existing saved MMDs from [0.1, 0.5), [0.5, 1.0), [1.0, 1.5). No re-sampling/refit/missing-bin zero. q common-three is better than 19f but worse than the selected default; no overall style gain or promotion is claimed.

| Speed bucket (m/s) | q generated / reference | Default | 19f | Reward 0.5 | q MMD² |
|---|---:|---:|---:|---:|---:|
| [0.0, 0.1) | 248 / 148 | 0.162036646 | unknown | 0.153941019 | 0.117147595 |
| [0.1, 0.5) | 3719 / 2251 | 0.045007834 | 0.099097514 | 0.097979391 | 0.116291770 |
| [0.5, 1.0) | 16479 / 6664 | 0.026610069 | 0.052449066 | 0.033691576 | 0.023770432 |
| [1.0, 1.5) | 3529 / 5324 | 0.051329780 | 0.052804274 | 0.060324877 | 0.062474363 |
| [1.5, 2.0) | 21 / 969 | 0.076918687 | unknown | 0.106086073 | unknown |
| [2.0, 3.0) | 0 / 4 | unknown | unknown | unknown | unknown |
| [3.0, ∞) | 0 / 0 | unknown | unknown | unknown | unknown |

45D fixed b25 AMASS centers and c274 held-out reference were reused; no fitting was performed. All failed/idle/reset data remain saved; only the original first-frame and cross-episode exclusions apply. Both original sources and actual new935/terminal/GPU tool closure/readonly seal are bound in receipts and sidecars. CPU ledgers finalized, with no CUDA/SDK or sealed test34 output use.
