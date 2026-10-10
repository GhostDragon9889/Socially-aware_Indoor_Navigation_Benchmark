# Actual q tracking CPU metrics

Both registered commands exited 0. All 22 motions, 11,982 metric frames and 11,960 legal style intervals were retained. Failed motions 20 and 257 contribute all 841 metric frames and 839 legal style intervals. No falls occurred.

| Saved run | Tracking success | Position error (m) | Rotation error (rad) | Inclusive 90D MMD² |
|---|---:|---:|---:|---:|
| Selected default 62c4 | 11/22 | 0.300708008 | 0.861454446 | 0.046821061261 |
| Public prior 19f / reward 2 | 17/22 | 0.209859460 | 0.810942963 | 0.042092339591 |
| Reward 0.5 / 09e | 15/22 | 0.251093953 | 0.891656786 | 0.110079945476 |
| Stratified q / reward 2 | 20/22 | 0.214709068 | 0.782872638 | 0.041569629732 |

q has 20 successes versus 17 for 19f. Its position error is slightly higher, rotation error lower, and aggregate MMD² slightly lower. This single seed does not establish an overall style gain. All four saved tracking reports support the same first five speed buckets.

| Speed bucket (m/s) | q generated | Reference | Default | 19f | Reward 0.5 | q MMD² |
|---|---:|---:|---:|---:|---:|---:|
| [0.0, 0.1) | 291 | 148 | 0.051339401 | 0.069109639 | 0.205521799 | 0.041025144 |
| [0.1, 0.5) | 1235 | 2251 | 0.087036412 | 0.080869815 | 0.127282551 | 0.099228406 |
| [0.5, 1.0) | 6025 | 6664 | 0.016421447 | 0.013908124 | 0.050139880 | 0.015948324 |
| [1.0, 1.5) | 3999 | 5324 | 0.023121926 | 0.017932584 | 0.061628692 | 0.012763659 |
| [1.5, 2.0) | 407 | 969 | 0.056186120 | 0.028641536 | 0.105826805 | 0.038882614 |
| [2.0, 3.0) | 3 | 4 | unknown | unknown | unknown | unknown |
| [3.0, ∞) | 0 | 0 | unknown | unknown | unknown | unknown |

All samples, failure frames and unsupported buckets remain in the saved reports. Fixed c274 reference, original 90D mathematics and original same-frame decomposition were reused. CPU ledgers finalized with 25 and 3 owned executions; no GPU/SDK or sealed test34 outputs were used.

Root GPU tool-session closure and readonly seal, actual terminal complete-payload audit, new935 and original metric sources are bound in both CPU receipts. One receipt assembly failed because it assumed Popen closure keys in a tool-session GPU receipt; that metadata failure is preserved. The successful metric computation was not rerun. No frozen source or metric mathematics changed.
