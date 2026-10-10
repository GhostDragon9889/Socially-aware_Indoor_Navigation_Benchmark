# Actual state-prior navigation CPU metrics

All three frozen commands (`navigation_raw`, `navigation90`, `navigation45`) completed on their first attempt with exit code 0 and CUDA disabled. The sealed [completion receipt](navigation_CPU_completion_v1.json) has SHA256 `e63423d4473cd1d55652d46199727c9865995f84a122cf355d1151d2b0bbe292`; the [independent navigation review](../state_prior_navigation_peer_review_v1/actual_navigation_review.json) passed with SHA256 `a05e8dbfc962138a946fe6bc9a01190545f06033fdd039d05280bd97a392519b`.

| Measurement | Actual result |
| --- | --- |
| Completed controls / duration | 6000 / 200 s, 4 actors |
| Falls / static and peer contact actor frames | 0 / 0 / 0 |
| Route vertices reached | 10 per actor; 40 total |
| Original raw consistency checks | 203 passed |
| Same-episode style intervals | 23996; initial 4 actor samples excluded |
| Idle intervals retained | 4; 0.13333333333333333 actor-seconds |
| Inclusive 90D MMD² | 0.06811695149147463 |
| Registered maximum 90D MMD² | 0.07238060315284697; threshold passed |
| Fixed-basis 45D pose MMD² | 0.04696745985114492 |
| K=50 occupancy / coverage / normalized entropy | 29 / 0.58 / 0.5375610946534968 |
| K=100 occupancy / coverage / normalized entropy | 52 / 0.52 / 0.5808928402928111 |

The primary 90D calculation retains every same-episode interval, including the four idle intervals. Its three supported speed bins are 0.1–0.5, 0.5–1.0 and 1.0–1.5 m/s. The remaining four bins retain unknown (`null`) MMD values. The 45D calculation uses the exact saved `b25dffa…` centers and `c274348…` reference; neither reference nor generated poses were fitted again.

The selected baseline supports five speed bins. On the three common supported bins, its secondary equal-bin mean is **0.04098256102946459**, compared with **0.06811695149147463** for this trial. The selected baseline has lower MMD² in each common bin. This secondary comparison leaves the registered primary aggregate unchanged and does not establish an overall style gain.

Each CPU calculation admitted the actual completed navigation result, registered terminal reader report (`ed47b114…`), immutable 934-file training source (`3db16db2…`) and observed GPU source records before processing the capture. The CPU import ledgers finalized successfully with 0, 30 and 20 owned executions for raw, 90D and 45D respectively. The frozen command catalog (`506d1066…`), executor and mathematical tools were unchanged. No GPU or Kit was started by these calculations; sealed 34-test outputs were not read. Default promotion remains false.
