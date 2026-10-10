# Four-actor mixed layout and group reset masks

Independent **NumPy layout/mask algorithm only**. No SDK, NN, weights, GPU, real capture or production185 code is used. Shared physics, collision filtering, runner wiring, group epochs, actual reset and curriculum integration are **not implemented or run**. Passing these checks does not establish a shared environment.

## Interface

`make_mixed_layout(N)` accepts a Python integer N>=4 divisible by4 and returns int64 arrays:

- `group_actor_rows[N/4,4]`: group g is `[2g,2g+1,N/2+2g,N/2+2g+1]`.
- `scene_ids[N]`: group row number assigned to each canonical actor row.
- `actor_ids[N]`: stable initial canonical IDs0..N-1. A future caller that permutes actor storage must preserve these identities and remap row indices, not regenerate IDs.

First N/2 rows remain AMASS, second N/2 remain procedural; each group contains exactly2+2. For N=256 this is64 groups of4 **logical actor rows**, not a claim that IsaacLab now creates64 four-person environments.

`expand_group_reset(group_actor_rows, terminated, truncated)` accepts a complete explicit partition `[N/4,4]` and exact bool masks `[N]`. Group members need not occupy contiguous or numerically ordered rows. Duplicate, missing/out-of-range, noninteger or non-four-member groups are rejected. Layout generation fixes the mixed2+2 assignment; the reset function itself accepts any valid explicit four-member partition.

It returns fresh arrays:

| Field | Meaning |
|---|---|
| `terminated[N]` | Original physical terminal flags, unchanged. |
| `truncated[N]` | Original truncations OR collateral reset members. An original simultaneous terminated+truncated flag is preserved. |
| `reset_mask[N]` | Every member of each group having any original terminated/truncated member. |
| `collateral_reset[N]` | Group-reset members that had neither original flag. These become truncated; they are not labelled falls or time limits. |
| `triggered_groups[K]` | Sorted int64 row indices of the supplied group matrix, not inferred actor-row ranges. |

No input is mutated. The helpers draw no random numbers and do not call `mixed.completed_step`, a planner, simulator, reset function or GAE.

## Future runner boundary

The existing GAE in `CrowdSim/humanoid/training/contracts.py:57–75` suppresses bootstrap for terminated rows, and stops carry across both termination and truncation. Collateral group truncations therefore retain terminal-observation bootstrap but end their old episode's carry. A future runner must compute these masks **before collecting the transition**, obtain next values on pre-reset terminal observations, and only then reset the complete group. It must keep group-reset counts distinct from real fall/cap/clip-end counts.

This file does not update actor or group episode IDs, AMP histories, reference plans, clocks, reset poses, source/RNG state or any curriculum statistics. Those are necessary future integration work. No source sampling, exposure equality beyond static layout, collision avoidance or physical effect is claimed.

## Bounded checks

Five synthetic NumPy cases cover N=4/8/256 layout, one fall with three collateral truncations and scalar GAE meaning, clip-end/no-reset/multiple groups, irregular explicit row membership and immutable inputs, and invalid masks/partitions. They do not test or change the original GAE implementation.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_group_reset_staging_v1 -p test_group_reset.py -v
```

Actual CPU check: tool chunk `961e16`, exit code0, all5 tests passed; the unittest runner reported **0.020s**. This was a synthetic NumPy/interface check, not a simulator or training run. Ruff checked `group_reset.py` and `test_group_reset.py`: tool chunk `551e53`, exit code0, all checks passed; elapsed time was not recorded. No checks were rerun for this documentation update.

Independent source review by `paper_method_next_iteration` returned **Ready / Required=[]** after reading these three files and the original GAE contract. Root also read the source/tests (`f58e64`) and found no source defect. This review adds no claim of physical integration or successful shared-group training.

Root whole-source review found no required changes. Primitive/test SHA identities were verified and set readonly in actual tool339616; README remains an updateable status page. Old source, training defaults, q937 and the formal40k protocol are unchanged.
