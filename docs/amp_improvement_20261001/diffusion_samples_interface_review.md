# Multiple diffusion samples: interface review

Read-only review of the current CLI, runner, and frozen planner, followed by
CPU inspection of actual completed two-sample records. No business source was
changed and no model/SDK process was launched by this review.

## CLI and returned field

No blocking interface defect was found:

- `--diffusion-samples` is an integer, defaults to one, accepts 1–20, and permits
  multiple samples only as an explicit evaluation deployment setting.
- CPU parser checks accepted evaluation with two samples and rejected zero,
  21, and multiple samples during training. A fresh uncreated output path was
  supplied so existing evaluation artifacts remained protected.
- The runner forwards `args.diffusion_samples` as `num_samples` to the planner
  and records it as `runtime_execution.diffusion_samples_per_actor`.
- `PaperDiffusionPlan.selected_sample` is the actual selected index tensor of
  shape `[B]`. It selects the same candidate used to construct returned XY.
- With joint peer guidance, all actors in a scene select one coherent joint
  future. The peer-cost helper includes only the upper triangle, avoiding
  duplicate pair penalties in selection. The adapter records the selected
  **raw** diffusion candidate alongside the corresponding actor ID and control
  step, before constructing the separate executed reference.

## Actual two-sample rendered capture

Run: `output/diffusion_amp/independent_amp_diffusion_render_s2_v1`.
Its **final** record has **900 frames / 30 seconds**, rather than the earlier
25-second progress snapshot. It contains **one fall (actor 2)** and arrivals
**[1,0,0,0]**, matching `result.json`; complete cycles are **[0,0,0,0]**.

Recorded selection checks:

| Field/check | Actual result |
| --- | --- |
| CLI and runtime sample counts | Both 2 |
| `selected_sample` | int64 `[248]` |
| Paired `actor_ids` | `[248]` |
| Joint planning batches | 62 |
| Four distinct scene actors in each batch | All 62 |
| One coherent sample index for each scene batch | All 62 |
| Selected indices within `[0,2)` | All 248 |
| Joint futures selecting candidate 0 / 1 | 27 / 35 |
| Raw diffusion XY | finite `[248,51,2]` |
| Adapted reference XY | finite `[248,151,2]` |

Correct forwarding and sample recording do not establish improved navigation
or robustness from additional samples. The one fall remains part of this short
demonstration's result. This experiment is separate from the prior 200-second
single-sample timed/causal evaluation with zero falls.

## Historical field availability

The earlier continuous 200-second capture's archived runner predates the
`--diffusion-samples` forwarding and `selected_sample` recorder. Its reference
archive therefore lacks `selected_sample`; that field's verification remains
**null**, rather than fabricated from the newer source or the default sample
count. Its route fields are present and were independently verified.

Evidence:

- [Actual sample records and result hashes](diffusion_samples_interface_review.json)
- [Short-demo route event analysis](../../output/diffusion_amp/independent_amp_diffusion_render_s2_v1/route_completion_analysis.json)
- [Long evaluation verification](continuous_timed_route_verification.json)

The frozen `paper.py` source remained unchanged during this review.
