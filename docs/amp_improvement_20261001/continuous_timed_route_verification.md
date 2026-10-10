# Continuous timed deployment: verified physical routes

Input: `output/diffusion_amp/shared_independent_amp_continuous_timed_v1`.
The completed capture contains 6000 frames at 30 Hz for four actors: 200 seconds
of shared physics. This analysis used CPU arrays only, without loading models,
CUDA, or the SDK.

## Actual arrivals and complete cycles

All **26** recorded arrivals agree with `result.json`: **[8,6,6,6]**.
Ordered four-vertex cycles within individual episodes are **[2,1,1,1]**,
for **five complete cycles**. There are no index-event or reset-boundary
consistency errors.

| Actor | Episode | Actual arrivals | Full cycles | Completion times (s) | XY distance (m) |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 1 | 8 | 2 | 86.37, 179.23 | 147.871 |
| 1 | 1 | 6 | 1 | 145.07 | 129.930 |
| 2 | 1 | 6 | 1 | 107.87 | 124.263 |
| 3 | 1 | 6 | 1 | 109.60 | 127.537 |

The capture contains **zero falls and zero time-limit terminations**. All four
episodes remain right censored at the capture boundary: this is verified
200-second exposure without recorded falls, rather than an inferred terminal
episode-success flag.

Independent physical-root checks validate every arrival against its captured
pre-arrival vertex and the declared scene route. Distances are
**0.732847–0.748502 m**, within the **0.75 m** arrival radius. No arrival occurs
on a first post-reset frame or fallen frame. Reset jumps are excluded from
distance; no resets beyond initial episode initialization occurred in this run.

## Fixed weights, explicit deployment changes

The controller and diffusion paths match the raw independent-AMP evaluation.
Current immutable checkpoint bytes were hashed independently:

- Controller SHA256:
  `e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`.
- Diffusion step 5000 SHA256:
  `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`.

Declared deployment uses timed static A* waypoints, 0.5 m start-recovery radius,
0.8 m scene-guidance radius, 0.6 m peer-guidance radius, and the causal reference
adapter configured for 1.5 m/s speed and 2 m/s² acceleration. These settings
jointly differ from raw deployment. This run does not isolate the effect of
any one change. The limits describe the **reference adapter**, not hard bounds
on actual physical root speed or acceleration.

The preserved source tarball matches all **49** manifest entries by byte size
and SHA256, without extraction or execution. Later live changes to the runner
and CLI configuration belong to the two-sample rendered demo. The past run
continues to reference its archived source, including its absence of the later
`selected_sample` recording field.

## Scope

This is one newly authored four-actor obstacle fixture. It does not establish
NavIsaacLab's eight-scene benchmark result or recover the unpublished author
dataset/configuration. The short two-sample demo is a separate experiment;
its recorded fall is preserved in its own report.

Evidence:

- [Episode/event analysis](../../output/diffusion_amp/shared_independent_amp_continuous_timed_v1/route_completion_analysis.json)
- [Readable route summary](../../output/diffusion_amp/shared_independent_amp_continuous_timed_v1/route_completion_analysis.md)
- [Independent geometry, checkpoint and source verification](continuous_timed_route_verification.json)
- [Preserved source manifest](shared_continuous_timed_source.json)
- [Raw deployment verification](raw_route_completion_verification.md)
- [Separate two-sample interface review](diffusion_samples_interface_review.md)

The existing [CPU analyzer](analyze_route_completion.py) can re-run this audit
using `--run-dir output/diffusion_amp/shared_independent_amp_continuous_timed_v1`.
