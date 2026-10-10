# Raw independent AMP: physical route completion

CPU analysis of the completed `shared_independent_amp_raw_v1` capture:
6000 frames at 30 Hz × four actors = 200 seconds of shared physics.
No model, CUDA, or simulation was started by the analysis.

## Verified results

The 12 recorded vertex-arrival events match `result.json` exactly:
`[1,3,4,4]`. Post-arrival route indices are continuous, restart at zero in
new episodes, and produce **two complete four-vertex cycles within individual
episodes**. Counts from different attempts were not combined.

| Actor | Episode | Actual arrivals | Complete cycles | Episode outcome in capture |
| --- | ---: | ---: | ---: | --- |
| 0 | 1 | 0 | 0 | Fell at 66.60 s |
| 0 | 2 | 1 | 0 | Right censored at 200 s |
| 1 | 1 | 3 | 0 | Fell at 102.17 s |
| 1 | 2 | 0 | 0 | Right censored at 200 s |
| 2 | 1 | 4 | 1 | Cycle completed at 188.47 s; unfallen through capture |
| 3 | 1 | 4 | 1 | Cycle completed at 86.13 s; later fell at 156.60 s |
| 3 | 2 | 0 | 0 | Right censored at 200 s |

Actor 3's completed traversal is real; its later fall prevents calling this
a 200-second episode without falls. Four final episodes are right censored,
so their absence of recorded termination is not an episode-success label.

Independent checks against the physical root and captured pre-arrival route
vertex show distances **0.739804–0.749859 m**, all within the declared
0.75 m arrival radius. All captured vertices match the scene's corresponding
route vertices. There are **zero arrivals on the first post-reset frame,
zero arrivals on a fallen frame, and zero index/reset continuity issues**.
Displacement between episodes is excluded from walked distance.

## Preserved comparison and source verification

Old `shared_obstacle_clean_sdf_5000_v2` reported cumulative arrivals
`[3,2,1,2]` (eight), but did not capture `route_vertex_reached` or
`route_indices`. Its per-episode arrivals and complete-cycle counts are
**null**, rather than reconstructed from cumulative counts or assumed to be
zero. This comparison also changes the controller and route counting semantics;
it does not isolate a single cause of improvement or establish paper benchmark
performance.

The preserved raw-run source archive matches all **49** manifest entries by
size and SHA256. The verification only reads tar members; it does not extract
or execute them. The live runner and configuration have since changed for the
next experiment, so their current hashes are recorded separately and are not
substituted for the past run's source hashes.

Artifacts:

- [New event/episode analysis](../../output/diffusion_amp/shared_independent_amp_raw_v1/route_completion_analysis.json)
- [New readable route summary](../../output/diffusion_amp/shared_independent_amp_raw_v1/route_completion_analysis.md)
- [Independent geometry and archive verification](raw_route_completion_verification.json)
- [Old V2 unavailable metrics](legacy_v2_route_completion.json)
- [Preserved source manifest](shared_independent_amp_raw_source.json)
- [Analyzer regression results](route_completion_analysis_tests.xml): 10 CPU tests passed; Ruff passed.

## Re-run for the next capture

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
docs/amp_improvement_20261001/analyze_route_completion.py \
--run-dir output/diffusion_amp/shared_independent_amp_raw_v1
```

Change only `--run-dir` for the next completed capture. The script verifies
the runner status before reading its route events and writes a separate
`route_completion_analysis.json` and `.md` alongside the capture.
