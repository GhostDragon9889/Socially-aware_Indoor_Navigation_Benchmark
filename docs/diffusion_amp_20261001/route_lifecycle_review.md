# Route and control lifecycle review

Review date: 2026-10-01. Scope: the current `scripts/train_paper_amp.py`,
`CrowdSim/humanoid/amp_world.py`, `CrowdSim/humanoid/reconstruction_scene.py`,
and frozen diffusion output contract. There is no `validation_scene.py` in
this checkout; `world.validation_scene` is a `ReconstructionScene` instance.
No Kit, CUDA inference, or additional model training was started for this review.

## Reproduced lifecycle defects and authorized scene API fix

### 1. Arrival checked only at replanning can miss a traversed vertex

The reviewed runner updates route progress in `new_goals()` (runner lines
234–246), called during replanning (263–264), rather than after every actual
physical step. The old scene methods check only the position passed to that
query against the arrival disk. A 3 m/s path from `(5.25, 6.3)` to `(6.75, 6.3)`
over 0.5 seconds passes a vertex at `(6,6)` with radius 0.75 m. Both replanning
endpoints are outside, but 14 intermediate 30 Hz positions are inside.
The old checks therefore keep the old destination and can command a return
to an already traversed vertex.

**Fix supplied:** `advance_route(positions, route_indices, *, actor_ids=None)`
returns `(updated_indices, reached_bool)` using one actual control-frame
position, advancing at most one vertex. It performs no A*, counter accumulation,
or physical state mutation. The runner owns calling this after nonterminal
physics, and accumulating the returned mask exactly once.

### 2. Reset can count an untraversed route vertex as reached

`AMPWorld.reset()` places the root at its fixed origin (`amp_world.py:169–176`).
The reviewed runner keeps its route index and then calls `new_goals(ids)`
(`train_paper_amp.py:460–463`). Each fixture route's final vertex is that
actor's origin. With actor 0 at `(15,18)`, route index 3 refers to `(6,6)`.
The old pre-reset query reports False; querying the reset origin immediately
reports True and wraps index 3 to 0, adding a reached vertex without a physical
return journey. It also carries a partially completed route into a new physical
episode without declaring the route-attempt semantics.

**Fix supplied:** `route_goal()` and `route_waypoints()` now accept
`advance=False`. This keeps indices and returns an all-False reached mask;
planning queries no longer need to act as progress checks. The default remains
True for compatibility. The parent-owned runner integration resets the route
index to zero for a new attempt, queries goals with `advance=False`, and retains
only counts earned at actual post-step positions before reset.

**Observed versus possible:** CPU scans of the old corrected-timed capture and
the new 5000-step-model capture found **zero** observed intermediate arrival
misses under the stated 16-frame follow-up check. The new capture also had
zero reset-related route changes into a vertex near the reset root. These are
real, reproducible conditional defects; they do not invalidate the new run's
eight reported vertices.

## Control contracts checked

- **Position timing:** policy inputs use pre-step physical state; `plan_age`
  increments after `world.step()`, and the position reward compares post-step
  root XY to the interpolated position at that updated time. No one-frame
  alignment error was found in this path.
- **Interpolation:** plan time is 0–5 seconds at 0.1-second samples. Index 50
  is correctly obtained from lower=49, upper=50, alpha=1. Bounds are held at
  the final point, rather than read past the buffer.
- **Future tail:** the 52 predicted future states cover 0.1–5.2 seconds, but
  the runtime returns 50 future positions plus the current point, ending at
  5 seconds. At plan age 14/30 seconds, a task query for five seconds ahead
  requests time 5.4667 and holds the endpoint. This shortens the last preview
  interval. It is explicitly declared in provenance and used during both
  training and evaluation; it is a horizon assumption, not an indexing defect.
- **Five state channels versus XY commands:** the denoiser predicts
  `[x,y,vx,vy,yaw]`; the runner consumes only `prediction.xy`. Its PACER task
  observation is ten heading-relative XY points plus terrain
  (`amp/observations.py:132–172`), and the position tracking reward also uses
  XY. Predicted velocity/yaw are not PD velocity or facing commands. Replacing
  this observation with five-channel commands would change the controller's
  input/checkpoint contract. Soft dynamics costs appropriately use finite
  differences of the consumed XY rather than claiming that the unused
  velocity head limits physical or commanded speed.
- **A* specification:** the endpoint is at up to five metres of path arc,
  queried at t=5 seconds. Timed mode uses the same path's arc proportion at
  t=0.5…5 seconds. This is a nominal 1 m/s geometric goal schedule. A* points
  guide diffusion; they do not replace the denoiser output or move the root.
  A failed path holds the actual query position. CPU examples of free roots
  `(9.7,12)` and `(12,2.3)` fail due to occupied inflated start cells and
  produce that hold. Endpoint mode does not use timed-mode start recovery.
  This is an existing explicit failure policy that can stall near obstacles,
  not a newly introduced silent teleport or map inconsistency.
- **Resume/configuration:** current guards compare all optional collision and
  dynamics weights/radii and trajectory-source settings during training resume.
  Joint peer guidance is limited to shared evaluation and a batch containing
  the whole scene. Scene fixtures are evaluation-only, so missing route arrays
  in a training runner snapshot do not currently corrupt a supported resume.

## New completed physical capture: reference discontinuity

Read-only input:
`output/diffusion_amp/shared_obstacle_clean_sdf_5000_v2/physical_states.npz`
(6000 frames × 4 actors; result status `passed`). This capture records 11
falls and route counts `[3,2,1,2]`. The source lifecycle fixes above were
implemented afterward; the capture and its metrics were left unchanged.

The runner replaces an actor's entire XY plan at each replan (`runner:280–281`),
anchoring the new t=0 point at the current root but imposing no continuity on
the first future segment. In the actual capture, excluding episode crossings,
the diagnostic `30 * ||target[t] - target[t-1]||` has:

| Statistic | m/s |
| --- | ---: |
| All-frame median | 0.567 |
| All-frame 95th percentile | 2.071 |
| Global 15-step replanning ticks, 95th percentile | 8.930 |
| Other steps, 95th percentile | 1.404 |
| Maximum | 26.732 |

These are reference-position difference values, **not actual root velocity**.
They demonstrate that reference jumps remain substantial after better goal
guidance. They do not establish that a jump caused a particular fall. A
controller curriculum exposing replanning transitions, or an explicitly
declared continuity objective tied to actual current velocity, needs a
separate experiment. Increasing existing soft speed/acceleration weights
was not reliably beneficial in the prior CPU experiments and is not silently
recommended as a fix.

## Validation and freeze

Authorized source changes in this follow-up:

- `CrowdSim/humanoid/reconstruction_scene.py`
- `tests/test_paper_route_lifecycle.py`

Focused command:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m pytest \
tests/test_paper_route_lifecycle.py tests/test_paper_route_waypoints.py \
tests/test_paper_reconstruction_scene.py -q \
--junitxml=docs/diffusion_amp_20261001/route_lifecycle_tests.xml
```

**58 passed in 1.66 seconds.** Existing frozen endpoint bit comparisons,
default route progression, timed path/recovery, and map tests remained green.
New cases cover missed intermediate arrival, reset queries with advancement
disabled, subsets/wrap, empty queries, and explicit boolean flags.
Ruff `--target-version py312` passed for both changed files.

Scene source frozen SHA256:
`a352c5f18529c61e15703e33e44392b5bded087aacb2431658602e0aa43b1803`.
Diffusion source remained frozen at
`7d99be6fa58364b243ec5ebe551c490594068cd019e2da89d42b3f8927c235e5`.
The JSON evidence stores hashes, exact CPU reproduction values, and capture
checks. Runner integration and its focused tests remain parent-owned.

Evidence: [route_lifecycle_review.json](route_lifecycle_review.json),
[route_lifecycle_tests.xml](route_lifecycle_tests.xml).
