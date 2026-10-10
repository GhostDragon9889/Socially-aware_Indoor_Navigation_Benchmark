# Same-bank candidate selection entry

Prepared source only. No preflight resources, weights, numerical callback, NN,
SDK, GPU or future captures were opened by this preparation.

## Executable entry and boundary

`evaluate_candidates.py --selection-domain {matched_raw50,reachable50}` takes
the original `simulate_paper_pedestrians` arguments after that option. It
requires the four-person validated fixture, 200 seconds, seed7, inference-only,
alpha0 and no donor/native scene. Append `--preflight-only` for the original
SDK-free resource/opaque checkpoint checks. This does not validate model
tensors or physical success.

The original deployment materializes its complete driver argv. The entry changes
only `--diffusion-samples 1` to `10`, verifies every other parsed field, and calls
the original resource function for the actual S10 request. It records the entry,
runtime, isolated runner, unchanged parent and selection-helper source records.
No selection flag is added to the original driver parser. The selector travels
as the isolated runner's keyword-only `reference_selector` argument.

The actual branch delegates exactly once through
`deployment.invoke_existing_driver`. Its actor/native factory kwargs pass
unchanged into the isolated runner, which owns the sole App/world. The entry
has no control loop or second App. Original SDK fast-exit behavior can bypass
Python cleanup; preflight and this source preparation do not assert actual
cleanup or a closed process.

## Callback and timing

`ReferenceSelector(mode).describe()` uses only the standard library.
`__call__(prediction, inputs, adapter, previous_xy, previous_velocity,
scene_ids, guide, waypoint_times, kwargs)` imports Torch/helper only when the
real runner calls it after its single App has started.

The root runner inserts this callback after its original previous-reference
state selection and reset initialization, before writing the new reference.
Ordinary replans use the last executed reference XY/velocity; only real
initialization/reset uses physical root XY/velocity. Replan remains before the
next `world.step`; history, AMP actions, control clocks, contacts and PD/physics
are unchanged. Positive peer weight promotes each due replan to all four actors,
and the existing planner chunk size includes all four.

The original `paper.plan` is called once. Its `sample_states` is already
denormalized **LOCAL physical** `[4,10,52,5]`; it must not be denormalized again.
`_world_prediction_xy` rotates the first50 local XY nodes and adds current world
XY. Prepending actual current XY gives raw `[4,10,51,2]`. Both arms build every
candidate with identical previous reference states; only the scoring geometry
is raw-dense or adapted-dense. Goal times retain the original .5..5.0s indices;
peer/static costs use exactly the common50 future .1..5.0s nodes. Original
generation clean guidance still has52 future nodes. Thus both arms differ from
historical52-node selection, and their mutual difference is the scoring domain.

The frozen `PaperDiffusionPlan` is replaced consistently: selected RAW51 XY,
selected local50 yaw transformed/wrapped to world, selected local velocity
rotated to world with current velocity prepended, its scalar speed, and joint
selected sample. Its full `sample_states`, time and reference remain unchanged.
The returned `DenseReferencePlan` selects the original adapter's computed
candidate geometry/velocity/flags/deviation; it is not adapted a second time.
There is no second NN call, sample draw, teleport or fallback.

## Candidate archive semantics

The callback returns exactly the nine tensor keys requested by the runner:
raw51, adapted151, adapted velocity151, goal/static/pair/joint costs, world
waypoints and scene IDs. All ten candidates, including unsafe/unselected ones,
are retained per actor using the original archive append flow. Goal/static/pair
costs are **unweighted**. Pair cost assigns half each original upper-triangle
pair contribution to each endpoint, so summing actors recovers the original
unordered scene cost. Joint cost already includes weights20/5/10 and repeats
the scene total on each actor row; those repeated rows must not be summed again.

Equal initial seeds do not promise identical later physical inputs or candidates
after the arms choose different references, fall/reset or reach route vertices
at different times. Report actual candidates and complete runs; do not infer
bitwise physical pairing or a safety guarantee from this source.
