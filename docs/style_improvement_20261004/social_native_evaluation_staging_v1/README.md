# Social actor in the original single native scene

This is **unexecuted source preparation** for 3–15 people. Only this new
directory is changed. The frozen 2379 producer, c31 four-person entry and
production185 remain unchanged. No candidate module, CLI, preflight, Torch,
NN, SDK, dataset, checkpoint or test has been imported/executed for this stage.
Source compilation and static AST/literal derivation passed in tool chunk
`4cb910` (exit0); Ruff passed in chunk `25dd47` (exit0). These checks did not
import or execute a candidate module/function. Root and independent source review passed with Required=[]/Optional=[].
Actual serial validation remains pending.

## Producer boundary

`native_producer.RawSocialNativeEvaluationObservations.from_world` accepts:

```python
producer = RawSocialNativeEvaluationObservations.from_world(
    world, actor_ids, scene_ids,
    native_request=actual_original_preflight_request,
    terrain_sampler=world.height_at,
    terrain_sampler_semantics=explicit_native_proxy_description,
    policy_description=actual_frozen_social_actor.describe(),
    grid_center="root", grid_heading="root",
    height_reference="static_ground", include_ego=False,
)
producer.begin_episode()  # After the world's actual initial reset.
packet = producer.observe(actual_world_future_xy)  # Tensor[N,10,2], +0.5..5s.
# After the original driver actually resets some rows:
producer.begin_episode(actual_reset_rows)
```

It reads the immutable 17726B original producer, SHA
`2379d65876775173fbdd111d846fd847beb6b30109ecb7e04aa30f7a569064f7`,
and replaces **one** exact lifecycle fragment. The replacement requires N3..15,
shared=True, 30Hz, a native provider and the same provider on validation_scene.
Full bytes must reverse to 2379. All other original methods retain exact AST,
including observe, explicit episode notifications, state ownership, original
self357, task12308, actual quaternion unit/fall/heading handling and clocks.
It continues the original per-actor lifecycle; b608 training World/group resets
are not used. The original producer's existing N4 gate already permits native
N4; N3/N15 require the new explicit native boundary.

The backend's `__file__` is the actual new adapter file, so its initial
`_EXECUTION_RECORD` identifies executed adapter bytes rather than pretending
the derived body is unchanged 2379. The facade's descriptor and each original
packet's metadata name the original source, derived SHA, new execution source
and native bindings. It delegates the intact methods and only augments describe.

NativeGridScene and DynBenchSceneProvider must be the actual source-pinned
classes. The selected configuration file must equal the request's configuration,
and the preflight/map/snapshot bytes remain bound. Actual provider/map/query,
single context/stage observation, N reset origins and exactly N dynamic robot
roots `/World/envs/env_i/Robot` must match. Actor IDs are unique stable IDs;
all actors have one explicitly shared scene ID. Runtime origins must equal
the registered absolute world coordinates and actual NativeGridScene.origins.
No additional scene.env_origins translation is applied. The facade rejects
provider/scene/origin replacement, and uses the original clock guards around
its metadata checks. Failure faults the producer; it never retries physics.

The sampler must be the original bound `world.height_at`. In native mode it is
the admitted static-grid **absolute floor or floor+1.8m** feature: blocked,
unknown and map-exterior cells use the cap. It is not measured obstacle tops,
an exterior support guarantee or body geometry. The adapter adds no second
1.8m band and does not substitute `native_surface_height_at`. Original 6975/234e
task/raster code adds the existing occupant1.7m once and retains its scaling,
velocity frame, ties and same-scene ownership. The height reference/fall query
uses original ground_height_at: actual current floor-only rays, excluding the
exact dynamic robot roots. Its finite-floor exterior failure remains intact.

## Native entry boundary

`evaluate_native_social.py` binds the complete immutable 27829B c31 parent,
SHA `c31c87391ab62c2c33369fc437c6b8967546987a505e885b7954ba9349107d8e`.
Nine uniquely reversible literal edits cover the native CLI/resources/argv,
producer admission, native factory invocation and final resource identity;
the remaining edits identify the actual new source/profile/report namespace.
These are six functional seams plus explicit source/profile/report identities,
not six literal replacements. The final static check retained ten original
producer methods and six original entry helper ASTs, and reversed both derived
sources to their complete pinned parents.
Every other helper AST is checked, including the complete actual-training/USD
gate and the parent's original physical-driver derivation. The latter still
performs its original ten changes on production driver d8b854; this stage does
not copy a new controller/planner/contact/reference/reward/cleanup algorithm.

The original cold `preflight_native_scene` and lazy `make_native_scene_factory`
are reused. The original `driver_arguments` supplies N actors, seed7, guide20,
one Diffusion sample, alpha0, peer5/radius0.6, static10/radius0.8, replan15,
causal reference speed1.5m/s/acceleration2m/s², and zero trajectory speed/accel
guidance weights. Native planner batch=max(8,N) includes the full N<=15 scene
in one joint-guidance call. The native factory is passed directly to the
original driver's native_scene_factory parameter after the caller's one App;
the original AMPWorld uses NativeScenePowerSimulator and shared collision
configuration, not the four-articulation training World. Original routes,
partial resets, contacts, four120Hz substeps/control, task metric and captures
remain untouched. Native sources/resources and selected artifacts are checked
before/after execution. App/partial-World cleanup and failed-status exit remain
the c31 implementation.

The unchanged cd227 loader must strictly admit the selected new social
checkpoint. The unchanged c31 `_training_run` additionally requires a completed
actual shared-training result, configuration equality via canonical JSON,
matching model counters, source/profile stability, cleanup and final model
inventory. **Four** actual training USD prototypes must match the actual native
evaluation `_robot.cfg.spawn.usd_path` bytes/SHA. Four refers to training
prototypes, not the number of native people, and is not changed to N. A mismatch
fails; no USD equivalence is guessed. The retained actor supports arbitrary B,
while saved actor_count remains the training count. Grid/root/static-ground/
include-ego=False must match its descriptor; no task/RMS conversion is made.
No old native raw audit or legacy1044 checkpoint certifies this new policy.

## Future invocation and limitations

Root sealed the reviewed sources/profile/README readonly. Selected weights and
training-result SHA are unknown; no future artifact/pass record is generated.

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/social_native_evaluation_staging_v1/evaluate_native_social.py \
  --actors 15 \
  --scene-config ACTUAL_REGISTERED_NATIVE_SCENE_CONFIG \
  --scene-sha256 ACTUAL_NATIVE_CONFIG_SHA \
  --amp-checkpoint ACTUAL_IMMUTABLE_FINAL_SOCIAL_MODEL \
  --amp-sha256 ACTUAL_MODEL_SHA \
  --training-result ACTUAL_COMPLETED_SHARED_TRAINING_RESULT \
  --training-result-sha256 ACTUAL_RESULT_SHA \
  --diffusion-checkpoint ACTUAL_SELECTED_IMMUTABLE_DIFFUSION_MODEL \
  --diffusion-sha256 ACTUAL_DIFFUSION_SHA \
  --control-steps 6000 \
  --output output/diffusion_amp/social_CNN_native15_200s_v1
```

`--preflight-only` remains source/resource inspection; it cannot validate model
tensors, current SDK geometry, new-model navigation or performance. Controls4
or30 are bounded smokes, not the6000/200s comparison. All failed/idle/reset rows
are retained by the original driver; neither MMD nor routes filter a run.
Future execution requires the root's serial scheduling after formal50502,
terminal audit, primary comparison and the necessary actual social training /
SDK gates. This code does not attest completed SDK validation, author benchmark
equivalence, complete old-run bit identity, physical success or promotion.

## Root review and seal

Root whole-source reads739cd6/7bf944/2f716e and relevant original interfaces b8dff0/ba6953 passed; final pinned compile/JSON check1af564 exited0 without importing or executing candidates. Independent whole-source review and static derivation31a60e exited0, Required=[]/Optional=[]. Root confirmed ordinary/nlink1 files and set0444. Producer13020B SHA `7d2a51563f59811a6c6d16d82f880e99accaffbfecc3074e44c48bad64666621`; entry14326B SHA `be9f4126dd9ff26ce6f1f5ebc7ab0da99e012dd2cb9298a34eb326bf7ec25e6d`. No native CLI/NN/SDK/model/data run is claimed. Formal50502 has separately actually closed0 in f10f35; the primary comparison and all actual new-policy gates still precede this future entry.
