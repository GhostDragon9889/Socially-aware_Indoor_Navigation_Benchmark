# Pending eight-actor, two-group PhysX smoke

`smoke_group_physics.py` is an independent future entry. Only source compilation
and Ruff have run. No App, SDK scene, physics step, tensor operation, policy,
checkpoint or contact probe has been run by this stage. No result/raw archive
exists or is claimed. Root must schedule its sole actual GPU execution after
formal training, the registered main evaluations and fixture work.

Future command (not executed):

```sh
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/group_articulation_physics_smoke_staging_v1/smoke_group_physics.py \
  --num-actors 8 --control-steps 8 --device cuda:0 --seed 7 \
  --output output/diffusion_amp/group_SMPL8_2env_PhysX_smoke_v1
```

The output directory must be absent or empty. Actual main invocation alone can
create `result.json` and NPZ files. It never loads NN weights or an AMP actor.
Current production185, SMPL files and all sealed training sources stay unchanged.

## One App and native config path

Before launch, this entry imports only stdlib. It uses the existing
`import_simulator_before_torch("isaaclab")` bootstrap and exactly one AppLauncher
call, with headless rendering/cameras disabled and `fast_shutdown=False`.
All physics, Torch, Warp and asset modules are imported after App creation.
The new entry has no `os._exit`/original training driver fast-exit path. It records
whether `sim.stop()` and `app.close()` actually returned; results are written
after those calls. An abrupt SDK/process exit may leave no result, and only the
root tool can establish the actual process exit.

`SceneCfg` reuses current ProtoMotions' SMPL MJCF-to-USD converter, named
articulation root, actuators and existing PD settings; no new asset is authored.
`robot.contact_bodies=None` avoids constructing stale single-Robot sensor cfgs;
the copied single-USD template's real `spawn.activate_contact_sensors=True`
authors ContactReportAPI before spawn. This additional instrumentation is not
a change to the production SMPL source. The grouped factory (`88e5a8d2…`) and
actor-row bridge (`2f5dc721…`) are pinned before import. The scene has four named
articulations with two native instances each, exactly8 policy actor rows.

The ground is a locally spawned static Cuboid,64×64×0.2m, topZ0 and global
collision_group−1; no remote ground USD, HDR or download is introduced. It spans
negative and positive clone origins. A single simulation context runs120Hz,
with four steps per30Hz control and `step(render=False)`. Absolute SDK step
counters before/after each phase must differ by exactly32 for the default8
controls. Startup counters are recorded as observed, never assumed to be0.

## Positions, state frames and canonical rows

The factory's four local root offsets are (−1.5,−1.5), (−1.5,+1.5), (+1.5,−1.5),
(+1.5,+1.5). Native `default_root_pose` is copied and actual group origin added
once; each asset receives explicit world root LINK pose/zero LINK velocity and
the existing default joint pose/velocity. This holds default PD targets without
an actor network. It is a short configuration/physics test, not a balance or
trained locomotion evaluation.

Each asset's actual24 unique body names,69 unique joint names and2 instances
are checked against the reviewed rig; four native orderings must agree. The
report lists native-to-common body/DOF indices. The bridge changes actor rows
only, not DOF, body, quaternion or coordinate order. Native quaternions are
XYZW. Snapshot arrays separately identify root LINK pose/velocity, body LINK
pose, body COM positions and root COM linear velocity; COM is not substituted
for a link. Installed `ProxyArray.torch.detach().clone()` makes one explicit
Warp-to-Torch snapshot without retaining live buffer aliases.

Canonical mixed rows are group0 [0,1,4,5], group1 [2,3,6,7]. Ordinary separated
positions are recorded independently before and after the first phase.

## Contact API and two distinct phases

The force reader follows the already used production exact-path native API:

- `usd_body_paths.resolve_body_prim_paths` opens the generated SMPL USD and
  resolves real rigid-body relative paths by body name; it never guesses an
  immediate-child body layout.
- `physics_views.resolve_native_contact_factory(sim)` selects the initialized
  Lab physics view and checks manager ownership. No additional SimulationView
  or legacy/private fallback is created.
- `contact_filter_layout` excludes the floor and self, preserves all8 actor
  IDs,24 bodies per actor and168 exact peer-body filter columns. Each actor's
  native view must actually report sensor_count24/filter_count168. Public native
  `sensor_paths` must match the24 submitted source paths in order; `filter_paths`
  must match the168 submitted columns in order, either one shared list or the
  identical list repeated per sensor. The actual representation is recorded;
  unknown paths/order/representation fail rather than guessing a force mapping.
  These public properties are defined in installed `omni.physics.tensors`
  `api.py:4891–4902`; their actual runtime representation has not yet been queried.
- `create_rigid_contact_view` receives one filter list per exact sensing body,
  just as current `PedestrianContactMonitor` does. This avoids incorrectly using
  `ContactSensorCfg`'s filtered multi-body-per-env interface: its documented
  one-body restriction is not silently ignored.
- Every completed120Hz substep uses `get_contact_force_matrix(dt=1/120)` and a
  cloned finite float32[24,168,3] tensor. Raw normal force vectors are retained
  as [control,substep,actor,sensor_body,filter_body,xyz]. Layout metadata assigns
  each filter column to its peer actor/body. They are not net-body force sums,
  impulses, a PACER termination metric or a causal/quality measurement.

**Separated phase:**8 controls/32 substeps from default separated poses. Any
peer body-pair magnitude strictly>1N fails this phase. The report additionally
records whether every raw peer force is exactly0; subthreshold is not relabeled0.

**Deliberate overlap phase:** all poses/joints are explicitly reset. Actor1 is
placed0.2m from actor0 in the **same** group, while actor2 in another group gets
actor0's exact world position and orientation. The initial observed native root
poses must confirm that cross-group equality. Same-group union force must be
strictly>1N to establish a working positive contact probe. All cross-group
pair maxima must be **exactly0**, with a separate reported raw-force condition;
below1N does not satisfy that zero check. Thus disabled force reporting cannot
make the negative probe alone pass. This overlap is intentional collision-filter
instrumentation, not a legal training spawn or a policy-quality comparison.

Both phases retain unselected/failed force and state arrays once a phase has
completed. If a shape/API/clock/runtime error occurs before a phase completes,
the result stays failed and contains the traceback; a completed earlier phase
is still saved. No source pass or cfg attribute is substituted for measured
contacts. The8-view contact-data capacity8192 is recorded, but no unexposed
buffer-overflow observation is fabricated.

Final source-only checks: compile `190e74` exited0 without executing the module;
Ruff `a0a9cd` exited0, all checks passed. The final raw cross-group check tests
every recorded force-vector component for exact0 separately from magnitude
thresholds. No new smoke tests or SDK call ran.
