# Prepared shared-world SDK probe

`probe_shared_world.py` is source preparation. It has not been imported or run;
no App, SDK scene, Torch tensor, dataset payload, model, checkpoint or GPU was
constructed for this stage. Only source compilation and Ruff are authorized
now. Root owns the later serial GPU schedule after formal training and its
registered main comparisons. This file introduces no trainer or policy.

## Future invocation

Run in a fresh SDK Python process with an explicit local manifest path and a
new or empty output child of the repository's `output` or `/tmp`:

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/shared_group_world_SDK_probe_staging_v1/probe_shared_world.py \
  --manifest /absolute/path/to/the/fixed/v7/manifest.json \
  --num-actors 8 --control-steps 8 --device cuda:0 --seed 7 \
  --output output/diffusion_amp/shared_group_world_SDK_probe_v1
```

The manifest placeholder must be replaced with its actual existing path;
the fixed SHA is `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`.
Only its train2848 bank is loaded in the future run. The bank's full SHA is
declared by that manifest, not recomputed or claimed as independently audited
by this short probe. The original train-only AMASS source uses CPU mmap and
child-joint DOF velocities. No held-out bank or policy weights are requested.

An optional `--fall-probe` adds exactly one control after moving actor0 to
world Z=.3m and XYZW `[1,0,0,0]`, with zero native root velocities. It is a
deliberately manipulated pose, distinct from the initial zero-action phase.
It must produce a real end-of-control physical-fall flag and complete group0
reset; otherwise the probe fails. The history is not rewritten to pretend
that the external pose write was a physical transition. No AMP reward or
policy is evaluated on that manipulated input. The total including this
extra control must remain at most12. Default phase controls must be at least4.

## Reused sources and physical contract

The entry pins and executes the existing `SharedGroupWorld` b608cae6 and
native normal-view factory517afb01 after the caller launches one App. Both
must remain ordinary readonly files with one link. The former already pins
its scene/actor bridge/state adapter/task/contact/sampler and original
observation, PD and curriculum sources. The eight-actor d000200 smoke is
pinned as source setup evidence; its `main` or physics phases are never
called. Solver4/4, DT1/120, four substeps, headless renderer-disabled setup,
SMPL local single-USD template and a global collision-group−1 Cuboid floor
are retained. The Cuboid covers world `[-32,32]²`, is .2m thick, centered at
Z−.1m, and its top is exactly0. This covers negative as well as positive
scene origins; ground callbacks reject queries outside the finite floor.

The original `MixedTrajectoryCurriculum` constructor is executed unchanged
inside a private package namespace so its relative imports do not execute
the NN-bearing training package initializer. First-half AMASS and second-half
procedural assignments produce groups `[0,1,4,5]`, `[2,3,6,7]`, each with two
real AMASS and two original smooth procedural command actors. Its original
default procedural config and `linear_velocity` command tail remain. Bounds
are `None`, matching the constructor default: the probe does not introduce
an extra whole-remaining-clip rejection/resampling rule. The physical floor
and every actual static-height query are nevertheless finite and checked.
No Diffusion replanner, model, PPO, D, replay or expert sampler is constructed.

`SharedGroupWorld.create_after_app` constructs the sole SimulationContext and
four Articulations over two group environments; it owns actual origin
addition once and actual current link↔COMMON mapping. The entry records
all native body and joint name orders, group/world origins and spawn XYZ.
Each asset's actual `cfg.spawn.usd_path` is resolved and fully hashed; the
actual MJCF bytes and exact robot/SceneCfg/converter/D6-workaround source
identities are also recorded. Identical cached USD paths across four assets
remain visible rather than being represented as independent asset files.
Names and file identities establish traceability, not pose or collision
correctness. No new Euler conversion or prior D6 numeric proof is repeated.
COMMON Pelvis and independent native root-link states have separate NPZ keys.
The bootstrap callable's actual module origin is checked before launching
the App; the actual imported corpus, robot/config/SceneCfg and converter
origins are checked before construction. A previously cached module from
another checkout is rejected rather than being authorized by `sys.path`.
The D6 workaround is checked if actually loaded; a cached-USD path that never
imports it is explicitly recorded as not loaded, not forced to execute an
unneeded converter or represented as observed execution.

Grid callbacks explicitly declare root XY center, root heading and absolute
static-center-ground Z reference, with `include_ego=False`. The returned
raw self357/task12308 and newest-first AMP history `[8,10,195]` are verified
as finite F32 on the actual device. This choice is an explicit probe geometry
mode, not an assertion that another head/root or root-Z training mode is
equivalent. Static height is zero only on the actual finite floor.

## Probe values, ordering and evidence

Applied actions are exactly zero69 with the canonical57 effective mask.
They are untrained PD commands, not a Gaussian actor or locomotion quality
measurement. Every completed control must have exactly four actual physics
steps, with rendering disabled. Initial SDK/reset steps are not assumed
zero: the entry records the actual post-initialization baseline.

Each attempted control writes owned all-N pre-state and raw inputs. After
the real control, it writes owned all-N physical/post-input data, flags,
target XY, four-substep signed mean joint power, sampled current native
normal forces, and explicit positive synthetic nextV on every original
episode **before** any group reset. These values only test the handoff;
they are not predictions from a learned critic. Original termination is
the union of physical-fall and nonfoot net-normal contact. Cap/source-end
and collateral truncation stay distinct. Group reset coverage, episode and
age/history refresh, and no extra reset physics step are checked afterwards.
At least one true cap4 event is required; early actual failures are retained
and cannot be silently substituted for observing the cap.

The normal getter is an actual current all-partner world normal-force sample
after the fourth substep; its source metadata is retained as declared by the
frozen sampler. The entry checks the real counter around control/reset.
It does not turn this into a force temporal maximum, peer pair measure,
friction/impulse channel or an independently audited synchronization claim.

Every NPZ uses exclusive creation, numeric arrays, flush and fsync, and is
hashed after writing. Existing partial or failed outputs are never replaced.
If an exception occurs, the entry attempts one native failure snapshot,
labels it as **not a validated control endpoint**, records the actual
observable counter and preserves all partial file names/bytes. Incomplete
steps do not increment the completed-control count. The report does not
claim independent durability, physical checkpoint resume or model quality.

`fast_shutdown=False` allows normal `finally` cleanup; the entry attempts
simulation stop and App close separately, retaining each error. Status0
requires actual probe checks and both cleanup calls to return. External
process termination or SDK fast-exit can still prevent Python finalization;
missing evidence must remain missing and root must observe the actual tool
exit. The entry never retries or reconstructs a faulted World automatically.

## Remaining real SDK risks

Native queued replication, clone ContactReportAPI readiness, actual names
and path order, current force availability, grouped collision filtering,
device/Warp access, reset/FK refresh, actual episode-cap exposure and zero-PD
behavior are not established by source compilation or prior mock tests.
An actual construction failure retains honest failure evidence. This probe
does not replace the earlier separated/same-group/cross-group contact smoke,
and does not claim that source collision filtering alone proves physical
cross-group exclusion. No physical or training result exists for this stage.
