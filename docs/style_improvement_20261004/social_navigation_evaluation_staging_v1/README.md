# Social AMP in the original four-person navigation evaluator

**Unexecuted source preparation.** This directory contains an independent
evaluation-only profile and entry for the new social CNN checkpoint. It does
not contain trained social weights or a physical result. Production185,
the registered AMP62/40k Diffusion primary comparison and existing defaults
remain unchanged. No Torch/NN/SDK/data/checkpoint import, preflight, model load,
simulation or test has been executed for this entry.

## Source integration

`evaluate_four_actor_social.py` reads the complete frozen73954B original
`scripts/train_paper_amp.py`, SHA `d8b85401ace1da7d0039dcad4e023590577944e487789e742019c363a602e3d3`.
Ten uniquely counted substitutions preserve the original complete bytes by
reverse transformation. All non-main helper functions retain their exact AST.
The physical/planning/contact/reward metric equations remain in that source.
This is a bounded derivative, not an import hook that mutates a production
module, not a new physical controller, and not a training entry.

The substitutions are limited to the profile reader, normal App shutdown and
ownership tracking, controller admission, the one pre-action observation,
actual partial-reset notification, declared training counters, interrupted
array preservation and final cleanup/reporting. The original per-actor
`world.step`/`world.reset`, route handling, replan15 clock, reference adaptation,
four120Hz substeps, signed effort-power and static/peer/floor capture remain.
The b608 grouped-training lifecycle and its50N whole-group termination are not
used here. Temporary full network construction by the strict loader is declared;
the retained actor does not evaluate critic/D or construct optimizer/replay.

The independent JSON profile declares only the evaluator's child-frame reference
and357/12308/root/root/static-ground/include-ego=False contract. It is not an
old1044 TrainingConfig or a requested new training recipe. The actual complete
social configuration and saved RMS come from the selected model. The old
configuration gate and old model loader remain unchanged.

## Explicit artifacts and actual physical training admission

Every selected policy, Diffusion model and training result needs its explicit
full SHA and an ordinary readonly single-link file. The policy goes through the
sealed cd227 checkpoint loader: restricted CPU/mmap admission, complete model
and four RMS validation, retained actor path, CPU initializer RNG fork and
explicit device move. No training RNG or SDK state is restored.

The entry additionally requires the actual completed f12a shared-training
`result.json`, not just positive model counters. Its schema/status, completed
updates/transitions, configuration, fixed v7 manifest, source/profile stability,
training-loop and cleanup return flags must match the selected final model.
The artifact inventory must name exactly that model and its bytes/SHA.
Configuration comparison uses exact canonical JSON to handle saved tuple/list
representation without changing values. All four actual training USD prototypes
must match the evaluation articulation's actual USD bytes/SHA. A synthetic CPU
PPO test model alone cannot pass this gate. These checks do not attest paper
budget, training quality or the authors' private policy.

The model's train metadata is retained directly. The entry does not fabricate
the old `provenance['physics']`, a Diffusion training checksum, or old
`legacy_rng_preparation` to satisfy the old evaluator. Training counters in the
report are explicitly declared model counters matched to the actual training
receipt; evaluation controls and actor-controls are independently6000/24000
for the main run.

The actual CPU checkpoint loader preserves the caller CPU stream. After actor
admission, before Diffusion planner construction, this entry explicitly reseeds
Python/NumPy/Torch with7. It performs no old MLP initializer draws and makes no
claim of complete old-run bit identity. This is the new evaluator's fixed RNG
recipe; actual policy-dependent states, resets, replans and captured plans remain
observable and must be considered when comparing results.

## Observation phase and captures

The sealed2379 observation producer binds the actual original shared N4
AMPWorld, four stable distinct actor IDs and one scene ID. It takes one owned
COMMON state per observation and returns self357/task12308 together with that
state. The actor therefore receives the original actual ten futureXY samples
at+.5..5s, the declared current static/cuboid height callback, and all four
current peer states in the same phase. It adds no peers, state teleports, task
normalization, default obstacle bands, motion reference oracle or fallback
controller. Sampler semantics are declared features, not measured collision
truth. The producer strictly binds the model's grid geometry and include-ego
choice.

`begin_episode` is called once after actual initial reset and only for the
actual reset rows after each successful original partial reset. Actual unit
Pelvis quaternions remain unchanged. A singular projected task heading can
continue a prior heading only for a genuinely fallen actor in the same episode;
this does not change the physical SO(3) metric or self357 convention.

The first actual raw self/task/owner arrays are stored in
`social_initial_inputs.npz`. Every pre-action control writes its actual
physics/age/episode IDs, Pelvis quaternion, used/defined/fallback heading and
physical-fall diagnostics to `social_observations.jsonl`. Full12308 inputs are
not duplicated across6000 controls. Original post-physics `physical_states`,
reference plans, ground support, routes and all failed/idle frames retain their
original phase and measurement meanings. Initial inputs are not a replacement
for complete physical captures or a forward-equivalence check.

Unexpected failures preserve actual available physical/reference channels in
separate partial NPZ files with each channel's observed length. Missing
substeps/half controls are not padded, retried or marked complete. These partial
files cannot supply a successful200s denominator. Original successful capture
files are never overwritten by partial preservation.

## Ownership, checks and future invocation

The sole App is launched with `fast_shutdown=False`. The entry refuses a
pre-existing SimulationContext before the sole World constructor. It tracks
returned App/World objects and construction attempts so even an early Torch
import failure or a partially created World context can be disposed. Final
identity checking, owned context stop and App close are independent; failures
cannot retain passed status. No cleanup or physics retry is performed.
Nonfinite report numbers, such as the original pre-loop minimum-height sentinel
on an early failure, are preserved as labelled strings and force failure.
An OS abort before these paths return is not certified by an output header.

Source compilation and isolated stdlib-only `_derive` succeeded in tool
`0ca3f9`: ten changes, whole-source reverse equality, untouched-helper AST
equality, derived SHA `6fd9c1002c2ceedf04975289215e1118872c09cfc2fad0c66b322c08880188a6`.
Ruff passed `4f2937`. This checks code construction only; the entry itself,
argument parser, preflight, App, producer and policy were not invoked.
Root final whole-source reads fc8375/b3439f and independent state peer are complete: Required=[] / Optional=[]. Peer stdlib-only literal/AST reconstruction7e093d confirms all ten deltas, five unchanged helper ASTs and the original status Return. The code, profile and README are now ready for readonly sealing. No additional runtime or test was executed.

After actual formal50502 exit0, terminal audit, fixed Diffusion paired/AMP62
primary, serial NN checks, old8-person collision smoke, new World probe and
shared-training smoke/pilot, a future explicit command is:

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/social_navigation_evaluation_staging_v1/evaluate_four_actor_social.py \
  --amp-checkpoint ACTUAL_IMMUTABLE_FINAL_SOCIAL_MODEL \
  --amp-sha256 ACTUAL_MODEL_SHA \
  --training-result ACTUAL_COMPLETED_SHARED_TRAINING_RESULT \
  --training-result-sha256 ACTUAL_RESULT_SHA \
  --diffusion-checkpoint ACTUAL_SELECTED_IMMUTABLE_DIFFUSION_MODEL \
  --diffusion-sha256 ACTUAL_DIFFUSION_SHA \
  --control-steps 6000 \
  --output output/diffusion_amp/social_CNN_fixed_four_actor_200s_v1
```

`--control-steps 4` or30 is a bounded smoke, not the200s primary. `--preflight-only`
inspects declared local resources and source derivation without creating output
or importing Torch/SDK; it cannot validate model tensors or physical training
quality. None of these future commands has run.

Actual loader/forward/RNG parity, SDK lifecycle/episode notifications, the shared
model's physical training, full22 tracking, source-admitted CPU metric reading,
200s results and native-scene adaptation remain incomplete. This source entry
does not show achieved or exceeded paper performance and does not promote a
default model.
