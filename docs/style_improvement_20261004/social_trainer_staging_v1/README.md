# Independent social CNN/PPO/AMP connection

This is source preparation, not an executed trainer or a completed shared
physical environment. No Torch import, network construction, forward,
checkpoint loading, SDK/App, GPU, training or NN test was performed for this
stage. Existing production185 files, defaults and frozen experiments remain
unchanged. Root reviews this mutable stage before any source freeze.

## Interface

Future callers import `SocialTrainingConfig` and `SocialAMPTrainer` from
`social_trainer.py`, and supply the required explicit raster choices
(`grid_center`, `grid_heading`, `height_reference`, `include_ego`) and flat
`actor_count`. `SocialAMPTrainer(config, device=..., provenance=...)` takes a
detached dataclass copy; the original constructor's action-mask fill does not
mutate the caller's configuration. There is no default legacy configuration.

The parent `PaperAMPTrainer` source is pinned to SHA
`707f355acbd8b5d33aa40c3299a2540610ea695ae4bb9705ede3498e27224f9d`.
`SocialAMPNetwork` directly constructs the sealed shared social actor/critic,
then adds the original `_mlp` discriminator and `disc_logits`. It never
constructs an old task MLP and replaces it afterward. The original D factory
requires the float32 default dtype; another default dtype is rejected.

| Boundary | This stage |
| --- | --- |
| Raw self / task | 357 / 12308; `[trajectory20, HWC64×64×3]` |
| Encoded policy input | `[self357, map256, normalized_trajectory20]` = 633 |
| Policy | One CNN encoder shared by separate actor/critic MLPs |
| Discriminator | State-action2019 = 10×195 history + 69 applied actions |
| Actions | Raw Gaussian69, fixed log std −2.9, clamp then physical mask57 |
| RMS | Self357, **entire task12308**, value1, and declared AMP dimensions |
| AMP RMS default | Original `per_frame`: **195**, not 1950 |
| Optional declared AMP RMS | `per_history_coordinate`: 1950; no public moments copied |
| Initialization | Fresh local CNN/MLP defaults; no actor/D/RMS transfer |

State-only D is rejected here and remains a separate experiment. The local
history/action expert proxy remains the original declared inverse-PD or
user-supplied action contract; this stage supplies no new expert generation
or timestamp alignment. It does not claim the private Nav policy or training
recipe has been reproduced.

## Bounded reuse

`DERIVATIONS` declares exact allowed string counts. Before a future import
creates callable methods, each derivation is compiled and must reverse to
the complete original method bytes and AST:

1. Constructor: exactly **one** `PaperAMPNetwork` → `SocialAMPNetwork` factory
   substitution. Original seed, mask, all four RMS, two Adam optimizers and
   replay construction remain unchanged. The shared encoder's parameters
   occur once in the policy optimizer; D parameters belong only to D Adam.
2. Update: exactly **one** mirror import → fixed social paired-mirror function
   assignment. The two original positional self/task arguments still work.
   Raw self/task are mirrored together before the same rollout-frozen RMS.
   Original residual, physical mask, `/57` reduction and outer configuration
   coefficient remain intact. Default coefficient50 is applied once; the
   helper's separate `effective_symmetry_loss` is not called.
3. Save: replace one format literal and add one contract/validation hook before
   the unchanged temporary Torch save/replace.
4. Load: replace one format literal and replace the one configuration/constructor
   line with preconstruction metadata admission plus postconstruction payload
   validation. Original strict state loads, replay copy and deferred RNG restore
   remain intact.

Unmodified inherited methods are `_observations`, `act`, `_policy_log_prob`,
`value`, `_amp_inputs`, `amp_reward`, `_discriminator_update` and `restore_rng`.
Original `RunningNormalizer`, `generalized_advantages`, rollout validation,
policy replay and terrain-statistics helpers are reused from pinned sources.
The only update wrapper addition requires rollout width = declared actor count.
No GAE, discriminator loss, reward transform, PPO objective/gradient handling,
replay sampling or moment-update formula is replaced. This retains the local
BCE discriminator; it does not claim equivalence to original AMP LSGAN.

Raw self/task moments are updated once from original rollout samples after
PPO. Mirrored data are not counted a second time. Value RMS, normalized critic
targets and denormalized rollout values follow the original algorithm. Public
PACER also has effective symmetry weight50, but its69-coordinate mean differs
from this local masked57 reduction.

## New checkpoint admission

Format: `local.neutral_smpl.social_CNN_PPO_AMP_training.v1`, version exact int1.
Contract: `local.neutral_smpl.social_training_checkpoint_contract.v1`.
Old1044 format, unknown/partial fields, state-only D, altered configuration,
architecture, source identities or derivation identities are rejected before
the new network constructor. The complete new configuration document retains
raster frame/height/self choices, actor/group count, raw/encoded dimensions,
seed, all numerical hyperparameters and expert-sampling declaration.

After construction and before state copies, admission checks exact model keys,
shape/dtype/finite tensors and fixed log std; all four RMS shapes/dtypes/finite
moments, nonnegative variance/count; both Adam group options/parameter IDs,
complete state after updates, steps and moments; complete replay counters,
shapes, finite values and normalized/frozen action columns; counters and full
Python/NumPy/CPU/CUDA RNG structure. Python/NumPy and CPU generator state are
validated with private objects. CUDA bytes receive structural validation, not
a CPU CUDA-generator restoration claim. Original `restore_rng` requires a
CUDA host when saved CUDA states exist.

Fresh/no-warm-start self, task and value RMS counts must each equal the saved
`physical_transition_count`: the unchanged update consumes all original flat
observations and returns exactly once per successful update. AMP RMS count has
additional expert examples and, in per-frame mode, ten frames per history; it
is not assigned this same equality.

Replay admission also requires `size = min(capacity, accepted)`. The original
algorithm's `accepted` is cumulative. A retained input block at least as large
as capacity writes its last capacity rows and resets cursor to0; subsequent
smaller blocks advance from that rebased cursor. Therefore cumulative
`accepted % capacity` does **not** determine a full buffer's cursor. Admission
preserves the original bounded-cursor/partial-buffer rules without inventing
a false modulo invariant or changing replay math.

The adapter record is captured once at initial module import. Contract creation
compares the current file record with that initial record and rejects a change;
it returns a copy of the initial identity. Old loaded functions cannot silently
acquire a newly edited source SHA at save/load admission.

`warm_start_pacer` always raises. There is no `strict=False` migration or hidden
named-block transfer. Absolute source locations are part of this stage's saved
contract; relocation/adaptation would require a declared later source revision.

`save(path, runner_state=None)` and `load(path, restore_rng=False)` retain the
original complete trainer state. `runner_state` is either an opaque supplied
dictionary or None; the inclusion flag reports only that fact. Its contents
are **not** checked for SDK/PhysX restoration, group epochs, planner state,
history or expert-bank private RNG. Those future runner responsibilities cannot
be inferred from a valid trainer checkpoint.

## Remaining integration

The four-articulation scene/actor bridge, real collision/contact handling,
group reset/curriculum, target/expert sampling, raster observation producer,
rollout timestamps and physical runner save/restore are not connected here.
The existing driver still constructs its original trainer; this stage provides
no CLI training launcher. A differentiated value-guidance facade is also not
provided: inherited trainer `value` remains `no_grad`, while the sealed network
has a normalized, critic-only differentiable `value` method. Future guidance
must explicitly freeze input/value RMS and denormalize without disabling
input gradients.

Only source compilation, Ruff and inverse source metadata are allowed now.
NN tests and complete save/resume validation remain unexecuted, pending GPU-free
root authorization. Source/AST equivalence is not numerical training evidence.

Initial source checks: tool chunk `b7c9de`, exit0, compiled the initial new
source and four derived methods without executing them; all four inverse
methods matched complete original bytes/AST and13 dependency pins matched.
Ruff chunk `546915` exited0. Root review then required the count and import-time
identity refinements above. Initial source bytes are preserved at
`before_resume_consistency_revision.py` (SHA01c0976e…), and the initial report
is preserved verbatim at `source_derivation_before_resume_consistency.json`;
its original executed source path refers to those initial bytes. Current
`source_derivation.json` must bind the final checked source. No NN tests were
authored or run, and strict schema validation does not establish training or
SDK restoration.

Final source revision checks: compile/inverse-method invariance tool `703d03`
exit0, Ruff `13ac20` exit0. The four `DERIVATIONS`, both derivation helpers and
method compiler retain their initial exact AST; the original parent source SHA
is unchanged. The existing full inverse method proof is therefore reused in the
updated `source_derivation.json`, explicitly labelled as reused. Both tools
completed directly without a yielded session; no stage import or NN execution.

## Root source seal and newly authored future NN checks

Root whole-source reads `420a85` / `e2f23c`, followed by necessary-delta reads
`c85b2d` / `0828c3`, accepted the contract. Root seal `e63cce` froze the final
24078-byte trainer SHA `81a6319f…` and its derivation/history records. This seal
is source evidence; it did not execute a network or a physical restoration.
The sealed trainer and prior source proofs have not been edited afterward.

After that seal, `test_social_trainer_future_CPU.py` was authored separately.
It remains **UNEXECUTED**. The earlier statement that no NN tests were authored
describes the initial source phase; four future checks now exist:

1. Fresh CNN / D2019 / four RMS / physical57 mask / detached caller config.
2. One cached synthetic4-control ×8-actor PPO/D update, one epoch/minibatch32,
   D batch32, replay64. Self/task/value counts32 and per-frame AMP count640;
   raw coupled mirror is observed once, and the entire raw task12308 moment
   vector is checked without a second mirror-statistics update.
3. Full synthetic CPU trainer state roundtrip: all model tensors, two Adam
   states, four RMS and replay; explicitly restore pending RNG after
   `load(restore_rng=False)`, then compare same-input sampled actions and
   Python/NumPy/Torch CPU draws. The constructor's reseeding is not described
   as global RNG preservation by `load(False)`.
4. Exactly the requested malformed-bundle risks: old schema, incorrect RMS
   transition count, model dtype mismatch, and missing policy Adam state.

The synthetic self/history/raster values are finite arbitrary covariates, not
valid SMPL physics, measured contact, raster geometry or AMASS reference data.
Expert actions are explicitly `user_supplied_normalized_actions`. An opaque
synthetic caller dictionary roundtrip does not prove physical SDK resume.
Only one cached PPO/D update is used collectively by these cases; no extra
big-block replay case, fifth case or existing NN suite is added.

Authoring checks only: test-source compile chunk `8a4eb4` exit0, Ruff chunk
`acaad0` exit0. Neither imported/executed the file, Torch or any NN. Both tools
completed directly without a yielded session. The earlier three standalone
social-policy NN tests are still pending and are not included here.

Future command, **only after root explicitly authorizes a serial invocation
following formal50502 completion and required primary comparison**:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_trainer_staging_v1 -p test_social_trainer_future_CPU.py -v
```

Future four NN cases received source-only independent review (`Required=[]`) and complete root read `a91a88`; root `27cb66` sealed the12606-byte test source SHA `98de7ef9e8183b62b4b2019736086402faf27f8f25138c21510aae790207ec00` as ordinary single-link mode0444. The tests remain unexecuted; compilation/Ruff and source review are not NN evidence.
