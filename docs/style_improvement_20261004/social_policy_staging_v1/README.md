# Neutral-SMPL social task encoder and actor/critic boundary

Standalone executable Torch class definitions, **source-only preparation**. Only stdlib compilation and Ruff may run while formaltraining50502 is live. Torch import, module construction, tensor forward/backward, model/checkpoint loading, SDK/GPU, shared physics and training have **not been executed**. The authored tests are unexecuted. Production185, old models/defaults and formal40k/q937 plans are unchanged.

## Exact interface

| Boundary | Contract |
|---|---|
| Raw self observation | `[B,357]`, fixed neutral SMPL COMMON self features. |
| Raw task observation | `[B,12308]` = trajectory20 followed by HWC/C-order `[64,64,3]` raster. |
| Trajectory | Ten root-relative, root-heading XY targets at +0.5…+5s, in metres, **no ×10 scale**. Current `observations.py:174` concatenates localXY directly. |
| Raster | Channels `[height, localVx, localVy]` from the separately staged social raster contract. Grid center/heading and height-reference choice must be explicit in future observation/checkpoint config. |
| Normalization | External self RMS357 and **whole raw task RMS12308**, then clip5 as selected by the future trainer, before splitting/reshaping. This class does not normalize, update RMS or scale observations. |
| Encoder | Three Conv2d layers, 16 filters each, kernel4/stride2/padding1/ReLU:64→32→16→8. Flatten16×8×8=1024→Linear512/ReLU→Linear256/ReLU. |
| Encoded task | `[map256, normalized trajectory20]` =276. |
| Actor/critic input | `[normalized self357, encoded task276]` =633. |
| Heads | Separate633→2048/ReLU→1024/ReLU→69 actor mean and →1 critic. One shared task encoder; no unused second CNN. |
| Policy output | `policy(self, task)` returns `(Normal(mean69, exp(log_std)), normalized_value[B])`; `forward` has the same contract. Fixed float32 log_std buffer−2.9. No action is sampled inside. |
| Value-only output | `value(self, task)` calls shared encoder+critic only, without actor/Normal creation. It remains differentiable; there is no no_grad/decorated denormalization. |

Construction requires `SocialAMPActorCritic(task_encoder_sharing="shared")`. Separate **heads** do not mean separate task encoders. The stage accepts only finite materialized float32 2D inputs. Fresh nn.Linear/Conv2d defaults create all trainable weights/biases; no global RNG is reseeded by the class. Caller owns construction seed, device, parameter freezing and saved RNG. This is the current local fresh-initialization convention, not a claim of bitwise public PACER initialization. The official base builder zeroes head Linear biases; this standalone stage leaves Torch constructor biases, explicitly.

The external trainer still samples Gaussian actions, computes masked likelihood on57 effective coordinates, clamps/masks the69-action PD request, and denormalizes critic values with value RMS. None of those responsibilities moves into this network. `value` returns the **normalized** value used by PPO's normalized return target; a future differentiable guidance facade must freeze model parameters, apply frozen input RMS and value denormalization while retaining input gradients. Wrapping that facade in no_grad would break trajectory guidance.

## Primary source and paper distinction

Fixed public PACER commit `baeb6241f03fc9d78370a64e3c2dcef9d830d97f`:

- `third_party/pacer/pacer/learning/amp_network_sept_cnn_builder.py`, SHA `8b2d335de582eea98706f1ea097df3b548e9150879ecfa1e90b4061f0c9f0120`: `eval_task` reshapes normalized task HWC, calls `_task_actor_cnn` plus `_cnn_mlp`, returns `[map features, trajectory]`. Both `eval_actor` and `eval_critic` call that same method. A `_task_critic_cnn` is allocated under `separate` but not used by these paths; this local stage does not allocate those dead parameters.
- `third_party/pacer/pacer/data/cfg/train/rlg/amp_humanoid_smpl_cnn_task.yaml`, SHA `3941d6e96c4f914046c6ec3a152fe5989ba6fa16abfc93eec97eda54cd891edf`: three16-filter4/2/1/ReLU convs, projection512/256, separate2048/1024 heads, fixed−2.9 sigma, normalize_input/value.
- Cached TRACE/PACE supplement B.1 (`paper_fidelity_research_v1/trace_and_pace_text.txt:866–908`) describes64²×3 and root-centered4×4m/root-height-relative observations; it says four-level CNN and256-dimensional task latent and includes shapeβ. The fixed code instead uses the three convs plus projection and concatenates trajectory20 afterward. This stage follows the **fixed code**, with neutral self357 and no β conditioning. It is not a claim that those paper and code descriptions are identical.
- NavIsaacLab IV-B (`docs/full_audit_20261001/navisaaclab_paper.txt:427–434`) describes task encoding and actor/critic but supplies no private network config, social checkpoint or normalization schema. This stage is a local PACER-based social adaptation, not strict Nav private-model reproduction. Its raster geometry/ownership choices remain the separate declared local contract.

## Required integration work, deliberately not implemented

Current `PaperAMPNetwork` builds task1044→512→256 and actor input613. Trainer constructor and resume hard-build that class (`trainer.py:132,525–533`), and its policy optimizer lists task_encoder/actor/critic parameters once (`:138`); the new shared-encoder names make a future isolated integration coherent but **are not installed there**. This class has no discriminator. AMP history/D/replay, both optimizers, complete RNG, world/terrain/group reset and curriculum state remain separate future responsibilities.

Current full-public warm start `trainer.py:186–207` requires357/1044 and copies the complete old task MLP and613-column first actor/critic weights. Those dimensions and semantics differ; no prefix/padding/strict=False migration is provided. `load_state_dict` explicitly rejects strict=False, but strict module keys/shapes alone do not prove full saved tensor dtype, moments or provenance. A future new social checkpoint/config/reader must establish all of them. Fresh training is the declared stage initialization; a later explicit transfer strategy would need its own review.

The old task RMS1044 and protected suffix20:1044 (`terrain_statistics.py:19`) cannot be reused as12308 social moments. New whole-task moments/layout identity must be trained/serialized; unchanged357 self or1D value moments are not transferred by this source. Old `checkpoint_contract.py:78` and `amp_inference.py:199–204` correctly reject non1044 tasks, and old inference constructs its old model. They stay unchanged.

Future RMS integration should retain the **local** trainer's float64 population-moment mechanism, count0 initialization, explicit raw-rollout updates and frozen statistics while collecting/evaluating one rollout. This is not a claim of equivalence to public count1/forward-update statistics. Mirror branches use the same saved moments and must not update RMS a second time from augmented observations. The whole12308 input is normalized before the CNN, while scalar value normalization remains the existing separate value RMS mechanism.

Potential later initialization must be a reviewed, named-block adaptation, not a whole-actor warm start: unchanged357 self-feature statistics may be compatible if their exact feature/frame contract is retained; hidden2048→1024 and output1024→69/1 blocks have matching shapes but are still operating on a new preceding representation. The first actor/critic layer changes613→633 columns, and the old task MLP/RMS has different raster dimensions and semantics. A state-only1950 D and its AMP-history RMS may admit a separate strict transfer if the same10×195 feature/history contract is retained. None of these possible transfers is implemented, loaded or validated here, and shape matching does not establish equivalent policy, initialization RNG or physical outputs.

Current symmetry config and `training/symmetry.py` allow only357/1044. A future social mirror must mirror **raw** self/task/action before the same RMS: trajectoryY sign reversal; raster localY-axis flip; height/Vx unchanged and Vy sign reversal. The existing official velocity-map flip does not reverse Vy, whereas the declared social raster fixes that parity. No mirror adapter has been installed in PPO and the existing1044 checks must not simply be bypassed. A future checkpoint must identify chosen grid-center/heading/height-reference and include_ego, not infer them from12308 alone.

Only these standalone classes, source contract and pending tests are implemented. Setting an existing YAML task_dim to12308 is not sufficient to train or deploy them. No source/seam pass here is an actual social policy or physical-quality result.

Actual final source checks: stdlib compile only `98148d`, exit0 (no source execution/import); Ruff `3ab724`, exit0, all checks passed. Earlier source-only checks `df065a`/`eff39e` are retained as preparation history. Independent whole-source review returned Ready/Required=[], and root also read source/tests/README (`119d3e`, `c47979`, `a02450`, final status paragraph`b6321d`). Root verified exact identities and made source/test readonly in actual tool`5b6434`: source6763B SHA`2a4007f732f0854d05b784debcbd0798e0c587149ba8b63d623378ad1ab3ac15`, test2707B SHA`76b9ed7e4587fa18b3667c9c27bf4945990576eb53d13b52d0f78396fb7f62e3`. The three authored Torch tests remain unexecuted; no module import, network construction, forward pass or checkpoint loading was performed. Compile/lint/source review is not evidence of neural execution or trained behavior.
