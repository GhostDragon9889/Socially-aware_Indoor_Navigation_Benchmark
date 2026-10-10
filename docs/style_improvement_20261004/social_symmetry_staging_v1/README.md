# Coupled raw social symmetry

Pure Torch math with four new CPU tests passed in the one approved invocation. Independent source peer reported `Required=[]`; root whole-read and approval preceded execution. No NN, policy, SDK, weights, capture or physical execution is used. Production185 and earlier sources/proofs remain unchanged.

## Interface

`mirror_raw_observations(self_obs[B,357], task_obs[B,12308], device=...)` returns a new coupled raw pair in canonical PACER body order. Apply **before normalization**, then use the same externally frozen self/task RMS on original and mirrored branches. New12308 task moments are required; no1044 dummy task or old terrain normalizer is used. A supplied device must be CPU or indexed CUDA and match both inputs. For compatibility with the existing two-positional trainer call, an omitted device uses only the already materialized self tensor's device, with strict task agreement; it never selects or transfers data. Raw self/task/action inputs are finite float32, nonempty2D batches.

Self357 preserves the current named bilateral permutation and reflection equations:23 root-relative XYZ positions,24 tangent-X/normal-Z vector pairs,24 linear velocities are polar `(1,-1,1)`;24 angular velocities are axial `(-1,1,-1)`. Pelvis stays first. Original `mirror_actions` is reused exactly for69 canonical DOF triples, bilateral exchange and axial signs. No action clipping, projection, detach or normalization occurs in the mirror.

Task12308 reuses the sealed task function: prefix20 trajectory Y negated; HWC raster localY/axis2 flipped; only velocity-channel Vy negated. Grid and velocity frames remain the already declared producer contracts. Owners/physical bodies are not moved. This mirrors observations/actions, not a simulator state or rig, and does not prove symmetric physical responses.

`canonical_action_mask(device=...)` derives the exact hand/toe four-triple freeze in canonical named order:69 raw coordinates,57 active. Its absolute mirrored mask must equal itself. `effective_symmetry_loss(mean, mirrored_mean, action_mask, device=...)` returns **unweighted** `mean_batch(sum69(((mirrored_mean−mirror_actions(mean))*mask)^2))/57`. Both branch gradients remain. Caller/trainer multiplies the configured coefficient once; the default social setting is50. The existing PPO update may instead retain its own identical residual/mask/57 formula and use only the two mirror APIs. No critic, discriminator, PPO rollout or optimization is created here.

## Exact source adaptation

All original sources are fully byte-hashed before loading selected code. Source facts:

- Original `CrowdSim/humanoid/training/symmetry.py`,4897B SHA`af8dfd1cc04f42c03ee538c16543455466dea7067c6e04c3c494c16ce3e14373`.
- Original `CrowdSim/humanoid/amp/observations.py`,10438B SHA`a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba`: its original PACER_BODY_NAMES/PACER_DOF_NAMES declaration AST is loaded, without importing observation code or rotation utilities.
- Sealed `../social_task_observation_staging_v1/social_task_observation.py`,8512B SHA`6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1`, ordinary readonly single-link, is loaded from its exact bytes and filename. Its own raster dependency pin remains effective.

The old mirror function has20 top-level body statements, indexed0..19. The self adaptation retains **14 original statements**, indices2 and4..16, unchanged. It deletes six indices0,1,3,17,18,19 (old mixed docstring, old coupled1044 validation, old1044 validator, old trajectory/grid calculations and old paired return). It adds one `return mirrored_self`, yielding15 statements; removes the task argument and changes the function name/return annotation. It keeps the body's original permutation, positions/rotations/velocity/angular equations exactly. `_counterpart`, `_name_permutation`, `_validate_tensor` and `mirror_actions` function ASTs are unchanged.

The complete derived helper/self/action module has normalized `ast.dump(include_attributes=False)` SHA`1ba13634dd6649781bcff160c861de4058dd300721f713ca426894fc846eee86`. Runtime verifies this derivation SHA. Actual stdlib-only source checks`612dbc` verified four original helper ASTs exact and the14-retained/6-deleted/1-added counts; SDK Python3.12 check`14d813` confirmed the same derived SHA without importing Torch or running the mirror. It is explicitly a source adaptation, not execution of the old1044 function unchanged.

## Method limits

The local task mirror's Vy sign correction is a vector-consistent social convention. Fixed PACER's published task mirror flips its grid but does not negate Vy; this stage **does not claim author task-mirror equivalence**. Public `_sym_loss` already multiplies its69-coordinate mean by50 (with external `symloss_coef=1`); this stage instead leaves its57-active mean unweighted and applies50 once externally. The effective coefficient50 is consistent, while the57 versus69 reduction differs. The57 choice preserves the declared neutral SMPL hand/toe decoder and current social settings, not an asserted identical paper loss.

Canonical body/DOF order is required; arbitrary native Articulation order must be converted earlier. GPU rounding/device execution, normalized branch behavior, trainer linkage, policy training and physical bilateral outcomes remain pending. Matching two-sided tensor gradients is not evidence of improved gait.

## Actual CPU check

Four tests cover closed-form named polar/axial/action signs and HWC mirroring; twice-mirror identity and input/RNG preservation; tensor gradients on both branches,57-mask invariance and a single external50 multiplier; rejection of old1044 input, shape/batch/dtype/device/nonfinite/alternate57-mask errors. They did not rerun old tests or construct a policy. The single approved command was:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_symmetry_staging_v1 -p test_social_symmetry.py -v
```

Actual tool `5b2c14` completed directly with exit0, without a yielded session:4 tests /0.020s, tool wall0.872219839s, `OK`. Tests asserted that CUDA remained uninitialized before and after. `actual_CPU_checks.json` retains the exact command/output, source/test identities, root approval and source peer evidence. This is a CPU tensor/autograd result, not SDK, policy-training, GPU or physical proof. Source and tests were unchanged after source review; root freezing remains a separate step.
