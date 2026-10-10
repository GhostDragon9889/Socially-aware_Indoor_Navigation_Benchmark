# Public PACER state prior with zero action extension

Verdict: the proposed fresh initialization is mathematically valid and worth an
isolated engineering trial. It is not evidence that motion quality improves.
Production 181 and isolated 931 were not edited. No GPU, Kit, simulator,
physical rollout, training run, optimizer step, or trained checkpoint was used.
The only writes in this review are the new files in this directory.

## Verified local resource and semantic match

The restricted CPU loader accepted the installed
`data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth`
under its expected SHA256
`bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615`.
It checks complete model keys/shapes/dtypes, finite tensors, and all three
normalizers. This is a public PACER reference checkpoint; it is not an author
NavIsaacLab checkpoint or a strict paper resource.

| Item | Installed public source | Proposed fresh destination |
| --- | --- | --- |
| D first weight | float32 `[1024,1950]` | float32 `[1024,2019]`, prefix copied, final69 zero |
| D first bias | float32 `[1024]` | exact copy |
| D second weight/bias | `[512,1024]` / `[512]` | exact copy |
| D output weight/bias | `[1,512]` / `[1]` | exact copy |
| AMP mean/variance | float64 `[1950]` | exact copy; no action RMS |
| AMP original count | float64 scalar `15482880001` | record original, then existing cap `200000` |

The source AMP features are ten frames at30Hz, newest first, each frame195:
heading-local root rotation6, root linear velocity3, root angular velocity3,
19 selected joint exp-map rotations represented by114 tan/normal values,
57 raw DOF velocities, and12 positions for four key bodies. There is no root
height or body-shape feature. Hand/toe joint triples are excluded and the key
order is R_Ankle, L_Ankle, R_Wrist, L_Wrist.

Primary local sources are `third_party/pacer/pacer/data/cfg/pacer_no_shape.yaml`
(`numAMPObsSteps10`,30Hz, rootHeightObs false, localRootObs true), public
`humanoid_amp.py` (`current` history index0, demo time offsets `0,-dt,...` and
the SMPL feature equation), and the corresponding local port in
`CrowdSim/humanoid/amp/observations.py`. The native AMASS corpus also indexes
`indices[:,None]-arange(history_frames)` and the physical world prepends the
current state. The current discriminator flattens this10x195 history then
appends the effective normalized69 action coordinates.

The new initialization preserves this existing observation convention. It
does not establish that public training motion statistics are the same as the
real2848 AMASS bank, nor change the current child-joint expert velocity or
inverse-PD next-frame pose proxy into measured expert commands.

## Initial function equivalence and gradients

For identical raw history `h` and valid finite normalized applied actions `a`,
with the same AMP mean/variance, epsilon1e-5 and clip5:

```
z_new(h,a) = [W_public, 0] [normalize(h), a] + b_public
           = W_public normalize(h) + b_public
D_new(h,a) = D_public(h)
```

All downstream weights/biases and ReLU placements must match. The count cap
does not enter inference, so it leaves initial logits unchanged. Later moment
updates differ from an uncapped public normalizer by design. The existing cap
must be applied after copying AMP count, not to an empty count0 normalizer.

The actual CPU proof used64 synthetic histories drawn using public mean and
variance and checked zero, random and extreme effective actions. It reported
max absolute public/new logit difference `2.384185791015625e-7` in all three
cases, action-change difference0 and normalized-history difference0. Exact
bit equality to the1950-wide GEMM is not required: the2019-wide multiply can
use a different floating-point reduction. A bounded tolerance is appropriate.

One backward pass used the production classification, gradient penalty,
logit regularization and L2 formula. All57 active new action columns had
nonzero parameter gradients, with norm `2.099243640899658`; all12 frozen
columns had zero gradient. Actor/critic/task and self/task/value moments were
unchanged and the protected terrain cold checkpoint gate passed. No optimizer
step was performed and all physical/update/replay counters stayed zero.

Zero action weights give `dD/da=0` initially, but parameter derivatives are
`dLoss/dW_action = sum(delta * a.T)`. Classification can therefore immediately
learn active action columns. The action component of the input-gradient
penalty is initially zero. Nonzero learning is not guaranteed for every batch:
all-zero actions, cancellation, dead activations, or saturated loss can remove
the signal. Frozen hand/toe inputs remain identically zero and cannot provide
classification information.

Function equivalence is for the same history. The native state_action runner
uses pre-step `h[t..]+a[t]`; public PACER publishes its AMP observation after a
physics step, and the local state-only runner uses post-step history. The
candidate retains the current state_action timing. It is not complete public
rollout/reward equivalence.

## Required implementation and checkpoint guards

1. Add an explicit typed initialization strategy or strict boolean distinct
   from the old state-only `warm_start_discriminator`. Default false preserves
   legacy behavior. Reject unknown/nonboolean values and simultaneous old/new
   transfer flags. Do not silently relax the old state-only gate.
2. Require state_action, history10x195, discriminator1024/512, history-coordinate
   AMP RMS, normalized observations/values, clip5 and the verified public SHA.
   The trial profile preserves terrain protection, actor/critic, data,
   reward, PPO/replay and command curriculum. Bind metadata to the actual flag.
3. Reject from-scratch under this pretrained strategy. Require cold counters0,
   empty policy/discriminator optimizer state, empty replay and a fresh
   terrain policy binding. Initialize only once. Build no additional randomly
   initialized layer; overwrite the existing layer so random draws match the
   unchanged2019 constructor and loader path.
4. Perform exact source-key/shape/dtype/finite validation before copying. Copy
   prefix1950, first bias, later D layers/head and AMP1950 moments; make only
   the appended69 weights zero. Preserve gradients and optimizer parameter
   identities. Record original AMP count before the old count cap.
5. Checkpoint the new strategy plus public source SHA, source input1950,
   target input2019, layout/newest-first convention, transferred prefix/RMS,
   zero-action initialization and original/capped count provenance. Keep
   `author_navisaaclab_weights=false` and no strict-author promotion.
6. Reject config/provenance conflicts on save/load and requested-profile vs
   checkpoint conflicts before training resume. A legacy random2019 checkpoint
   has the same shapes and cannot be promoted by shape matching. The old
   state-only1950 checkpoint cannot be resumed as this candidate.
7. A resume restores trained new action weights, complete optimizer/RMS/replay,
   counters and runner RNG/physical state. Never call warm-start on load or
   require trained action weights to remain zero. Zero-column equality is a
   cold-initialization assertion, not a generic resume invariant.
8. Preserve old evaluation readers and protected snapshots. The candidate
   needs explicit readers recognizing the new strategy; an old reader that
   requires transferred-discriminator=false cannot be treated as the new gate.

## Counterexamples and practical limits

- Copying D but keeping zero/unit history RMS fails: this CPU example produced
  logit max difference `0.545336127281189`.
- Reversing history violates the source coordinate convention: observed
  logit max difference `0.43242132663726807`.
- Per-frame195 RMS,11 frames, shape2060 public model, nonzero extra columns,
  altered layer activations or a mismatched key-body order defeat equivalence.
- Passing Gaussian raw samples instead of clamped/masked physical actions
  violates the existing native contract. Ignoring masks can let D identify
  expert/policy from joints which physics never applies.
- Imported prior plus imported AMP RMS is one bundled initialization change;
  a quality result cannot attribute improvement to D weights alone.
- Public D may favor the source corpus over the2848 bank or learn the expert
  pose-command proxy rather than natural motion. These are reasons to measure
  held-out deployment quality, not evidence that this candidate will fail.

## Minimal next checks, without starting training here

First run source-bound CPU regression checks for cold copy/logit parity,
nonzero active-column gradients, negative configuration cases, full trained
resume preserving nonzero action columns, and unchanged legacy RNG/model
provenance. Bind the report to the exact candidate source freeze hashes; the
concept proof in this directory cannot satisfy that implementation gate.

Use the already scoped fresh physical pilot4, then complete resume4 to8 to
check finite gradients/rewards, moments/count updates, replay, protected
terrain invariants, physical exposure and full checkpoint restoration. A pilot
establishes that the training path works; it does not approve motion quality.
Only a separate authorized fresh2000 run preserves the planned256actors,
32steps/update and16.384M transitions budget. Do not convert either old run.

Reuse the preregistered native full22/navigation200/pose45 coverage/90D MMD
evaluation against the selected baseline and require the native follow-on
quality gate. Do not evaluate or select on test34. A different reward-time
convention, data split or threshold needs a separately declared experiment.

Evidence: `concept_cpu_proof.json` has `passed=true`, but explicitly
`source_candidate_sha256_bound=false` and `training_authorization_gate=false`.
The reproducible `check_candidate_cpu.py` makes only in-memory changes and
prints aggregate diagnostics. Run it with CUDA_VISIBLE_DEVICES empty,
PYTHONDONTWRITEBYTECODE1, four CPU threads and the SDK Python interpreter.
