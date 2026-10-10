# Frozen social critic value interface

Source-only independent stage. The adapter has not been imported, constructed,
forwarded, trained, or used for Diffusion guidance. No NN tests were authored
for this slice. Production185 and all sealed training/policy/task files remain
unchanged. The source is now sealed after root and independent source review; runtime checks remain pending.

## Exact boundary

`value_adapter.py` imports only stdlib. Its future constructor accepts an
**already constructed**, source-pinned `SocialAMPTrainer`; it does not import
or construct a trainer, load a checkpoint, or initialize a model/normalizer.
The caller must stop concurrent training updates while taking the snapshot.

`SocialValueSnapshot(trainer)` deep-copies one tuple containing the existing
task encoder, critic, self RMS357, task RMS12308, and value RMS1. It retains
their dtype/device, freezes parameters, and sets the copies to eval. It omits
the actor, log_std, discriminator, AMP RMS, optimizers, replay, RNG state, and
physical runner state. Deep-copying the complete network would include these
unneeded components, so that path is deliberately absent. No `.to()`, global
seed change, parameter sampling, or default RMS fallback occurs.

The constructor requires the declared new architecture/configuration, both
`normalize_observations` and `normalize_values` explicitly True, positive
integer update/transition counts, and self/task/value counts each equal to the
physical transition count. Encoder/critic tensors must be finite float32 on
the trainer's exact CPU or indexed CUDA device. The three RMS copies retain
finite float64 mean/variance/count, nonnegative variance, and clip5. These
admission conditions do **not** prove physical training, critic accuracy,
collision avoidance, or quality. They do not perform full checkpoint/Adam/
replay/RNG admission; an existing trainer is the input to this interface.

## Values and candidates

`snapshot.value_raw(raw_self, raw_task)` accepts finite float32 `[B,357]` and
`[B,12308]` on the snapshot device and returns denormalized `[B]` values. Raw
task layout is trajectory20 followed by the HWC64x64x3 C-order raster12288.
The copied encoder produces `[map256, normalized_trajectory20]`; concatenating
self357 gives critic input633. There is no actor/Normal call.

The original `RunningNormalizer` in `CrowdSim/humanoid/training/trainer.py`
keeps float64 population moments. Its `update` is no_grad; its `normalize`
and `denormalize` are differentiable. This adapter calls those exact methods:
input-dtype conversion of moments, epsilon1e-5, observation clipping to
[-5,5], and value denormalization are preserved. It never updates moments.
The original trainer's `value` is no_grad and calls the actor/policy path;
it is therefore not called here. Copied frozen parameters are compatible
with input gradients. Clipping can produce zero gradients.

`context = snapshot.bind_current(raw_self, raw_task)` owns detached clones of
current self and the raw raster suffix. The internal context is created only
through this public method. `context.candidate_value(local_root_xy)` accepts
finite float32 `[B,S,10,2]`, with S>0 and B exactly matching the bound current
batch. It replaces only the first20 raw task entries and returns `[B,S]`.
The candidate tensor is not detached and its requires_grad is not changed.
The body and raster do not vary across candidates.

These ten XY pairs are already-local root-heading coordinates in **metres**,
at +0.5,+1.0,...,+5.0 seconds, in the same XY order as the sealed raw task
assembler. There is no scale10. A caller using world trajectories must first
subtract the same current root XY and inverse-rotate by the same current
root/task heading; this adapter performs no world conversion, resampling,
extrapolation, or coordinate inference. Raster grid/head/root/height-reference
choices remain those explicitly captured in the trainer configuration.

Construction and inference reject inference_mode, including input inference
tensors. Calls use enable_grad so that an enclosing no_grad context does not
silently remove candidate gradients. Callers remain responsible for supplying
requires_grad inputs when gradients are wanted and for using float32 without
autocast. No gradient/step-size optimizer, score combination, sample selection,
Diffusion sampler, or collision mask is supplied by this stage. Denormalized
critic value alone is not certified as a safe trajectory objective or as a
mandatory private Nav mechanism.

## Fixed provenance

The adapter pins exactly four direct dependencies: the sealed social trainer
81a6319f, policy2a4007f7, raw task layout6975d95f, and original normalizer source
707f355a. The trainer's own captured checkpoint contract separately covers
its declared configuration and dependencies. There is no recursive directory
inventory. The adapter captures its own import-time source record and checks
that current bytes still match at construction.

After a successful copy, `metadata()` returns isolated captured source records,
configuration, trainer provenance/contract, update count, and the three RMS
counts. Later calls read only owned copies, not later trainer weights/moments
or file contents. No actual future root tool receipt, checkpoint SHA, physical
clock, or training quality is invented. A legacy357/1044 model or RMS cannot
be passed through this boundary, and strict=False migration is not provided.

The implementation is an adaptation of the staged shared PACER-style CNN and
the local population-RMS mechanism. It is not a reconstruction of unpublished
Nav critic configuration. Actual snapshot ownership, numerical forward and
input-gradient behavior remain untested until root authorizes bounded NN
checks after the formal GPU workload and its required comparisons finish.

## Actual source checks

- SDK Python, hidden CUDA and PYTHONDONTWRITEBYTECODE=1: `compile` of source
  bytes only, tool chunk65f02f, exit0. The adapter module was not executed.
- `/home/lfm/anaconda3/bin/ruff check .../value_adapter.py`: chunk135006, exit0,
  `All checks passed!` No model/module import or NN test was performed.

Final root complete read1744d4 and independent full reviews found Required=[]. Root42131f verified all four direct pins and compiled bytes only, without importing the adapter or NN. Root0780fd sealed the **12573B SHA `9875c58b90e23ab6613ee7298bb4a199295435ad7f540f5b760e524a8f3b4f76`** source as an ordinary single-link0444 file. Existing65f02f/135006 compile/Ruff remain source-only results; numerical values, copy ownership and input gradients have not been executed.
