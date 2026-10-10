# Shared social training AMASS expert connection — source preparation

The adapter and four fake CPU cases now pass one root-authorized CPU execution. Original methods sample synthetic strided tensors; real manifest metadata and bank stat are read. No real bank/checkpoint deserialization, NN/SDK construction, physical sampling or training occurred. Production185,
frozen World, all existing corpora and earlier trials remain unchanged.

## Future API

```python
expert = SocialAMASSExpert(world, world.curriculum.corpus)
batch = expert.sample(expert_batch_size)
# Existing trainer.update receives batch.features and batch.actions.
# Keep batch.motion_ids/frame_indices/history and expert.describe() as evidence.
```

The caller supplies an already constructed exact `SharedGroupWorld` (b608cae6)
and its same original train `AMASSCorpus`, on the same device. This slice creates
neither object, calls no reset/step/forward, and creates or reseeds no generator.
It requires an idle World (`_prepared`/`_pending` both `None`) and checks the
stationary native counter before and after sampling. Draws after a prepared
control or before terminal archival/reset completion are rejected before RNG
consumption. This adapter does not schedule those control or reset operations.
It neither archives a physical transition nor implies that a physical actor
followed the returned expert motion.

The default is the original `AMASSCorpus.sample_reference(batch, 10)` call:
uniform selected training clip followed by its original legal-center draw. It
is **not** uniform over all windows or balanced by dataset/style. Existing
shared corpus RNG is consumed exactly by that single original sample call;
no extra original draw is added. Expert batches remain unrelated to the
fixed-half AMASS/procedural assignment of physical actor rows.

The original [`AMPWorld.expert_batch`](../../../CrowdSim/humanoid/amp_world.py:268)
function is extracted as one exact AST node, rather than reimplementing its
arithmetic. Normalized AST SHA:
`dbf7c3d3419e4c9cb6a4be571a726a854f41c9f6f9fd633d703093256c5be787`.
Its recursive global inventory is only `torch`, `build_amp_observation`,
`action_to_pd`, and builtin `getattr`. These are supplied from the source-pinned
World's existing pure tensor dependencies; the NN/SDK-bearing original
`amp_world.py` module is never imported. A recording source facade forwards
the one exact call and retains its actual return values.

Returned fields:

- `features`: owned float32 `[B,10,195]`, newest source frame first, from the
  unchanged named COMMON-to-PACER observation builder.
- `actions`: owned canonical `[B,69]`, normalized next-frame **mocap joint
  position / actual named PD scale**, clipped to [-1,1]. Frozen hand/toe
  coordinates are exactly zero; the returned bool69 mask has57 active entries.
- `history`: owned detached tensors for the source history; these are corpus
  observations, not preceding controls from the shared simulator.
- `next_position_common`: owned next source-frame joint positions before the
  original COMMON-to-PACER action ordering and normalization.
- `motion_ids`, `frame_indices`: owned original CPU int64 sampled IDs/centers;
  the current center has nine earlier frames and at least the next frame in
  the same admitted train clip. No heldout arrays are read.
- `batches`, `windows`: cumulative successful adapter sampling exposure.

The proxy is **not measured torque, measured policy action, or a disclosed
Nav author demonstration action**. The source keeps the original knee scale,
hand/toe exclusion, clipping and joint ordering. It reports the existing
corpus requested/stored/output DOF-velocity frames without converting them
again. Input gradients are deliberately irrelevant for expert examples.

## Corpus and optional sampling admission

The constructor verifies the real bytes of fixed local v7 manifest SHA
`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`,
its exact loaded manifest document and train entry,2848 ID pool and caller-loaded
timing metadata counts.
This is the local reconstructed2848 segment set, **not the author's private
2848 list**. The manifest stat/read in source preparation was metadata only:
9,507,694 bytes; train bank declared1,155,817,682 bytes, SHA
`46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0`.
The adapter checks bank size and reports this hash as **declared, not recomputed**.
It does not rehash the1.1GB bank, deserialize it, or replace the caller's
prior resource/payload admission. Manifest equality is not bank content proof.
Before each draw the adapter checks the live shared corpus/CPU generator identity,
corpus sampling configuration and loaded manifest/ID pool, named mapping/dependency
identity, and exact saved float32 lower/upper PD limits. It never clones, reseeds
or rolls back a generator.

Opt-in q sampling requires an already constructed exact original
`ExpertReferenceSampler` and `ExpertWindowSampler`, the identical corpus, an
explicit `sampling_config` normalized by its existing source, and explicit
`sampling_provenance` equal to `_sampling_identity(prepared_sampler)` from this
adapter. That identity contains configuration plus the existing window schema,
configuration/registry/probability hashes and source provenance. Shared and
private generators must be distinct; the private window generator must be
the prepared reference sampler's same CPU generator. The identity and private
generator object are checked again before future draws; changes are rejected
before consumption.

Original q semantics are retained: one legacy corpus call is intentionally
consumed and discarded, followed by the prebuilt private sampler's replacement
draw. This changes the expert prior, applies no importance correction, and is
an explicit experiment. Its existing `state_dict()`/RNG counters must be saved
and restored by the future complete runner; this slice does not implement
checkpoint/SDK recovery. The default does not enable q or silently construct it.

Invalid batch/configuration fails before sampling. Exceptions or interrupts
after possible sampling RNG consumption fault this adapter, and it cannot be
manually retried. It does not roll back RNG or mark the World physically faulted.

## Four synthetic CPU cases

`test_expert_batch_CPU.py` uses exact original class AST methods on synthetic
strided tensors and bypasses corpus/World constructors with `__new__`. Only
manifest metadata and existing bank stat are consulted. It cannot prove
actual AMASS values, physical geometry, FK, native-device behavior or quality.

1. One original generator draw and real newest-first history/ID/frame output.
2. Noncanonical named DOF ordering, original knee scaling, clipping and57 mask.
3. Owned outputs, unchanged fake clock and synthetic source tensors, honest
   metadata labels; a specific packet-construction `KeyboardInterrupt` after
   the draw must propagate unchanged, fault the adapter, claim zero completed
   batches and refuse retry without another shared-generator draw.
4. Train-only/explicit-sampler/type/batch admission and idle World, mutated PD
   limits/manifest/mapping or replaced generator rejection before RNG consumption.

The optional q positive path is source-prepared, not executed by these cases.
No separate earlier suite, NN, SDK, dataset loading or physical test is added.
The actual execution below establishes synthetic CPU behavior only; full training integration remains pending.

## Actual source checks

First stdlib compile/AST plus Ruff tool `fc614f` exited0. It compiled both files
without executing them, proved the original expert AST/global inventory and
reported `torch_imported=False`. The later source-only tool `17ff7d` compiled
the prepared-generator admission revision, reported no imported Torch or SDK,
and Ruff passed. These checks preceded the root-required idle/live-decoder guard
revision; its final source-only check is recorded separately below. No test main or sampling call had been run at those source-only checks; later actual execution is recorded below.

Final guarded source/test stdlib compile/AST tool `1ca29e` exited0 and reported
`torch_imported=False`, `SDK_imported=False`; Ruff tool `6cdaab` exited0.
These checks parse/compile source without execution. The four cases were unexecuted at those source-only checks; later actual execution is recorded below. Live bank, optional q positive-path and SDK integration
behavior remain unverified.

Root's source review found that final owned packet construction had originally
been outside the post-draw fault guard. The unchanged pre-fix source is retained
readonly as `before_packet_fault_guard.py` (c95e77e1). The corrected source
constructs the complete owned packet, assigns successful counters and returns
inside the same `BaseException` guard; successful sampling/formulas are unchanged.
Final guarded packet source/test compile-only tool `baa594` exited0 with no
Torch/SDK import; Ruff `6b317a` exited0. The interrupt subcase is authored and
unexecuted at that source-only stage; subsequent synthetic execution is recorded below and is not full runtime recovery evidence.

## Actual root CPU execution

Root complete source633f1f, testcf1421 and packet-guard delta61043a reads preceded independent final whole review50f13c/4094a3: Required=[]/Optional=[]. Root958182 sealed source15279B/48bc4976 and test10758B/baa3f62a readonly/nlink1. One CUDA-hidden execution **567839/session95888 →33fabb closed0; 4 tests/.674s/OK**. [Actual receipt](actual_CPU_checks.json) is readonly/nlink1.

All cases use synthetic strided bank values and constructor bypass with real metadata admission. They prove original draw/order, named decoder/action proxy, owned outputs, idle/live guards and interruption fault/retry handling at that seam. CUDA remains uninitialized, global Torch RNG unchanged and SDK/NN-bearing module imports absent. No real AMASS data/FK/native source, optional positive q path, policy update, complete checkpoint recovery or paper result is established.
