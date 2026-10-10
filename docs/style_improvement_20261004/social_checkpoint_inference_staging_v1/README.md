# Social checkpoint inference reader — source preparation

Only source was authored/read/compiled/linted. The reader and its parent have
not been imported here; no Torch, NN, checkpoint, dataset, SDK, test, inference
or physics operation was executed. No actual checkpoint identity/result is
invented. Production185 and the sealed052618 actor snapshot remain unchanged.

```python
actor = FrozenSocialCheckpointInference.from_checkpoint(
    selected_readonly_checkpoint, expected_sha256=actual_complete_SHA256,
    device="cpu",  # or an explicit indexed CUDA target after caller SDK launch
)
prediction = actor.act(raw_self_B357, raw_task_B12308, deterministic=True)
```

[checkpoint_inference.py](checkpoint_inference.py) subclasses the exact sealed
052618 actor snapshot. It inherits its normalization, deterministic actor-head
forward, arbitrary positive B, strict F32/device/storage checks and clamp×57
mask; no duplicate policy forward or fabricated `__new__` trainer is used.
Saved training actor_count remains provenance, not inference batch size.

## Future admission and owned runtime path

1. Require an explicit local, nonsymlink, independent regular file with nlink1,
   no write mode bits and mandatory complete selected SHA256. Hash its full
   bytes through O_NOFOLLOW and compare descriptor/path device, inode, size,
   mtime, ctime, mode and nlink before/after. Keep the admitted descriptor open
   through loading/copy and load its Linux `/proc/self/fd/...` filename to bind
   mmap to that inode across pathname replacement. Reject LFS/non-Torch-ZIP.
2. Verify readonly actor052618/helper9875/trainer81a byte identities before
   executing dependency definitions. Only the explicit loader call imports
   the trainer's Torch definitions. Restricted `torch.load(map_location='cpu',
   weights_only=True,mmap=True)` reads the complete container; unused training
   tensors may be mapped/deserialized, but are never restored or retained.
3. Reuse exact81a `_admit_metadata` before the first network constructor. Require
   its social format, complete source/config/derivation contract, state_action
   D2019, fresh initialization declaration and positive declared update/physical
   counters. Legacy1044, unknown/partial schemas and changed feature contracts
   are rejected rather than migrated.
4. Explicitly construct one temporary full SocialAMPNetwork and four original
   RMS objects under CPU placement and `fork_rng(devices=[])`. This **does
   initialize NN parameters temporarily**; caller CPU RNG is preserved on
   construction/validation failure and success. No SocialAMPTrainer constructor,
   Adam/replay, global seed or saved training RNG restoration is called.
5. Validate every complete model key, shape, dtype, finite CPU/strided storage
   and exact fixed log_std−2.9. Validate all four RMS complete float64 mean/
   variance/count tensors and configured clip5. Counts/variance are nonnegative;
   original self/task/value counts equal positive declared physical transitions.
   AMP count keeps81a's separate nonnegative contract and is not assigned the
   physical count. Strict loading copies weights/buffers rather than retaining
   mmap aliases; original epsilon1e−5/population moments remain unchanged.
6. Retain only encoder, actor, self/task RMS and canonical57-active/69 mask.
   Discard critic/D/log_std, AMP/value RMS and the checkpoint container. Freeze
   the owned path; optionally move only that path to the explicit target. Recheck
   storage and full checkpoint bytes/stat/source identities before returning.

`describe()` distinguishes checkpoint-declared counters from independently
observed physical transitions, records temporary full NN construction and the
source/config/grid_center/grid_heading/height_reference/include_ego contract.
It does not attest optimizer/replay/RNG validity, terminal training completion,
SDK restoration, collision safety or motion quality. Changing saved RMS or
building a new12308 moment estimator at evaluation is unsupported.

The reader supports raw357/12308 deterministic actor inference only. It does
not decode PD, produce observations, run Diffusion/value guidance, export ONNX,
or launch full22/four-person200s/native evaluations. Those external producers
and comparison-preserving lifecycle adaptations are specified in the existing
[integration boundary](../social_deployment_integration_v1/README.md).
The descriptor filename is Linux-specific. Installed primary source
`/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/torch/serialization.py`
is87326 bytes, SHA `ce5bc5ca6a8faa2b5aee2c72ff1f1d4b02c2a2d82e96886de74ae9fc436a7f67`;
lines1540–1578 use the supplied path for ZIP reading, getsize and
UntypedStorage.from_file in the mmap branch. Source read86dafd and
compile-only747d68/Ruff550f29 exited0 without Torch/SDK import. This is source
grounding, not an executed loader/NN test or cross-platform compatibility claim.
Source checks are not runtime/load/network tests or a receipt for a future run.

Independent full-source peer completed with Required=[] / Optional=[] (0aabdb source-only). Root whole-source reads de27a1/acf9b5 and README406385 confirm the same final source. Files are sealed readonly after review; actual checkpoint/model/SDK execution remains pending.
