# Prepared terminal state-prior evaluation

Status: **prepared; future_effect=false**. This directory admits no future
checkpoint or motion quality. Preparation used CPU only, without Torch, Kit,
GPU, sealed test34 outputs, or a mutable training checkpoint.

## Registered protocol

`registration.json` retains the previous protected-terrain protocol's parsed
values except the new checkpoint/output/profile/scene paths. Its actual
configuration-only import ledger records eight executed modules from the new
934-file immutable code bundle.

- Tracking: the complete 22 validation motions and all 11,982 controls, with
  failed motions retained in the full-sequence denominator.
- Navigation: four people, 6,000 controls, 200 seconds; the original physical
  fall definition, route fixture, causal guide and contact measurements.
- Primary style: the original 90D pose/velocity metric, frozen c274348f AMASS
  reference, sigma=1, minimum bin size=30, maximum samples per bin=256. Failed,
  reset and idle frames remain included. Unsupported bins remain unknown.
- Secondary pose diversity: the saved AMASS-only b25dffa6 basis with 50/100
  centers in 45D. The basis is never refit and does not replace primary 90D.

The registered navigation minimum is zero falls, at least 21 route vertices
and inclusive primary MMD² at most **0.07238060315284697**. Complete full22
tracking and fixed-basis 45D reports are also required. Execution success alone
does not establish gait quality or admit native follow-on training.

## Terminal admission and execution

`run_registered_evaluation.py` requires the actual sealed `step_2000.pt`, its
full SHA, and the independent new reader's `terminal2000/audit.json`. It checks
the reader/helper against frozen source bytes, the new source-manifest SHA,
2,000 updates/16,384,000 transitions, complete resume payload, D2019, coupled
public D/history-RMS transfer and actual AMP count 24,776,000. Pilot/milestone,
old 931 source, legacy D/RMS-false reports and mutable checkpoint aliases are
rejected. Alias rejection happens before any checkpoint bytes are read.

`commands.json` contains the exact root-only GPU argv for tracking and
navigation. Replace only `ACTUAL_SEALED_STEP2000_SHA256` with the sealed SHA;
the placeholder itself is invalid. Run protocols sequentially after the
terminal reader passes and the training App closes, with root's single GPU.
The registered output and import-report paths must be fresh. The SDK Python
path is lexical; resolving it to a bare interpreter loses the SDK environment.

The executor uses the immutable bundle's original strict launcher and full
network inference adapter. A guarded import observes the selected entry's
module execution before `runpy`; this records an execution boundary without
claiming physical completion. Runtime import finalization is recorded only
when actually observed.

`evaluation_evidence.py` is the CPU source-admission bridge for later metric
readers. It requires completed captures with every registered argument intact,
the actual synchronous import event prefix and all 934 copied source hashes.
Large shared banks are checked by link/stat there; the launcher performs its
own resource verification before execution. This bridge does not compute or
invent future quality scores. The original 90D and 45D source identities are
recorded in the registration and commands.

## CPU preparation gate

55 contract tests passed with pytest plugin autoload disabled. They cover
coupled transfer and count semantics, reader/source mismatches, real pilot8
rejection, partial/test/output substitutions and mutable-alias refusal before
file reads. The actual SDK-free parser ledger finalized successfully and all
eight owned imports came from the immutable new code. Ruff passed.

Historical failed test XML files are retained: the first attempts used an
incorrect fixture expectation for the launcher's `finalized`/`phase=passed`
ledger fields; the final test file uses the observed schema. Default pytest
plugin autoload initially failed before collection on an unrelated ROS `lark`
dependency. No training or evaluator source changed in response.

`source_freeze_v1.json` and its source archive bind this prepared executor,
registration, evidence bridge, commands, tests and CPU proof. The old frozen
registration, watcher, evaluator tools and production files were not edited.
