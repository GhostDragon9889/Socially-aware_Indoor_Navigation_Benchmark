# Independent public state prior checkpoint auditor

`audit_state_prior_training.py` reads actual sealed training artifacts on CPU.
It derives from the frozen protected terrain auditor V4, whose source SHA is
`9966d01208da07cee9a6422e4a9a2cdfff205a14ea61fc3e62058fd31b4bcf89`.
The old auditor and old 931 source copy remain unchanged.

The new contract requires the explicit state prior flag, both public D/history
RMS transfer flags, the complete zero action initialization descriptor, the
actual original public counts, and the count cap 200000. It verifies the
2019 input D and 1950 history RMS, AMP count `200000 + updates * 12288`,
57 trained nonzero action columns and 12 frozen zero columns. The zero action
initialization descriptor describes the initial state; restoring a trained
checkpoint must preserve its learned columns.

Full model/optimizer/RMS/replay/RNG/physical runner state restores through the
actual new isolated loader. Real AMASS and mixed curriculum restore all three
private RNG streams and repeat the next reference/expert/plan draws exactly.
Inference loads only the selected checkpoint, preserves trained D weights,
and validates then discards AMP/value moments as the original full network
adapter does. The CPU synthetic inference inputs are not saved GPU observations.

All 934 immutable code files and the independent source archive are checked.
The archive also contains source manifest, launcher, commands and watcher.
The previous 931 manifest and the 927 unchanged source files are verified.
Actual synchronous import ledger snapshots and committed event prefixes are
retained. Root launch commands are matched exactly to their frozen plan;
closed phases require an observed process exit 0. Pilot artifacts must be
root sealed, immutable, and match their recorded bytes and SHA.

## CLI

Use the SDK Python with `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`.

```text
audit_state_prior_training.py --phase pilot4
  --checkpoint <new-pilot>/step_4.pt
  --result <new-pilot>/sealed_step4/result.json
  --metrics <new-pilot>/sealed_step4/metrics.jsonl
  --seal <docs-actual-runs>/pilot4_seal.json --output <fresh-docs-directory>
```

For `pilot8`, supply checkpoints 4 then 8 and both root seals, with the sealed
step8 result and metrics. The first four metric rows must equal sealed step4.
For `milestone`, supply exactly one watcher snapshot at update 500/1000/2000,
live committed metrics, and omit `--result`. For `terminal2000`, supply the
watcher step2000 snapshot, actual closed result and complete metrics. Terminal
admission additionally requires the registered full fresh process to exit 0.
The full fresh command contains no resume or from scratch flag.

`audit.json` schema is `public_state_prior.actual_training_checkpoint_audit.v1`.
Admission requires `status=passed`, exact `source_manifest`, exact selected
checkpoint record, all checks passed, and the source freeze records matching
the running reader. Top and per checkpoint public transfer/init flags are
true. `actual_terminal_fresh2000_confirmed` is true only for terminal2000;
`milestone_only_admission` must be false for terminal admission.

## Actual evidence

`actual_pilot4_final_v2/audit.json` passed 2068 checks on the actual immutable
step4 checkpoint: 4 updates, 32768 physical transitions, AMP count249152,
replay32768, discriminator Adam8 steps. `guard_tests.json` records28 passing
tests, including actual frozen old checkpoint rejection and actual new
checkpoint acceptance, provenance/tensor/flag/count mutations and committed
metrics/phase rejection. Weakening one strict boolean guard in a separate
`/tmp` copy makes two rejection tests fail; the review source is unchanged.

An earlier enhanced audit mistakenly expected inference to retain AMP RMS;
the framework AttributeError report is retained in `actual_pilot4_final`.
The actual adapter validates and discards AMP/value RMS. The corrected audit
checks this behavior and complete trainer RMS restoration separately.

These checks establish saved state and execution provenance, not improved
motion quality, author checkpoint compatibility or bitwise PhysX resume.
CUDA RNG bytes are preserved without execution. Shared large resources are
checked through frozen declared size/mtime/link contracts, not rehashed here.
