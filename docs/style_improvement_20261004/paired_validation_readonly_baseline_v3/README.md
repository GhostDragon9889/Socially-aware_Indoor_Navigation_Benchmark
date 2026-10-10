# Paired validation: independent OLD baseline snapshot v3

This is source-only preparation. The frozen original reader, EMA-normalization
v2 reader, original selection/registration, production185, formal training and
default baseline are unchanged. No candidate module/function, NumPy/Torch,
model, dataset, checkpoint tensor, SDK or GPU was executed/read for this stage.
Only source files and the root's small actual copy receipt were inspected.

## Declared source and actual model load

The frozen registration continues to declare
`social_diffusion_training_extended_v2/step_5000.ckpt`: 279052920B,
SHA `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`.
Root's actual receipt records this original as mode0600/nlink2. This reader
does not chmod it, replace its inode, or alter either hardlink.

The actual OLD model load uses only:

`output/diffusion_amp/formal_EMA40000_vs_63e_baseline_snapshot_v1/OLD_step5000.ckpt`

This independent copy must have the same complete byte count/SHA, readonly
mode0444, nlink1, and a different device/inode pair from the original. The
root-created `snapshot_receipt.json` is pinned in the source registration;
its exact original/copy identities and preservation flags are checked. The
reader rehashes both artifacts before the model load and after evaluation,
checking opened-file/current-path metadata stability and unchanged observed
identities. Original mode/link/device/inode must still equal the actual root
observation. This exception is limited to hashing the declared original; the
global original `verify` remains unchanged and the loaded copy remains strict.

The result's `OLD` names the actual loaded copy, `OLD_declared_original` keeps
the original declaration, and `OLD_snapshot_provenance` records both full
hashes/metadata and the actual root receipt. Receipt binding does not authenticate
the root tool's exit or validate checkpoint tensors; those are separate claims.

## Exact derivation

The entry pins the complete original 22889B reader (`9f7732c9…`) and complete
8360B v2 reader (`1cca02c0…`). It parses v2's four literal changes without
importing or executing v2. Five additional exact replacements cover:

1. OLD admission in `admit` (one I/O line).
2. OLD model load path.
3. Actual-copy/declared-original report provenance.
4. An independent default output directory.
5. The result schema's v3 identity.

All nine changes must reverse to the entire original bytes. Ten original
functions retain exact bytes/AST, including `verify`, source admission,
validation loading, selection, array records and metric aggregation. The
eleventh previously unchanged function, `admit`, differs only in its OLD line;
the actual77025 terminal audit and closed0 owner checks are otherwise exact.
The normalization callback is AST-identical to v2 and uses the same fixed
4079B `normalization_contract.py` (`2bfb3f97…`). No stored buffer is rewritten.

Actual source checks: chunk `2fc12a` exit0 verified literals, full inverse,
ten untouched functions, one-line admission scope, the normalization callback
and compilation. Ruff chunk `e438aa` exit0 passed. These checks did not import
or call candidate code. No tests or new runtime receipt framework were added.
Root and independent whole-source review passed with Required=[]/Optional=[].
Sources and registration are now sealed readonly for the actual invocation.

## Future root invocation

The same already frozen main200 population and separate504 diagnostic stratum,
selection/loss/plan seeds, model APIs, paired noise/timesteps, batch size,
per-window metrics, failed samples, empty-peer handling and denominators remain.
The new entry cannot prepare/reselect windows. NEW admission remains the original
terminal contract; no unknown future checkpoint identity is filled in here.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/paired_validation_readonly_baseline_v3/evaluate_paired.py \
  --mode evaluate --terminal-audit-sha256 ACTUAL_SEALED_TERMINAL_AUDIT_SHA \
  --output-dir output/diffusion_amp/formal_EMA40000_vs_63e_validation_readonly_baseline_v3
```

This preparation is not an actual model comparison, physical validation,
author-benchmark reproduction, automatic selection or promotion.

## Root source review and seal

Root read the final entire entryaa1ba8 and registration/READMEce8c1f. Independent whole-source and static review7438a5 exit0 confirmed Required=[]/Optional=[]. Root confirmed final15138B source SHA `adeaea4178284887bad77db8e8cf795d66c74e52b0812562e45617f528fa5e99`, ordinary/nlink1 sources and set0444. Actual terminal tuple-adapterauditor31365/e87839→999fed closed0, report6478B SHA `fbe1880fae2b4687165d7caeffd9bf6fd8c3cdedf57a44b26d0030185e5c1e44`; actualOLD copyb3f1f8 is already sealed. Actual model execution/closure will be recorded independently; source review alone is not a result.
