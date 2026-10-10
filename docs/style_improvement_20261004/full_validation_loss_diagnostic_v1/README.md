# Complete validation loss diagnostic (prepared; models not executed)

The fixed all-temporal validation population is **Maps 10,566 + Interact
19,386 = 29,952 windows**, not 29,952 Maps windows. Each domain has 100 episodes.
The formal training curve instead uses 4 × 200 fixed with-replacement windows
per domain. A full-population result must be reported alongside that curve,
without treating their different populations as identical.

`evaluate_full_validation.py` reuses frozen paired source `9f7732c9` for source
and terminal admission, the all-temporal v3 corpus, `make_training_batch`,
original `training_loss`, and original `validate_resume`. No loss kernel,
planner sampling, geometric metric, physical variable or training update is
introduced. The default CPU batch is 32; final batches are Maps 6 and Interact
26 windows. Each original batch mean is weighted by its window count. The
union result is window weighted, while both domain results remain separate.

Every canonical window has a private CPU generator seeded by
`20261101 + domain_index*1000000 + local_window_index`: randint100[1], then
randn[52,5] float32. Thus batch size only controls throughput, not the t/noise
assigned to a window. Whole-domain t/noise array byte hashes are retained.
Loss is one fixed realization per window, not the full noise expectation.
All zero-neighbor and legal empty-map windows remain. `trace_unknown_raster`
uses the original conditioning policy and loss remains eval/no-dropout.

## Actual CPU inventory

Initial `actual_window_inventory.json` was created by real tool `75bb25`, exit 0,
0.1656 seconds; 232,375 B, SHA
`928829658c8499f6c47e84792ec38cf1e38ce1c4cf28706fa1dfed2baa9cbf7f`.
It verifies all 200 validation episode index artifacts against the canonical
agent/anchor order, records source identities and fingerprints, and verifies
original readonly state/map bytes. It loads NPY indices only: no NPZ state/map
arrays, raster crops, checkpoints, Torch, NN, SDK or GPU.

Two pure scalar aggregation tests passed (`85e5c0`, exit 0); Ruff passed
(`4f22dc`, exit 0). These check tail weighting, missing/overlapping coverage and
invalid scalar inputs, and are not model validation.

The initial entry bytes are preserved readonly as
`before_normalization_revision.py` (SHA `b92dd4d2…1129370`). After the necessary
normalization admission revision, real tool `cd1c35` exited 0 in 0.1814 seconds
and wrote `inventory_v2.json` (232,659 B, SHA
`491f461887ba40cf158e5f395b01e11ee64c454faaab56231dc7857136cc3abf`).
Its complete population and source artifacts match the initial inventory
exactly; its entry/helper identity is the revised source. The final four scalar
tests passed (`b2f868`, exit 0), and Ruff passed (`5c9bf5`). Details and both
retained source identities are in `CPU_preparation.json`.

Root independently reviewed and froze the final sources, both inventories,
checks and shared helper (`c7945c`, exit 0). The source peer reported
Ready / Required=[] after actual source-only check `23b996`; see
[root_source_freeze.json](root_source_freeze.json). Its pending model/audit
fields remain null. The earlier preparation record retains its historical
`source_freeze_pending_root_review` value; this later root record closes that
source review without changing the earlier evidence.

## Future execution resources and inputs

Use the existing SDK Python, hidden CUDA and four CPU threads. CPU model memory
and run time have not been measured; the batch may be reduced while retaining
the same per-window t/noise. No full 29,952-window raster array is constructed.
Root must first finish the fixed 40k comparison, observe formal training
closed0, and obtain actual PASS from terminal reader `77025`. The existing
paired `admit` checks that root closure, frozen plan, terminal source and model.

Root then supplies the readonly inventory SHA, actual terminal audit SHA, and
optionally a readonly historical records JSON + its SHA:

```json
{"checkpoints": [{"checkpoint": {"path": "ACTUAL_READONLY_COPY", "bytes": 0,
"sha256": "ACTUAL_SHA"}, "optimizer_updates": 15000}]}
```

The example is a shape illustration, not admissible actual evidence. Each
history byte identity and nlink1 is checked; the complete original
`validate_resume` binds config, recipe, raw/EMA/Adam finite state, counts and
corpus metadata before any forward. Source bytes are checked again afterward.
The shared standard-library `normalization_contract.py` admits raw mean/std
only when exact to config converted to FP32. Each EMA value must be either
that config value or the FP32 value computed by the original recipe for its
saved update count. This permits the specifically justified constant-buffer
rounding path, not arbitrary 1-ULP shifts or unrestricted allclose. Actual,
expected and configured values, signed deltas and ULP distances are reported;
buffers are never rewritten. Tests cover the known 1.5 blend result, exact raw,
both supported EMA variants, zero subnormal rejection, unsupported 5.0 ULP,
wrong shape and nonfinite values. Real model normalization remains unobserved.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/full_validation_loss_diagnostic_v1/evaluate_full_validation.py \
  --mode evaluate --terminal-audit-sha256 ACTUAL_TERMINAL_AUDIT_SHA \
  --inventory-sha256 ACTUAL_FINAL_INVENTORY_SHA \
  --history-records ACTUAL_ROOT_RECORDS.json --history-records-sha256 ACTUAL_RECORDS_SHA
```

Future identity values are unknown. No models have run for this diagnostic.
History is exploratory evidence, never test-based checkpoint selection or
automatic promotion. The registered 40k main result and production185 remain
unchanged. No test-domain data or sealed34 outputs are evaluated.
