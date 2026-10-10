# Prepared CPU metrics after real terminal evaluation

Status: **prepared_not_executed; future_effect=false**. No future checkpoint,
terminal report or capture was read. No actual score has been computed. The
previous evaluation registration/source freeze remains unchanged.

## Commands

`commands_CPU_metrics_v1.json` contains complete lexical SDK Python argv,
the CPU environment, inputs, outputs, real GPU import reports and exact
metric source identities. Set `CUDA_VISIBLE_DEVICES=''`, OMP/MKL=4,
OPENBLAS=1 and `PYTHONDONTWRITEBYTECODE=1`. Run each command in a separate
process, in this order:

1. `tracking90`: the original 90D evaluator on the complete registered
   22-motion/11,982-control tracking capture.
2. `tracking_decomposition`: the original same-frame root/yaw/wrist/body
   decomposition and the frozen comparison tool's summary function.
3. `navigation_raw`: the original f851 ground/contact/COM/clock audit and ee0
   idle/phase diagnostics on the complete 4-person/6,000-control capture.
4. `navigation90`: the original 90D evaluator on that navigation capture.
5. `navigation45`: the original pose-only evaluator with the saved AMASS-only
   b25dffa6 50/100 centers. The original fitting function is replaced with an
   exact saved-center lookup; no generated samples fit a reference.

Outputs are separate fresh directories under
`docs/style_improvement_20261003/state_prior_metrics_v1`. Tracking decomposition
requires the preceding tracking90 result. Each metric saves its actual CPU
import ledger and the actual GPU ledger/event prefix it consumed. Failed
directories/logs/ledgers are retained; existing outputs are never overwritten.

## Source admission before metrics

The runner verifies its frozen catalog, the existing 0042d4 evaluation source
freeze, every frozen reader/bridge dependency and all 934 copied source files.
It then calls the existing new934 `evaluation_evidence.admit_actual_evaluation`
on the real registered `result.json`, the registered independent terminal2000
reader report and the actual GPU executed-import ledger. This requires new934,
complete registered CLI/counters, coupled public D+AMP RMS initialization and
the sealed terminal checkpoint SHA. Only after this admission does it process
a physical archive, reference population, basis or measurement function.

90D uses the immutable CODE module directly under the original strict owned
import guard. The original documentation tools are copied byte-for-byte to
read-only `/tmp` files before execution so the guard's prohibition of
executable production-checkout paths remains intact. Their actual staging
paths/hashes are recorded and retained. Owned libraries always resolve from
the new CODE. The wrappers redirect source roots and the old raw auditor's
epoch hook; metric function bodies are unchanged. The old full comparison
tool's `main` is not invoked because its subprocess path targets production.

## Sample and metric rules

The frozen c274348f validation reference, sigma=1, minimum bin size=30 and
maximum samples per supported bin=256 remain fixed. Primary physical-speed
buckets are `[0,.1,.5,1,1.5,2,3,∞)`; the original decomposition's reference-speed
buckets start `[0,.5,1,1.5,2,3,∞)`. These are separate original diagnostics.
Unsupported bins stay unknown.

Failed, fallen, reset and idle source frames are retained. The original helpers
exclude the first frame and cross-episode/reset derivative intervals, use the
actual post-step endpoint and retain all legal interval endpoint poses. Raw
idle/phase calculations keep the original saved before-control plan versus
post-step endframe relationship. No upright, idle or outlier filter is added
to either primary score or pose-center assignment.

45D uses the unchanged fixed-chain-scale local pose and deterministic inclusive
MMD sample cap. Its output centers are checked elementwise against the saved
basis. This remains a diagnostic, does not replace primary 90D and is not an
author Table III score. A metric execution pass alone is not a quality pass;
the existing navigation/full22/45D registration remains authoritative.

## Preparation evidence

`--prepare-only` hashes frozen sources without reading a real result, admission,
capture or checkpoint and without importing Torch or SDK. Nine CPU tests cover
source/input/output/terminal/parameter changes, saved-center reuse and stopping
before metric processing when source admission fails.

`actual_owned_imports_CPU_v2.json` and its finalized synchronous ledger prove
six actual owned imports from CODE and execution of five exact staged tool
modules. They also check the original primary speed buckets and the unchanged
tracking decomposition's analytic rigid-transform invariance. This is source
and mathematical preparation evidence, not a future model-effect report.
The prior source-only proof and initial test XML are retained.
