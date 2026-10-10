# Mixed curriculum style comparison preparation

**Prepared; no new model metrics have been computed.** Actual mixed500/1000/2000 validation captures are required before comparison. The new training is a fresh public PACER warm start, separate from the four-plus-four pilot.

## Frozen comparison contract

- Reference artifact SHA: `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`; the same validation22 source and selected arrays.
- Native child-joint velocity v7 manifest SHA: `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`.
- Existing `evaluate_pedestrian_style.py`, style feature/kernel module and same-frame decomposition script remain unchanged and are checked by SHA before execution.
- The complete nine historical `runs` dictionaries in `amass_root_style_v3/evaluation.json` and `comparison.json` are copied exactly; historical metrics are not recalculated. New runs are appended in a fresh output directory.
- Inclusive90D local pose/velocity Gaussian MMD² remains primary, with the same kernel, speed bins and sampling caps. All fall frames remain included; only declared initial frames and cross-sequence/reset derivative intervals are excluded by the existing reader.
- Upright-only results, speed distribution, yaw/wrist decomposition and near-ground body-origin toe metrics are secondary diagnostics. Toe/body-origin statistics do not establish actual contact or true COM.
- No sealed34 test outputs are read. The22 validation split has been repeatedly exposed and cannot be described as a blind test.

## Execution after root announces actual captures ready

Use the existing IsaacLab venv Python with CUDA disabled and four CPU threads. Pass any completed milestones, with fixed names `mixed500`, `mixed1000`, `mixed2000`, and a **fresh descendant** output directory. For example, with an actual completed validation capture path:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=.:src ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/mixed_curriculum_style_v1/compare.py \
  --run mixed500=/absolute/path/to/completed/full22_validation \
  --output docs/style_improvement_20261002/mixed_curriculum_style_v1/step500_comparison_v1
```

Later comparisons can supply multiple `--run` values and a separate fresh output. The tool rejects unfinished captures, other split names, different22 identities/protocols, changed reference/source hashes, checkpoint updates that differ from their milestone name, or unequal actual group-transition counters.

Milestone budgets at256 actors ×32 rollout steps:

| Milestone | Total physical transitions | AMASS transitions | Procedural transitions |
|---|---:|---:|---:|
|500|4,096,000|2,048,000|2,048,000|
|1000|8,192,000|4,096,000|4,096,000|
|2000|16,384,000|8,192,000|8,192,000|

These are acceptance expectations, not evidence that the run has reached any milestone. Actual reset/plan counts need not be equal. The tool records actual full-checkpoint metadata and preserves three-stream curriculum state identities but does not execute a physical or optimizer restore.

Outputs include new-only frozen evaluator results, a combined evaluation with the old nine dictionaries preserved, exact paired initial/reference checks, existing same-frame decompositions, and a combined comparison. This is the truefutureXY low-level tracking benchmark. Closed-loop learned Diffusion navigation requires a separate200s capture and will retain the old guidance20 clock/reporting contract.

## Prepared evidence

- [Preparation and exact hashes](preparation.json)
- [CPU comparison tool](compare.py)
- [Ruff](ruff.log)

The preparation gate passed without loading a new checkpoint or generating any new metrics. The new-run execution path remains unexecuted until genuine captures exist; it must pass before an actual comparison is published.
