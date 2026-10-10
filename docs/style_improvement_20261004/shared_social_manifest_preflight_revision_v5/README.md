# Validate AMASS data before starting Isaac Sim

The v4 probe failed before World construction because the fixed AMASS manifest
is a regular file with mode `0664`, while the entry required a readonly file.
The actual process exited `139`; its result and log are retained separately.

This revision accepts the existing data file based on its fixed SHA256 and the
complete reconstructed train2848 schema, checks for a concurrent file change,
and validates it before starting the App. Source files and training profiles
retain their strict checks. The manifest is checked again after controls or
training. No dataset permissions or training equations change.

This prevents the observed data-validation error from creating an App. Cleanup
of unrelated errors after App startup and before World construction remains
unverified. Native lifecycle evidence from the successful v18 physical smoke
does not by itself prove this shared World or training entry works.

## Bounded checks

```bash
PYTHONDONTWRITEBYTECODE=1 python docs/style_improvement_20261004/shared_social_manifest_preflight_revision_v5/check_manifest_preflight.py

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/shared_social_manifest_preflight_revision_v5/probe_shared_world.py \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --num-actors 8 --control-steps 8 --fall-probe --device cuda:0 --seed 7 \
  --output output/diffusion_amp/shared_social_world_probe_preflight_v5
```

Run the shared training smoke only after the probe completes with a real process
exit `0`. Every output directory must be new; failed runs must be preserved.

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/shared_social_manifest_preflight_revision_v5/train_shared_social.py \
  --config docs/style_improvement_20261004/shared_social_training_entry_staging_v1/smoke.json \
  --device cuda:0 --output output/diffusion_amp/shared_social_training_smoke_preflight_v5
```

These checks use the local reconstructed AMASS list, not the unpublished author
clip list. They do not establish learned locomotion quality or paper results.
