# Experimental AMP upper-limb blend: CPU readiness

This is an isolated, deterministic evaluation controller. It combines two trained reconstruction AMP actors and is **not** the paper's single trained actor. It is prepared for a future explicit evaluation hook; this report contains no physics deployment, balance, navigation, style-quality, or safety result.

## API and action contract

```python
from CrowdSim.humanoid.runtime.arm_style import ExperimentalArmStyleInference

controller = ExperimentalArmStyleInference.from_checkpoints(
    base_checkpoint,
    style_checkpoint,
    alpha=0.0,
    device="cpu",  # Explicit target device; this readiness proof uses CPU only.
    base_sha256=base_expected_sha256,
    style_sha256=style_expected_sha256,
)
output = controller.act(self_obs, task_obs, deterministic=True)
# At a future AMPWorld integration, consume only output["applied_actions"].
# AMPWorld performs the one PD decode; this module never decodes PD targets.
metadata = controller.describe()
```

Inputs use the existing reconstruction contract: FP32 self observations `[B,357]`, task observations `[B,1044]`, history 10, control interval `1/30 s`, and 5-second future XY horizon. Each actor retains its own frozen self/task RMS. Parameters/mask must be FP32, RMS buffers FP64, and both actors must use one device. Manual dtype/device/schema/mask changes fail before inference; stochastic actions and training mode are rejected.

Only the XYZ axes of `L/R_Thorax`, `L/R_Shoulder`, `L/R_Elbow`, and `L/R_Wrist` can change: 24 active normalized coordinates. The remaining 33 active coordinates and 12 frozen hand/toe coordinates retain the baseline raw means. There is no branch identifier or new policy input.

For the selected 24 coordinates, `mean = (1-alpha)*base_mean + alpha*style_mean`. Other coordinates retain the baseline mean. The composite then applies `clamp(mean,-1,1) * canonical57_active_mask` once. Mixing already clamped actions gives a different controller and is prohibited by the tests. At `alpha=1`, the style's 24 raw means are assigned directly; at `alpha=0`, the baseline output dictionary is returned directly and no style forward is called. Consequently alpha zero preserves all baseline output bytes, including signed zeros.

The returned keys are only `actions`, `mean_actions`, and `applied_actions`. They are normalized action-space tensors, not decoded physical joint targets. The composite has no defined Gaussian likelihood, critic value, AMP reward, discriminator substitution, optimizer, replay state, pose injection, or reset behavior. Do not reuse baseline log probabilities to train this composite.

## Physical compatibility admission

Matching action dimensions alone is insufficient. Construction rehashes the exact artifacts loaded by the restricted v2 loader, binds size/SHA and unchanged file stat, and reads their finite JSON physics provenance. Both require canonical body/DOF order, 69 finite DOF bounds, named PD/armature/effort/velocity controls, floor material, and PhysX120/30 Hz. All recorded physics fields must match except the explicitly reported `terrain_extent_m` and `isolated_spawn_margin_m` training-domain fields. A future world integration still needs to compare its live rig and material to this recorded contract.

The actual selected artifacts both record the current neutral SMPL rig, canonical names, identical DOF limits/PD/material/body masses and a 192 m isolated training domain with 48 m spawn margin:

| Role | Artifact | Updates | Real training transitions | SHA256 |
|---|---|---:|---:|---|
| Baseline | `output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt` | 1000 | 8,192,000 | `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b` |
| Upper-limb style | `output/diffusion_amp/amp_style_mixed_curriculum_snapshots_v1/step_2000.pt` | 2000 | 16,384,000 | `28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72` |

These are existing independently trained artifacts. The composite performs no additional training and this proof does not designate the style model as a successful deployed controller.

## Executed CPU gates

- `tests/test_amp_arm_style.py`: **29 passed**, final run after physics validation tightening, 5.50 s; JUnit results in [tests.xml](tests.xml).
- Ruff target Python3.12 for the new module, tests and proof tool: **passed**.
- [Actual pair verification](actual_pair_verification.json): two immutable checkpoints; **3 mathematical FP32 observation batches** `(B,scale)=(1,0),(8,1),(32,50)` from a private CPU generator seed20261002; three alpha conditions per batch; **27 composite output tensor records**. Alpha zero matches all three baseline outputs bytewise. Alpha one takes only the 24 style raw coordinates; other 45 raw/applied coordinates remain baseline bytewise. Alpha0.25 matches the raw-mean blend and final projection bytewise. Repeated calls are bytewise identical.
- Python, NumPy and Torch caller RNG, every model parameter, and every RMS/mask buffer remain bytewise unchanged through loading, construction and calls. CUDA was not initialized; no `omni`, `isaaclab`, or `isaacsim` module was imported. No heldout34 outputs were read.
- Unit tests cover incompatible schema/mask/dtype/device, invalid alpha, incompatible same-dimensional physics, explicit allowed domain differences, invalid input, stochastic/training rejection, and the observable difference between raw-before-clamp and projected-action blending. Synthetic fixtures prove contracts only.

The proof tool initially failed while byte-viewing a scalar FP64 RMS buffer; its byte-comparison helper was corrected to flatten scalar buffers. No controller change was needed. The final actual proof completed exit0.

Reproduce from the repository root:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest tests/test_amp_arm_style.py -q

env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/arm_style_inference_v1/verify_actual_pair.py
```

## Deployment boundaries

Baseline legs and torso do not guarantee baseline stability after arm forces change. Upper limbs can affect whole-body balance, ground contact, peer collision, and navigation. No choice of alpha is recommended as physically safe by this CPU readiness report. Future real evaluation must explicitly record both immutable checkpoint hashes, alpha, the composite source hash, the live rig contract, and every fall/contact/navigation result; preserve the original baseline and failing historical runs.

Only the new module/test/proof/documentation were added. Source135 training business, nav138, thin-entry5, weights, datasets, existing control loops and deployment defaults were not modified. This module is opt-in and the existing main discriminator is unchanged. The source archive records this readiness implementation; it contains no checkpoints.
