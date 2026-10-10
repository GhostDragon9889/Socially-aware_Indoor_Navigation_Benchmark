# Explicit deterministic evaluation controls

Implemented 2026-10-02 for controlled evaluation after the unresolved S1
render/physics divergence. This change does not identify its cause or establish
a successful fix; a paired actual run is still needed.

## Opt-in behavior

`--deterministic-evaluation` defaults to **False** and requires positive
`--evaluate-steps`. Training requests using it are rejected by the SDK-free CLI.

When enabled, the runner validates `CUBLAS_WORKSPACE_CONFIG` **before resource
preflight or Kit launch**. It accepts existing `:16:8` and `:4096:8` values,
defaults an unset value to `:4096:8`, and raises `ValueError` for any other value
without overwriting it. This avoids changing the workspace after CUDA has
already initialized inside the SDK.
The shared CLI also performs the same read-only validation, so a pure
preflight rejects invalid opt-in values without setting an absent variable.

After Torch import and before world construction, the runner enables
`torch.use_deterministic_algorithms(True, warn_only=False)`, disables cuDNN
benchmarking, and enables cuDNN determinism. It does not change TF32 or matmul
precision, sampling count, model weights, diffusion step count, seeds, or
physics configuration. Unsupported deterministic CUDA operations fail through
the existing error-reporting/cleanup path; there is no fallback.

The false/default branch does not set an environment variable, reset inherited
Torch flags, or impose these controls. The runtime report records actual process
state in `runtime_numerics` and `runtime_execution.numerics`: deterministic and
warn-only flags, cuDNN flags, float32 matmul precision, CUDA matmul TF32 and
reduced-precision reduction flags, and cuBLAS workspace value. It explicitly
records `physx_warp_bitwise_guarantee: false`.

PyTorch treats random seeds and deterministic algorithms as separate controls.
[PyTorch 2.11 reproducibility documentation](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
NVIDIA documents the supported workspace values for cuBLAS reproducibility.
[NVIDIA cuBLAS reproducibility documentation](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).
These controls do not independently guarantee native simulator reproducibility.

## CPU verification and freeze

Focused regressions test the real CLI, supported/invalid environment values,
unchanged default environment and inherited flags, strict real Torch settings,
RNG/TF32/precision preservation, and the actual runner rejecting an invalid
workspace before output setup or SDK imports.

**105 CPU tests passed in 1.65 seconds**, including existing application CLI
tests. Ruff passed for the two assigned source files and new test file.
The CPU setting probe left CUDA uninitialized. No Kit, GPU, model inference,
training configuration, frozen diffusion source, or checkpoint was changed.

Changed files:

- `CrowdSim/humanoid/application/configuration.py`
- `scripts/train_paper_amp.py`
- `tests/test_amp_deterministic_evaluation.py`

Artifacts:

- [Actual settings, source hashes and test command](deterministic_evaluation_change.json)
- [Focused test results](deterministic_evaluation_tests.xml)
- [Unresolved S1 divergence evidence](render_s1_divergence.md)

For the paired actual evaluation, add this flag to **both** explicit physics-only
and rendered commands while keeping their controller, diffusion, seed, scene,
and guidance settings fixed. Keep earlier captures/source archives unchanged.

## Screenshot state diagnostics

Only `--deterministic-evaluation --capture` enables these additional reads.
Immediately around `_capture_frame` (after the normal mesh update and before
PNG encoding), the runner reads `world.state()` and independently
`detach().clone()`s the real common `RobotState` fields `root_pos`, `root_rot`,
`root_vel`, `dof_pos`, `dof_vel` and `rigid_body_pos`. Cloning prevents a live
Warp/Torch view from silently changing the earlier snapshot. It records each
field's maximum absolute delta without requiring zero delta.

Each image also records native
`omni.physx.get_physx_interface().get_simulation_time()` and
`world.simulator._sim._physics_step_count` before/after/delta, plus equality
and SHA256 hashes for cloned CPU and already-initialized current-device CUDA
RNG states. It does not initialize CUDA for a diagnostic or store full RNG
states. Nonfinite buffers/clocks/deltas fail clearly. Records appear under
`result.json:capture_state_diagnostics`; default runs perform no extra reads.

Native time is optional: the preserved first actual deterministic rendered
attempt failed because the Isaac Sim 6.0.0.1 PhysX binding did not expose
`get_simulation_time`, although the pinned SDK facade referenced that method.
The revised diagnostic resolves the public method with `getattr`/`callable`;
when unavailable it records `observed: false`, `before/after/delta: null`, API
source, installed SDK distribution version, module version if exposed, and
the missing-method reason. It keeps physical buffers, actual step count and
RNG diagnostics. It does not substitute a facade/proxy time or label zero as
native time. An available method returning a nonfinite value still raises.

Real CPU `RobotState` tests verify unchanged state/RNG and a deliberate
in-place mutation of shared body/DOF tensors. They also test a deliberately
changed CUDA RNG API using a CPU mock; this is not a GPU execution test.
State getters may use native timestamp caches: these records measure observed
common buffers and explicit native clocks, not all hidden solver state.
No screenshot logic or physics stepping was changed. These focused regressions
did not launch the SDK; the actual failed rendered attempt and its log remain
preserved at `output/diffusion_amp/shared_independent_amp_deterministic_render_v1`
and `/tmp/shared_independent_amp_deterministic_render_v1.log`.
The diagnostic does not establish the cause of the earlier render divergence.
