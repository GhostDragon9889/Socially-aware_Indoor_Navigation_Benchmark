# Independent raw social-task assembly

Four bounded synthetic CPU tensor tests actually passed in **0.012s**, after independent source peer Required=[] and root whole-read approval. The sole invocation completed directly in tool`c35147`, observed exit0 (wall0.880s; no ongoing session was returned). Source/test bytes stayed unchanged. [CPU checks](actual_CPU_checks.json) records their identities and exact invocation. This directory does not connect a simulator, train a critic, normalize observations or load weights.

## Executable interface

`assemble_raw_task(...)` returns raw float32 `[E,12308]` and diagnostic int64 owners `[E,64,64]`, in the explicit `ego_rows` order. Supply the complete current actor pool (`roots_xy`, root headings, actual world XY velocities, globally unique stable actor IDs, scene IDs), selected ego rows, and one same-observation-time trajectory `[E,10,2]` of **absolute world XY metres**. `trajectory_offsets_s[10]` must explicitly equal `+0.5,+1,...,+5s`; the adapter does not sample plans, advance their phase, verify absolute clock provenance or choose a lag.

The first20 coordinates are trajectory minus selected current root XY, inverse-rotated by the explicit `task_heading[E]`, then flattened `[time,XY]`. **There is no ×10 scale**:20 means ten XY points. Task heading and raster `grid_heading` are separate explicit inputs. The latter may describe a head or root grid; no center/heading is silently selected. `grid_center_xy[E,2]`, `grid_heading[E]` and `height_reference_world_z[E]` are mandatory. RootZ or static-ground reference selection belongs to the caller and is not an asserted fact about private NavIsaacLab code. `include_ego` is a mandatory bool.

The terrain callback is called **once** with float64 world XY `[E,64,64,2]` at the frozen raster's exact64² locations, on the declared device. Return finite absolute static world Z `[E,64,64]` there. Query mutation, wrong shape/device, nonfinite output and nonfloat output are rejected. A future SDK sampler must associate queries with these ego scenes; the adapter does not infer geometry or add future terrain changes. CPU tests use only deterministic tensor planes/zeros, with no SDK or NN.

The suffix12288 is the frozen float32 raster flattened HWC: local grid axes `X,Y`, channels `[height,Vx,Vy]`. It retains scene/self selection, continuous actor rectangles, stable-ID overlap ownership, background−egoV, inverse **root** velocity heading, staticZ+1.7 once and clip×5. These are observation features, not collision truth. All physical-source geometry/trajectory inputs must be float32 or float64 with a common dtype and already reside on explicit `device="cpu"` or indexed CUDA. Terrain output may independently be float32/64. Internal trajectory/grid arithmetic is float64; raw output is float32. Input tensors and RNG are unchanged.

`heading_from_pelvis_xyzw` uses the current projected world+X quaternion equation, unit norm tolerance `atol=1e-3,rtol=0`, no renormalization, and rejects horizontal projection norm≤1e-12. It is a tensor helper, not a world-state extractor. Actual runtime producers must obtain Pelvis from the explicit SMPL body-name mapping; do not substitute head orientation or assume an arbitrary simulator body index.

`mirror_raw_task` produces a new tensor: trajectory Y negated, map localY flipped, Vy negated. It neither normalizes nor mirrors self/action/body tensors. **Coupled full social self/action mirror integration remains pending.** `replace_candidate_trajectory` changes only20 prefix values using current roots/headings; the raster suffix is copied unchanged. The current self/body observations must remain fixed separately. No future peer motion is inserted. This supports a future candidate interface; it does not implement a trained critic or value guidance.

Normalization stays outside. The separate planned network interface normalizes the complete rawtask with new12308 task moments, reshapes HWC→CHW for CNN, and encodes `[map256,normalized_trajectory20]`. Old1044 moments/weights cannot be reused. SDK AMPWorld integration, shared-group training, normalized policy inputs, differentiable guidance and GPU equivalence/performance are still pending. Production185, defaults and every prior source/report are untouched.

## Source basis and pin

Only runtime dependency is the readonly original `../social_raster_torch_staging_v1/social_raster_torch.py`,9125 bytes, SHA256 `234e6366a00301998349dbc837fae5878cac4c506e29ad605ab8b2023a6c5dbe`. Its ordinary readonly single-link file and full bytes are checked once before executing those verified bytes; no dependency source is copied to this directory.

Current sources were read (not imported or executed) for the contract:

- `CrowdSim/humanoid/amp/observations.py:132–174`,10438B SHA`a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba`: canonical-state root is mapped Pelvis; ten future XYs are root-relative projected-heading coordinates, unscaled. Existing suffix is32² head-centered/head-heading terrain with nine root-ground samples, **not this new explicitly chosen64² social grid**.
- `scripts/train_paper_amp.py:593–619,744–787`,73954B SHA`d8b85401ace1da7d0039dcad4e023590577944e487789e742019c363a602e3d3`: action observations query pre-control `plan_age/30 + .5..5s`; after stepping, `plan_age` increments before next-value observations. AMASS/mixed branches instead use their explicit native curriculum future-target interfaces.
- `CrowdSim/humanoid/amp_world.py:206–223,263–266`,24834B SHA`1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83`: state root pose/velocity and explicit terrain sampling interfaces. Its planar quaternion yaw is the unit-quaternion equivalent of projected+X.
- `protomotions/utils/rotations.py:124–145,704–735`,28136B SHA`ef1475d33d6f941ccdea4a612477483db4fad8236ba7786fbe8bf91a65c1cef8`: XYZW projected+X heading/inverse-heading equation.

These context sources are not runtime imports; any future production wiring must recheck its then-current clock/body contracts. FP32 source rounding cannot be undone; FP64 staging arithmetic and raw output need not bit-match existing32-grid computations. GPU rounding at geometry boundaries and scalar-validation synchronization remain untested; no speed or physical-quality claim follows from CPU assembly checks.

## Actual bounded CPU verification

Four tests passed: one batched callback/layout/closed-form/input-RNG immutability case; mirror/candidate suffix preservation; tilted Pelvis heading and undefined/unit rejection; shape/dtype/offset/ID/device/callback rejection. Actual tool`c35147` returned exit0 directly (no yielded session), wall0.880475251s, unittest4 tests/.012s. CUDA stayed uninitialized before and after. They did not rerun the seven raster tests or read captures/models. Final source Ruff`a39b6c` and AST parse`b9186b` had passed before execution; no source changes were needed. Root read the full source/test/CPU record, verified source6975d95f/testa39767/CPUproof4fb5e04d identities and set them readonly. The [actual check record](actual_CPU_checks.json) remains the original execution evidence; README records this subsequent root freeze.

Exact approved and completed command (one invocation):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_task_observation_staging_v1 -p test_social_task_observation.py -v
```
