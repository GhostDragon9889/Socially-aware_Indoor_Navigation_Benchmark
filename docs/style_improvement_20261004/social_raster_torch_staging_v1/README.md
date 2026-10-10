# Independent Tensor social raster

Pure Torch geometry, isolated in this directory. Seven bounded synthetic CPU cross-backend tests actually passed in **1.460s**; independent and root source review found Required=[]. Root verified exact source/test/CPU-proof identities and set them readonly (actual toola1e54f). GPU execution, NN/policy use, real captures, SDK integration and shared-group training have not been run. Production185, the frozen NumPy primitive and existing reports are unchanged. The CPU-proof's earlier review-pending field remains a historical execution record.

## Contract and device

The reference is the readonly 7218-byte NumPy source at `../social_raster_staging_v1/social_raster.py`, SHA256 `ff9927e33935318b7e3cafa8cda919a7c8aa2dc77014aeb356bcfc5c405e33c8`. Tests verify its complete bytes before executing that reference source. The backend itself imports only Torch and does no NumPy or host-array conversion.

`make_grid`, `render` and `mirror` accept **Tensor inputs**, with a mandatory explicit `device="cpu"` or indexed CUDA device such as `device="cuda:0"`. Every supplied tensor must already be on that device; mismatches are rejected before tensor arithmetic. CUDA without an explicit index is rejected. No device is auto-selected and no input is silently moved. Validation uses scalar finite/ID checks, which can synchronize a CUDA device; no GPU timing or throughput claim is made.

Numeric inputs support float16/32/64 and signed integer tensors, as the NumPy contract accepts floating/signed integer numbers. Geometry is promoted to **float64 on the same device**, matching the NumPy calculation dtype. IDs accept representable integer types, become int64 and must meet the original nonnegative/unique rules. Bool/complex numeric values, bool IDs and unsigned overflow above int64 are rejected. Output raster is float32; owners are int64. `mirror` preserves its float32/64 raster dtype.

The original inputs and units remain:

- Complete `roots_xy[N,2]` world metres, `root_headings[N]` radians and actual `world_velocities[N,2]` m/s.
- Globally unique stable `actor_ids[N]`, nonnegative `scene_ids[N]`, unique valid `ego_rows[E]`.
- Explicit `grid_center_xy[E,2]` and `grid_heading[E]`; head or root choice belongs to the caller.
- Caller-sampled absolute `static_heights[E,64,64]` at those exact query locations, and explicit `height_reference_world_z[E]` from rootZ or sampled static-center groundZ.

Empty actor arrays and empty ego selections are rejected, matching NumPy. A scene containing only its ego with `include_ego=False` is valid: all pixels remain background. Scenes without selected egos are skipped. Actor order, ego order and noncontiguous scene rows do not determine ownership.

## Tensor mathematics

The64² grid uses `ij=X/Y`, inclusive `[-2,+2]m` samples and `4/63m` spacing. Occupants are continuous **0.5m×1m full-size** rectangles in their own root-heading frames, with inclusive boundaries. Only the ego's scene participates; `include_ego` is a required explicit bool.

Overlaps select the nearest actor root to each query point; strict distance comparison in ascending stable-ID order resolves exact ties. Background owner is−1/world velocity zero. Occupied absolute top is sampled staticZ+**1.7m once**, irrespective of overlap count. Channels are height-reference−top and `(occupant_worldV−ego_worldV)` rotated into the ego **root** heading, independent of the grid heading. Thus background velocity is−egoV. All channels clip to `[-3,3]` then multiply by5. `mirror` flips localY/axis2, negates Vy only and flips owners. No input or RNG is mutated.

`render(..., ego_chunk_size=16)` groups selected egos by scene, chunks egos, then streams one sorted actor at a time. Temporary pixel geometry is bounded by `[ego_chunk_size,64,64]` rather than `[N,N,64,64]`. Final output still necessarily contains all selected egos. The actual test covered **256 scenes×4 actors=1024 complete egos**, plus the independently supplied mixed256-row layout `[2g,2g+1,128+2g,129+2g]` with reordered egos/actors and chunk sizes1/16. No group/reset helper dependency is introduced.

## Actual CPU verification and limits

Session **30104**, launch`3c2863`→completion`2c34a8`, observed exit0: **7 tests /1.460s**. Exact owners agree with NumPy for every tested pixel; raster agreement is checked with absolute tolerance **2e-6**, not a blanket bitwise claim. Closed-form channels, self/background, overlap/nearest/stable ties/height once, rigid world covariance, reflection/mirror twice, clipping, mixed scenes, input immutability and Python/NumPy/Torch RNG preservation passed. Explicit dtype/shape/finite/ID/device and overflow rejection checks passed. Final Ruff`dcec26` passed; the first lint attempt`a51c19` rejected test formatting, corrected by`20ca7f` before the actual test run. [CPU checks](actual_CPU_checks.json) records exact sources and invocation.

Both backends compute supplied inputs in FP64, but **FP32 input rounding remains irreversible**: the test shows a root just outside an inclusive boundary in float64 becomes on-boundary after float32 conversion, changing the owner in both implementations. No epsilon repairs or widens boundaries. GPU trigonometric/arithmetic rounding may also affect boundary ownership or near ties; GPU equivalence, performance and memory profiling have not been tested. Finite inputs whose arithmetic overflows are rejected before clipping can hide the overflow.

The flattened task dimension is12308 (20 trajectory coordinates+64²×3), incompatible with the existing1044-task policy/normalizer contract. This isolated primitive does not select a training center/reference mode, construct a policy, load old weights or implement shared-group training. No physical improvement, author overwrite equivalence or default promotion is claimed.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_raster_torch_staging_v1 -p test_social_raster_torch.py -v
```
