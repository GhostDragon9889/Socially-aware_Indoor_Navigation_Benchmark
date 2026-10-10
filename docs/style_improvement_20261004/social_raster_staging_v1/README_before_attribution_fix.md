# Independent social raster primitive

Prepared NumPy geometry only. No real capture, model, weights, SDK, GPU or shared-group training is used. Current185, q937, old62c4 inference and the40k experiment are unchanged. Source is pending independent/root review and remains writable.

## Interface

`make_grid(grid_center_xy[E,2], grid_heading[E])` returns world XY `[E,64,64,2]`. `render` takes complete actor rows and returns `(raster[E,64,64,3], occupant_ids[E,64,64])`. `include_ego` is a mandatory keyword bool. `mirror` reflects raster localY and occupant ownership.

| Input | Units / meaning |
|---|---|
| `roots_xy[N,2]`, `root_headings[N]` | Actual actor root worldXY metres and root-heading radians. |
| `world_velocities[N,2]` | Actual root worldXY velocity, metres/second; not planned/reference velocity. |
| `actor_ids[N]`, `scene_ids[N]` | Nonnegative integer IDs. Actor IDs must be globally unique and stable across row permutation; scene IDs group valid occupants. |
| `ego_rows[E]` | Unique row indices into the complete actor arrays. |
| `grid_center_xy[E,2]`, `grid_heading[E]` | Explicit caller-chosen spatial center and heading. They can come from head or root; the primitive does not infer which. |
| `static_heights[E,64,64]` | Caller-sampled absolute static worldZ in metres at the exact `make_grid` query points. |
| `height_reference_world_z[E]` | Explicit reference worldZ in metres: actual root height OR sampled static-center ground height, chosen and recorded by the caller. |

Grid axes use `ij`: axis1 is localX and axis2 localY. The64 sample locations include both endpoints of `[-2,+2]m`, so sample spacing is `4/63m`. Channels last: `[height, localVx, localVy]`, final dtypefloat32; occupant IDs int64, background `-1`. A flattened task using20 trajectory coordinates would have **12308 dimensions**, incompatible with the old1044-task policy/normalizer interface.

## Geometry and channels

Only actors in the ego's scene are eligible. Each root carries a continuous rectangle with **full root-frame X/Y sizes0.5/1m** (half-sizes0.25/0.5m), rotated by that actor's root heading; boundaries are included. This declared proxy is not a measured body AABB, mesh or contact footprint.

For overlaps, the actor whose root is nearest the query point owns the pixel. Exact distance ties select the smallest stable actor ID. The winner supplies world velocity. This is a local deterministic attribution correction, not an assertion about author overwrite behavior.

An occupied pixel's absolute top is `sampled_static_Z +1.7m`, added **once**, regardless of occupant count. Background top is staticZ. Height is `height_reference_world_z - absolute_top`.

Every pixel's velocity is `(occupant_worldV - ego_worldV)` rotated by the inverse **ego ROOT heading**, regardless of the chosen spatial-grid heading. Background occupant worldV is zero, hence background relative velocity is `-ego_worldV`, not zero. All three channels are clipped to `[-3,3]` and multiplied by5. Inputs are not mutated and no RNG is drawn.

`mirror` flips localY (array axis2), preserves height/Vx, negates Vy, and flips the occupant-ID array. Applying it twice restores the original. This sign correction is declared explicitly; the paper/official velocity-flip descriptions do not establish the same corrected operation.

## Paper and code distinctions

- Nav paper B.1 describes root-centered64²×3 over4×4m with height relative to current root height. That mode requires caller-supplied rootXY/root heading/rootZ.
- The inspected official `group_cnn` source uses a±4m grid (8m span); current local1044 observations can use a head-centered spatial grid and static-center ground reference. Those are distinct choices, not paper B.1 equivalence.
- This primitive allows either explicit center/reference pair. **The CPU cases use a head-derived grid and caller-sampled static reference**, with root heading different from grid heading to test velocity-frame separation. Future actual training mode has not been selected.
- Continuous metric rectangles here replace the official10×20 discrete-pixel stencil. Grid size, footprint and tie attribution do not claim exact author reproduction.

## Bounded CPU checks

`test_social_raster.py` uses fixed synthetic geometry: closed-form grouping/self inclusion and nonzero relative velocity; nearest versus stable-ID overlap and height addition once; actor permutation; global rigid-frame covariance; world reflection with localVy sign; clipping and invalid finite/shape/ID contracts. It checks Python/NumPy RNG and input immutability. These are geometry/interface checks, not NN, body-shape, physical-contact or training-effect validation.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_raster_staging_v1 -p test_social_raster.py -v
```

Shared-group collision training, social policy/normalizer integration and checkpoint loading are **not implemented or run**. New input dimensions and new observation semantics require an explicit independently trained policy contract. No physical improvement or promotion is claimed.
