# Independent social raster primitive

Implemented NumPy geometry only. Four synthetic CPU cases passed; independent and root source review found Required=[]. The primitive, tests and historical CPU proof are now readonly. No real capture, model, weights, SDK, GPU or shared-group training is used. Current185, q937, old62c4 inference and the40k experiment are unchanged.

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

- [TRACE/PACE supplement B.1](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf) (PACER low-level control) describes root-centered64²×3 over4×4m with height relative to current root height. That mode requires caller-supplied rootXY/root heading/rootZ.
- The inspected official `group_cnn` source uses a±4m grid (8m span); current local1044 observations can use a head-centered spatial grid and static-center ground reference. Those are distinct choices, not paper B.1 equivalence.
- This primitive allows either explicit center/reference pair. **The CPU cases use a head-derived grid and caller-sampled static reference**, with root heading different from grid heading to test velocity-frame separation. Future actual training mode has not been selected.
- Continuous metric rectangles here replace the official10×20 discrete-pixel stencil. Grid size, footprint and tie attribution do not claim exact author reproduction.

## Bounded CPU checks

`test_social_raster.py` uses fixed synthetic geometry: closed-form grouping/self inclusion and nonzero relative velocity; nearest versus stable-ID overlap and height addition once; actor permutation; global rigid-frame covariance; world reflection with localVy sign; clipping and invalid finite/shape/ID contracts. It checks Python/NumPy RNG and input immutability. These are geometry/interface checks, not NN, body-shape, physical-contact or training-effect validation.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m unittest discover -s docs/style_improvement_20261004/social_raster_staging_v1 -p test_social_raster.py -v
```

Shared-group collision training, social policy/normalizer integration and checkpoint loading are **not implemented or run**. New input dimensions and new observation semantics require an explicit independently trained policy contract. No physical improvement or promotion is claimed.

Root verified exact source/test/CPU-proof byte identities and set them readonly in actual tool a47c94. The original CPU proof records the then-current README and pending review truthfully; `README_before_attribution_fix.md` preserves that exact old documentation. Its old attribution to Nav paper B.1 was corrected to TRACE/PACE, without changing the primitive or rerunning tests.

The [Torch backend](../social_raster_torch_staging_v1/README.md) now implements the same contract with explicit same-device tensors and scene-local ego chunks. Seven real CPU cross-backend checks include1024 selected egos; tested owner pixels match exactly and raster values pass absolute tolerance2e-6. GPU equivalence and performance remain untested. [Group/reset helpers](../social_group_reset_staging_v1/README.md) separately implement the64×4 mixed layout and termination/truncation masks; actual collision groups and training integration are still pending.

## Remaining policy and physical integration

Read-only inspection of current training establishes separate issues that geometry alone does not fix:

- Ordinary training creates256 individually filtered actor/env rows. A64×4 experiment should preserve flat actor tensors and explicitly introduce64 collision groups, with genuine within-group contact, between-group filtering and shared ground. Disabling all filtering would allow every group to collide and is insufficient.
- The mixed curriculum keeps the first128 AMASS and last128 procedural rows. Group `g` can take `[2*g,2*g+1,128+2*g,129+2*g]`. A member's fall or curriculum boundary invalidates the entire group's episode/plan; actual fall flags remain terminal while collateral resets are truncations evaluated on the physical terminal observation before reset. The existing GAE can bootstrap truncations while cutting carry; runner integration remains required.
- Dynamic raster uses actual root velocities from one pre-action snapshot. Static height queries remain static for reset Z/fall/ground support; peers must not be painted into that ground query. Training on existing mixed target plans is possible without concurrently adding Diffusion as a new training variable.
- The inspected [fixed PACER CNN builder](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_network_sept_cnn_builder.py#L49) actually uses three16-filter `k4/s2/p1` convolutions, `1024→512→256`, then concatenates20 trajectory coordinates. This differs from supplement B.1's four-layer description. The executed map encoder is shared by actor/critic, whose MLPs are separate. A neutral variant would use357+256+20=633 features rather than the old613.
- Official CNN input is standardized per flattened observation coordinate before reshaping; no map unnormalization occurs. A faithful code-derived variant needs new12308 moments, frozen within rollout/PPO and at evaluation. It cannot reuse old1044 suffix moments or the protected32² terrain policy. Raw geometry is retained separately for diagnostics and mirrored before normalization.
- Official shape-conditioned group code has self368 and10×206 state-only discriminator input; the current neutral SMPL has self357 and10×195 history. Keeping the current neutral rig is an explicit local adaptation. Its state-action D also consumes69 applied-action coordinates, as declared by the local Nav Eq2 reconstruction; it does not become the official group discriminator.

The [NavIsaacLab method](https://arxiv.org/html/2606.26265v1#S4.SS2) specifies a task encoder and MLP actor/critic but does not expose all these PACER CNN/group/shape choices. No private Nav configuration is inferred from the public upstream code. New policy/normalizer, symmetry adapter, collision construction, group-reset runner and checkpoint/deployment admission must be implemented and physically verified before any effect claim or default replacement.
