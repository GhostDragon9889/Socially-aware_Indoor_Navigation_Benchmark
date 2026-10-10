# Single-sample rendered versus physics-only divergence

Reviewed 2026-10-02, CPU only. No source changes, model inference, Kit launch,
or CUDA initialization were performed by this review.

Inputs:

- Physics-only: `shared_independent_amp_continuous_timed_v1` (200 seconds).
- Rendered S1: `independent_amp_diffusion_render_s1_v1` (30 seconds).
- Comparison: the first 900 control frames and reference plans keyed by
  `(control_step, actor_id)`, including the initial four-actor plan.

## Earliest observed differences

Initial **XY anchors and physical reference velocities are bit equal** for
all actors. Both start at `[[6,6],[6,18],[18,6],[18,18]]`. The initial raw future
XY already differs slightly, before any physical control transition:

| Recorded quantity | Maximum initial difference |
| --- | ---: |
| First raw diffusion future XY | 0.000463486 m |
| First adapted future XY | 0.000581741 m |
| First adapted future velocity | 0.00308037 m/s |
| First post-step body position | 0.00000762939 m |
| First post-step pelvis position | 0.00000572205 m |
| First post-step executed target | 0.00000190735 m |
| First post-step joint position | 0.0000236034 rad |
| First post-step joint velocity | 0.0247951 rad/s |

These traces are close initially, but are not bit identical. The first saved
physical frame precedes the first `_capture_frame(..., warmup=True)` call.
Consequently, that subsequent 12-update warmup cannot have caused the difference
already present in the saved frame. Renderer/mesh/camera initialization occurs
before this initial plan and is not excluded by this timing observation.

Maximum XY separation across actors grows as follows:

| Compared control frames | Maximum pelvis XY separation (m) |
| --- | ---: |
| 1 | 0.00000572 |
| 30 (1 s) | 0.000878 |
| 150 (5 s) | 0.237693 |
| 300 (10 s) | 1.55309 |
| 450 (15 s) | 3.59788 |
| 485, immediately before the recorded fall frame | 4.80946 |

Rendered S1 actor 2 falls at zero-based frame **485**, control time **16.20 s**.
Its episode/reset markers diverge at frame 486. Static contact flags first
diverge at frame 345; route-index flags first diverge at frame 650. These
differences follow the earlier plan/physical differences rather than locating
their original cause.

## Configuration and code inspection

Reported argument differences are output path, capture enabled, renderer
enabled, capture interval (2 control steps), capture length (900 versus 6000),
and the new explicit `diffusion_samples=1` versus the older missing CLI field.
The archived older caller used the planner's one-sample default. This comparison
therefore does not show that extra diffusion samples caused the rendered fall.

The local screenshot helper calls `physics_manager.forward()`, sets
`/app/player/playSimulations=False`, pumps 12 initial or three later Kit
updates, then restores the prior setting. The pinned PhysX implementation of
`forward()` calls articulation kinematics synchronization and fabric update
with `(0.0,0.0)`; it does not explicitly call the physics step. The official
RTX render helper uses the same nonstepping forward plus guarded app-update
pattern. The source review found no explicit extra physics transition in this
helper, although a before/after measurement of all physical buffers was not
recorded in these AMP captures.

Mesh appearance uses an explicitly seeded local NumPy Generator, not a global
Torch noise generator. The selected SMPL constructor creates its default pose,
translation and shape parameters with zeros. A fresh **CPU import-only** probe
of `CrowdSim.tools.smpl_mesh_visualizer`, `CrowdSim.protomotions_runtime`,
`CrowdSim.humanoid.simulate`, and `smplx` preserved Torch CPU, global NumPy and
Python RNG states. It imported neither Isaac Sim nor Omni and left CUDA
uninitialized. This does **not** measure CUDA construction, third-party camera
initialization, or the original live process's RNG state.

## What can be concluded

Small initial planning and physical differences amplify under closed-loop
control and replanning. The observations do not establish changed full diffusion
noise, an added full physics step, or a specific renderer bug. Same seeds alone
do not guarantee deterministic CUDA operations; PyTorch distinguishes random
seeding from deterministic algorithm controls. This is a possible mechanism,
not an identified cause for this run. [PyTorch 2.11 reproducibility documentation](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

The captures did not save exact pre-plan full five-dimensional contexts,
pre/post-render CUDA RNG states, denoising noise, or immediate pre/post-capture
velocities/efforts. A controlled diagnosis would need those observations and
cloned pre/post-capture buffers so a shared tensor view cannot hide changes.
No such instrumented run was performed here and existing evidence was preserved.

For presenting the already validated 200-second trajectory, a separately labeled
**recorded-state replay** can render its captured motion. It should cite the
original capture hash and be distinguished from a new closed-loop physical
evaluation. The 30-second S1/S2 closed-loop render results retain their actual
falls; a replay would not replace their measurements.

Evidence: [render_s1_divergence.json](render_s1_divergence.json). The scientific
cause remains unresolved; no business-source patch is justified by this CPU
comparison alone.
