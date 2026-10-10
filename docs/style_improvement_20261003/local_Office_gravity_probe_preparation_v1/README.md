# Local Office gravity and support probe: preparation

This is an **unexecuted local Office3/10s plan**. It creates no simulator,
changes no production file and reads no future controller/capture. It does not
identify Office as one of the authors' eight evaluation scenes.

## Existing evidence and remaining measurement

The existing Office V9 map has a source-bound floor/resource gate. Its saved
physics metadata records gravity `[0,0,0]` and two snapshot steps. The floor
is a finite engineering reference mesh; selected original flat-ground
collisions are disabled in the runtime session. Those static map observations
do not establish pedestrian stability under Earth gravity.

The installed IsaacLab source declares `SimulationCfg.gravity=(0,0,-9.81)`
and creates USD PhysicsScene gravity direction/magnitude from that vector
([configuration](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_cfg.py:43),
[context](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/sim/simulation_context.py:253)).
The existing ProtoMotions simulator constructs that configuration at 120Hz.
The added observer reads the **future actual** singleton config, active stage,
one active PhysicsScene and unchanged submitted-step counter. It labels these
as configured gravity observations, not native-engine gravity or timestamps.

## Observer integration

`observe_configured_gravity.py` imports only the standard library when cold.
`wrap_native_scene_factory(existing_factory)` delegates the original factory,
then decorates one newly created Office provider's live observation callback.
The original floor/source/raycast verifier runs unchanged. Its returned record
gains `local_configured_gravity_observation` after the gravity observation.
The wrapper does not write USD properties, construct App/context/views,
advance physics or change policy actions.

The future root driver adapter must pass that wrapped factory through the
existing `native_scene_factory` keyword. The materialized standalone deployment
command in the plan **does not attach this observer by itself**. It is an
explicit future integration recipe, not an executable claim of coverage.

## Future short-run protocol

The existing Office3 config, map, three spawns/routes and 0.8m clearance are
bound by exact source records in `plan.json`. The plan uses the same trained
PACER reward controller/profile, Diffusion checkpoint, guide20/eta0, seed7 and
arm alpha0 as the Warehouse comparison. The controller SHA/bytes remain null
until the terminal reader admits the actual sealed checkpoint.

The short probe is 300 controls at30Hz: 1,200 shared120Hz substeps, 900 actor
controls and 14,400 feet/substep observations. Existing capture channels record
all three actors' pre-reset falls, static/peer contacts, floor-only normal
forces and true COM. Failed controls and negative heights remain in the result.

A future Office3 short-capture CPU adapter is needed for independent reading;
the registered Warehouse4/15/200s reader does not accept this case. Preserve its
raw contact/support mathematics and adapt only source/case/count metadata.
Keep the actual root process closure separate from the CPU preparation receipt.
Style scores are diagnostic and do not filter completed results.

## CPU evidence

The actual source-only preparer closed with exit0 in0.216s. It successfully used
the current production185 roster and existing Office cold scene preflight;
Torch/IsaacSim/IsaacLab/Omni imports were absent. Three CPU analytic tests passed
in0.02s, exercising gravity consistency, singleton/zero-gravity/different-gravity/
clock rejection and exact original-verifier delegation. Fake SDK observations
in those tests are explicitly not actual Office evidence.

The single GPU remains root-owned. This plan allows no concurrent launch and
makes no claim of navigation quality,200second stability or paper equivalence.
