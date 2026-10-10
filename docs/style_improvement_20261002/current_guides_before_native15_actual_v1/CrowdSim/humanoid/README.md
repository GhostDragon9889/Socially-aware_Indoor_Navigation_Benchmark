# SMPL pedestrian simulation

## Diffusion + AMP

The paper-method reconstruction uses direct SMPL PhysX control, a separately
trained five-state Diffusion planner, and PPO/AMP. The current AMASS v7 manifest
contains 2,848 genuine segments from 1,853 original trials, with explicit
child-joint velocities and unchanged held-out source IDs.

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python scripts/train_paper_amp.py \
  --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --trajectory-source pacer_curriculum --curriculum-mode smooth \
  --curriculum-speed-max-mps 2 --curriculum-accel-max-mps2 1.5 \
  --num-envs 256 --updates 1000 --rollout-steps 32 --episode-steps 300 \
  --terrain-extent-m 192 --spawn-margin-m 48 --checkpoint-every 100 \
  --synchronized-replanning --physics-only \
  --output output/diffusion_amp/amp_MY_NEW_RUN
```

Use a fresh output directory. Diffusion is deployed after this independent
low-level training; it is not called for procedural training commands.
The negative-backward reward is an explicit optimization of the printed
direction equation. The public PACER state-only discriminator, mirrored action
regularization and policy replay are recorded adaptations.

| Module | Responsibility |
| --- | --- |
| `application/` | SDK-free CLI, configuration and local resource checks |
| `amass_corpus.py`, `training/amass_curation.py` | Motion sources, resampling, FK and velocity conventions |
| `amp_world.py`, `amp/` | Physical SMPL control, observations and public-model mapping |
| `training/` | PPO/AMP, replay, symmetry, trajectory curriculum and causal references |
| `diffusion/` | Five-state denoising and explicit waypoint/collision guidance |
| `motion_tracking.py`, `style_evaluation.py` | Complete-clip tracking and frozen held-out diagnostics |
| `contact_monitor.py`, `foot_support.py`, `foot_capture.py` | Body contacts, floor-only normal forces and real COM |
| `ground_support_archive.py` | Separate four-substep force/state/clock archives |

Optional evaluation flags `--pairwise-contacts --measure-ground-support` save
distinct body-contact and floor-support channels. Foot-origin tangent speed is
a contact-conditioned proxy, not measured contact-point slip.

See [the current training and physical evidence](../../docs/AMP_STYLE_PROGRESS_20261002_ZH.md).
The current matched reward experiment reached 11/22 complete tracking successes
and no falls; it does not establish the authors' full results or superiority.

### Experimental AMASS root trajectory curriculum

Use `configs/diffusion_amp/amp_style_amass_root_curriculum.yaml` with
`--trajectory-source amass_root`; omit the procedural `--curriculum-*` options
from the command above. Every core TrainingConfig value stays at the negative
baseline. The curriculum binds a real training clip and source phase at reset,
translates it once, and uses its native 30Hz root XY targets. It never sends
future 3D poses to the actor or reanchors targets to the live physical root.
Only future XY commands outside the source receive the declared endpoint
velocity extension; physical episodes truncate at the actual source end.

The real 4+4-update resume pilot and fresh 1,000-update training completed.
The sealed step 1000 reached 17/22 complete tracking successes with no falls,
0.229 m position and 0.832 rad rotation error. Frozen style MMD² worsened
despite a better speed distribution. A new step500→750 continuation completed
and reached 16/22 successes with two fallen sequences; it is distinct from the earlier fresh training run.
Deployment uses the separately trained Diffusion planner. See the current
stage report for the separate navigation and style acceptance evidence.
The new root-trained controller's full200s navigation trial had 13 falls and
five arrivals versus zero and 21 for the baseline. Its once-only full34 test
reached 11/34 complete successes with zero falls. The earlier zero-fall
backward-penalty controller remains the deployment baseline.

### Mixed AMASS / navigation-command curriculum

The opt-in profile `configs/diffusion_amp/amp_style_mixed_amass_procedural.yaml`
uses `--trajectory-source mixed_amass_procedural`. Fixed actor slots split the
256 environments into 128 real-AMASS and 128 original smooth-procedural commands;
keep the speed/acceleration arguments at 2 and 1.5 as in the baseline command.
Use 2,000 updates and a fresh output directory. The actor sees the same observation
contract, without branch IDs or future 3D target poses. Independent RNGs and
actual branch transition counts are part of the complete resume state.

The 700-check CPU gate and real 4+4-update resume pilot passed. Fresh full training
completed 16,384,000 total transitions, with 8,192,000 per branch; all three
registered milestone checkpoints were sealed.
This is twice the earlier total training budget. Step 2000 completed full validation:
19/22 successes, no falls, position error 0.200 m and rotation error 0.768 rad.
Its complete four-person, 200-second navigation test had 11 falls and 19 arrivals;
step 1000 had 13 falls and 15 arrivals. Both fail deployment selection. The
current stage report retains the inclusive style scores and exact source bindings.

### State-and-action discriminator

`amp_style_mixed_state_action.yaml` independently initializes a 2019-input
discriminator and its history normalizer, while retaining the public actor/critic
warm start. Expert actions are next-frame AMASS PD position-target proxies, not
measured controls. Fresh 2000-update training completed 16,384,000 transitions
from an isolated, read-only source copy; steps 500/1000 and the terminal payload
passed complete recovery audits. Full tracking completed 17/22 successes with
zero falls; four-person navigation for 200 seconds had two falls, only two
arrivals and about 74% idle time. This model is rejected for deployment. The
[registered evaluation](../../docs/style_improvement_20261002/state_action_evaluation_registration_v1/README.md)
keeps all failed frames and does not consult the 34-sequence test results.

## Explicit pedestrian deployment

`scripts/simulate_paper_pedestrians.py --inference-only` uses the evaluation actor
without training objects. The real 30-second baseline and two-network alpha0
pilots are bitwise equal to the existing 200-second fixture prefix.
`--experimental-arm-style-checkpoint` and `--arm-style-alpha` opt into a
24-coordinate arm-only experiment; nonzero alpha is not a validated model.

`--native-dynbench --scene-config configs/diffusion_amp/native_warehouse_pedestrians_v1.yaml`
connects the native USD, conservative A*, SDF-guided Diffusion and the actor
in the existing sole world. The first five-second test had three falls. With the separate0.8m-clearance
configuration, the completed 30-second test had zero falls and contacts, eight
vertex arrivals and no complete route cycles; 865 independent checks passed.
The completed four-person, 200-second native test had zero falls, zero measured
static/peer contacts, 57 arrivals and 12 complete route cycles. Its 983-check
independent audit passed. Fifteen-person physics and the authors' eight-scene
protocol remain separate pending evaluations.
The native alpha .25 arm experiment also completed 30 seconds with zero falls and
8 arrivals. Frozen pose MMD improved slightly, while speed W1 and K100 coverage
worsened; it remains optional. The actual v9 native map resource gates now pass
for all four supplied scenes, using owned anonymous runtime roots and unchanged
original source layers. Independent raw admission passed 4,441 checks over 547,321 native cells.
See the [fixed pose diagnostics](../../docs/style_improvement_20261002/pose_diversity_candidate_v1/README.md)
and [terrain RMS review](../../docs/style_improvement_20261002/navigation_terrain_rms_review_v1/README.md).

The source-bound [30-second indoor replay](../../output/diffusion_amp/native_warehouse_recorded_replay_indoor_30s_v1/pedestrian_motion.mp4)
shows visible pedestrians in the original Warehouse. It renders saved states only:
450 frames, 15 FPS, zero physics/policy steps; the original source layers remain unchanged.
See the current stage report for source-bound evidence and separate quality gates.

## Existing SFM / MaskedMimic entry

From the repository root, using the pinned Isaac Sim 6 / IsaacLab 3 environment:

```bash
python -m CrowdSim.humanoid.simulate --num-humanoids 4
python -m CrowdSim.humanoid.simulate --headless --steps 240 \
  --num-humanoids 4 --reset-at 120 --capture
```

The default uses SFM, frozen SMPL MaskedMimic, AMASS validation and a local SMPL
neutral mesh in a shared PhysX world. The bounded command saves result/config
JSON, physical NPZ, RGB PNG and an MP4 under `output/pedestrian_simulation`.
Use `--no-mesh` explicitly for physical skeleton visualization.

See [the current completion report](../../docs/PROJECT_COMPLETION_20261001_ZH.md)
for resource locations, verified results and remaining paper reproduction work.

## Humanoid avoidance diagnostic

This test removes navigation robots and runs MaskedMimic humanoids in a fixed
bidirectional encounter. It measures whether the physical pelvis follows the
SFM command and its first future target.

```bash
python CrowdSim/humanoid/run_avoidance_test.py
```

For a short smoke test:

```bash
python CrowdSim/humanoid/run_avoidance_test.py --steps 300
```

Outputs are written to `output/humanoid_avoidance_eval/`:

- `metrics.json`: distance, collision, velocity, heading and target tracking metrics;
- `humanoid_avoidance.gif`: offline bird's-eye rendering;
- `navigation/trajectory_latest.jsonl`: per-step vectors for deeper inspection.

The GIF uses blue for actual pelvis velocity, green for the filtered SFM
velocity, orange for TTC force, and yellow for the first MaskedMimic future
target.
