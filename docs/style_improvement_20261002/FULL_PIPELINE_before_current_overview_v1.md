# Full Isaac Lab PPO pipeline

The current [AMASS motion-style training stage](docs/AMP_STYLE_PROGRESS_20261002_ZH.md)
uses 2,848 real nonoverlapping segments from 1,853 original trials, explicitly
corrects expert velocities to native child-joint coordinates, and adds policy
history replay and mirrored action regularization. Two matched 1,000-update
physical training runs are complete. Changing only the backward-direction term
to a penalty improved full-clip tracking from 5/22 to 11/22 successes, removed
the two fallen sequences, and reduced position error from 0.431 to 0.301 m.
The [four-model analysis](docs/style_improvement_20261002/backward_penalty_style_v1/report.md)
also records remaining rotation, wrist, speed-distribution and foot-proxy gaps.
The physical navigation baseline is `amp_style_amass2848_backward_penalty_v1/step_1000.pt`.
The new AMASS-root tracking candidate is `amp_style_amass_root_curriculum_v1/step_1000.pt`:
17/22 complete successes, zero fallen sequences, 0.229 m position and 0.832 rad
rotation error. These local results do not establish paper parity or superiority.

The selected backward-penalty baseline's four-person, 200-second Diffusion deployment had zero falls and
21 route-vertex arrivals. The [independent floor-support audit](docs/style_improvement_20261002/ground_support_200s_actual_verification.md)
verified all 24,000 physics substeps and live COM data. The
[navigation style and stall analysis](docs/style_improvement_20261002/backward_penalty_navigation_style_v1/comparison.md)
records more idle time and fewer arrivals than the preceding controller;
the new candidate is not uniformly better. The guidance strength 20→40
ablation reached 29 vertices but introduced one fall. The completed stratified
expert experiment had 8/22 tracking successes and two falls, so it does not
replace the candidate. The fixed-total-weight near-horizon experiment reached
20 vertices and had two falls, so the default remains unchanged. The next
training experiment uses actual AMASS root trajectories with a fixed episode
translation and explicit command-only continuation outside the source clip.
Its real 4+4-update resume pilot and fresh 1,000-update run completed. The latter
performed 8,192,000 physical transitions. Complete tracking improves, but the
frozen inclusive style MMD² worsens from 0.04682 to 0.05607 and local body
position error increases. A new continuation from the sealed step 500 added
250 updates; full validation reached 16/22 successes and two fallen sequences,
so it does not replace the root step1000 tracking candidate. It is not a recovered
snapshot of the earlier run. The root step1000 navigation trial completed with
13 falls and five arrivals versus the baseline's zero falls and 21 arrivals;
the previous zero-fall navigation baseline remains selected. Once selected,
the fixed root candidate's full34 test reached 11/34 successes and zero falls
versus 4/34 and one fall for the baseline. This is separate generalization
evidence and does not reverse the closed-loop navigation failure.
A fixed 128-AMASS / 128-procedural mixed curriculum now has 700 passing CPU
checks and a verified real 4+4-update resume pilot. Fresh 2,000-update training
completed under `amp_style_mixed_curriculum_v1`; its actual 16,384,000 total
transitions are twice the earlier total budget. Step500's complete validation
reached 16/22 successes without falls. Step1000 reached 19/22 without falls,
with position error 0.2035 m and better speed distribution; inclusive pose style
remains worse than the navigation baseline. Step2000 reached 19/22, zero falls,
position 0.2001 m and rotation 0.7684 rad, with better wrists and joint rotations;
its inclusive MMD remains worse. Its complete 200-second navigation trial had
11 falls, 19 arrivals and 130 static-contact actor frames. The 2,259 independent
measurement checks passed, but navigation quality failed. Step1000 completed
the same 200-second protocol with 13 falls and 15 arrivals. Both failed quality
selection; the original zero-fall deployment baseline remains selected.
The [frozen navigation style comparison](docs/style_improvement_20261002/mixed_curriculum_navigation_style_v1/README.md)
retains all failed frames: step2000 moves more, but pose MMD² and the foot-speed
proxy worsen. An independently initialized state-and-action discriminator profile
now has a completed real 4+4-update resume pilot; full quality results remain pending.
The source, recipe, snapshots and acceptance protocol
are recorded in the current stage report. The [root controller's full 200-second video](output/diffusion_amp/recorded_amass_root_curriculum_200s_v1/pedestrian_motion.mp4)
replays the failed 13-fall deployment; it adds no physics or policy execution.
See the current stage report for frozen sources,
actual gates and separate tracking, style and navigation results.

Start with [the current structure, resources and physical evaluation index](docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md).
The preceding [paper research, independent AMP training and closed-loop improvements](docs/PEDESTRIAN_AMP_DIFFUSION_PROGRESS_20261001_ZH.md)
include the fixed Diffusion5000 model, clean prediction guidance and independently
trained low-level procedural commands. The preceding single-candidate timed-route
deployment with a causal reference completed four people for 200 seconds with
zero falls, 26 route arrivals and five complete cycles in the authored obstacle
fixture. Static contacts and pose coverage remain below the intended quality;
this single-seed result does not establish the authors' eight-scene results.
The [200-second recorded-state video](output/diffusion_amp/recorded_pose_replay_200s_v1/pedestrian_motion.mp4)
replays that physical run with a permanent source label. A separate
[30-second deterministic rendering comparison](docs/amp_improvement_20261001/deterministic_render_pair_verification.md)
verified identical saved states and plans with and without capture; it is a
short closed-loop diagnostic, not a new 200-second rendering validation.
It distinguishes the source CrowdSim application, installable ProtoMotions core,
simulator-free PPO package and separately supplied licensed assets.

## Run the selected physical navigation baseline

Run from this checkout using the pinned environment and the existing supplied
assets. Select a new output directory for every run.

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python scripts/simulate_paper_pedestrians.py --inference-only \
  --amp-checkpoint output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt \
  --config configs/diffusion_amp/amp_style_amass_backward_penalty.yaml \
  --diffusion-checkpoint output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt \
  --manifest data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json \
  --duration-seconds 200 --output output/diffusion_amp/navigation_NEW_RUN
```

Add `--preflight-only` to inspect the resource contract without starting Kit.
For native Warehouse, add `--native-dynbench` and
`--scene-config configs/diffusion_amp/native_warehouse_pedestrians_clearance08_v1.yaml`.
Its completed 30-second test had zero falls, zero contacts and eight vertex
arrivals, with no complete route cycles. The 865-check independent audit
verified the live stage, floor rays and full physical capture; full200s native
quality remains under evaluation. Neither this authored fixture nor the available four native
USD scenes supplies the authors' complete eight-scene evaluation protocol.

## Paper pedestrian method: Diffusion + AMP

The new method entry points are `scripts/train_paper_diffusion.py` and
`scripts/train_paper_amp.py`, with `configs/diffusion_amp/method_reconstruction.yaml`.
They train a five-state trajectory x0 diffusion model and PPO/AMP using direct
SMPL PhysX simulation. The original method profile supports an action-conditioned
discriminator; the current public-PACER transfer experiments explicitly use a
state-only discriminator adaptation. The standalone pedestrian command
in the later section retains its existing SFM/MaskedMimic configuration.
See [the Diffusion/AMP implementation and training guide](docs/DIFFUSION_AMP_RECONSTRUCTION_20261001_ZH.md)
and `output/diffusion_amp/` for actual training artifacts and declared assumptions.

```bash
source ../IsaacLab-Protomotions/.venv/bin/activate
python scripts/prepare_diffusion_amp_data.py
python scripts/train_paper_diffusion.py --steps 1000 --batch-size 32
python scripts/train_paper_amp.py --updates 1000 --num-envs 64 --physics-only \
  --synchronized-replanning --output output/diffusion_amp/amp_NEW_RUN
```

The derived AMASS manifest, training parameters and new weights do not establish
equivalence to the authors' unpublished 2,848 clips, eight scenes or checkpoints.

The AMP configuration and file checks can run before importing Torch or Kit:

```bash
../IsaacLab-Protomotions/.venv/bin/python -m CrowdSim.humanoid.application.preflight \
  --config configs/diffusion_amp/method_reconstruction.yaml \
  --diffusion-checkpoint output/diffusion_amp/diffusion_training/step_1000.ckpt \
  --output output/diffusion_amp/PREFLIGHT_NEW_RUN
```

This applies the real CLI and configured local resource paths without launching
simulation. Checkpoint signatures/file presence do not establish tensor/model
compatibility; the runtime retains its dedicated loading gates. CLI paths use
cwd; relative AMASS package paths in a manifest use that manifest's directory.

The historical three four-person 200-second obstacle evaluations each had 16
falls. Later independently trained AMP and explicit deployment changes improved
this fixture; see the latest guide above for the fixed models and full command.
The separate bounded 32-person
50-second straight-path diagnostic had no falls for either public PACER or the
corrected trained AMP controller. These are distinct physical tasks. Exact
filtered normal contacts are now verified on controlled wall/peer/floor probes;
historical captures without contact channels retain unknown physical contacts.
See [the evaluation index](docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md#已有物理评估能说明什么)
for the actual reports and comparison limits.

The standalone pedestrian path has been migrated and tested with Isaac Sim 6 / IsaacLab 3.
Start with [the current completion report](docs/PROJECT_COMPLETION_20261001_ZH.md)
for the environment, fixes, packaging, resources and physics/render evidence.

The repository provides two complementary training paths:

1. `src/crowdsim_ppo/`: a simulator-free reference pipeline for quickly reading
   and testing the algorithm.
2. `CrowdSim/` + `protomotions/`: the original Isaac Lab crowd-navigation
   pipeline with humanoids, robot sensors, maps and differential-drive control.

## Included dependency closure

```text
CrowdSim/ppo/train_ppo.py
├── CrowdSim/ppo/ppo_policy.py
├── CrowdSim/world/builder.py
│   ├── CrowdSim/protomotions_runtime.py
│   │   └── protomotions/
│   ├── CrowdSim/scene_setup.py
│   ├── CrowdSim/nav_manager.py
│   │   ├── CrowdSim/world/{map,planning,sfm}.py
│   │   └── CrowdSim/control/drive.py
│   └── CrowdSim/utils/
├── CrowdSim/tools/render_navigation_fast.py
└── CrowdSim/config/env.yaml
```

Preference PPO additionally uses `CrowdSim/pref/`. Expert trajectory collection
uses the selected files under `CrowdSim/flow/`. PPO-specific tests and evaluation
utilities are included under `CrowdSim/tests/` and `CrowdSim/tools/`.

## External files

The following large or separately distributed files are intentionally not stored
in this Git repository:

- MaskedMimic checkpoint, normally
  `data/pretrained_models/masked_mimic/smpl/last.ckpt`
- `resolved_configs_inference.pt` in the same directory as that MaskedMimic
  checkpoint (required by the runtime)
- motion file referenced by `CrowdSim/config/env.yaml`
- warehouse or other scene USD referenced by `CrowdSim/config/scenes/*.yaml`
- trained robot PPO and reward-model checkpoints

Put those files at the configured paths or update the YAML files. ProtoMotions
robot definitions and mesh assets required by its simulator backends are included
under `protomotions/data/assets/`.

Some media and USD files in the source checkout were available only as Git LFS
pointer files. Their pointer metadata is retained, but the corresponding binary
objects must be downloaded from the upstream ProtoMotions distribution when a
simulator configuration needs them.

## Environment

The tested path uses **Isaac Sim package 6.0.0.1**, **Python 3.12**, the PhysX
backend, and IsaacLab commit `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`.
That checkout has product `VERSION=3.0.0` and Python package `isaaclab==12.0.0`.
It is a pinned development commit; the product and package numbers differ.

For an existing checkout with the specified Sim and Lab already installed,
apply and check the recorded dependency profile from this repository:

```bash
../IsaacLab-Protomotions/.venv/bin/python scripts/isaaclab3_environment.py apply \
  --lab-root ../IsaacLab-Protomotions --sync
../IsaacLab-Protomotions/.venv/bin/python -I scripts/isaaclab3_environment.py check
source ../IsaacLab-Protomotions/.venv/bin/activate
```

The [environment profile](env/isaaclab3_sim6/README.md) includes the complete
manifest/lock, pytest and a check for the repaired USD provider conflict. The
script validates the existing installation, backs up its dependency files and
does not perform a new Sim installation. Run the commands below with that Python.

The scene builder checks required files before launching Isaac Sim and reports
missing files and unresolved Git LFS pointers. Scene-only `--empty` mode does
not require humanoid checkpoints, motion files or a navigation map.

## Standalone pedestrians

The default uses existing SMPL MaskedMimic weights and AMASS validation motions,
SFM targets, a local neutral SMPL mesh, and a shared PhysX world. It requires no
robot policy. GUI mode runs until the window closes:

```bash
python -m CrowdSim.humanoid.simulate --num-humanoids 4
```

Bounded headless execution saves physical states, a result report, RGB frames
and an MP4 before Kit shuts down:

```bash
python -m CrowdSim.humanoid.simulate \
  --headless --steps 240 --num-humanoids 4 --reset-at 120 \
  --capture --output-dir output/pedestrian_simulation
```

The earlier four-person 8-second run passed for SFM + MaskedMimic.
The Diffusion + AMP implementation, new training artifacts and its independent
evaluation commands are documented in the guide linked above. Author experiment
equivalence still requires the unpublished motion subset and evaluation protocol.
Robot training/evaluation below also requires its configured assets and policies;
it was not included in the standalone pedestrian runtime acceptance.

## Train

```bash
python CrowdSim/ppo/train_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --total-steps 5000000 \
  --headless
```

Resume:

```bash
python CrowdSim/ppo/train_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --resume output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --headless
```

Collect expert trajectories:

```bash
python CrowdSim/flow/collect_ppo_buffer.py \
  --env-config CrowdSim/config/collect/ppo.yaml \
  --ppo-ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt
```

Preference-reward fine-tuning:

```bash
python CrowdSim/ppo/train_ppo_finetune.py \
  --env-config CrowdSim/config/env.yaml \
  --train-config CrowdSim/config/reward_model_ppo.yaml \
  --base-ppo PATH/robot_ppo_latest.pt \
  --reward-ckpt PATH/reward_model.pt \
  --headless
```

## Evaluate and test

Grid evaluation preserves the configured humanoid count, runs one robot, and
disables the training curriculum. Goal distances come from `navigation.path`,
so evaluation with the default config uses 7–10 m goals.

```bash
python CrowdSim/tools/eval_policy.py \
  --ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --env-config CrowdSim/config/env.yaml --num-episodes 200 --seed 42 --headless
```

Cached plots can be regenerated without starting Isaac Sim or supplying a
checkpoint:

```bash
python CrowdSim/tools/eval_policy.py --load-data PATH/policy_eval.pkl
```

Core tests do not start Isaac Sim but require PyTorch, pytest, and the map/plot
Python dependencies in the environment profile. The simulator-free package is
tested directly from this source checkout:

```bash
PYTHONPATH=src:. python -m pytest --import-mode=importlib tests CrowdSim/tests
```

Tests under `protomotions/tests` additionally cover other simulator backends.
The earlier completion-stage full regression passed 1,886 tests with two skips and six LFS
deselections, including Newton FK. Two local SMPL inference configuration tests
were also run separately and passed. This historical result predates the latest
pedestrian changes. The current AMP/Diffusion core gate passed 463 tests in 20
files; see [the final focused log](docs/amp_improvement_20261001/final_core_tests.log).
The old G1 motion cache is inconsistent with
the current asset; the backend agrees with independently computed current-asset
FK. SMPL exp_map is verified on PhysX; Newton/MuJoCo explicitly reject that
three-axis contract unless configured for matching hinge data and policies.

The simulator-free package can also be installed independently; its default
configuration ships in the wheel and works outside a source checkout:

```bash
python -m pip install -e ./src
crowdsim-ppo-train --device cpu --total-steps 4096 --output output/point_goal
crowdsim-ppo-eval --device cpu --checkpoint output/point_goal/latest.pt --episodes 20
```
