# Full Isaac Lab PPO pipeline

Use Isaac Sim 6.0.0.1, the pinned Isaac Lab 3.0 development checkout, and the supplied SMPL assets. Run from this source checkout. The installable ProtoMotions core and the CrowdSim pedestrian application have separate packaging boundaries.

## Current pedestrian results

| Component | Current result | Selection |
| --- | --- | --- |
| AMASS style data | 2,848 genuine, nonoverlapping segments from 1,853 trials | Fixed v7; the authors' private IDs are unavailable |
| Diffusion | Trained five-state social trajectory model, step 5000 | Fixed deployment model; ORCA-generated interactions are an engineering dataset |
| Navigation AMP | Backward-penalty step 1000; four people for 200 s, zero falls, 21 arrivals | Selected deployment baseline |
| Mixed AMASS/procedural AMP | Step 2000 tracking 19/22, zero falls, position 0.200 m; navigation 11 falls | Tracking improves; navigation candidate rejected |
| State-and-action AMP | Fresh 2000 updates / 16.384M transitions; full tracking 17/22, zero falls, position 0.208 m / rotation 0.726 rad | Inclusive MMD worsens; registered 200 s navigation in progress |
| Native Warehouse | Actual 30 s, zero falls, zero contacts, eight arrivals | Full 200 s and 15-person quality tests pending |
| Native map resources | Actual v7 Warehouse / Full Warehouse admitted | Office / Hospital fail negative timeline; session-range correction pending SDK validation |
| Optional arm-style blend | alpha .25 actual 30 s moves more but inclusive pose MMD worsens | Experimental; default unchanged |

These local results do not establish the paper's eight-scene performance or superiority. Tracking, pose style, speed, falls and contacts are reported separately, including failed frames.

- [Current project structure and resource/evaluation index](docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md)
- [AMASS, training experiments, fixed checkpoints and physical evidence](docs/AMP_STYLE_PROGRESS_20261002_ZH.md)
- [Read-only AMASS style coverage audit](docs/style_improvement_20261002/amass_style_coverage_v7_read_only_v1.md)
- [Registered full 22 / 200 s state-action evaluation](docs/style_improvement_20261002/state_action_evaluation_registration_v1/README.md)
- [Historical implementation and training stage](docs/PEDESTRIAN_AMP_DIFFUSION_PROGRESS_20261001_ZH.md)

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
