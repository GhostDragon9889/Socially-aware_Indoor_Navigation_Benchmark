# Full Isaac Lab PPO pipeline

## Paper pedestrian method: Diffusion + AMP

The new method entry points are `scripts/train_paper_diffusion.py` and
`scripts/train_paper_amp.py`, with `configs/diffusion_amp/method_reconstruction.yaml`.
They train a five-state trajectory x0 diffusion model and an action-conditioned
AMP policy using direct SMPL PhysX simulation. The standalone pedestrian command
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
The current complete regression passed 1,886 tests with two skips and six LFS
deselections, including Newton FK. Two local SMPL inference configuration tests
were also run separately and passed. The old G1 motion cache is inconsistent with
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
