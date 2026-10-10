# Usage, training, evaluation and recording

[简体中文](usage.zh-CN.md) · [Documentation index](README.md) · Updated: 2026-10-10

## 1. Method and modules

The four-scene deployment uses public TRACE trajectory diffusion and a state-only PACER AMP controller fine-tuned on local AMASS. Planned trajectories enter policy observations; AMP outputs PD position targets and PhysX advances pedestrians. Simulation does not move roots along prescribed paths or interpolate joint poses. Recording subsequently replays measured physical poses with camera cuts.

The NavIsaacLab five-state Diffusion/history-plus-action AMP reconstruction remains a separate experiment. The public TRACE/PACER adaptation is not the authors' private model or a complete reproduction.

| Module | Responsibility |
| --- | --- |
| `src/social_navigation/` | Dispatch, root discovery and additional native scene preflight |
| `src/CrowdSim/humanoid/diffusion/` | TRACE networks, candidates and trajectory guidance |
| `src/CrowdSim/humanoid/amp/`, `training/` | AMP interface, expert observations, PPO/discriminator and gait fine-tuning |
| `src/CrowdSim/humanoid/runtime/` | Lifecycle, actual state, map/contact data and shutdown ownership |
| `src/CrowdSim/humanoid/analysis/` | Posture, speed, coverage and contact evaluation |
| `src/protomotions/` | IsaacLab backend, SMPL configuration, MotionLib and upstream training |
| `src/crowdsim_ppo/` | Independent PointGoal navigation baseline |

## 2. Prepare a terminal and inspect commands

Complete [installation](installation.en.md) and [configuration](configuration.en.md) first.

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
export NAV_LAB_ROOT="$(cd ../IsaacLab-Protomotions && pwd)"
export NAV_PY="$NAV_LAB_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$NAV_PY" -m social_navigation status
```

```bash
"$NAV_PY" -m social_navigation --help
"$NAV_PY" -m social_navigation train-amp --help
"$NAV_PY" -m social_navigation record --help
"$NAV_PY" -m social_navigation evaluate-physics --help
"$NAV_PY" -m social_navigation evaluate-posture --help
"$NAV_PY" -m social_navigation evaluate-speed --help
"$NAV_PY" -m social_navigation evaluate-stasis --help
"$NAV_PY" -m social_navigation audit-speed --help
"$NAV_PY" -m social_navigation audit-feet --help
"$NAV_PY" -m social_navigation audit-camera --help
```

Forwarded subcommands retain the original tool's arguments. Run native GPU/Kit jobs serially, use fresh output paths and preserve failure logs.

## 3. AMP fine-tuning

This is the existing 2026-10-10 conservative recipe. It needs fixed labeled v7, the public PACER anchor teacher, real training gait priors and the trained walking_turn parent. With 128 updates, 160 environments and 32 rollout steps, it collects 655,360 transitions. New-run falls and quality must be evaluated again.

```bash
cd "$NAV_ROOT"
export NAV_CFG="$NAV_ROOT/configs/diffusion_amp/four_scene_speed_20261010"
export NAV_PARENT="$NAV_ROOT/data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth"
export NAV_TRAIN="$NAV_ROOT/output/amp_finetune/my_new_128_update_run"
"$NAV_PY" -m social_navigation train-amp \
  --output "$NAV_TRAIN" --updates 128 --num-envs 160 \
  --parent-checkpoint "$NAV_PARENT" \
  --optimization-config "$NAV_CFG/locomotion_conservative128_v3.json" \
  --gait-prior "$NAV_CFG/gait_conservative_prior_v3.json" \
  --temporal-gait-prior "$NAV_CFG/temporal_speed_prior_v1.json" \
  --walking-turn-windows "$NAV_CFG/walking_conservative_windows_v3.json" \
  --task-horizons-s 0.5 1.0 2.0 4.5
CUDA_VISIBLE_DEVICES= "$NAV_PY" scripts/audit_pacer_amass_finetune.py \
  --run "$NAV_TRAIN" --output "${NAV_TRAIN}_cpu_audit.json"
```

Outputs include `last.pt`, `Humanoid_AMASS.pth`, `metrics.jsonl` and `result.json`, with adjacent `_execution.json`, `.log` and `_sources/`. CPU export auditing checks parameter changes, origins and training records; it does not replace independent style/navigation evaluation.

If the parent is unavailable, explicitly select public PACER weights for a new training experiment. That does not reproduce the validated parent chain or automatically admit the resulting controller. Changing the manifest/data count requires corresponding corpus and prior bindings; the current wrapper does not support arbitrary training counts.

Diffusion research entry: `scripts/train_paper_diffusion.py`. Paper-style AMP: `scripts/train_paper_amp.py`. Shared SocialCNN: `scripts/train_shared_social.py`. Inspect their `--help` and experiment documents; inputs, discriminator interfaces and checkpoint formats differ. The example above does not retrain TRACE.

## 4. A 60-second physical scene test

Validate the fixed resources and native map first:

```bash
cd "$NAV_ROOT"
"$NAV_PY" -m social_navigation preflight --scene YOHO
```

Then create a new physical source:

```bash
export NAV_SCENE=YOHO
export NAV_CFG="$NAV_ROOT/configs/reproduction/four_scene"
export NAV_RUN="$NAV_ROOT/output/physics/${NAV_SCENE}_new_60s"
"$NAV_PY" -m social_navigation simulate \
  --scene "$NAV_SCENE" --steps 1800 --seed 7 --output "$NAV_RUN"
```

1,800 controls at 30Hz correspond to 60 seconds. Measured-contact mode records 7,200 physical substeps at 120Hz per actor. Physical CLI budgets range from 1 to 6,000 controls; current full-video admission requires 1,800.

The fixed `simulate --scene` command cannot substitute a custom profile. Use the verified launcher directly for a candidate:

```bash
"$NAV_PY" scripts/run_trace_pace_verified.py --output "$NAV_RUN" -- \
  --config configs/local/my_YOHO_profile.json \
  --scene configs/local/my_YOHO_scene.yaml \
  --steps 1800 --measure-contacts --seed 7
```

Physical outputs include `result.json`, `physical.npz`, `physical_states.npz`, `planning.json`, `ground_support.npz` and `contact_measurements.npz`. The parent preserves the actual exit code, source/input snapshots and logs. Stop and retain evidence if exit/status indicates failure.

## 5. Evaluate each aspect

The block below enables `set -euo pipefail`. A failure can exit an interactive shell; alternatively save the block as a script. Evaluate the same completed physical source throughout.

```bash
set -euo pipefail
"$NAV_PY" -m social_navigation evaluate-physics \
  --run "$NAV_RUN" --execution "${NAV_RUN}_execution.json" \
  --output "${NAV_RUN}_quality.json"
"$NAV_PY" -m social_navigation evaluate-posture \
  --run "$NAV_RUN" --require-walking-screen --output "${NAV_RUN}_posture.json"
"$NAV_PY" -m social_navigation evaluate-speed \
  --run "$NAV_RUN" --scene "$NAV_CFG/${NAV_SCENE}_scene.yaml" \
  --output "${NAV_RUN}_speed.json"
"$NAV_PY" -m social_navigation evaluate-stasis \
  --run "$NAV_RUN" --output "${NAV_RUN}_spatial_stasis.json"
"$NAV_PY" -m social_navigation audit-speed \
  --run "$NAV_RUN" --output "${NAV_RUN}_substep_speed.json"
CUDA_VISIBLE_DEVICES= "$NAV_PY" -m social_navigation audit-feet \
  --run "$NAV_RUN" --steps 1800 --output "${NAV_RUN}_foot_surface.json"
```

| Check | Current local gate or interpretation |
| --- | --- |
| Physics/contact | Complete budget, no falls/resets, measured contact conditions and a successful actual exit |
| Posture | `--require-walking-screen` checks actual walking poses across the captured frames |
| Speed | Normal cruise target 0.5–1.5 m/s; starts, turns and yielding may slow down; all slow frames remain reported |
| Instantaneous speed | Actual XY speed ≤1.5 m/s across all 7,200 substeps |
| Spatial stasis | Complete 5s root-position windows determine spatial confinement; longest stasis ≤10s |
| Foot surface | Neutral SMPL skin relative to the declared support surface; not a contact-patch slip or friction measurement |

Style MMD/frozen-cluster coverage is separate analysis. Passing speed/safety screens does not establish greater realism or superiority over the paper. The 22 validation clips have already informed selection; they are not fresh unseen tests. Keep the 34 test clips independent. Detailed local definitions are in the [four-scene results](../FOUR_SCENE_SPEED_20261010_ZH.md).

## 6. Cameras and 60-second video

Visible actor lists must match each camera shot. Fixed configurations:

| Scene | Actors | `NAV_ACTORS` |
| --- | --- | --- |
| YOHO | 4 | `[[0,2],[1],[3]]` |
| NewMetro | 4 | `[[0,1,2],[3]]` |
| SecondFloor | 1 | `[[0],[0]]` |
| ThirdFloor | 1 | `[[0],[0]]` |

For YOHO, continue with the same `NAV_RUN`:

```bash
export NAV_ACTORS='[[0,2],[1],[3]]'
CUDA_VISIBLE_DEVICES= "$NAV_PY" -m social_navigation audit-camera \
  --run "$NAV_RUN" --camera-plan "$NAV_CFG/${NAV_SCENE}_cameras.json" \
  --actors "$NAV_ACTORS" --output "${NAV_RUN}_camera_skin.json"
"$NAV_PY" -m social_navigation record \
  --run "$NAV_RUN" --quality "${NAV_RUN}_quality.json" \
  --posture "${NAV_RUN}_posture.json" --speed "${NAV_RUN}_speed.json" \
  --substep-speed "${NAV_RUN}_substep_speed.json" \
  --spatial-stasis "${NAV_RUN}_spatial_stasis.json" \
  --spatial-stasis10 "${NAV_RUN}_spatial_stasis.json" \
  --foot-surface "${NAV_RUN}_foot_surface.json" \
  --camera-skin "${NAV_RUN}_camera_skin.json" \
  --camera-plan "$NAV_CFG/${NAV_SCENE}_cameras.json" \
  --output-dir "$NAV_ROOT/output/video/${NAV_SCENE}_new_60s" --tag "$NAV_SCENE"
```

The recorder also verifies physical/FK correspondence and camera-child execution. Shots use contiguous intervals from one physical source, without pose interpolation, time scaling or mixing physical runs. Rendering requires complete body/skin assets and scene materials.

Check the entire video rather than only its header:

```bash
export NAV_VIDEO="$NAV_ROOT/output/video/${NAV_SCENE}_new_60s/${NAV_SCENE}_60s.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_read_frames \
  -show_entries format=duration -of json "$NAV_VIDEO"
ffmpeg -v error -i "$NAV_VIDEO" -f null -
```

Fixed output is H.264, 960×720, 30FPS, 1,800 frames and 60 seconds, with `<scene>_video_receipt.json`. Skin-projection admission does not test every geometric occlusion; view the continuous video as well.

For each other scene, change `NAV_SCENE`, `NAV_RUN`, video output and `NAV_ACTORS`, then repeat preflight, physics and all evaluations serially. The `EVT-EXP.usd` robot is not part of this closed loop; pedestrian videos do not certify robot social navigation.

## 7. Lightweight development checks and evidence scope

```bash
cd "$NAV_ROOT"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$NAV_PY" -m pytest \
  --import-mode=importlib -p no:cacheprovider -q tests/test_project_layout.py
"$NAV_PY" scripts/verify_upstream_integration.py
```

Existing 2026-10-10 results include four local scene videos and bounded metrics. They do not establish the authors' complete eight-scene protocol, arbitrary crowd/seed generalization or better-than-paper performance. Source status, successful exit, full-frame quality, recording checks and human realism evaluation are different evidence. This documentation update checks commands/configuration/dependencies; it does not retrain or record new videos.
