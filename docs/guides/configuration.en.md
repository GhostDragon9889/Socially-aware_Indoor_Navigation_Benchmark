# Configuration and resources

[简体中文](configuration.zh-CN.md) · [Documentation index](README.md) · Updated: 2026-10-10

## 1. Working directory and environment variables

Use the pinned environment from [installation](installation.en.md), starting at the checkout root:

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
export NAV_LAB_ROOT="$(cd ../IsaacLab-Protomotions && pwd)"
export NAV_PY="$NAV_LAB_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$NAV_PY" -m social_navigation status
```

| Variable | Role |
| --- | --- |
| `NAV_ROOT` / `NAV_PY` | Shell variables used by these examples; the application does not read them as configuration |
| `SOCIAL_NAVIGATION_ROOT` | Override checkout discovery; must point to a complete source tree |
| `PROTOMOTIONS_ASSET_ROOT` | Override `protomotions.assets`; the tree must contain mjcf/mesh/usd subdirectories |
| `PYTHONPATH` | Make `src` and the checkout root available for source commands |
| `PYTHONDONTWRITEBYTECODE` | Avoid pyc files in source and SDK trees |
| `CUDA_VISIBLE_DEVICES` | Select a GPU; standalone CPU analysis examples clear it |
| `OMNI_KIT_ACCEPT_EULA` | Kit license setting after accepting NVIDIA's terms, described in installation |

Explicit source and asset locations:

```bash
export SOCIAL_NAVIGATION_ROOT="$NAV_ROOT"
export PROTOMOTIONS_ASSET_ROOT="$NAV_ROOT/src/protomotions/data/assets"
```

These variables do not rewrite absolute paths in JSON/YAML. Migrate those entries individually; a new machine does not need to recreate the old `/project/EXP/...` layout.

## 2. Resource inventory

| Resource | Location or entry point | Role and availability |
| --- | --- | --- |
| Upstream models/examples | `data/pretrained_models/`, `examples/data/` | Original Git/LFS references restored; download selectively |
| TRACE model/config | `data/pretrained_models/paper_references/trace/ckpt/trace/orca_mixed/{iter40000.ckpt,config.json}` | Public diffusion model used by the four-scene profiles; included in this project’s private LFS release |
| PACER anchor teacher | `data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth` | Teacher for local AMP fine-tuning; included in the private release, with an original downloader available |
| Selected AMP, YOHO/NewMetro | `data/pretrained_models/local_amass_finetune/speed_conservative_20261010/Humanoid_AMASS.pth` | Local fine-tuned artifact included in this project’s private LFS release |
| Selected AMP, upper floors | `data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth` | Current narrow-scene controller and fine-tuning parent, included in the private release |
| Fixed AMASS | `data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json` and train bank | 2,848 training, 22 validation and 34 test segments |
| General AMASS libraries | Adjacent `Assets/motion/` | Authorized archives, per-motion conversions and train/test/validation MotionLib |
| SMPL/body assets | `src/protomotions/data/assets/` and body parameters referenced by rendering | Physics skeleton, visible skin and body parameters are distinct resources |
| Indoor scenes | `/project/EXP/Sence/Sences/` or your new resource location | Preserve USD together with materials and referenced layers |
| Native map evidence | Scene `preflight`, `native_map`, `probe_script_snapshot` | External artifacts binding the scene, PhysX queries, support surface and generator source |

Raw scene directory names are `Rest_Room_HongKong_YOHO`, `Rest_Room_HongKong_NewMetro`, `Rest_Room_Second_Floor`, and `Rest_Room_Third_Floor`. CLI names are `YOHO`, `NewMetro`, `SecondFloor`, and `ThirdFloor`.

This project’s [private asset release](../../ASSET_RELEASE.md#english-instructions) contains the selected TRACE/AMP/teacher weights, training parameters and resume checkpoints, EVT robot, four scenes and their material references. Raw AMASS, licensed SMPL parameters, SDKs and videos remain separate. The release’s 467 file records were verified; see the [publication receipt](../ASSET_PUBLICATION_20261010.json). The old 47-item asset ledger describes the earlier upstream merge; see the [merge record](../UPSTREAM_MERGE_20261010_ZH.md).

Restore published resources from the checkout root:

```bash
python scripts/manage_release_assets.py fetch
python scripts/manage_release_assets.py verify
python scripts/manage_release_assets.py install-models
```

The fetch helper explicitly selects the private project LFS endpoint and requires an authorized GitHub account; the default `.lfsconfig` download endpoint remains NVlabs upstream. install-models restores seven model/config/manifest items, without motion banks or native-map migration. The three data/smpl pkl files are currently symlinks to external arena_assets; prepare their actual parameters separately on a new host. Published USD dependencies were verified, while relocated native-map admission and physics require new validation. See the [comprehensive local audit (Chinese)](../RESOURCE_DEPENDENCY_MAP_20261010_ZH.md).

### Public PACER download

Use a separate download environment. Google Drive availability and limits may affect downloads.

```bash
cd "$NAV_ROOT"
uv venv --python 3.12 .venv-download
uv pip install --python .venv-download/bin/python gdown==6.2.0 requests
.venv-download/bin/python scripts/fetch_pacer_reference.py --roles no_shape
```

Obtain TRACE through the release instructions in its [pinned original repository](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c). PACER files and ProtoMotions SMPL trackers are not interchangeable. Verify model identity against the profile's actual SHA.

## 3. General motion banks and two curated manifests

The recorded general MotionLib banks contain 11,303 train / 345 validation / 138 test motions. The earlier amass_manifest.json selects only 319 / 22 / 34. Current fixed v7 selects 2,848 train / 22 validation / 34 test segments. Its 2,848 training segments come from 1,853 real parent motion files; source availability and current bank SHAs were verified. Validation/test still point to complete general `.pt` banks; selection is defined by `clips`, not by the total number of motions inside those files.

Suggested layout:

```text
Assets/motion/
  raw/                              Authorized archives and extracted data
  processed/smpl/                    Converted .motion files and provenance
  amass_smpl_train.pt
  amass_smpl_validation.pt
  amass_smpl_test.pt
  preparation/                      Hashes, conversion and coverage records
<repo>/data/diffusion_amp/amass_style_2848/
  manifest_labeled_v7.json
  manifest_labeled_v7_train.pt
```

Use the following only when creating new libraries. First extract authorized archives into `raw/` as described by the [AMASS preparation record](../AMASS_PREPARATION_ZH.md). Scripts do not download licensed data and refuse to overwrite final banks.

```bash
cd "$NAV_ROOT"
"$NAV_PY" data/scripts/prepare_amass_dataset.py \
  /absolute/path/to/Assets/motion --splits validation test train --workers 4
"$NAV_PY" data/scripts/prepare_amass_dataset.py \
  /absolute/path/to/Assets/motion --splits validation test train --stage pack
"$NAV_PY" data/scripts/prepare_amass_dataset.py \
  /absolute/path/to/Assets/motion --splits validation test train --stage verify
"$NAV_PY" scripts/prepare_diffusion_amp_data.py \
  --motion-root /absolute/path/to/Assets/motion \
  --output data/diffusion_amp/amass_manifest.json
"$NAV_PY" scripts/curate_amass_pedestrian_data.py \
  --base-manifest data/diffusion_amp/amass_manifest.json \
  --config configs/diffusion_amp/amass_curation.yaml \
  --output output/amass_curation_new/manifest.json
```

Curation creates a new local subset; it does not guarantee the IDs, labels or SHA of the existing labeled v7, or recover the authors' unpublished list. Keep fixed v7 intact. New data requires a separate manifest/config and validation of split isolation. The current fine-tuning wrapper defaults to fixed v7 and requires exactly 2,848 training IDs.

## 4. Four-scene configuration

Directory: `configs/diffusion_amp/four_scene_speed_20261010/`.

| File | Contents |
| --- | --- |
| `<scene>_profile.json` | Four resource paths/SHAs, diffusion guidance, replanning, braking, body map guards and candidate admission |
| `<scene>_scene.yaml` | Native scene/map evidence, support surface, spawns, cyclic routes, clearance and lookahead; current files are JSON-compatible YAML |
| `<scene>_cameras.json` | Shot eye/target, start control and focal length |
| `locomotion_conservative128_v3.json` | PPO/AMP learning rates, objective weights, symmetry and gait priors; schema v5 |
| `gait_conservative_prior_v3.json` / `temporal_speed_prior_v1.json` | Spatial/temporal gait statistics fitted only on training data, with source SHAs |
| `walking_conservative_windows_v3.json` | Speed windows and initialization segments from the real training bank |

Profile schema: `crowdsim.trace_pace_reliability.v1`. Native scene schema: `crowdsim.native_pedestrian_scene.v1`.

| Setting | Meaning |
| --- | --- |
| `resources.*.path` / `.sha256` | Resource location and complete file SHA256 |
| `replan_controls` | Number of 30Hz controls between plans; currently 15 for YOHO |
| `num_samples` | Diffusion candidates per batch |
| `guidance.max_speed_m_s` / `target_speed_m_s` | Planning cap/target, not measured body-speed certification |
| `max_acceleration_m_s2` / `max_yaw_rate_rad_s` | Candidate motion constraints |
| `map_weight` / `agent_weight` | Map and peer guidance costs |
| `origins_xy` / `routes_xy` | Scene-world XY coordinates in meters; current measured-contact entry admits N1/N4 |
| `clearance_m` / `route_comfort_clearance_m` | Required/preferred clearance, checked with actual body envelopes |
| `preflight` / `native_map` / `probe_script_snapshot` | External evidence and SHAs that must be migrated together |

## 5. Custom models and scenes

Unified `preflight` / `simulate` commands use the fixed four-scene directory and do not accept custom `--config`. Put a new configuration under `configs/local/` and use `run_trace_pace.py` / `run_trace_pace_verified.py` directly.

Example candidate binding, after the new weight exists:

```bash
"$NAV_PY" - <<'PY'
import hashlib
import json
from pathlib import Path
source = Path('configs/diffusion_amp/four_scene_speed_20261010/YOHO_profile.json')
profile = json.loads(source.read_text())
weight = Path('output/my_verified_training/Humanoid_AMASS.pth').resolve()
profile['resources']['amp_checkpoint'] = {
    'path': str(weight),
    'sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
}
profile['local_finetune_admission'] = {
    'candidate': True,
    'physical_quality_pending': True,
    'purpose': 'New candidate; independent physical/style evaluation required',
    'author_experiment_reproduction': False,
}
destination = Path('configs/local/my_YOHO_profile.json')
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise FileExistsError(destination)
destination.write_text(json.dumps(profile, indent=2) + '\n')
PY
```

A matching path/SHA establishes file identity, not quality. Do not retain an old `tested_execution` as evidence for the new candidate. Copy the scene to `configs/local/my_YOHO_scene.yaml`, migrate its evidence paths, and verify USD references, map coordinates, support surface, clearance, spawns and routes again.

For a new collected scene, first create a static reference copy. Input must be Z-up, use meters and have a valid default prim; replace `scene.usd` with the actual entry file:

```bash
"$NAV_PY" scripts/prepare_collected_pedestrian_scene.py \
  --source /absolute/path/to/Collected_scene/scene.usd \
  --output "$NAV_ROOT/output/scenes/my_scene_static.usda" \
  --default-physics-owner
```

This does not generate a PhysX query map. The four-scene query tools, finite support surfaces and logs are source-bound external archives. Carry those archives to another machine, or rerun the matching probes and regenerate the complete evidence chain. There is currently no one-command converter from arbitrary USD to an admitted pedestrian scene.

```bash
"$NAV_PY" scripts/run_trace_pace.py \
  --config configs/local/my_YOHO_profile.json \
  --scene configs/local/my_YOHO_scene.yaml \
  --steps 1800 --measure-contacts --seed 7 --preflight-only \
  --output output/custom_preflight/new_run
```

The low-level resource preflight checks model/manifest identity only. Unified `social_navigation preflight` additionally checks native scene configuration. Custom scenes need equivalent native-map admission; a successful low-level preflight is not a physics result.

## 6. VS Code and configuration maintenance

Open root `Socially-aware_Indoor_Navigation_Benchmark.code-workspace` and select `.venv/bin/python`. `.vscode/settings.json` supplies source paths and excludes data/generated directories; `launch.json` and `tasks.json` expose status, preflight, baseline and native commands. A debugger still needs the same resource configuration and SDK environment.

Update `profile.json` hashes with manifest/lock changes, then run the environment checker. When changing a file listed in the upstream mapping, maintain its Git blob record and retain the initial merge value for traceability. Keep generated data, credentials and training weights outside the source publication manifest.

Next: [Training, scene evaluation and video](usage.en.md).

Current clone entry and path migration: [CLONE_REPRODUCTION.en.md](../../CLONE_REPRODUCTION.en.md).
