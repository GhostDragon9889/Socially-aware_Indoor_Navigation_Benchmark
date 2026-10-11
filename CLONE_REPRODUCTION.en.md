# Reproduce the four-scene pedestrian pipeline from a Git clone

[中文](CLONE_REPRODUCTION.md) · 2026-10-10

## Included and external inputs

The current pipeline keeps Diffusion planning, AMP control, training, simulation adapters, resource restoration, evaluation, rendering, configurations, dependency locks and the native shutdown patch in this repository. It does not require code copied from NavIsaacLab, separate TRACE/PACER checkouts or the chat workspace.

| Component | Location |
| --- | --- |
| Diffusion, AMP, humanoid world and native scene adapters | `src/CrowdSim/humanoid/` |
| ProtoMotions, IsaacLab backend and asset conversion | `src/protomotions/` |
| CLI, exact resource relocation and restoration | `src/social_navigation/` |
| Independent PointGoal PPO baseline | `src/crowdsim_ppo/` |
| Training, evaluation, recording and data conversion | `scripts/`, `data/scripts/` |
| Portable scene profiles, cameras and resource identities | `configs/reproduction/` |
| Gait priors and completed AMP fine-tuning recipe | `configs/diffusion_amp/four_scene_speed_20261010/` |
| Default SDK and exact project overlay | `env/isaaclab3_ea_sim61/`; historical `env/isaaclab3_sim6/` |
| Selected weights, resume checkpoints, EVT robot and four scenes | `resources/releases/pedestrians_20261010/` |
| SMPL UV and attributed textures | `resources/pedestrian_visuals/` |

Obtain licensed **AMASS motion datasets** and **SMPL parameter files** separately. Git, Git LFS, uv, NVIDIA drivers and Isaac software are installation dependencies; recording also requires ffmpeg/ffprobe. Existing four scenes are supplied through the project's private LFS release. Additional scenes require separate downloads and fresh map generation.

This reproduces the implemented public TRACE Diffusion + locally fine-tuned PACER AMP pipeline, with documented method adaptations. It does not establish equivalence to the authors' unpublished models, motion list or eight-scene protocol. The EVT asset is available; a complete robot social-navigation benchmark is not integrated into the current four-scene recording entry. Historical `docs/style_improvement_*` experiment scripts and receipts are not maintained fresh-run inputs.

## 1. Clone and install the exact SDK

Use Linux x86_64 and an NVIDIA GPU supported by Isaac Sim 6. Authenticate Git access to the private repository.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git
cd Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
git lfs install --local
uv python install 3.12.13
uv run --no-project --python 3.12.13 python scripts/audit_checkout.py
# Option A: reuse an installed matching SDK
python3 scripts/bootstrap_reproduction.py configure --lab-root /absolute/path/to/IsaacLab
# Option B: full installation inside this project (choose instead of A)
# python3 scripts/bootstrap_reproduction.py install
export NAV_PY="$(python3 -c 'import json; print(json.load(open(".runtime/external_sdk.json"))["python"])')"
export NAV_ADDONS="$NAV_ROOT/.runtime/external_sdk_packages"
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT:$NAV_ADDONS"
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
```

The default pins Python3.12.13, Lab3EA source `ae37b028ea415c91ea2bc32609efcd759ed2b974` (source package17.0.2), Sim6.1.0.0 and Torch2.11.0+cu128. The interpreter belongs to the selected SDK. Project additions are exactly locked ordinary packages, with no second Python or simulator provider. The installer also restores Git-lost read-only permissions after checking all frozen input hashes. If an existing additions lock differs, `--rebuild-addons` builds and validates a replacement while retaining the previous directory. Read and accept NVIDIA EULA before Kit startup. The full fresh SDK installation has now been executed; see [setup and validation](docs/guides/external_sdk.zh-CN.md). Download resources below; licensed AMASS/SMPL inputs remain external.

## 2. Restore project resources

```bash
"$NAV_PY" scripts/manage_release_assets.py fetch
"$NAV_PY" scripts/manage_release_assets.py prepare --check-usd
git -c lfs.url=https://github.com/NVlabs/ProtoMotions.git/info/lfs \
  lfs pull --include='src/protomotions/data/assets/mesh/smpl/**,data/yaml_files/amass_smpl_*.yaml,src/protomotions/data/assets/usd/terrain_material.usda' --exclude=
```

The fetch command detects the project remote by URL, including `origin` in a fresh clone. Preparation verifies 467 release records and 14 original USD snapshots, restores weights, the fixed motion manifest, UV/textures and exact tested scene layers. Legacy file locations are translated in memory and in owned anonymous USD copies. Original receipts, geometry, physical opinions and checksums remain unchanged. Unregistered or missing dependencies fail explicitly.

## 3. Prepare licensed datasets

Download the matching SMPL motion sets from [AMASS](https://amass.is.tue.mpg.de/) into `<dataset>/raw/*.tar.bz2`. The committed split YAMLs and `ARCHIVES` mapping in `data/scripts/prepare_amass_dataset.py` define datasets and segments. Source-version substitutions do not satisfy the fixed identities.

```bash
export NAV_AMASS=/absolute/path/to/AMASS
"$NAV_PY" data/scripts/prepare_amass_dataset.py "$NAV_AMASS" \
  --stage prepare --splits train validation test --workers 4
"$NAV_PY" scripts/rebuild_fixed_amass.py --source-bank "$NAV_AMASS/amass_smpl_train.pt"
"$NAV_PY" scripts/rebuild_fixed_amass.py --split validation --source-bank "$NAV_AMASS/amass_smpl_validation.pt"
"$NAV_PY" scripts/rebuild_fixed_amass.py --split test --source-bank "$NAV_AMASS/amass_smpl_test.pt"
```

Existing generic banks produced by the repository converter can start at the last three commands. Outputs go to `data/datasets/amass/`. The rebuild retains the fixed **2,848 / 22 / 34** train/validation/test segments, order, ranges, labels, fingerprints and child-joint angular-velocity convention. Held-out banks contain the full generic population; the manifest selects evaluation clips. Container templates store serialization metadata, not motion tensors. Full output SHAs must match the original banks; mismatches are retained and rejected. See the [Chinese guide](CLONE_REPRODUCTION.md#3-准备-amass-和-smpl) for the three SHA256 values.

The fixed training-bank rebuild from the local generic bank was verified. Full conversion of all raw archives was not repeated in this change. Different source versions or numerical results fail the exact hash gate. The local fixed list is based on real AMASS; the authors' own motion IDs remain unpublished.

Obtain the neutral model under the [SMPL](https://smpl.is.tue.mpg.de/) license and place `SMPL_NEUTRAL.pkl` in `data/smpl/`. UV/textures come from step 2. Physical rig meshes use the project's MJCF/STL assets; SMPL parameters are needed for skinning and foot/camera audits.

An ignored `data/external_resources.json` can explicitly mount the three already verified banks using keys `amass_train`, `amass_validation`, `amass_test`. Values are absolute file paths; content checks remain mandatory.

## 4. Run, evaluate and record

```bash
"$NAV_PY" scripts/audit_checkout.py --require-resources --output output/checkout_audit.json
"$NAV_PY" scripts/run_four_scene_benchmark.py --preflight-only --output output/four_preflight
"$NAV_PY" scripts/run_four_scene_benchmark.py --record --output output/four_scene_run
```

Use `--scenes SecondFloor` for one scene; omit `--record` for physical simulation and evaluation only. A short physical smoke is available through `"$NAV_PY" -m social_navigation simulate --scene SecondFloor --steps 60 --output output/smoke`; it does not replace the 60-second quality gates.

Each scene uses 1800 controls at 30Hz and retains source/config snapshots, actual return codes and independent posture, speed, contact, foot-surface, spatial-stasis and 120Hz speed-cap checks. Videos are written to `<output>/<scene>/videos/<scene>_60s.mp4`. Recording uses exact measured poses and times without interpolation. Failed gates preserve evidence and stop recording. Always use fresh output directories.

## 5. Fine-tuning

The repository includes the completed recipe, parent weights, public anchor teacher, real-training gait priors and resume checkpoints. The AMP command and arguments are shown in [usage instructions](docs/guides/usage.en.md#3-amp-fine-tuning). New weights require independent evaluation and a new profile with a pinned SHA; they are not automatically admitted.

Diffusion research training entries are `scripts/train_paper_diffusion.py` (AMASS manifest) and `scripts/train_orca_diffusion.py` (separate ORCA dataset manifest). Their training protocols differ from the public TRACE model selected for the four-scene pipeline.

Validation scope and limitations: [CLONE_VALIDATION_20261010.json](docs/CLONE_VALIDATION_20261010.json). Fresh SDK installation, complete new 60-second four-scene runs and author-protocol equivalence require separate evidence.

UV/textures are ordinary Git blobs; existing weights, robot and scenes use Git LFS.
