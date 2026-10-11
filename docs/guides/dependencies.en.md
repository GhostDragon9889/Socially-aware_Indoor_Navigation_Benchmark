# Project dependencies

[中文](dependencies.zh-CN.md) · [Guide index](README.md) · Updated 2026-10-10

## Default: IsaacLab 3.0 EA / Isaac Sim 6.1

The [official EA documentation](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html), pinned source and checked-in locks define the target. Product VERSION, package version and source revision are separate identities.

| Component | Pin |
| --- | --- |
| Platform / Python | Linux x86_64 / 3.12.13 |
| IsaacLab source commit | `ae37b028ea415c91ea2bc32609efcd759ed2b974` |
| IsaacLab source tree | `27e5b57ad0928e0339e121a06be040ae234a99ed` |
| Product VERSION / package | `3.0.0` / `isaaclab==17.0.2` |
| Isaac Sim package | `6.1.0.0` |
| Torch / NumPy / Warp | `2.11.0+cu128` / `2.5.1` / `1.16.0` |
| Standalone USD provider | `usd-exchange==2.3.0`; USD `25.5` |
| Project interpreter | `.runtime/isaaclab3_ea_sim61/.venv/bin/python` |

## Authoritative files and installation boundaries

| File | Purpose |
| --- | --- |
| [Root pyproject.toml](../../pyproject.toml) | Core ranges, five source packages and optional backends |
| [SDK manifest](../../env/isaaclab3_ea_sim61/pyproject.toml), [SDK uv.lock](../../env/isaaclab3_ea_sim61/uv.lock) | Complete snapshot for a new pinned SDK |
| [profile.json](../../env/isaaclab3_ea_sim61/profile.json) | Source, binary providers and manifest/lock hashes |
| [Dependency inventory](dependency-inventory-isaaclab3-ea-sim61.json) | Machine-readable SDK, overlay and core declarations |
| [requirements.overlay.lock](../../env/isaaclab3_ea_sim61/requirements.overlay.lock) | 32 exact project additions |
| [Runtime instructions](../../env/isaaclab3_ea_sim61/README.md) | Reuse, install and read-only validation |
| [Constraint receipt](../../env/isaaclab3_ea_sim61/runtime_constraints_receipt.txt) | Observed visible packages, not a standalone SDK installation lock |

The SDK and project additions use separate virtual environments. A project `.pth` references SDK site-packages and source while prioritizing this checkout's `src/`. Torch, NumPy, Warp, MuJoCo, Isaac packages and USD providers are not installed twice. Installed SDK files are not changed. Validation checks provider origins, 169 USD RECORD entries, default overlay dependencies and 18 project core requirements: 116 requirements satisfied locally. Failure preserves the selected `.venv`; changes to the SDK, lock or checkout path reject an old receipt.

## Project additions

| Package | Exact version | Role |
| --- | --- | --- |
| `omegaconf` / `hydra-core` | `2.3.0` / `1.3.2` | Project and saved configurations |
| `tensordict` | `0.9.0` | Policy tensors and batches |
| `lightning` / `pytorch-lightning` | `2.5.6` / `2.5.6` | Training |
| `wandb` / `protobuf` | `0.23.0` / `6.33.6` | Logging and SDK compatibility |
| `dm-control` / `dm-env` | `1.0.44` / `1.6` | MuJoCo motion and kinematics |
| `smplx` / `chumpy` | `0.1.28` / `0.70` | Licensed SMPL parsing and skinning |
| `scikit-image` | `0.26.0` | Image processing and evaluation |
| `moviepy` / `imageio-ffmpeg` | `1.0.3` / `0.6.0` | Video handling |
| `easydict` / `termcolor` | `1.13` / `3.3.0` | Reference controller settings and logging |
| `pytest` | `9.0.3` | CPU functionality tests |

The overlay lock contains the full list. `dm-control==1.0.40` failed with a missing `flex_bandwidth` on the target MuJoCo 3.11. The [official 1.0.44 release](https://github.com/google-deepmind/dm_control/releases/tag/1.0.44) passed actual CPU contact, stepping, copy and reset checks. Successful imports alone do not establish this compatibility.

## Physics and asset contracts

- Preserve trained AMP dynamics explicitly: `use_newton_actuators=False`, `enable_external_forces_every_iteration=False`.
- The new importer defaults root USD to `Physics=physics`; robot spawning and path inspection explicitly select `physx`, preserving SMPL's 24 bodies and 69 DOFs.
- Warp/ProxyArray/Torch conversions use explicit views; state writes and targets use current official `_index` methods.
- The [upstream reference registry](../../data/pretrained_models/upstream_reference_registry.json) archives only two old NavIsaacLab branch-specific configs as unsupported. Original weights/sidecars remain intact. Current TRACE/PACER and local fine-tuned releases are outside this exclusion.
- Software and CPU checks do not certify new-SDK four-scene 60-second physics, videos or paper performance. Existing four-scene recordings used Sim 6.0. New runs require their own exits, states and quality reports.

## System tools and other backends

Install Git/Git LFS, uv, NVIDIA drivers and compatible hardware following the [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html). Recording requires system `ffmpeg` and `ffprobe`; Python video packages do not provide proof that both commands are available. AMASS, SMPL parameters, weights, scenes and materials are external resources: see [configuration](configuration.en.md).

The default `isaaclab` extra targets the new stack; `isaaclab6` targets Lab12/Sim6.0. The [old snapshot](../../env/isaaclab3_sim6/README.md), dependency inventory and historical experiments remain for replay. Do not run root `uv sync --all-extras` inside the SDK or runtime: MuJoCo, Newton, Genesis, IsaacGym and the standalone PPO baseline have separate environment boundaries.

The root metadata supports Python3.12 and limits uv universal resolution to Linux x86_64. Its Lab packages use portable pinned official Git subdirectories; Sim6.1 explicitly declares `tinyobjloader==2.0.0rc13`. All six simulator extras have pairwise conflicts. Real uv metadata resolution with both pinned Git sources passed under the default prerelease policy; this dry-run does not install or validate their physics. Root MoviePy ranges allow the supported 1/2 writer APIs, while this runtime remains locked to1.0.3. The historical standalone `newton[sim]==1.0.0` extra selects Warp1.12 and is separate from the SDK-owned Newton1.5.2 / Warp1.16 stack.

Next: [installation](installation.en.md) → [configuration](configuration.en.md) → [usage](usage.en.md).
