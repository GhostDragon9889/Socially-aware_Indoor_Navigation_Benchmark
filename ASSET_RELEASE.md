# Current model and scene resources / 当前模型与场景资源

The `pedestrians_20261010` resource release supplies the selected four-scene inputs. Source, SDK installation and resource downloads are managed separately. See [installation](docs/guides/installation.en.md), [resource preparation](CLONE_REPRODUCTION.en.md) and [current validation limits](docs/validation.md).

| Resource | Release path | Use |
| --- | --- | --- |
| TRACE | `models/trace/` | Trajectory diffusion |
| PACER teacher | `models/pacer_teacher/` | Frozen AMP teacher |
| Conservative AMP | `models/amp/speed_conservative_20261010/` | YOHO and NewMetro |
| Walking/turning AMP | `models/amp/walking_turn_20261009/` | SecondFloor, ThirdFloor and current training parent |
| Selected training state | `training/` | Current selected weights' recipes and complete recovery state |
| Robot | `robot/EVT-EXP.usd` | Navigation robot asset |
| Four scenes | `scenes/` | USD layers, materials and textures |
| Native maps | `evidence/` | Required frozen geometry and resource identities |

Release paths above are relative to [resources/releases/pedestrians_20261010](resources/releases/pedestrians_20261010/). [manifest.json](resources/releases/pedestrians_20261010/manifest.json) records payload sizes, hashes and provenance; [catalog.json](resources/releases/pedestrians_20261010/catalog.json) declares USD entry points. Historical unused training outputs are not included in the source tree. The selected current release remains available for reproduction.

Configure either the external SDK or the uv-installed SDK, then run:

```bash
python3 scripts/run_external_python.py scripts/manage_release_assets.py fetch
python3 scripts/run_external_python.py scripts/manage_release_assets.py prepare --check-usd
python3 scripts/run_external_python.py scripts/manage_release_assets.py catalog
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
```

The downloader selects the benchmark's private LFS endpoint; access to the private repository is required. Existing upstream LFS pointers retain their original NVlabs endpoint. Resources are checked against their manifest before installation. `prepare --check-usd` also verifies local USD reference closure; SDK material modules are resolved by Isaac Sim.

SMPL body parameters, original AMASS motion data and SDK binaries require separate preparation under their respective terms. Local resource bindings belong in ignored `data/external_resources.json`. Recorded provenance may retain original resource locations; executable code resolves resource references from configured bindings instead of requiring a workstation directory.

Keep the licenses in the release's `LICENSES/` directory and the project `legal/` notices. The working runtime targets IsaacLab 3.0-EA and Isaac Sim 6.1. Resource integrity and short runtime checks do not establish convergence or four-scene physical quality; see the validation page for the remaining gaps.

中文：仅保留当前四场景选用的 TRACE、PACER 教师、两组 AMP 权重及其训练状态、机器人、USD 和必要地图资源。旧实验输出、重放工具和历史报告已从工作树移除。环境只通过外部 SDK 配置或 uv 安装，下载和运行命令同上。SMPL/AMASS 等许可数据按各自授权独立准备，资源完整性检查不能代替物理质量验收。
