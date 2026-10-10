# 配置与资源准备

[English](configuration.en.md) · [文档目录](README.md) · 更新：2026-10-10

## 1. 运行目录和环境变量

使用[安装指南](installation.zh-CN.md)中的固定环境，从源码根运行：

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
export NAV_LAB_ROOT="$(cd ../IsaacLab-Protomotions && pwd)"
export NAV_PY="$NAV_LAB_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$NAV_PY" -m social_navigation status
```

| 变量 | 作用 |
| --- | --- |
| `NAV_ROOT` / `NAV_PY` | 本文 shell 示例使用的路径变量，不是程序自动读取的配置 |
| `SOCIAL_NAVIGATION_ROOT` | 包内源码根定位覆盖；必须指向完整 checkout |
| `PROTOMOTIONS_ASSET_ROOT` | `protomotions.assets` 的资产树覆盖，需包含 mjcf/mesh/usd 等子目录 |
| `PYTHONPATH` | 从源码运行时提供 `src` 和项目根目录 |
| `PYTHONDONTWRITEBYTECODE` | 避免在 SDK/源码树生成 pyc |
| `CUDA_VISIBLE_DEVICES` | 选择 GPU；独立 CPU 分析示例置空 |
| `OMNI_KIT_ACCEPT_EULA` | 在已接受 NVIDIA 条款后用于 Kit 启动，见安装指南 |

需要显式定位源码/资产树时：

```bash
export SOCIAL_NAVIGATION_ROOT="$NAV_ROOT"
export PROTOMOTIONS_ASSET_ROOT="$NAV_ROOT/src/protomotions/data/assets"
```

这两个变量不会自动重写 JSON/YAML 中的绝对资源路径；应逐项迁移配置。新机器无需构造 `/project/EXP/...` 旧目录。

## 2. 资源清单

| 资源 | 路径或入口 | 用途与状态 |
| --- | --- | --- |
| 上游模型/示例 | `data/pretrained_models/`、`examples/data/` | 已恢复 Git/LFS 引用，按需从上游下载 |
| TRACE 权重和模型配置 | `data/pretrained_models/paper_references/trace/ckpt/trace/orca_mixed/{iter40000.ckpt,config.json}` | 四场景使用的公开扩散模型，需另行准备 |
| PACER 锚定教师 | `data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth` | 本地 AMP 微调教师，下载器支持 |
| 选定 AMP，YOHO/NewMetro | `data/pretrained_models/local_amass_finetune/speed_conservative_20261010/Humanoid_AMASS.pth` | 本地训练产物，不由上游 LFS 提供 |
| 选定 AMP，二/三层 | `data/pretrained_models/local_amass_finetune/walking_turn_20261009/Humanoid_AMASS.pth` | 本地父权重，需保留或重新训练 |
| 固定 AMASS | `data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json` 及对应 train bank | 训练 2,848、验证 22、测试 34 条选定片段 |
| 通用 AMASS 库 | 相邻 `Assets/motion/` | 原始归档、逐动作转换与 train/test/validation MotionLib |
| SMPL 与人体资产 | `src/protomotions/data/assets/`、录制器引用的人体模型参数 | 物理骨架、可视皮肤、人体参数是不同资源，按各自许可准备 |
| 室内场景 | `/project/EXP/Sence/Sences/` 或本机对应的新位置 | USD 与所有材质/引用层应一起保留 |
| 原生地图证据 | scene 配置中的 `preflight`、`native_map`、`probe_script_snapshot` | 绑定场景、PhysX 查询、支持面和生成源码，属于外部运行产物 |

原始四场景目录分别为 `Rest_Room_HongKong_YOHO`、`Rest_Room_HongKong_NewMetro`、`Rest_Room_Second_Floor`、`Rest_Room_Third_Floor`。CLI 名称分别为 `YOHO`、`NewMetro`、`SecondFloor`、`ThirdFloor`。

源码仓库不包含这些本地训练/许可数据和全部运行归档。旧资产清单的 47 条记录中只有 8 条路径与 OID 匹配固定上游，另 39 条来源未核实；不要把 LFS pull 当作全部生成资产的恢复工具。详见[合并记录](../UPSTREAM_MERGE_20261010_ZH.md)。

### 公开 PACER 下载

使用独立下载环境；原始 Google Drive 下载可能受到网络和限流影响。

```bash
cd "$NAV_ROOT"
uv venv --python 3.12 .venv-download
uv pip install --python .venv-download/bin/python gdown==6.2.0 requests
.venv-download/bin/python scripts/fetch_pacer_reference.py --roles no_shape
```

TRACE 从固定[原始仓库](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c)的发布说明获取。PACER 下载结果和官方 ProtoMotions SMPL tracker 不能互相替换；用 profile 中的真实 SHA 核对模型。

## 3. AMASS 的两种清单

通用 MotionLib 与固定导航风格子集不同。通用库历史数据为 train 11,303 / validation 345 / test 138；固定 v7 只选择 train 2,848 / validation 22 / test 34。验证/测试字段仍引用完整通用 `.pt`，实际选择由 `clips` 列表决定，不能按 `.pt` 的全部动作数解释划分。

建议目录：

```text
Assets/motion/
  raw/                              原始授权归档及其解压目录
  processed/smpl/                    转换后的 .motion 与来源记录
  amass_smpl_train.pt
  amass_smpl_validation.pt
  amass_smpl_test.pt
  preparation/                      来源、哈希、转换与覆盖报告
<repo>/data/diffusion_amp/amass_style_2848/
  manifest_labeled_v7.json
  manifest_labeled_v7_train.pt
```

只有准备新资源库时才执行以下流程；原始归档需要按 [AMASS 准备记录](../AMASS_PREPARATION_ZH.md)先解压到 `raw/`，脚本不会下载授权数据。准备脚本拒绝覆盖已存在的最终库。

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

该筛选命令生成新的本地子集，不能保证与既有 labeled v7 的 ID、标签或哈希相同，也不能复原作者未公开的私有清单。不要覆盖固定 v7 冒充相同数据；新增数据应使用新配置/清单并分别验证训练、验证和测试隔离。当前微调包装器默认固定 v7 路径并要求正好 2,848 个训练 ID。

## 4. 四场景配置

目录：`configs/diffusion_amp/four_scene_speed_20261010/`。

| 文件 | 内容 |
| --- | --- |
| `<scene>_profile.json` | 四个资源路径/SHA、扩散引导、重规划、制动、身体地图保护和候选准入说明 |
| `<scene>_scene.yaml` | 原生场景/地图证据、支持面、出生位置、闭合路线、净空和展望距离；当前内容是 JSON 兼容 YAML |
| `<scene>_cameras.json` | 分段相机 eye/target、起始控制步和焦距 |
| `locomotion_conservative128_v3.json` | PPO/AMP 学习率、损失/奖励权重、对称和步态先验配置，schema 为 v5 |
| `gait_conservative_prior_v3.json` / `temporal_speed_prior_v1.json` | 仅训练集拟合的空间/时间步态统计及源 SHA |
| `walking_conservative_windows_v3.json` | 真实训练库中的速度窗口和初始化片段 |

profile 的 schema 为 `crowdsim.trace_pace_reliability.v1`；native scene 为 `crowdsim.native_pedestrian_scene.v1`。

| 关键项 | 含义 |
| --- | --- |
| `resources.*.path` / `.sha256` | 真实资源位置与完整文件 SHA256 |
| `replan_controls` | 两次重规划之间的 30Hz 控制步；YOHO 当前为 15 |
| `num_samples` | 每批扩散候选数量 |
| `guidance.max_speed_m_s` / `target_speed_m_s` | 规划上限/目标速度，不是实际身体速度验收结果 |
| `max_acceleration_m_s2` / `max_yaw_rate_rad_s` | 候选运动约束 |
| `map_weight` / `agent_weight` | 地图与同伴引导代价 |
| `origins_xy` / `routes_xy` | 场景世界 XY 米坐标；当前测量接触路径只准入 N1/N4 |
| `clearance_m` / `route_comfort_clearance_m` | 可行及偏好净空，与实际身体包络共同核验 |
| `preflight` / `native_map` / `probe_script_snapshot` | 必须成套迁移的外部证据及 SHA |

## 5. 自定义权重与场景

统一 `preflight` / `simulate` CLI 使用固定四场景目录，不接受自定义 `--config`。创建 `configs/local/` 中的新配置后，使用底层 `run_trace_pace.py` / `run_trace_pace_verified.py` 的参数。

例如将新权重绑定到候选 profile（先保证示例权重存在）：

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

更新路径与真实 SHA 只证明文件身份，不证明物理质量。不要复制旧 `tested_execution` 当作新候选的成功凭证。同时复制对应 scene 为 `configs/local/my_YOHO_scene.yaml`，逐项迁移外部证据路径，并重新核对 USD 引用、地图坐标、支持面、净空、出生与路线。

对新的 collected 场景，可先创建静态 reference 副本；输入需 Z-up、米单位及有效 default prim。示例 `scene.usd` 需替换成实际入口文件：

```bash
"$NAV_PY" scripts/prepare_collected_pedestrian_scene.py \
  --source /absolute/path/to/Collected_scene/scene.usd \
  --output "$NAV_ROOT/output/scenes/my_scene_static.usda" \
  --default-physics-owner
```

这一步没有生成 PhysX 查询地图。当前四场景的查询工具、有限支持面及日志来自 source-bound 外部归档；跨主机运行还需携带这些归档，或重新执行相应 probe 流程并更新整个证据链。仓库目前没有“一条命令自动转换任意 USD 为已验收场景”的入口。

```bash
"$NAV_PY" scripts/run_trace_pace.py \
  --config configs/local/my_YOHO_profile.json \
  --scene configs/local/my_YOHO_scene.yaml \
  --steps 1800 --measure-contacts --seed 7 --preflight-only \
  --output output/custom_preflight/new_run
```

底层资源预检只核对模型/清单身份；统一 `social_navigation preflight` 会额外核对 native scene。自定义场景需执行同等原生地图/场景准入，不能以底层预检退出 0 声称场景已经通过物理验证。

## 6. VS Code 和配置维护

打开根 `Socially-aware_Indoor_Navigation_Benchmark.code-workspace`，选择 `.venv/bin/python`。`.vscode/settings.json` 提供 src 导入路径、排除数据/产物目录；`launch.json` 和 `tasks.json` 提供状态、预检、基线和原生入口。终端变量不等于被调试程序的全部配置；调试仍需相同的资源路径和 SDK 环境。

更新 manifest/lock 时同步更新 `profile.json` 的哈希并运行环境检查；更新被上游映射清单记录的文件时维护对应 Git blob 记录，保留原合并值的追溯字段。生成结果、授权数据、凭据和训练权重保持在源码发布清单之外。

下一步：[训练、场景评估与录像](usage.zh-CN.md)。
