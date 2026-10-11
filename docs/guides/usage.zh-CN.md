# 使用、训练、评估与录像

[English](usage.en.md) · [文档目录](README.md) · 更新：2026-10-10

默认目标为 Lab3.0 EA/Sim6.1。先[配置外部 SDK 或在项目内完整安装](external_sdk.zh-CN.md)，项目直接使用该 SDK 的解释器，附加包保存在项目目录。历史结果与本轮实测范围见[核验报告](../EXTERNAL_SDK_VERIFICATION_20261011_ZH.md)。

## 1. 方法与模块

当前四场景部署使用公开 TRACE 的轨迹扩散 + PACER 的 state-only AMP 控制器及本地 AMASS 微调。轨迹进入策略观测，AMP 产生 PD 位置目标，由 PhysX 推进行人；模拟不是通过轨迹直接移动根部或插值播放关节。录像阶段则回放已测量的物理姿态，仅切换相机。

NavIsaacLab 的五状态 Diffusion / history-plus-action AMP 方法重建入口仍作为独立实验存在，不能把公开 TRACE/PACER 适配称为作者私有模型的完全复现。

| 模块 | 职责 |
| --- | --- |
| `src/social_navigation/` | 命令转发、根目录定位、额外原生场景预检 |
| `src/CrowdSim/humanoid/diffusion/` | TRACE 网络、候选采样、轨迹引导 |
| `src/CrowdSim/humanoid/amp/`、`training/` | AMP 接口、专家观测、PPO/判别器和步态微调 |
| `src/CrowdSim/humanoid/runtime/` | 生命周期、真实状态、地图/接触及关闭顺序 |
| `src/CrowdSim/humanoid/analysis/` | 姿态、速度、覆盖、接触等评估 |
| `src/protomotions/` | IsaacLab 后端、SMPL 配置、MotionLib 与上游训练框架 |
| `src/crowdsim_ppo/` | 独立 PointGoal 导航基线 |

## 2. 准备终端与查看命令

先完成[安装](installation.zh-CN.md)和[配置](configuration.zh-CN.md)。

```bash
cd /absolute/path/to/Socially-aware_Indoor_Navigation_Benchmark
export NAV_ROOT="$PWD"
export NAV_LAB_ROOT="$(python3 -c 'import json; print(json.load(open(".runtime/external_sdk.json"))["lab_root"])')"
export ISAACLAB_ROOT="$NAV_LAB_ROOT"
export NAV_PY="$(python3 -c 'import json; print(json.load(open(".runtime/external_sdk.json"))["python"])')"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NAV_ROOT/src:$NAV_ROOT:$NAV_ROOT/.runtime/external_sdk_packages:$NAV_ROOT/.runtime/rvo2_native${PYTHONPATH:+:$PYTHONPATH}"
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

所有 forwarded 子命令保留底层工具参数。原生 GPU/Kit 作业逐个运行，避免同时创建多个 Isaac 实例。每次使用新输出路径，保留失败日志。

## 3. AMP 微调

下面是已有 2026-10-10 保守配方，依赖固定 labeled v7、公开 PACER 锚定教师、真实训练步态先验和已训练的 walking_turn 父权重。128 更新、160 环境、32 步 rollout 产生 655,360 次交互；新运行的跌倒和质量须重新测量。

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

输出包含 `last.pt`、`Humanoid_AMASS.pth`、`metrics.jsonl`、`result.json`，相邻保存 `_execution.json`、`.log` 和 `_sources/`。CPU 导出审查核对参数变化、来源和训练记录，不替代独立动作风格/导航验证。

父权重缺失时可明确选择公开 PACER 权重启动新训练，但该过程不等于已验证的父模型链；不能直接给新模型标记“部署通过”。更换 manifest 或增加动作需同步修改固定语料入口和先验绑定，当前包装器不支持任意动作数。

Diffusion 研究训练入口为 `scripts/train_paper_diffusion.py`；论文式 AMP 为 `scripts/train_paper_amp.py`；共享 SocialCNN 入口为 `scripts/train_shared_social.py`。先查看其 `--help` 和专门实验文档，输入、判别器接口及权重格式不同。当前示例不重新训练 TRACE。

## 4. 60 秒物理场景测试

先完成固定资源和原生地图预检：

```bash
cd "$NAV_ROOT"
"$NAV_PY" -m social_navigation preflight --scene YOHO
```

然后运行一个新物理源：

```bash
export NAV_SCENE=YOHO
export NAV_CFG="$NAV_ROOT/configs/reproduction/four_scene"
export NAV_RUN="$NAV_ROOT/output/physics/${NAV_SCENE}_new_60s"
"$NAV_PY" -m social_navigation simulate \
  --scene "$NAV_SCENE" --steps 1800 --seed 7 --output "$NAV_RUN"
```

1800 个 30Hz 控制步对应 60 秒，接触模式每人保存 7200 个 120Hz 物理子步。CLI 物理预算为 1–6000 个控制步；视频的当前完整准入固定为 1800 步。

自定义候选不能通过固定 `simulate --scene` 覆盖 profile；改用 verified 底层启动器：

```bash
"$NAV_PY" scripts/run_trace_pace_verified.py --output "$NAV_RUN" -- \
  --config configs/local/my_YOHO_profile.json \
  --scene configs/local/my_YOHO_scene.yaml \
  --steps 1800 --measure-contacts --seed 7
```

物理源保存 `result.json`、`physical.npz`、`physical_states.npz`、`planning.json`、`ground_support.npz`、`contact_measurements.npz` 等；父进程另存实际退出码、源码/输入快照及日志。退出非零或 `status` 失败应停止并保留记录。

## 5. 逐项评估

下面脚本启用 `set -euo pipefail`。如果交互终端直接运行，非零退出可能结束当前 shell；也可存为脚本后执行。只评估刚完成的同一个物理源。

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

| 项目 | 当前本地门槛 / 解释 |
| --- | --- |
| 物理与接触 | 完整预算、零跌倒/重置和测量接触等质量条件；实际退出必须成功 |
| 姿态 | `--require-walking-screen` 执行实际全帧行走姿态筛查 |
| 速度 | 正常巡航目标 0.5–1.5 m/s；起步、转弯和避让允许减速；全帧速度仍保留 |
| 瞬时速度 | 所有 7200 子步实际 XY 速度不超过 1.5 m/s |
| 空间停留 | 连续完整 5s 根位置窗口的空间范围判断，最长停留 ≤10s |
| 脚部表面 | 实际姿态下中性 SMPL 皮肤对声明支持面，不等于测量接触斑滑移/切向摩擦 |

风格 MMD/固定聚类覆盖及冻结验证动作属于独立风格分析；通过速度/安全筛查不等于更自然或优于论文。历史验证 22 条已用于选型，不应称作新的未见测试；测试 34 条应保持独立。详细本地口径见[四场景实测说明](../FOUR_SCENE_SPEED_20261010_ZH.md)。

## 6. 相机与 60 秒录像

相机皮肤检查中的行人列表必须对应每个镜头。固定配置为：

| 场景 | 人数 | `NAV_ACTORS` |
| --- | --- | --- |
| YOHO | 4 | `[[0,2],[1],[3]]` |
| NewMetro | 4 | `[[0,1,2],[3]]` |
| SecondFloor | 1 | `[[0],[0]]` |
| ThirdFloor | 1 | `[[0],[0]]` |

YOHO 示例继续上节同一个 `NAV_RUN`：

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

录制器还执行物理/FK 一致性与相机子进程核验。所有镜头来自同一物理源连续区间，没有姿态插值、时间拉伸或拼接不同物理运行。完整模型、可视皮肤和场景材质必须存在。

完整核对视频，不只读取文件头：

```bash
export NAV_VIDEO="$NAV_ROOT/output/video/${NAV_SCENE}_new_60s/${NAV_SCENE}_60s.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_read_frames \
  -show_entries format=duration -of json "$NAV_VIDEO"
ffmpeg -v error -i "$NAV_VIDEO" -f null -
```

固定配置输出为 H.264、960×720、30FPS、1800 帧、60 秒，并有 `<scene>_video_receipt.json`。镜头皮肤投影检查不检测所有场景几何遮挡；仍需观看连续视频。

另三个场景逐个更换 `NAV_SCENE`、`NAV_RUN`、视频输出和 `NAV_ACTORS`，先预检再重复物理与所有评估。机器人 `EVT-EXP.usd` 并未进入当前示例闭环，不能把行人录像称作已验证的机器人社交导航。

## 7. 轻量开发检查与证据范围

```bash
cd "$NAV_ROOT"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$NAV_PY" -m pytest \
  --import-mode=importlib -p no:cacheprovider -q tests/test_project_layout.py
"$NAV_PY" scripts/verify_upstream_integration.py
```

2026-10-10 已有四场景视频及本地有限场景指标，但未证明作者完整八场景协议、任意拥挤场景/随机种子泛化或超越论文。源码状态、成功退出、全帧质量、录像检查和人工真实感评价是不同证据。此次文档更新只核验命令/配置/依赖与说明，没有重新训练或录制。
