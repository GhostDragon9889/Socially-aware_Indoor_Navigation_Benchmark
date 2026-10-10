# 同一 AMP 的新鲜 SFM／Diffusion 配对：初态接线已准备，SDK 待执行

新入口 [`evaluate_shared_social_high_level.py`](../../../scripts/evaluate_shared_social_high_level.py) 显式选择 `--planner diffusion` 或 `--planner sfm`。两臂复用相同冻结 Social AMP、四人物理 fixture、控制循环、因果参考、原始捕获和退出清理。依据[论文 IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)的同 AMP 高层消融要求，补上规划器构造前的初态记录；不将旧 DP 闭环自动当作新 SFM 的匹配基线。

这是本地 N4 接线准备。作者八场景名单／任务／SFM 参数及完整指标分母仍未恢复；本入口不支持 native N3–15，也没有执行新分支 SDK smoke 或 200 秒物理评价。S1 已知质量不合格，选它开展后续诊断并非模型晋升。

本轮重新查阅[arXiv 当前摘要及提交历史](https://arxiv.org/abs/2606.26265)，页面仅列 2026-06-24 的 v1；[公开仓库 README](https://github.com/yh83305/NavIsaaclab2.0)继续介绍机器人导航 PPO 入口。GitHub commits 页面／API 本次读取失败，未据此声明已核对 main 最新 SHA，也未把 README 描述当作作者 AMP／Diffusion 私有资源已发布的证明。实现继续绑定已核验的具体公开源码。

## 代码与记录边界

- [paired_initial.py](../../../CrowdSim/humanoid/analysis/paired_initial.py) 仅在实际 actor admission／reseed 后、规划器构造前记录一次初态。现有 parent 的 Social 派生保持；另加两个可逆 capture 替换与三个可逆 entry hook。原 helper AST、world step/reset、奖励、AMP、参考、路线、接触计数和清理均未修改。
- COMMON 通道包含全部 24 身体的位置、XYZW、线／角速度，以及 69 关节位置、速度、力；另保留 AMP 历史、episode／reference ID、planar history 和布尔历史 mask。不能用首次动作输入替代完整身体初态。
- Native link／COM／质量通道来自现有 `FootCaptureAdapter._state_snapshot()`；继续使用其名字映射和 COM 组合核验。COMMON 与 native 同帧 link 通道必须一致。没有另建 view、调用 contact force sampling 或推进物理。
- 同时记录相同 scene／actor 顺序、全部路线、物理配置、actor checkpoint／RMS／动作合同、评价参数、因果参考、运行数值设置、数据 provenance、manifest／scene／profile SHA 和正式 Diffusion 的 used／unused 配对身份。仅输出目录从配对参数中排除。
- [Python 3.12 getstate](https://docs.python.org/3.12/library/random.html#random.getstate)、[NumPy 2.3 get_state](https://numpy.org/doc/2.3/reference/random/generated/numpy.random.RandomState.get_state.html)、[Torch 2.11 CPU state](https://docs.pytorch.org/docs/2.11/random.html)及 AMASS 私有 generator 均使用 state getter，无 reseed／restore／draw。CUDA 已初始化时才调用 [get_rng_state_all](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.get_rng_state_all.html)；CPU fixture 不初始化 CUDA。
- 读取两次的初态必须保持，读取及发布前后的随机状态／Lab submitted-step counter 必须保持。这个 counter 不是 PhysX native simulation time。报告不保留 world／native owner，数组均为脱离原存储的 CPU 副本。
- [compare_shared_social_initial_conditions.py](../../../scripts/compare_shared_social_initial_conditions.py) 核验 archive SHA 及每个通道的 shape、dtype、payload bytes，然后要求全部合同与通道精确一致。NPZ 容器本身的压缩字节不用于两臂相等判定。空／部分记录、浮点 ID、错 mask、未知通道和不同私有 RNG 都不接纳。

初态匹配只说明记录的起点／协议匹配。两臂未来随机消耗、重置次数和物理轨迹可以不同；它不证明完整运行正常退出或表现合格。实际退出回执、所有失败帧、原生子步／接触／参考数组及完整 200 秒曝光仍是独立要求。CPU mock 不能作为 native 状态来源证明；pure metadata 工厂不能认证 live world。

## 已实际执行的检查

| 检查 | 结果 | 实际 tool chunk／exit |
| --- | --- | --- |
| 最终 CPU 契约 | 7 passed；detached 通道、时钟／native 一致、无 RNG 消耗、缺失／篡改拒绝、两臂资源归一与源派生 | `390dac`／0 |
| 最终四个文件 Ruff | passed | `28bc49`／0 |
| 最终 DP source | 完整 inverse、既有 helper AST 不变；Torch=false、SDK=false | `d4a34c`／0 |
| 最终 SFM source | 保留原 SFM 九项替换，再加共同 capture；Torch=false、SDK=false | `01c17d`／0 |
| 最终真实 S1 DP 资源预检查 | 原终态、formal DP40000、v7 清单通过；没有启动 SDK／加载模型 tensor／创建输出目录 | `b4d57c`／0 |
| 最终真实 S1 SFM 资源预检查 | 同一资源集合与 30 controls／120 actor-controls 计划通过；未执行物理 | `615739`／0 |
| S2 来源保护检查 | 22 项全部匹配；新工作未修改训练 owner 的来源 | `310731`／0 |

第一次 6 项 CPU 测试和两臂 source 检查已通过，第一次 Ruff 因测试中的三个分号退出 1（`252f5c`），已保留日志与确切源码。随后补上关节力、资源 SHA、pure metadata 边界与第七项测试；所有版本、tool exit 和最终精确源码见 `execution.json`。这里没有新增训练，也未读取 test34 动作数组。

## 已登记的下一物理工作

先等待 S2 owner 的训练、自动评价与 CPU 完成回执，独占 GPU 流程结束后才启动新物理任务。后续固定 S1（`29b590…`）、seed7、正式 DP40000（`4581f29…`）和相同 N4 fixture，开展新鲜两臂诊断；不依据对照结果换 AMP checkpoint。SFM 继续使用[公开方程及本地已声明的五秒常目标适配](../shared_social_sfm_integration_v1/README.md)。

依次执行两臂 30 controls smoke；观察真实进程退出 0、所有原生 owner 释放，并核对 snapshot、raw51／reference151 和完整 native captures。每条命令都需新的输出目录。下面两条 **当前未执行**：

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_high_level.py --planner diffusion \
  --execution-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json \
  --diffusion-checkpoint output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt \
  --diffusion-sha256 4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff \
  --control-steps 30 --output output/diffusion_amp/shared_social_S1_paired_DP_smoke30_v1

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_high_level.py --planner sfm \
  --execution-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json \
  --diffusion-checkpoint output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt \
  --diffusion-sha256 4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff \
  --control-steps 30 --output output/diffusion_amp/shared_social_S1_paired_SFM_smoke30_v1
```

两臂 smoke 的实际成功与捕获核验完成后，才以 `--control-steps 6000` 各自创建新的 200 秒输出。初态接纳命令为：

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/compare_shared_social_initial_conditions.py \
  --diffusion-initial output/diffusion_amp/shared_social_S1_paired_DP_smoke30_v1/paired_initial_condition.json \
  --sfm-initial output/diffusion_amp/shared_social_S1_paired_SFM_smoke30_v1/paired_initial_condition.json \
  --output output/diffusion_amp/shared_social_S1_paired_smoke30_initial_admission_v1.json
```

此命令也未执行。初态文件名不是实际成功回执；完整 raw 审计入口仍要求由外部工具观察到的 navigation200s 退出回执。任何源码／时钟／native 字节／RNG 漂移或预测失败均保留失败和部分数据，不回退，不删帧，不宣称达到论文效果。
