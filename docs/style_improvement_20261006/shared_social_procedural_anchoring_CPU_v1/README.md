# 程序轨迹重锚定：实际CPU对照与独立候选

**原命令／奖励对照和新episode参考模块已实际CPU退出0；训练与物理改善尚未验证。**

## 方法与边界

NavIsaacLab [IV-B／Eq4](https://arxiv.org/html/2606.26265v1#S4.SS2)使用同时间的根部XY误差，
轨迹项为`exp(-2*delta)`。公共[PACER任务](https://github.com/nv-tlabs/pacer/blob/main/pacer/env/tasks/humanoid_traj.py)
在episode重置时生成路径，随后按`progress_buf*dt`查询奖励目标。
本地共享World的程序分支正常每15控制步（.5秒）从当前物理根部生成新的smooth路径；
AMASS半数分支另有固定源绑定，不能混为程序轨迹规则。

本批采用固定2848 train bank每条动作的最早合法重置帧9：真实根部位置／XYZW姿态保留。
每条恰好一次、等权；这不是实际随机重置曝光、独立验证或论文作者2848清单。
声明两种反事实根部运动：原地静止、按该帧真实AMASS初始XY速度匀速移动；Z不动、姿态不变。
没有物理积分、策略或完整人体运动。连续150个POST控制步覆盖5秒，不触发跌倒／整组重置。

三种目标机制均调用原51节点插值和原奖励实现：

1. 原程序生成器每15步重生成，后续使用其原私有RNG抽新路径。
2. 重复同一条首路径的前.5秒，并从每段当前根部重新起步。这是隔离时钟／锚定的反事实。
3. 保留同一条首路径，按episode时钟走完5秒。

首15个POST目标逐值相同。机制2与3不再生成新路径；机制1额外生成九次，
完整私有RNG和10份实际路径均保存，不能称后续随机路径仍完全配对。
共2×3×150×2848＝2,563,200个轨迹奖励行；只比较轨迹项，不推断AMP／能量／总回报或实际策略排名。

## 实际全量结果

| 声明根部运动 | 原周期重生成 | 同首路径周期重启 | 同首路径episode时钟 |
| --- | ---: | ---: | ---: |
| 原地静止：平均轨迹奖励 | .964443 | .964326 | .234614 |
| 初始XY速度匀速：平均轨迹奖励 | .967266 | .967149 | .244269 |
| 原地静止：平均根部目标距离，m | .106182 | .106566 | 2.777718 |
| 初始XY速度匀速：平均距离，m | .101175 | .101440 | 2.553004 |

同首路径对照表明，周期重启会保留较高的静止轨迹奖励；原生成器的实际CPU结果与之接近。
这是下一次独立训练消融的依据，尚未证明共享策略失败因果，也不能据此声称新行人更会走路。

[绑定实际S2的结果](actual_bound/result.json)确认其已完成2000更新／16.384M转换及真实训练／CPU审计退出0，
程序配置与本批逐字段一致。7项实际输入与10项直接实现来源前后保持；S3的23项来源guard保持。
初版`actual/`与当时的`initial_source_snapshots/`保留；最终绑定版本为`actual_bound/`。
两版全量NPZ数值逐值相同，差别仅是补齐实际S2配置及退出绑定。

## episode参考候选

[EpisodeProceduralReference](../../../CrowdSim/humanoid/training/episode_trajectories.py)
只在显式新episode重置时用原smooth生成器生成路径；后续查询没有当前根部参数，不消费RNG。
支持按actor部分重置、严格episode ID、时钟／范围拒绝、独立路径与私有RNG状态恢复。
默认10秒episode生成15秒参考，以支持末控制步完整5秒未来；这是明确的本地课程选择，
不是公共PACER／作者路径分布的精确复制。延长路径会改变后续重置的私有RNG曝光。

[实际2848检查](episode_reference_candidate.json)／[逐行目标](episode_reference_candidate.npz)：
首51节点及前150个POST目标与上述原首路径／episode对照逐值相同，455,680个目标查询完成；
真实根部字段保持，query不抽样，状态恢复逐值一致，CUDA／SDK未初始化。
这里的状态恢复仅是参考模块；完整SDK／课程／AMP历史恢复未证明。
**尚未接入共享World或开启训练。** 后续需独立World／入口和原奖励配方对照，
与future81候选分开登记，S3完成后再执行SDK检查、训练和完整动作／风格／物理评价。

## 复现与验证

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_procedural_anchoring.py --output-directory /tmp/procedural_anchoring_new
```

[工具记录](tool_execution.json)保留真实进程退出：初版13881／60974f、绑定版本15431／96780f、
候选68823／967857均退出0。49项检查（本批20、原轨迹／奖励29）及Ruff通过。
解析检查覆盖POST重规划边界、原smooth起步加速度、持久目标推进、部分重置与查询越界／状态恢复。
运行环境Python3.12.13／PyTorch2.11.0+cu128／NumPy2.3.1，CPU四线程。
bank加载遵循[PyTorch2.11](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)的CPU／weights_only／mmap接口。

[首轮封存核验失败](initial_seal_failure.json)保留：系统NumPy1.26.4与实际运行的2.3.1在两项float64均值归约中
相差最多5.551115e-17，严格相等核验因此拒绝。改用原SDK的NumPy2.3.1及原归约实现核验；
原始数组／数值保持，没有放宽容差或改动质量标准。

当前默认权重、原奖励、固定2848清单和在训S3配方均保持；本文不发布论文效果达标结论。
