# 新 ORCA 地图／邻居条件数据与实际微调（2026-10-01）

## 结论

已补充**真实 RVO2 专家**的同步多行人、障碍地图条件数据，并完成独立五状态 Diffusion 的 **1,000 次实际 GPU 微调**。使用新输出目录，原 AMASS checkpoint 与正在执行的 AMP 实验保持冻结。

这补齐了“AMASS 只有单人轨迹、没有真实地图或邻居条件”的训练缺口。**未达到稳定避障，也不是作者未公开的 NavIsaacLab 数据／权重复现**。32 个独立保留窗口中，最终障碍物碰撞点比例为 16.19%，必须如实保留这一结果。

## 公开来源与 Python 3.12 兼容性

| 来源 | 固定版本 | 本轮实际使用 |
|---|---|---|
| [TRACE ORCA 生成器](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim) | `594b2af0824c5853e1a1884e167cd5ed097dd81c` | 官方障碍形状／尺寸／位置采样、行人属性范围、偏好速度扰动与 2–4 秒重采样 |
| [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2/tree/c2c46ba8d59556aa10faf03479293236efea154d) | `c2c46ba8d59556aa10faf03479293236efea154d`，与 TRACE gitlink 一致 | 真实 C++ ORCA 线性规划与 `PyRVOSimulator.doStep()` |
| [TRACE ORCA dataset parser](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset_specific/orca_peds/orca_dataset.py) | 同上 | 0.1 秒状态间隔、`atan2(vy,vx)` 方向代理、90m 双层地图坐标约定 |

在 `data/tools/orca_generator/.venv` 独立 CPython **3.12.13** 环境中，以 Cython **3.1.3**、宿主 CMake **3.28.3**、GNU C++ **13.3.0** 编译固定 RVO2 源码，构建并安装 `pyrvo2-0.0.0-cp312-cp312-linux_x86_64.whl`。实际四人迎面避碰／障碍物停止与多 episode 测试通过；没有改 vendor tracked source，也没有向 IsaacLab/Isaac Sim 环境安装依赖。

没有直接使用旧 `gen_dataset(cfg)` 库入口：它读取全局 `args`；原记录循环第一次保存 t=1/60 秒、末段间隔不统一。新 wrapper 使用官方采样器和真实求解器，**先记录 t=0，每六个 60Hz 物理子步保存一次**。不运行旧 trajdata 全依赖栈；其旧 `np.bool` 地图缓存路径不是本轮依赖。

TRACE 为 **NVIDIA Source Code License-NC**，RVO2 为 **Apache-2.0**。完整许可证随新数据置于 `LICENSES/`；TRACE 衍生生成约定和网络不能被重新标为主项目 MIT。环境包版本、wheel SHA256、源文件 SHA256 保存于 [数据验证 JSON](trace_orca_data_verification.json)。

## 新数据与明确的重建假设

入口：`scripts/generate_orca_diffusion_data.py`。生成／加载模块：`CrowdSim/humanoid/orca_corpus.py`。数据目录：`data/diffusion_amp/orca_social_pilot_v1/`。

Manifest SHA256：`c4ae294295d7bf2649f84ea64d50f9bc9e79d32d3174ae07624b4d211896462c`。

| 划分 | 独立障碍布局 | Episode | 条件窗口 |
|---|---:|---:|---:|
| train | 16 | 32 | 26,597 |
| validation | 4 | 8 | 6,560 |
| test | 4 | 8 | 11,118 |
| 合计 | 24 | 48 | 44,275 |

每 episode 20 秒、201 个同步 10Hz 帧、4–15 个半径 0.4m 的真实 ORCA 圆盘行人；实际专家速度上限 2m/s。每场景 1–5 个非空障碍，官方 ellipse/box/triangle，尺度 3–10m、初始位置范围 ±15m；没有添加虚假的边界墙。原全局 90m 栅格覆盖所有本轮轨迹（最大绝对位置 27.80m）。磁盘数据文件共 **1,533,232 bytes**（不含独立工具环境）。

状态为 `[world_x,world_y,world_vx,world_vy,yaw]`。位置与速度直接记录真实 RVO2 求解器；yaw 按官方 parser 使用 `atan2(vy,vx)`，静止初始帧为 0。**圆盘没有身体朝向，此 yaw 不是 SMPL 姿态真值**。数据也没有身体关节、动作或力矩，不能替代 AMP 的 AMASS 身体参考。

每个训练窗口有 31 个历史／当前帧（过去 3 秒）和 52 个未来帧（+0.1…+5.2 秒）；运行时规划取前 50 帧并含当前状态，输出 5 秒局部路径。窗口不会跨 episode。场景 ID、障碍布局 SHA、episode ID 与种子全局唯一，同一场景所有 episode 均属于同一划分，防止邻近窗口穿越训练／保留边界。

邻居按当前 15m 内真实距离选取、排除 ego、稳定排序，返回同步历史与真实 agent ID；最多 14 个槽位，未用槽位为零且 mask=False。筛选窗口至少有一个真实邻居，且局部 crop 与障碍相交面积 ≥0.25m²。地图真实栅格为 `[walkable,obstacles]`，224×224、12px/m，局部 `col=12*x+56,row=12*y+112`，按 ego yaw 旋转，正 y 增加行，域外填 .5。

三划分各抽查 512 个窗口：所有状态有限、每窗口至少一个真实邻居且均在 15m 内；最少实际障碍像素分别 70／59／71。padding 没有冒充邻居。无 unsafe pickle/object array；NPZ 用 `allow_pickle=False`，加载时校验源 SHA。

## 专家 QA

对每个真实 **60Hz** 子步，而非只对 10Hz 保存帧，检查圆盘间距和中心到真实多边形的有符号距离。48 段已接纳 episode 共 **57,600** 个物理子步；最低 pair surface gap **−0.0784mm**，最低 obstacle surface gap **−0.00912mm**，**没有超过 0.1mm 容差的穿透**。不能将浮点近接说成严格零穿透。

两段失败 episode 的种子、失败间距和原因保留在 manifest 中，按确定性重试种子重新生成，没有丢弃失败证据。它们最小 pair gap 为 −0.187mm／−0.297mm。完整统计见 [trace_orca_data_verification.json](trace_orca_data_verification.json)。

## 实际条件训练与恢复

入口 `scripts/train_orca_diffusion.py` 以独立 AMASS `step_1000.ckpt` 的五状态模型和 normalization 初始化；**新 Adam 优化器、RNG 与社交训练阶段计数**。后续 resume 恢复新阶段的 optimizer、Torch／CUDA RNG 与每划分 numpy sampler RNG，并核对阶段／manifest SHA；保护冻结 AMASS 输出目录。

原 AMASS SHA：`4b6f41d0f55146bd12376db1fac0806a4f1b69ae4a12bdb1293e8baada576d7f`。开始／结束再次校验一致。

先执行 2 次真实更新容量探测，再从保存的社交 checkpoint 恢复 998 次更新，阶段共 **1,000** 次。batch=8，Adam lr=1e−4，PyTorch CUDA allocator 上限为显存的 .15。两次运行的最大 reserved 仅 **775,946,240B**，低于约 2.4GB 硬限；没有启动 Kit。998 步运行含采样评估／保存用时 113.56 秒。

首次真实更新有 64 个有效邻居槽位、42,585 个实际障碍像素；clip 后 map encoder gradient norm **0.17964**、neighbor encoder **0.10583**。该次关闭 condition dropout 取得明确梯度证据，之后按模型配置进行条件 dropout。checkpoint 保存该证据。

| 指标 | 社交 phase2 初始模型 | 社交 phase1000 |
|---|---:|---:|
| validation normalized x0 MSE（4×8 保留窗口） | 1.04514 | 0.09420 |
| test normalized x0 MSE（4×8 窗口） | — | 0.15961 |
| 32 固定 test 窗口 ADE | 1.16291m | 1.06326m |
| 同窗口 FDE（5秒） | 2.68316m | 2.47138m |
| 邻居碰撞点比例 | 15.56% | 7.00% |
| 障碍物碰撞点比例 | 15.50% | 16.19% |
| 至少一次邻居碰撞的轨迹比例 | 37.50% | 28.13% |
| 至少一次障碍物碰撞的轨迹比例 | 28.13% | 37.50% |

这里是 100 步 DDPM、无 waypoint guidance 的**开环 ego 路径**对全部真实保留邻居未来轨迹和固定多边形的时间对齐比较，绝非多行人联动仿真或闭环避障结果。初始比较来自实际已更新两步的新阶段，不冒称未经微调的完整 AMASS baseline。训练 loss 改善没有转化为充分的障碍避让。

新固定权重：`output/diffusion_amp/social_diffusion_training/step_1000.ckpt`；动态原子保存：同目录 `last.ckpt`。固定 SHA256 **`69d4336d002c84fa848a93a65dd622e5222a663fac63f61b3e94c6872b8c9d15`**。训练结果在同目录 `training_result_step_1000.json`。这是本地新生成 ORCA 数据上实际训练的独立模型，不是作者 AMP／Diffusion checkpoint。

安全加载、所有模型／优化器有限性、CPU 恢复、实际条件参数变化与保留输入消融记录于 [trace_orca_checkpoint_verification.json](trace_orca_checkpoint_verification.json)。

该 CPU 核查安全加载了 384 个 model state 项／1,247 个模型、优化器与 RNG Tensor，均有限；恢复模型、Adam、RNG 没有初始化 CUDA。8 个固定 test 窗口在同一 t=80／noise 下，full loss=0.06215、移除邻居=0.06018、map 改为 unknown=0.06769。条件梯度与参数更新是真实的，但这个小样本中的邻居消融结果也说明**邻居条件尚不能被当作提升预测质量的充分证据**。

## 验证与运行

8 个实际 CPU 回归通过（2.13s）：RVO2 四人迎面避碰／障碍停止、确定性均匀时间与速度 heading、旋转地图、真实邻居／未来帧身份对齐、场景泄漏／源 SHA 篡改拒绝、真实专家 QA 与人工穿透检出、冻结目录／显存限制保护、五状态 training batch 互操作。测试不以 stub 代替 ORCA；没有扩展的环境会明确 skip 相应 optional tests。JUnit：[trace_orca_tests.xml](trace_orca_tests.xml)。四个新增源码文件 Ruff PASS。

```bash
# 独立生成环境；固定 RVO2 source build wheel 已在工作区，不改 SDK env
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
data/tools/orca_generator/.venv/bin/python scripts/generate_orca_diffusion_data.py \
  --output data/diffusion_amp/orca_social_NEW_RUN

# 新阶段训练（默认输出已有本轮权重，重新开始必须选新目录）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
../IsaacLab-Protomotions/.venv/bin/python scripts/train_orca_diffusion.py \
  --output-dir output/diffusion_amp/social_diffusion_NEW_RUN \
  --steps 1000 --batch-size 8 --cuda-memory-fraction .15

# 继续同一来源/同一阶段；保存所有优化器和 RNG，保持旧 AMASS 固定
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
../IsaacLab-Protomotions/.venv/bin/python scripts/train_orca_diffusion.py \
  --resume output/diffusion_amp/social_diffusion_training/last.ckpt \
  --steps 1000 --batch-size 8 --cuda-memory-fraction .15
```

## 后续边界

pilot 只有 24 个障碍布局，validation/test 各仅 4 个；各 episode 的邻近窗口相关，密度分布也未做完全分层平衡。应扩大**新路径**中的训练布局／种子并保留独立测试布局，再按保留避障指标选择更多训练与采样目标，不能只追求 loss。当前不改 root 正在进行的静态 SHA AMP 实验。

此版本已经具备真实 map/neighbor 条件输入、训练与来源绑定，但剩余问题是开环障碍碰撞率、长期闭环与 SMPL 身体控制质量、作者未公布的数据/参数。后续使用新权重必须作为独立实验，不能追溯修改已有 AMP 训练 provenance。
