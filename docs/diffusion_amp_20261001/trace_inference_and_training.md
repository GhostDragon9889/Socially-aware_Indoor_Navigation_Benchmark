# 轨迹 Diffusion 实现与实际训练

日期：2026-10-01。实现目录：`CrowdSim/humanoid/diffusion/`。当前 Python 3.12、PyTorch `2.11.0+cu128`、torchvision `0.26.0+cu128`，真实 CUDA 设备为 RTX 4080 SUPER。

## 1. 两条明确区分的实现

| 实现 | 输入/输出状态 | 权重来源 | 用途 |
|---|---|---|---|
| `DiffusionPlanner` | `[x,y,speed,yaw,a,yawrate]`，网络输出干净动作，unicycle 重建状态 | [官方 TRACE](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c) `orca_mixed/iter40000.ckpt` 的完整 EMA | TRACE 参考推理与兼容性验证 |
| `PaperDiffusionPlanner` | 直接 `[x,y,u,v,theta]`，网络直接输出五维干净状态 | 从随机初始化实际训练；不装载 TRACE 权重 | 论文五维状态约束的独立方法实现 |

TRACE 原来的速度标量与 unicycle 动力学不能完整表示行人侧向移动，因此没有将其重命名为论文五维模型。五维版本对位置、二维速度共同执行世界/局部坐标转换，没有调用 SFM 或用规则轨迹替换扩散网络输出。

作者没有提供专属源码、完整网络超参数、定制 diffusion/AMP 权重及 2,848 条动作清单。这份实现明确固定所选配置，不能据此声称与作者 checkpoint、训练数据或论文数字完全一致。

## 2. 固定输入输出与坐标约定

`PaperDiffusionPlanner.prepare_context(states, history, neighbors, raster, history_mask=..., neighbor_mask=...)`：

| 字段 | 形状 | 语义 |
|---|---|---|
| states | `[B,5]` | 当前世界坐标 `x,y,vx,vy,yaw` |
| history | `[B,31,5]` | 30 帧过去 + 当前，最旧→最新，间隔 0.1 秒 |
| history_mask | `[B,31]` bool | 明确历史有效性；当前帧必须有效并与 states 一致 |
| neighbors | `[B,N,31,5]` | 其他行人的同一时间网格世界状态 |
| neighbor_mask | `[B,N,31]` bool | 缺失帧掩码；没有邻居时补一个全无效槽位，max pooling 输出零 |
| raster | `[B,2,224,224]` | 局部 `walkable, obstacles`，值域 `[0,1]`，不是历史轨迹通道 |
| training future | `[B,52,5]` | 世界坐标未来，0.1、0.2、…、5.2 秒 |

`world_to_agent_state` 用当前朝向旋转 XY 和速度向量；yaw 使用相对角并 wrap 到 `[-pi,pi)`。masked 历史允许 NaN，但进入网络前清零；可用位置、速度、地图等出现 nonfinite 时明确失败。

地图转换来自固定 TRACE config/原 trajdata：12 px/m，`col=12*x+56`，`row=12*y+112`，局部 +y 对应增加 row。原 ORCA `orca_dataset.py` 明确层名为 `[walkable,obstacles]`。`rasterize_navigation_map` 兼容四维/五维状态，将世界坐标映射交给现有 `NavigationTask.world_to_pixel`，保留地图 origin yaw、floor 边界；地图域外默认为两层 0.5。

`plan` 返回 `PaperDiffusionPlan`：

- `xy[B,51,2]`、`yaw[B,51]`、`velocity[B,51,2]` 为世界坐标，包括当前 t=0 与未来至 t=5.0 秒。
- `vel[B,51]` 是二维速度模长，兼容需要标量的调用方。
- `time[51]` 为 `0..5.0` 的 0.1 秒网格。
- `sample_states[B,S,52,5]` 保留完整网络局部物理状态；`selected_sample[B]` 明确选中的样本。
- 没有完成训练或加载训练 artifact 的五维模型拒绝运行 plan，防止把随机初始化输出当成可用权重。

## 3. 网络、损失和采样

五维版本的可复核配置：history/state+mask MLP，邻居独立 MLP 后 masked max pool，ResNet18 + grid decoder，Temporal U-Net base dim32、倍数 `[2,4,8]`、五维直接输出。历史特征128、条件特征256、地图 grid 特征32；100 步 cosine DDPM，输出 x0。这些具体宽度/ResNet 选择属于依据 TRACE 的实现假设；论文作者相应完整超参数未公开。

训练采用：

`z0 = (local_state - train_mean) / train_std`

`zt = sqrt(alpha_bar_t)*z0 + sqrt(1-alpha_bar_t)*epsilon`

网络预测 `z0`；前四维使用归一化 MSE，heading 残差恢复物理角度后 wrap，再除 heading std 做 MSE。均匀采样 t。训练 map/neighbor conditioning dropout 均为 0.1；ego 历史保持可用。validation 使用 eval 模式和关闭 dropout，不更新 BatchNorm 统计量。

DDPM posterior 使用标准 clean/noisy 系数和 posterior variance；100 步实际网络调用。可选定时 waypoint 使用归一化位置软梯度：每个 trajectory 先平均 K/XY，再对 B/S 求和，使指导强度不随 planning chunk 大小变化；每步使用 posterior-variance scaling。这是明确实现的指导选择，不宣称其与作者未给出的目标指导超参数完全相同。

显式 seed 使用临时/局部 RNG，实现重现并保留周边 RNG；seed=None 正常推进 RNG，重复规划不会因回滚状态重复同一噪声。

## 4. 已完成的真实训练

命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../IsaacLab-Protomotions/.venv/bin/python -u scripts/train_paper_diffusion.py \
  --steps 1000 --batch-size 32 --device cuda \
  --normalization-batches 32 --eval-batches 8 \
  --eval-interval 100 --checkpoint-interval 250
```

实际 1,000 次 Adam 更新，lr `2e-4`，grad clip1，耗时 **80.49 秒**，没有 nonfinite loss/gradient。数据来源为父代理生成的 `data/diffusion_amp/amass_manifest.json`：train319/可用diffusion228 clips、validation22/20、test34/15；源文件 hash 与 clip-level disjoint 划分在 manifest 和运行配置中保存。

训练数据假设明确：AMASS 单人 root trajectory；无同步邻居观测，所以 neighbors 为空；无场景占据地图，所以 raster 固定 walkable=1/obstacles=0。该训练证明五维 locomotion 预测链路实际可训练，尚不证明有拥挤行人/地图避障能力。

| 指标 | 数值 |
|---|---:|
| 初始固定验证损失 | 1.5976815224 |
| 1,000 步后固定验证损失 | 0.1774934027 |
| 独立测试损失 | 0.3889817446 |
| 最后训练批 loss | 0.0608994402 |

训练归一化只来自 train 的 32×32 个窗口，包含 53,248 个未来状态；angle mean 使用 circular mean，std floor0.25：

```text
mean = [0.42052856, 0.38731366, -0.05089650, 0.08227959, 0.82448280]
std  = [0.89337498, 1.28139687,  0.60243964, 0.55017805, 1.70393336]
```

## 5. 权重、恢复与验证证据

主 artifact：`output/diffusion_amp/diffusion_training/last.ckpt`；固定 1,000 步快照：`step_1000.ckpt`。大小 **279,065,717 B**，SHA256 `4b6f41d0f55146bd12376db1fac0806a4f1b69ae4a12bdb1293e8baada576d7f`。

单文件 artifact 包含 schema/config/normalization、384 项 model state、optimizer、step、dataset provenance、CPU/CUDA/Python/NumPy RNG、corpus sampler RNG。保存使用现有 atomic_torch_save；恢复严格校验 schema/config。CPU 恢复不初始化 CUDA；同 CUDA 恢复检查 RNG topology。

实际恢复该 checkpoint 后，在独立 `resume_cpu_verification/` 做了一次 CPU Adam 更新，step1000→1001，耗时4.03秒，主 artifact 未覆盖。CPU regression 还验证显式 RNG恢复、optimizer恢复和恢复后相同seed轨迹一致。

真实 GPU forward：

- 五维已训练模型：2 条真实 held-out AMASS 窗口，完整100步，0.596秒，所有输出有限。该两样本无引导预测 ADE5s=1.330m、FDE5s=1.740m，样本不足以代表泛化指标，误差仍说明需要更多训练/评估。
- TRACE reference：全部406/406 EMA 状态严格加载，完整100步，0.376秒，输出有限；CPU 同样实际运行完整采样和 goal guidance。

焦点 CPU 回归：原8项全部 PASS，另 batch-size-independent waypoint gradient 回归1项 PASS。Ruff Python3.12 gate PASS。没有运行 Kit，没有改第三方固定源码或环境依赖。

证据：`trace_training_verification.json`、`trace_gpu_forward_verification.json`、`trace_focused_tests.xml`、`trace_port_sources.json`、`trace_implementation_files.json`，以及训练输出中的 `run_config.json`、`training_result.json`。

## 6. 仍需完成的复现范围

1. 交互/障碍条件训练数据：当前训练没有同步邻居或真实地图标签，不能据 loss下降宣称实现社会避障能力。
2. 更长训练与足够 held-out trajectory/scene 评估；1,000 updates 是已完成的初始训练阶段，不是已证明收敛。
3. 与 SMPL navigation AMP 的真实闭环训练、goal/effort/heading reward 与 discriminator 训练由其他模块承担；当前报告没有给 AMP 稳定性结论。
4. 论文八场景、每场200秒、3–15行人和准确作者动作子集/训练设置对齐。数据和作者权重缺失的情况下必须保留 reconstructed 标识。

## 7. 来源与许可

固定源码为 [NVIDIA TRACE](https://github.com/nv-tlabs/trace/tree/594b2af0824c5853e1a1884e167cd5ed097dd81c)，实际仓库 LICENSE 为 NVIDIA Source Code License-NC。对应文件保存在实现目录 `LICENSE.trace`，Diffuser 的原 MIT notice 保存在 `LICENSE.diffuser`，来源和改动在 `NOTICE.md`/`trace_port_sources.json` 逐项记录。未修改主项目 LICENSE。TRACE 权重的 README 标识为 CC-BY-NC-SA-4.0。
