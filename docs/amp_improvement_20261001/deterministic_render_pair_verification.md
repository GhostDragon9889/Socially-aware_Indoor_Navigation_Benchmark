# 确定性物理与渲染对照：30秒实际运行

核查日期：2026-10-02。本报告仅使用CPU读取完成后的JSON、NPZ与源码归档；没有启动Kit、CUDA或模型，没有修改代码、权重或历史运行工件。

## 结论

两次运行均完成 **900个30Hz控制帧、4名行人、30秒**。全部已记录物理数组和规划数组逐字节一致，包括首批及最后一批规划；全部450次截图诊断的已观测状态、物理步数和Torch RNG保持不变。

这是本次固定种子和配置下的实际记录一致性证据。**不能推广为PhysX/Warp内部状态或任意机器、种子、时长的逐位一致保证，也不能据此确定此前非确定性渲染分歧的原因。**

## 核查对象与设置

| 项目 | 物理运行 | 渲染运行 |
|---|---|---|
| 目录 | `output/diffusion_amp/shared_independent_amp_deterministic_phys_v2` | `output/diffusion_amp/shared_independent_amp_deterministic_render_v2` |
| 状态 | passed | passed |
| 实际控制帧 | 900 | 900 |
| 渲染 | 启动时禁用 | 启用，450张PNG及MP4 |
| 截图间隔参数 | 15，截图关闭 | 2，每秒15张 |
| 总墙钟时间 | 116.6303秒 | 204.4250秒 |

命令参数仅有`output`、`physics_only`、`capture`和`capture_every_steps`不同。其余参数一致，包括种子7、4人、100步DDPM、FP32、单样本、未编译前向，以及同一控制器和Diffusion权重。

实际记录的Torch版本为`2.11.0+cu128`。两次均启用严格确定性算法，`warn_only=false`、cuDNN `benchmark=false`、`deterministic=true`；矩阵精度`highest`，CUDA矩阵TF32关闭，cuDNN TF32开启，cuBLAS workspace为`:4096:8`。运行报告均明确`physx_warp_bitwise_guarantee=false`。

## 完整数组与样本选择

| 工件 | 数组数 | 完整记录比较 |
|---|---:|---|
| `physical_roots.npz` | 1 | 数组字节一致，最大绝对差0 |
| `physical_states.npz` | 26 | 数组字节一致，最大绝对差0 |
| `reference_plans.npz` | 9 | 数组字节一致，最大绝对差0 |

所有数组的形状、dtype及内容均核查；浮点数组全部有限。JSON报告保留每个数组独立的SHA256，区分NPZ文件字节与解压后的数组字节。

首批4条`raw_xy`和连续参考轨迹均相同，起点分别为`[6,6]`、`[6,18]`、`[18,6]`和`[18,18]`。完整248条原始规划、适配轨迹和参考速度均一致；首帧和末帧物理位置、姿态、关节位置/速度/力，以及全部中间帧均一致。

248条记录来自62次联合规划，每批4名行人各出现一次。同批`selected_sample`一致，全部为0，符合单样本范围。这里没有多样本选择效果的比较。

## 450次截图前后诊断

实际文件为`frame_000000.png`至`frame_000898.png`，间隔2控制帧，共450张，均存在。

- `root_pos`、`root_rot`、`root_vel`、`dof_pos`、`dof_vel`、`rigid_body_pos`六组已观测缓冲区：每次截图的最大绝对差均为0。
- 所有物理步数前后差均为0；计数与四个120Hz子步对应，始终等于`4 × control_step`。首张为4→4，末张为3596→3596。
- CPU和当前CUDA设备RNG均在450次诊断中被观测，前后相等且SHA256一致；没有完整RNG状态写出。
- **原生PhysX时钟均未观测到**：`observed=false`，`before/after/delta=null`。SDK分发版本`6.0.0.1`的公开绑定没有可调用的`get_simulation_time`。没有用代理时钟替代或将缺失值解释为0。

状态来自`world.state()`的公共RobotState读取，并使用独立`detach().clone()`防止共享视图掩盖变化。这些getter可能使用时间戳缓存，因此诊断仅支持已观测缓冲区、控制侧物理步数及RNG的结论，不暴露所有原生求解器状态。

## 行为质量仍有边界

| 实际指标，两次相同 | 数值 |
|---|---:|
| 跌倒/时间上限终止 | 0 / 0 |
| 各行人真实到达顶点数 | `[0,1,0,1]`，总2 |
| 完整路线周期 | `[0,0,0,0]` |
| 静态接触行人帧 | 315 / 3600 = **8.75%** |
| 无序行人对接触帧 | 27 / 5400 = **0.50%** |

行人对接触矩阵对称：54个有向真值对应27个无序接触对帧，不能把有向真值和无序分母混用。两次真实到达分别为行人3在23.5667秒、行人1在28.9333秒，距当前顶点0.7367米及0.7448米，均位于0.75米到达半径内，与`result.json`的计数一致。

四个末尾episode均因采集结束而右删失，没有记录为时间上限成功。0次跌倒不等于无碰撞、路线完成或长期稳定。本次30秒结果也不能替代此前200秒闭环评估。

## 来源、权重及历史证据

两次运行由父任务在同一冻结driver下执行：

- `scripts/train_paper_amp.py` SHA256：`f0550ad9eb006bfb373ff507024a380de6e8533c4ec05de521a8f3752ef70b57`。
- 独立AMP step1000：`e0d6ed4a9acb39c581bed92fab722a426bf23c34a3949bc48712cc13bf04c333`。
- Social Diffusion step5000：`63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`。

本次CPU核查重新计算了两份权重、配置、AMASS清单、场景配置、两组result与NPZ文件的SHA256，并保存在JSON报告中，没有加载或执行checkpoint。

当前源码归档[final_runtime_source.json](final_runtime_source.json)及`final_runtime_source.tar.gz`包含75个文件：归档大小223599字节，SHA256为`7969f0e2320f7ba84bac4b4b212c1e5b3cd9ee88a9faa26aeb3a5da991227e6c`。归档整体及75个成员的大小/哈希均与清单一致，当前工作区对应文件也匹配。这是**运行后的源码归档刷新**：父任务还移除了`scene.py`中未使用的`ISAAC_NUCLEUS_DIR`导入。不能据此假定归档中每个当前文件字节都被此前两个进程使用；driver、确定性配置与测试未改变，运行参数和工件对照单独核查。

此前非确定性的S1/S2渲染跌倒记录和首次确定性渲染因原生时钟接口缺失而失败的v1均保留。此次对照没有覆盖、撤回或重新标记这些结果。

[完整数组哈希、450次诊断汇总与来源证据](deterministic_render_pair_verification.json)；[确定性开关与CPU回归](deterministic_evaluation_change.md)；[历史S1分歧](render_s1_divergence.md)。
