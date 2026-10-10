# 正式 Diffusion 验证回升：源码核对

本轮只读源码与原始资料；未导入 Torch、构造网络、读取数据或 checkpoint，也未改变正在运行的 40k 协议。固定 40k 终态仍是主结果；历史 checkpoint 的完整 validation 仅作另标诊断，不按终态 Test 选优。

## 三项有证据的风险

### 1. 有效 batch400 与 BatchNorm batch200 不等价

正式入口登记为 micro200×accum2。原 `optimizer_update` 每次执行两次训练前向、一次 Adam；地图 ResNet 与 decoder 含 BatchNorm，时序卷积使用 GroupNorm。默认 BN momentum=.1 时，两次前向使旧 running mean 的权重为 .9²=.81；一次 batch400 前向为 .9，且合并方差也不同。BN 原论文明确按 minibatch 计算统计量，PyTorch 明确使用 running estimates 做 eval。[BN 原论文](https://proceedings.mlr.press/v37/ioffe15.html)，[PyTorch BatchNorm2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm2d.html)。

本地 whole-state EMA 还平均这些 running buffers。此实现与固定 TRACE 的全 `state_dict` EMA、训练前更新时序相符，**不能称为 EMA 实现错误**。[TRACE EMA](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace_helpers.py#L600)，[更新时序](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/algos/algos.py#L149)，[公开 batch400](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json#L20)。

另核读了 OpenAI 的 Improved Diffusion 官方 `main` 源：其网络使用 GroupNorm32，EMA 对参数列表更新；这是另一种架构/训练选择，不能据此改写 TRACE 的全-state EMA。[官方 nn.py](https://raw.githubusercontent.com/openai/improved-diffusion/main/improved_diffusion/nn.py)，[官方 train_util.py](https://raw.githubusercontent.com/openai/improved-diffusion/main/improved_diffusion/train_util.py)。这两份是本轮成功访问的移动 `main`，未获得固定 commit 身份，不冒称固定来源。

先使用已准备完整 validation 的 raw/EMA 分域结果。它们的差异同时包含参数与 buffers，不能单独归因于 BN。若实际结果确有必要，后续最小探索诊断是固定同一份参数、仅互换 raw/EMA 的 BN running_mean/running_var，在共同窗口/t/noise 上报告 loss 差值；不重校准、不训练、不改部署或主结果。本轮未实现或执行该探索。

### 2. 五状态 clean-state MSE 与实际轨迹质量并非同一指标

本地损失对 normalized `[x,y,u,v,yaw]` 的 52 帧直接取均方，yaw 使用环绕误差；平面坐标误差在原单位下受 `1/state_std²` 加权。`state_mse` 合并四个位置/速度分量。它未通过动力学将预测速度与位置耦合；固定 TRACE 的 `state_and_action` 则通过 unicycle forward dynamics 构造状态，再计算 clean-state loss。[TRACE 动力学转换](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace.py#L390)，[损失](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace.py#L785)。Nav III-A 明确使用五状态，因此这属于独立适配边界，不能据此判定本地公式写错。

首先看现有 fullval 的 state/yaw parts，并结合既有 paired validation 的 ADE/FDE/净空结果；不得从 MSE 一项推断接触或动作风格。若四分量合并后仍无法解释，后续只增加同一残差的 XY/V 分项与扩散 timestep 分组即可，不改目标函数。原始 DDPM 与 Improved DDPM 使用不同损失/参数化；其 timestep importance sampling 结果不能直接作为本地 clean-state 目标重加权的理由。[Improved DDPM §2.2、§3.3/Eq.17](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)。

### 3. 小 monitor、逐场景 paired 与全窗口 validation 的分母不同

正式 `eval_batches=4`、micro200：每域固定 seed 抽 800 次有放回窗口，不保证 800 个 unique；每次评估重用该子集与随机过程。完整 validation 为 Maps10566 + Interact19386 =29952 个唯一窗口。paired 的主组每个 episode 抽一窗，属于逐场景口径；fullval 则按窗口计数。训练 union 的 Maps/Interact 窗口比例为 81396/151650，即约34.93%/65.07%，没有固定50:50。公开 TRACE 将两域交给 combined loader，也未要求窗口50:50。[TRACE 数据源配置](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json#L35)。

现有 fullval 已覆盖 unique 全集、共同逐窗口 t/noise、尾批加权、域分项和 raw/EMA，不需要重复建设。每窗口仍只有一个固定 t/noise 实现，应如实称固定 realization 的完整窗口诊断，而非精确噪声期望；也不能仅凭小 monitor 回升断言全域过拟合或选择历史最优。

## 本轮实际核读源码

- [正式入口](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/orca_official_all_temporal_formal_preparation_v1/train_formal_all_temporal.py:151)：17101 B，SHA256 `fc0b8f1a3e35a5c9e52aa7b49cdd59017213512bceb04d88ae3ddc368226c7b4`；实际预算/评估参数来自同目录冻结 `commands_and_audit_plan.json`。
- [EMA/optimizer_update](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/orca_official_mixed_training_staging_v2/training_state.py:24)：10425 B，`8dafe55397180d88f22f720d0ef0dfd270f9473ec528685bd2a86aba04913dd9`。
- [实际五状态模型](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/orca_interact_usable_staging_v1/paper_interact.py:487)：49029 B，`42abd2c7757fc3e1feac8a1b0645c86a2d831744f4f3de8567c0c62ecf0c18b3`。
- [已准备完整验证诊断](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261004/full_validation_loss_diagnostic_v1/evaluate_full_validation.py:219)：18655 B，`e82d4ae34f7dbea23c5372e0cbf6b9e71d41d9ead5d13ab1b38bdd208987a28a`。
- 固定 TRACE 来源与字节身份已在 [原来源记录](/project/EXP/SangforAndSaut/ProtoMotions/docs/style_improvement_20261003/trace_training_budget_source_review_v1/source_records.json) 核实；本轮读对应本地 `third_party/trace` 源码。未据链接推断私有 Nav 配置或作者效果。
