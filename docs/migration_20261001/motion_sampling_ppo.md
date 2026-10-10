# 动作采样与 PPO 单样本修复

日期：2026-10-01。未修改环境依赖，未启动 Isaac Sim。本轮使用固定 Python 3.12 环境执行真实 MotionManager 和真实 CPU PPO；动作库仅用确定的长度/权重 fixture。

## 动作采样

旧代码的短动作过滤只在初始化且配置了额外截断量时生效。更新/恢复权重、直接修改权重、显式/固定 ID、确定性子集均可绕过过滤。现在统一以 `clip_length >= env_dt + sample_time_truncate_s` 建立持久资格掩码，额外截断量为空时仍保留一个环境步。初始化、权重更新、checkpoint 恢复和随机抽样均应用该掩码。

显式、固定和确定性子集 ID 必须有合法范围与可用 reset 时间窗；无效请求在改写 `motion_ids/motion_times` 前失败。`exclude_motion_ids` 仍只排除概率抽样，因此用户仍可显式选择长度合法的 excluded 动作。所有随机候选失效时提供说明性的 ValueError，零样本抽取仍为空操作。公共 `sample_time()` 的过度截断 clamp 契约保持原样。

另修复两条已复现的入口：环境数大于动作数时，重复 ID 的显式 subset 列表仍按每个环境分配；Mimic 的部分 done 重采样同时筛选显式 IDs，保持其与原请求 env_ids 对应。

源码位置：

- `protomotions/envs/motion_manager/motion_manager.py:104`、`:136`、`:344`、`:460`、`:494`。
- `protomotions/envs/motion_manager/mimic_motion_manager.py:80`。
- 回归：`protomotions/tests/test_motion_manager.py:394` 起。

## PPO

两份实现的普通 update 与 KL update 均使用 `returns.var()` 计算每批 value loss 尺度。单样本尾批的无偏方差产生 NaN；普通分支传播到参数，KL 的 NaN guard 则静默跳过尾批。现改为整个 rollout 的总体方差 `buffer.returns.var(unbiased=False) + 1e-8`，整个更新过程采用同一 value loss 尺度，避免尾批放大或失效。

源码位置：`CrowdSim/ppo/ppo_policy.py:842,962` 和 `src/crowdsim_ppo/ppo_policy.py:842,962`。修复后两个副本仍字节相同。`tests/test_ppo_singleton.py` 覆盖两个副本、普通/KL 更新和 1/3 样本 rollout，均用真实网络、真实 buffer 和 Adam；逐步检查有限梯度，最终检查参数、Adam 状态、统计量，以及每个 epoch 的真实 optimizer step 数，确保没有静默丢弃尾批。

## 验证

- 新动作回归 17 项：修改生产代码前 15 失败、2 通过；修复后动作测试全文件 **39 通过**。
- 新 PPO 回归 8 项：修复前 **8 失败**（非有限梯度或 optimizer step 缺失）；修复后 **8 通过**。
- 联合验证：**47 passed，1.58s**；六个修改/新增 Python 文件 Ruff 检查通过；三个 tracked 动作文件 `git diff --check` 通过。
- 命令如下。显式限定 PYTHONPATH 并禁用 pytest 自动插件加载，防止登录 shell 的 ROS 插件污染专用测试环境。

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -m pytest -q \
  tests/test_ppo_singleton.py protomotions/tests/test_motion_manager.py --tb=short
```

边界：这些测试没有运行 Kit、行人物理或长期机器人训练。PPO 的全 rollout 方差会改变原先各 minibatch 的 critic loss 尺度，恢复训练时应关注 critic loss 和学习曲线；单样本或常量 return 的总体方差为零时仍由原有 epsilon 提供有限分母。
