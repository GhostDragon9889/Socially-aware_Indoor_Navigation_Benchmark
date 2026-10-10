# 显式同步规划时钟：纯 CPU 调度验证

## 接口与保持的约束

新增 `CrowdSim/humanoid/planning_clock.py`：

```python
due = due_agents(plan_age, control_step, replan_steps, synchronized=False)
```

- 默认使用已有 `plan_age >= replan_steps`，行为保持原 per-actor scheduling。
- 显式 `synchronized=True`：全局 `control_step % replan_steps == 0` 时全部 actor 规划；非全局 tick 仍立即规划 `age >= interval` 的 actor。
- 新 reset 继续沿用 runner 的 sentinel：首次后续动作前 `plan_age[id] = replan_steps`，不能等到下一个全局 tick。
- 每个成功规划 actor 的 age 由 caller 置零；每次物理控制步结束 age 与绝对 counter 各加一。因此最大间隔仍为 15 控制步，即当前 30Hz 下 0.5 秒。
- 返回原 Tensor device 上排序的 int64 IDs，不修改 age，不持有隐藏时钟、RNG、模型、仿真对象。热路径仅验证 metadata，不引入 GPU→CPU value checks。

root AMP runner 本轮未由此 agent 修改。环境、权重、optimizer、normalizer、dataset、control dt、DDPM schedule、重规划间隔均不属于这个 pure module。

## CPU 回归

`tests/test_paper_planning_clock.py`：13 项通过，Ruff PASS；JUnit 保存于 `planning_clock_tests.xml`。

覆盖原规则保留、全局 tick all、off-grid spawn immediate、随机 reset 下最大间隔、持久 absolute counter 跨任意 update 边界恢复一致、旧错相 age 的显式阶段切换及输入契约。

随机调度统计（**CPU 事件模拟，不是物理 rollout／模型推理性能**）：1200 个控制步，64 actors，seed=20261001，每 actor／step reset probability=0.006，实际 450 个 reset events。

| 模式 | planner 调用数 | 总 agent plans | planner batch 中位数 | full64 batch 数 | 最大间隔 |
|---|---:|---:|---:|---:|---:|
| 原异步 | 1119 | 5364 | 4 | 1 | 15步 |
| 同步＋立即 reset | 432 | 5546 | 1 | 80 | 15步 |

call 数减少 61.4%，agent plans 增加 3.4%。同步模式中位 batch 仍为1，因为强制立即 reset 的小批存在；收益来自另外80次全局 full batch 对原有错相规划的整合，不能将其描述为所有请求都已批量化。实际吞吐提升需由 root 新运行分段计时确认。

## Resume / 新阶段审查

**这不是 bit-identical resume。** 全局 tick 会提前刷新刚 reset／刚规划过的 actor，最短间隔可能只有一控制步；上限保持原15步。它改变局部目标刷新分布、planner 抽样顺序和 RNG 消耗次数，必须作为显式新的 planning stage 记录。

counter 是动作前的绝对控制步索引，初始0；每次 `world.step()` 成功结束后仅加一次。不能使用 `for step in range(rollout_steps)` 的局部循环索引，也不能在每个 PPO update 或 planner chunk 重置。

同一同步 stage 的 checkpoint 必须保存与恢复 absolute counter、plans、plan_age、goals、world histories、optimizer／normalizers 与 RNG。旧训练 checkpoint 若没有 counter，可在相同 num_envs 与完整 rollout 的严格 gate 下，以 `physical_transition_count // num_envs` 重建，并断言整除（当前 trainer 是 `steps * envs` 全量累积）。fresh evaluation world 的 counter 应从0开始，不可按改变后的 eval num_envs 除旧训练 count。

一次性 CPU 验证 checkpoint `plan_age` 为整数、shape 与 num_envs 相符、范围0…15，counter 为非负整数；不要每控制步同步 device 值来做这些检查。stage 切换只允许明确的 clock mode 差异，保留编译 flag、N、planner batch、manifest / Diffusion SHA、模型／optimizer配置、dt、interval。不要清空 goals/plans/history 或重新初始化 optimizer、normalizer／RNG 来掩盖 stage 切换。

频繁跌倒的 off-grid 即时请求无法由这个时钟消除；global tick 可能更改 task targets 的瞬时分布。后续必须同时检查模型训练指标、跌倒率与 closed-loop 路径质量，而不仅是运行速度。本轮没有运行 GPU／Kit，也没有修改运行中的模型、权重或 root runner。
