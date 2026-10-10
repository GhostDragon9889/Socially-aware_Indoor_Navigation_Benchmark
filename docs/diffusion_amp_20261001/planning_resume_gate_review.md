# 规划阶段恢复门禁：独立修复复核

状态：通过。针对当前 runner 源码 AST 执行 41 个 CPU 分支验证；没有修改业务源码、checkpoint 或环境，没有启动 Kit/GPU。

## 已修复的真实缺口

此前 `runner_state=None` 或空字典会绕过 `if saved` 的恢复／阶段合同，继续使用旧 trainer 计数。当前在加载后首先强制 training resume 为非空字典。两种原始绕过输入，以及 list/string 错误类型，现在均明确拒绝，且拒绝前没有物理恢复调用。

## 合法新阶段与状态保留

本次复核安全读取 root 自产的 update612 `resume.pt`（`weights_only=True`、CPU mmap、不加 allowlist）。真实 runner 包没有 `control_step`，以 1,253,376 transitions / 64 envs 得到 **19,584**；plan age 实际范围 **1…15**。

原样执行源码中的 `per_actor_age → synchronized` 显式转换与恢复分支：stage 写明 update612、source checkpoint SHA256、相同 interval/episode/planner batch；physical 六项、AMP/history episode、planar history、history mask、age/episode/reference ID、plans/plan ages/goals 和 corpus generator state，共 **17 项**逐 Tensor 相等。CPU sink 记录先 physical reset、恢复上述状态，再调用 `trainer.restore_rng()`；没有调用 new_goals 清空训练计划／历史。

默认 legacy resume 保留 per-actor clock；同步 stage 后续 resume 恢复持久 counter；fresh evaluation 的 counter 为0。`snapshot()` 源码保存 `control_step`。

## 失效分支

拒绝隐式 clock 切换、冗余 new-stage、未知 base-contract 键、interval／episode／planner batch／env count 改变、transition 不整除、布尔／浮点／负数／不一致 counter、错误 shape/dtype 或超范围 plan ages、seed／dataset三项 hash/split／diffusion改变，以及物理 clock、gain、material、mass、DOF 合同改变。

评估允许显式 shared world／新 scene geometry，与此同时保持训练 controller 的物理时钟、gain、material、mass、body/DOF/limits合同；所抽测的 controller 合同变化均拒绝。

## 序列化核查与边界

该 resume 文件 static unsupported globals 为 **空**，`runtime_execution.torch` 的精确类型是内置 `str`；递归检查未发现其他非 plain metadata 类型（Tensor 和标准容器允许）。root 的原始文件保留及120 Tensor全等迁移证据见 [runtime_restart_synchronized.json](runtime_restart_synchronized.json)。

物理 reset 与 RNG restore 在本复核中使用 CPU sinks：证实源码顺序／实参保留，不证明 PhysX contact/solver caches 可序列化，也没有执行 CUDA RNG恢复。同步阶段改变抽样分组，可能提前刷新 reset 后计划，不能声称 bit-identical。最大 .5s 为本次15步／30Hz合同；其他CLI参数应按实际 interval解释。

精确 source/hash、全部输入分支与逐 Tensor等值结果见 [planning_resume_gate_review.json](planning_resume_gate_review.json)。
