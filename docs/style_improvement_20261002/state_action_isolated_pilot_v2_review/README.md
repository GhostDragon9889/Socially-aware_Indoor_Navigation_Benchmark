# 实际隔离 state-action AMP：4→8 更新恢复审计

2026-10-02。**995 项 CPU 检查全部通过**，Ruff 通过。两个检查点及第8更新的 result/metrics 已独占封存。此报告只确认实际训练更新、恢复 payload 与隔离源码一致性，不声明收敛、导航效果或逐位 PhysX 恢复。

|项目|第4更新|恢复后的第8更新|
|---|---:|---:|
|累计 transitions|32768|65536|
|累计控制步|128|256|
|AMASS / procedural actor-transitions|16384 / 16384|32768 / 32768|
|PPO Adam step|12|35|
|D Adam step|8|16|
|AMP RMS count|49152|98304|
|replay size / capacity|32768 / 65536|65536 / 65536|
|本次执行段 fall|1|0|

两段合计观察到1次 fall，不能把恢复段的0误报为全部8更新0 fall。2019维 D 使用 history+action，未迁入公开 state-only D 或 AMP RMS；公开 actor/critic及self/task/value统计迁移独立记录。

## 恢复与来源

- 安全 `weights_only=True, mmap=True, map_location=CPU` 加载。模型、两个 Adam、四组 RMS、replay、runner state、Python/NumPy/Torch CPU及三个课程/语料 RNG 逐项核验。
- 真实v7、256 actors、192米地形/48米边界的完整 curriculum phase/native151/canonical51恢复通过；恢复后实际CPU subset reset→expert→plan 再恢复，返回张量和三个 RNG 字节精确相同。CUDA RNG字节保留，不在CPU运行。
- 隔离包926个普通文件全SHA验证、旧146登记源码逐项一致、只读权限与资源symlink/stat确认；1.15GB bank完整SHA没有重复重算。
- **历史实际GPU完整import map缺失**：`pilot4_v2_GPU_imports.json` / `resume4to8_v2_GPU_imports.json` 未生成。CPU59模块探测不能替代它们。V2 guard源码/命令已审查，仍不虚构运行时逐模块来源。后续 fresh v3 同步ledger属于新的修复证据。

## 不可变工件

- step4：`ee09977ebcc44ede2774ca95b36b7306a38aa387d20be7b1d45715a5782bf7d7`，388392861 bytes。
- step8：`f0c46bc07a7b50d6ba9e0d9d8fea7d1c7c8a5440a4520f237f7ed183286611ab`，653029149 bytes。
- [全部995检查与source/RNG/SHA](audit.json)、[实际CPU日志](audit.log)、[Ruff](ruff.log)、[可复跑工具](../audit_isolated_state_action_pilot_v2.py)。训练目录的两个 seal JSON 保留复制前后 inode/size/mtime和独占创建记录；未改 `last.pt` / result / metrics。

历史工具自身的字段查找、报告字典引用与lint修正失败日志保留，最终995项均通过。没有加载sealed34输出、启动Kit/GPU或修改业务/模型。未序列化求解器/接触缓存，因此不声称逐位物理轨迹恢复等价。
