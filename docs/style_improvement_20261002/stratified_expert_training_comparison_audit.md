# 分层专家采样训练：独立 CPU 对照审计

日期：2026-10-02。只读检查已完成训练、冻结源码归档和真实 full22 捕获；不启动 CUDA/Kit，不修改训练、评估指标或数据。完整检查见 [审计 JSON](stratified_expert_training_comparison_audit.json)，可重复工具见 [audit_stratified_tracking_comparison.py](audit_stratified_tracking_comparison.py)。

## 配置与实验边界

两轮 `training_config` 的共同字段逐值相同，唯一新增有效字段为 `expert_sampling`：低速10%、快速5%、横移10%、短转25%、其他移动50%；先按短窗类别，再按 dataset 的父源支持数平方根、父源和合法中心采样。它是明确改变专家 prior 的实验，不是保持原分布的无偏方差缩减。

以下记录完全相同：公开 PACER no-shape 初始化权重与完整暖启动声明、v7 manifest/bank SHA、child-joint 专家速度坐标、模型/优化器/奖励设置、seed7、独立 smooth curriculum、物理时钟与 measured projected power、训练 budget。两轮都使用256环境、32步 rollout、1000更新，实际 **8,192,000 transitions / 32,000 controls**；平地192 m、spawn margin48 m、episode300 controls=10 s，程序命令速度0–2 m/s、加速度1.5 m/s²，未使用 Diffusion 生成训练命令。

CLI 差异仅 config/output 路径及后来新增的禁用足部观测默认值 `measure_ground_support=false` / threshold1 N。归档共同93个文件中86个 SHA相同、7个不同；新增15个文件。因此不能宣称整棵源码逐字相同。核心 trainer、actor网络、奖励、curriculum、AMASSCorpus、AMP观测均逐字相同；`amp_world.py` 的差异仅安装、调用及保存/恢复新的专家 sampler。新增 foot/contact 接口在这两轮独立平地训练中未启用。

首个 rollout 的51项共同记录逐值相同，包括奖励、位置命令、PPO loss/KL/梯度、跌倒及命令计数；首个 D 更新随专家采样变化。**后续训练并不是逐步完全配对**：旧专家 batch 消耗同时供应 reset 的 corpus RNG，新 sampler 使用私有 RNG。相同 seed/reset规则不能保证后续 reset ID、solver/contact状态和物理轨迹逐值相同。报告保留这一限制，不作单种子统计显著性结论。

## 训练结果

| 实际指标 | negative1000 | stratified1000 |
|---|---:|---:|
| 更新 / transitions | 1000 / 8,192,000 | 1000 / 8,192,000 |
| 训练跌倒 reset | 313 | 203 |
| 每100,000 transitions 跌倒 reset | 3.8208 | 2.4780 |
| time-limit reset | 26,995 | 27,048 |
| 已结束 episode 中跌倒比例 | 1.1462% | 0.7449% |
| 更新751–1000 trajectory reward均值 | 0.985710 | 0.986230 |
| 更新751–1000 AMP reward均值 | 0.351682 | 0.371197 |
| 更新751–1000 D expert / live-policy概率 | 0.854624 / 0.157201 | 0.845056 / 0.165195 |
| 更新751–1000 backwards fraction | 3.7697% | 3.5295% |

313→203 的主要差别在更新1–250（292→185次）；最后250更新为4→5次，不能由累计下降推断末期更稳定。高 trajectory reward 与 D 分离同时存在；D的输入分布已改变，所以 AMP/D数值也不能独自充当 heldout 风格收益。

固定 step1000 完整 SHA：

- negative：`62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`
- stratified：`93c25e567d88cc4f4c639b1dea5e14494f0ef51dd304661fb23d36690e55328d`

本审计重新计算固定文件 SHA。完整 weights-only CPU 加载、优化器、normalizer、replay和私有 sampler RNG 检查见 [checkpoint审计](stratified_expert_checkpoint_review.json)。确切step200快照已被原子替换，报告明确缺失；没有重构或重标记后来的checkpoint。

## full22 输入严格配对

22个动作共11,982个真正 post-step metric帧：全部 `reference_*` 数组、全局 `trajectory_xy` 十个目标、30Hz时间/phase、初始 actual root/DOF/body pose与所有速度均逐值相同；不是仅比较动作ID。每个捕获文件 SHA 均与自己的 result一致，初始 reset frame 不进入主指标，后续跌倒帧全部保留。

benchmark、主 metric、trainer、AMP观测与底层 simulator 的捕获源码 SHA 一致；仅 `amp_world.py` 的专家接口接线 SHA 不同，并与各自预启动归档吻合。两轮 benchmark 参数只有 checkpoint路径/SHA及输出路径不同，protocol/physics/velocity/layout一致。

结果 **11/22成功、0跌倒 → 8/22成功、2跌倒**。位置误差0.300708→0.352941 m退步，旋转误差0.861454→0.791256 rad改善，不能推荐为整体更优控制器。具体分解、部位/速度 bins与下一步依据见 [跟踪分解](stratified_expert_tracking_decomposition.md)。风格/MMD由独立冻结评估负责，本审计没有重算或替换它。

## 重复执行

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_stratified_tracking_comparison.py \
  --output /tmp/stratified_comparison_audit.json
```

工具只读输入，输出必须是新文件。初次工具路径断言误写不存在的 `training/model.py`，日志保留；修正为真实 `amp/network.py` 后 [v2实际日志](stratified_expert_training_comparison_audit_v2.log) PASS，Ruff PASS。该失效属于审计工具路径，未改业务或实验产物。
