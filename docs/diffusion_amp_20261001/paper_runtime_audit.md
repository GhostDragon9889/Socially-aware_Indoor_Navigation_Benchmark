# 实际AMP训练入口只读审查

检查父代理的`scripts/train_paper_amp.py`、`amp_world.py`、AMASS corpus与本代理训练组件。源码SHA、逐项状态、当时的真实metrics快照见 [JSON](paper_runtime_audit.json)。未改父代理业务文件，未启动Kit/GPU，未降低失败门槛。此报告是训练进行中的快照。

## 已核实的合同

- 真实PhysX rollout进入PPO/D的两个Adam updater，raw69 Gaussian作likelihood，实际动作按clamp和冻结12维mask执行，D看到有效动作。
- Reward采用实际下一步时间对齐XY与实际力矩/角速度；原式反走奖励与负功奖励通过分量日志显式呈现。
- next_value在reset前读取；terminal不bootstrap，time limit bootstrap但截断GAE carry。
- AMP历史episode IDs真实随历史移位，reference reset历史填同一新episode。
- Actor/critic/D、Adam、各RMS、RNG、逻辑物理state、历史、planner目标/age与corpus RNG保存；模型构造后延迟恢复RNG。
- Resume核对数据/源/分割/planner SHA、身体/DOF/限位/PD/material与episode/replan/env数，拒绝训练optimizer CLI覆盖。

PhysX接触/solver cache没有序列化，因此完整逻辑状态保存不能宣称物理bitwise restart。29项CPU测试验证实际optimizer更新、下一次更新相同、延迟RNG、物理动作冻结合同等；真实采集是父代理正在执行的训练，而不是CPU夹具。

## 已向父代理报告的条件项

1. 当前默认30Hz/5s配置一致；自定义profile中的`control_dt`或`target_horizon_s`若变化，runner目前仍固定30Hz/5s，应显式拒绝。
2. 当前相同资产训练一致；live canonical masses已记录但resume比较未包含它们，资产质量变更需要额外身份gate。
3. 当前64环境初始位置在有限地形内；若增大num_env或长距离向负方向行走，应验证实际碰撞mesh边界，height query的clamp不能代表无限地板。该项是代码边界风险，不把观测到的跌倒误归因于越界。

这些建议没有擅自修改正在跑的训练。最终稳定性与收敛需heldout实际物理评价；训练跌倒reset必须保留计数。目前数据/地图/邻居条件不足以声称论文八场景的人群避障指标已复现。
