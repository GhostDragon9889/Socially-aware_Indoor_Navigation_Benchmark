# Episode候选：终态来源准入与独立评价入口

**来源接线、15项CPU检查及实际S2拒绝检查已通过；新候选尚未训练或评价。**

## 方法与范围

[NavIsaacLab IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)分别评价完整动作跟踪、动作风格和导航可靠性，
并用相同AMP控制器比较Diffusion与SFM轨迹生成。公共[PACER](https://github.com/nv-tlabs/pacer/blob/main/pacer/env/tasks/humanoid_traj.py)
在episode重置时生成参考，按episode进度查询奖励目标；这是本地固定参考候选的对照来源。
本地15秒参考、AMASS31重置先验和本地风格参数未证明与作者配方等同。

[此前课程／World入口](../shared_social_episode_reference_runtime_preparation_v1/README.md)
完成真实2848数据接口和CPU接线。当前补齐该候选的终态评价来源准入：

- 保留原实际训练退出、终态CPU审计、Pop-Art格式及只读终态权重条件。
- 匹配候选专属训练入口／World／课程来源、实际导入来源及训练前后完整来源清单。
- 实际reference配置须与已封存的真实2848接口报告一致；AMASS31、固定256人和2000更新保持。
- 完整64000控制步、16384000转换、两种来源各8192000转换和episode路径／视图计数分别核查。
- 保存的resolved configuration及其来源记录须匹配实际训练终态。此步骤不加载权重张量；后续原CPU actor检查负责数值加载。

[新入口](../../../scripts/run_shared_social_episode_evaluation.py)只改调用方私有模块中的准入函数，
随后调用原post-training代码。原子进程路径、命令、评分和完整失败帧保持；
新增来源在每个child前后与原数值来源一起核对，共22项guard。
实际评价包含CPU actor加载、完整validation22、冻结45D／90D风格、N4 Diffusion短检查／200秒及CPU原始数组审计。
相同新AMP的SFM配对、原生多地图／N3–15／多seed仍需后续独立执行，不能把本入口当作完整论文协议。

## 实际CPU证据

[工具退出](tool_execution.json)：最终15项pytest及11个subtest通过，Ruff退出0。
来源预检实际退出0且Torch／SDK未导入。用真实已完成的S2退出凭据执行负向检查：
原退出／CPU审计准入通过后，新候选终态选择被实际拒绝，预期退出1；没有构建NN或启动SDK。
没有新候选权重，测试中的接受分支只是元数据夹具，不是训练／评价成功证据。

首轮实际运行发现重复加载问题：来源预检留下私有helper别名，第二次来源检查先被原严格loader拒绝，
因此该次退出不能算作S2候选准入检查。现在在获取冷来源清单后仅释放自己加载的helper别名，
同进程两次完整准入通过，原loader拒绝已有模块的条件保持。初始源码／登记及失败均保留。

## 使用

```bash
python3 scripts/run_shared_social_episode_evaluation.py --check-sources
```

实际训练及CPU审计／封存完成后，再用该候选的真实退出凭据执行`--preflight-only`。
正常评价输出固定为`output/evaluation/shared_social_episode_reference_pilot2000_v1`，
要求S3训练和后续评价链的完整外部完成证书，拒绝已有输出目录。
当前没有新增SDK运行、动作质量测量或达到／超越论文的结论。
