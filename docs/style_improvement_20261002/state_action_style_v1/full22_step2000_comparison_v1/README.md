# State-action2000：真实 full22 固定风格比较

完整 GPU 捕获已结束：validation22、11,982 metric frames，17/22成功、0跌倒；CPU风格及独立矩阵审计通过。全部失败帧保留，只排除原每段初始reset帧；未读取34、未改feature/kernel/bins/阈值/reference。

| model | budget(M transitions) | success/22 | fall | position(m) | rotation(rad) | inclusive MMD² | speed W1(m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| backward_penalty1000 | 8.192 | 11 | 0 | 0.300708 | 0.861454 | 0.04682106 | 0.143767 |
| mixed1000 | 8.192 | 19 | 0 | 0.203485 | 0.875661 | 0.05496061 | 0.056390 |
| mixed2000 | 16.384 | 19 | 0 | 0.200096 | 0.768363 | 0.06044060 | 0.045834 |
| state_action2000 | 16.384 | 17 | 0 | 0.208382 | 0.725853 | 0.06252975 | 0.058332 |

- 旧12个模型的 evaluation/comparison 字典逐值不变；c274参考字节不变，sigma1/min30/max256原式。本次13模型JSON见 [comparison.json](comparison.json)、[evaluation.json](evaluation.json)。
- 相对同预算 mixed2000，state_action 全球旋转误差较小（.76836→.72585），但成功19→17、位置.20010→.20838、MMD².06044→.06253、speed W1.04583→.05833，不能称总体风格/跟踪改善。
- 根部yaw平均绝对误差.51422→.44764rad；去各自heading后的全身方向误差反而.50703→.53674，parent-local关节误差.25640→.26649。全球旋转下降与yaw下降同时出现，但这些诊断不是可相加的因果份额。
- 左/右Wrist parent-local方向误差改善：.49041/.45870→.37208/.32512rad；heading-local Wrist位置误差退步：.21095/.20386→.23462/.24304m。局部手腕角度改善不等于整臂姿态或整体运动风格均改善。
- 与 mixed2000 相比没有新增成功，丢失 motion20/261；五个失败序列仍完整纳入11,982帧。
- 现有近地Toe原点运动proxy：支持代理速度mean .66473m/s、超过.1比例26.0835%；没有full22原生ground/friction/contact-point测量，不能把它叫真实slip。

## 来源与独立复核

本次evaluation实际同步ledger观察69个owned executions/72个events，来自隔离v3/old146的926源码文件；`runtime_finalization_observed=false`保留，未补造SDK退出后的事件。entry/argv/checkpoint与真正terminal2000 admission绑定；不是当前工作树154/native分支或训练86-import ledger的替代证明。

Checkpoint SHA `eeac037798d2244e0cd4f0a81b40f51020c2aa3320106c7f10620e8c2ddde526`；真实fresh2000预算16.384M。D2019 history+effective action独立初始化，public state-only D/RMS未转移；expert action是next-frame normalized PD position proxy。因此与state-only warm-D是input+初始化/moments-transfer变体，不是仅action-input的单变量因果试验，也不是作者原权重/2848原ID/八场景复现。

独立 [NumPy paired审计](../full22_paired_tracking_review_v1/audit.json) 重新从全部raw frame用SO(3) matrix trace/skew atan2计算：四模型每个11,982帧，最大与原primary quaternion误差差异3.07e−13rad；三个baseline共66对clip的全部reference/clock及初始pose/velocity/AMP history逐元素相同。local/root/heading/wrist诊断与冻结decomposition一致，原primary从未替换或重新对齐。

这是反复使用的单seed validation，不是blind test/显著性；true futureXY跟踪不等于closed-loop Diffusion导航。新模型200s导航需独立完整记录才能判断稳定性。本报告不推广替换0-fall导航baseline。
