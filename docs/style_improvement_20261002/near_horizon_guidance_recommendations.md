# 近时域指令不足：下一轮单变量实验建议

2026-10-02，只读 CPU/源码研究；未修改代码、权重或运行 GPU。依据[已冻结 guide20 停滞报告](backward_penalty_navigation_style_v1/comparison.md)：actor0 连续 idle 38.267s，原始计划前0.5s 位移均值0.0342m、5s 位移0.9933m，执行 target 误差仅0.0423m。说明远端运动没有转化为当次近段指令；**不证明 mean loss 是唯一原因**。该完整4人200s采集为0跌倒、21顶点、4 cycles。guide40 尚未完成，不能据阶段进度宣称成功。

## 1. 优先：保持总强度，明确增加近时点的目标质量

当前 `paper.py:200` 对十个 timed waypoint 的欧氏误差取 mean；guide20 每点名义系数2。每0.5s重规划却等权约束到5s，允许反复把较大位移放到尚未执行的尾端。建议新增 **opt-in 时间权重**，目标仍是同一静态 A* 路线的十个时点，不替换 Diffusion 输出：

`J_goal=20·Σ_k a_k ||p(t_k)−q_k||₂`，`a_k=(1−η)/10+η/2·1[t_k≤1s]`。

默认 `η=0` 逐值保留旧 mean。首先仅比较 `η=0` 与 `.25`：前0.5/1s各从0.1→0.2，其余各0.075，总质量仍1；`.5` 只作为后续候选。**sampling gradient 和最终 sample filtering 必须使用同一权重**；有联合避碰时仍选一个 coherent scene sample。非uniform配置没有可用近时点应明确拒绝，不静默退化。

来源：TRACE 论文的[定时 local waypoint](https://arxiv.org/html/2304.01893v1#A2.SS2)与[官方独立目标加权](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/guidance_loss.py#L900)支持这一接口基础；上述 η 调度是**工程变体，非 NavIsaacLab/TRACE 发表的默认超参**。保留当前 clean Jacobian/方差模式、100 DDPM、FP32、t0无引导/噪声、RNG、0.5s cadence 与模型权重。

**主要风险：**近点压力可引发急转/急加速，破坏自然等待或避让；不能强制最小速度。验收须同时记录前0.5s沿路线进展、raw→causal reference→真实root位移、整段idle、真实顶点/cycles、跌倒和native碰撞，不能只看5s终点误差。

## 2. 次优先：需要时单独增加短段软参考速度一致性

若时间权重仍出现“早段停住、尾端赶路”，再单独试验 `J_ref=λ·mean_{t≤0.5s}||(p_t−p_{t−1})/0.1−v_ref(t)||²`；`p_0` 为实际root，`v_ref` 从**同一静态 timed 路线及当前实际速度**构造连续、可允许零速的参考，单位m/s，不能由模型已经停住的 XY 自我生成。建议显式 `λ=0/.01/.1` 小消融，保持第1项 `η=0`，不同时打开两项后归因。该项只改变采样软损失，不后处理位置、不写物理状态；邻居/几何约束与原 causal adapter 保留。

来源：[TRACE 官方 `TargetSpeedLoss(mode='use_position')`](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/guidance_loss.py#L148)用XY差分估速。这里改为路线方向的向量、限制到执行短段，是**工程扩展**，不是其标量速度默认；仅强迫预测 u/v 头匹配XY不会自动产生路线进展，当前 runner 实际消费的是XY。原 TRACE 从动作经动力学生成状态，而当前独立5D网络直接预测状态，两者不能假定相同运动学一致性。

**主要风险：**差分含 `1/dt`，梯度容易放大、路线拐点易造成冲击；正速度参考可能压制合法让行。因此必须是可退让的软项，不设硬最低速度；先在冻结idle及非idle上下文、固定噪声检查速度/加速度和避碰，再交给同协议4人200s物理评价。λ数字尚未经新实验验证。

## 共同控制与暂不优先的选项

两项实验均固定 controller `62c4edc…`、Diffusion5000 `63e7cad…`、原guide20/peer5/geometry10、地图半径、seed及足力监测协议；完整保留baseline。闭环重规划/reset时刻可能分化，不称bit-identical。

TRACE 的 global progress 仍主要评价计划尾端，直接搬入不能保证0.5s实际执行进展；其代码还有minimum-progress参数，不能偷偷当最低速度门禁。论文[价值引导](https://arxiv.org/html/2304.01893v1#S3.SS3)有真实依据，但当前critic未以真实完整路线进展校准，可能偏好易跟踪、低能耗的停驻；本轮不将它作为前两项优化。上述建议是待验证实验，不是论文达标或效果已改善的结论。

官方源码核对采用本地固定 `third_party/trace@594b2af0824c5853e1a1884e167cd5ed097dd81c`，包括 `guidance_loss.py:148,468,542,592,668,900` 与 `models/trace.py:634`；本轮已联网阅读原论文，GitHub文本端点读取失败时没有以其他实现替代。
