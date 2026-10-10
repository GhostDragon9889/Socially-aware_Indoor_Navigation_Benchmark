# 近段 waypoint ETA=.25：完整200秒原始物理观测对照

日期：2026-10-02。本报告仅 CPU 读取真实结束的4人、6000个30Hz控制帧及四子步测量；不运行Kit/GPU，不修改业务、采集、风格评估或历史报告。完整逐actor停滞、原计划/执行reference相位、接触事件和跌倒记录见 [comparison.json](comparison.json)。

## 结论

**ETA=.25缓解actor0停滞，但本次整体导航未改善，不应替代ETA=0基线。**相同权重和fixture下，路线顶点21→20，累计四顶点cycles同为4，跌倒0→2，同行人接触0→88个unordered pair-frames。actor0路线2→6、停滞58.11%→10.92%，同时actor2/3更常停滞、actor3在同行人接触中两次跌倒。它是一次部署参数消融，不是论文超参、统计显著性或作者八场景验收。

| 完整200秒、24,000 actor-frames | ETA=0 | ETA=.25 |
|---|---:|---:|
| 实际跌倒 / time-limit reset | 0 / 0 | 2 / 0 |
| 路线顶点（actor0–3） | [2,8,6,5] | [6,8,4,2] |
| 顶点总数 / 累计cycles | 21 / 4 | 20 / 4 |
| rootXY同episode位移速度<.1 m/s | 5843/23996=24.35% | 5459/23994=22.75% |
| 实际静态接触 actor-frames | 0 | 0 |
| 实际同行人接触 actor-frames / unordered pair-frames | 0 / 0 | 176 / 88 |
| 连续同行人接触事件 | 0 | 6 |
| 最低Pelvis高度 m | 0.71302 | 0.36542 |
| 双支持 / 无足支持（120Hz actor-substeps） | 48.0156% / 0.7479% | 48.2313% / 0.8542% |
| 有支持时足部link原点切向速度均值 m/s | 0.07314 | 0.07703 |

cycles按每actor累计顶点数整除4后相加；包含跨reset的累计计数，不等于无中断episode成功率。actor3的两次终止帧、reset边界及全部低速/不直立帧均保留。

## 1. 唯一有效参数改变与来源

实际runtime argv逐字段比较只有输出目录与 `near_horizon_waypoint_mass`：旧字段缺失对应原平均权重ETA=0，新ETA=.25。planner batch8、同步重规划、允许planner部署变化、seed7、guidance总权重20、agent5/radius.6、scene10/radius.8、causal reference、四人fixture与foot/contact阈值1 N都相同。

新 `training_config` 仅多一个未启用的 `expert_sampling=None` 元数据；共同值、physics、scene、runtime数值设置和真实contact契约相同。指定AMP、Diffusion、profile、manifest、scene完整资源SHA逐项相同：

- AMP negative1000：`62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`
- Diffusion5000：`63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`

ETA=.25不是“前1秒总权重只有25%”。公式 `(1−eta)/10 + eta/2·I(t≤1s)` 给前两个waypoint各.2、其余八个各.075，前1秒共40%系数质量；有效goal权重4/4/1.5…，总和20保持不变。gradient与sample过滤采用同一组权重，明确为工程变体。

实际运行前112文件归档SHA `2b6cbfcb67376ed7de7e7da5bcbe0b443e35d0558c5ad8f59fd0aa28c3fed6cf`；所有文件相对路径、regular member、字节数及SHA均验证。完整control loop AST与旧冻结foot循环一致；configuration只有4个明确near-horizon新增节点，移除它们后whole AST与旧冻结配置相同。后续AMASS opt-in课程可能修改当前工作树，本审计未把current-tree相同作为历史来源条件。

相同seed不等于相同闭环物理或后续随机噪声路径：不同计划、路线到达与fall会触发不同重规划和reset。这里只证明已声明参数、资源和测量契约的控制条件。

## 2. 测量通道严格审计

[最终raw audit](ground_support_actual_verification_v4.md) **PASS340项**；[完整JSON](ground_support_actual_verification_v4.json) 保存原始文件/模型/归档SHA及逐项检查。

- 6000控制帧、24,000个120Hz提交子步连续，每帧恰好4子步；episode/reset/termination与原physical捕获逐值相同，未补零或丢帧。
- 末子步body origin位置、Pelvis XYZW与原状态逐值一致；原4个static/peer字段与独立ground归档的peer投影逐值相同。
- floor-only sensor严格对应L/R Ankle+Toe四个完整body路径、唯一floor filter `/World/ground/terrain/mesh`；从raw normal-force向量独立重算每足、左右载荷、>1N mask、any/all支持、有效速度mask及四子步均值。
- 全body质量为正，总质量63.313949 kg；link世界姿态与局部COM偏置独立SO(3)重构真实COM，最大误差9.875e−7 m；质量加权COM另行重算。
- `actual_native_capture_validated=true`单列成立，历史 `native_capture_probe_passed=false` / neutral-stance probe=false保持原值。测量有效不等于姿势自然或稳定。

旧工具针对较小归档的路径域与configuration全文件SHA，在205/321项拒绝新公开路径和4个参数节点；失败报告保留。v3包装器AST常量缩进错误也保留。最终工具只允许固定manifest SHA的已知112成员，并证明configuration精确增量；**没有放宽任何测量检查**，未修改原工具或历史结果。

足部原点代理受滚动/转动影响，不是真实contact-point slip；摩擦力/接触点速度及AMASS trueCOM/接触参考未知，保持null或未测量声明，不推导论文滑足合格。

## 3. 停滞与命令关系

严格idle为同episode相邻位置差分<.1m/s，不合并短暂波动，不计reset跃迁。保存的1684个candidate计划逐控制重算：raw下一步、causal执行target及reference速度与before-control plan加age+1完全对应；计数与实际planning元数据以JSON为准。`global_goal_xy`是A* lookahead端点，最终顶点另看 `route_vertex_xy`。

| actor | ETA=0 idle | ETA=.25 idle | 最长idle s：旧→新 | ETA=.25该段执行target误差 m | raw .5s / 5s 位移 m |
|---|---:|---:|---:|---:|---:|
| 0 | 58.11% | 10.92% | 38.267→7.167 | 0.06047 | 0.05516 / 2.04676 |
| 1 | 5.03% | 11.32% | 1.767→13.467 | 0.05260 | 0.05950 / 0.94458 |
| 2 | 15.10% | 37.71% | 5.233→10.600 | 0.07400 | 0.06363 / 1.25330 |
| 3 | 19.15% | 31.07% | 8.600→8.367 | 0.06766 | 0.06252 / 1.58795 |

新四个最长idle段双支持均100%，native末子步root速度均值约.0167–.0377m/s；实际最终路线顶点仍距10.11–13.69m。raw近.5秒移动仅.055–.064m，5秒移动约.94–2.05m，执行target贴近root：**远端计划移动仍不保证每.5秒闭环重规划会执行足够近段运动**。

actor0改善同时，其他actor停滞加重，不能仅看旧最严重actor成功。四段root点到最近AABB最小距离约1.24–2.61m，这是几何点距离，不是身体碰撞距离；未由此建立terrain RMS或sole摩擦因果。

## 4. 跌倒与真正同行人接触时序

同行人事件按连续control帧定义，任何一方episode变化拆段；阈值为4子步normal-force最大body-pair magnitude >1N，floor/self excluded。两个fall都发生于actor3、与peer接触同帧出现；它是明确时序重合，**不能单独证明唯一跌倒原因**。

| actor3终止 | source frame / time | Pelvis m | peer事件 / 峰值normal force | 同帧foot支持 | 执行target误差 m |
|---|---|---:|---|---|---:|
| episode1 | 459 / 15.333s | 0.36542 | actor1–3，456–459，939.80N | 四足均无floor支持 | 1.34168 |
| episode2 | 1640 / 54.700s | 0.44454 | actor0–3，1635–1640，1172.41N | 有单侧支持、无双支持 | 1.01693 |

两次下一post-step frame都reset、episode分别变2/3；±.5s均无static接触，有peer接触。首次碰撞交互还有13.833–15.133s多个中断接触段；157.8–159.1s另有actor1–3接触40帧、峰376.42N，未发生第三次fall。所有6事件、force maxima、原episode ID与原帧索引完整存入JSON。

![原始停滞、actor3高度与路线计数](temporal_comparison.png)

## 5. 冻结风格与后续

风格只由 [唯一冻结style报告](../near_horizon_eta025_style_v1/report.md) 计算，本raw审计没有运行MMD或重拟合reference。该报告亦未支持整体推广ETA=.25：inclusive速度条件姿态+速度MMD² .0723806→.0778255，旧reference字节和guide20/40记录保持一致。它与本次跌倒、拥挤接触和命令停滞共同提示：本变体改善单个actor近段命令，却没有同时解决群体交互/完整行程质量。

保留ETA=0基线与所有失败证据，下一实验需继续单变量、固定controller/数据与完整200s/多seed协议，先检查peer交互时的身体接触和执行reference偏离，不能以本次原点代理或累计cycles宣称论文效果达成。

## 复算（只写新报告）

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/navigation_near_horizon_eta025_comparison_v1/audit_eta025_ground_capture.py \
  --output /tmp/eta025_ground_audit.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/navigation_near_horizon_eta025_comparison_v1/analyze_eta025_navigation.py \
  --ground-audit /tmp/eta025_ground_audit.json --output /tmp/eta025_raw_comparison.json
```
