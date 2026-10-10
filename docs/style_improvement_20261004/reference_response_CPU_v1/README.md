# 同控制相位的参考／身体响应关联

已由 root 实际执行，工具会话27686／完成块d834e8退出0，reader记录耗时4.217851秒。独立源码审阅 Required=[]；源码与结果已封存。它补充全 controls 的速度／方向响应关联，保留原同相位 XY 误差定义。

## 实际结果

三组均保留完整24,000 actor-controls。CSV共72,000行，root另逐行核对12条行人序列各从frame0到5999连续、control=frame+1，45个phase分组完整；所有组的有限值数量＋undefined数量等于该组分母（工具219f64退出0）。

| 同AMP62c4捕获 | 全程速度误差均值／P95（m/s） | 接触行人controls | 接触帧速度误差均值（m/s） | 参考方向undefined |
| --- | ---: | ---: | ---: | ---: |
| OLD63 S1 | .231013／.537961 | 0 | null | 2,401 |
| NEWb563 S1 | .258558／.647622 | 140 | .427503 | 2,322 |
| NEWb563 S10 | .245856／.585835 | 512 | .467333 | 2,118 |

这里的接触行人controls是每人参与任意peer接触后的计数，区别于原70／279个pair-controls；同一行人同帧多对接触只计一次。接触标记汇总四个120Hz子步，速度来自末子步，两者没有相同的子步定位。

S10全程速度误差低于新S1，同时接触帧误差、接触频率与原跌倒数更高，不能只依据全程平均误差宣告物理改善。低速方向undefined和全部跌倒／reset均保留。

phase1的native速度向量变化均值分别.081018／.096962／.096167m/s，phase15为.047655／.055325／.058653m/s；这是同episode相邻控制末端的观察差异，不是加速度、真实延迟或重规划导致偏离的证明。signed角度均值会正负相消，不应当作方向误差大小或圆周平均。

完整数据：[actual_diagnostic.json](actual_diagnostic.json)、[all_actor_control_rows.csv](all_actor_control_rows.csv)、[实际工具闭合记录](actual_CPU_tool_receipt.json)。未加载模型、使用GPU或修改默认权重。

## 原来源与时序

`diagnose_response.py` pin 并复用封存的 `actual_tracking_contact_CPU_v1/diagnose_tracking_contact.py` 完整字节 SHA `33dd5405…`（26264B）。其原 admission 在 NPZ 读取前核对真正 closed0 的 GPU/CPU owner、raw audit、result、完整 physical/reference NPZ 和原 producer/comparison；不调用旧 main。新增 ground 门读取原 raw audit 的封存完整字节身份，并与原 GPU owner 的 artifact 和原 result 的 capture 身份相等；统计后再次核对来源，拒绝覆盖任何已有输出。

三组 OLD63 S1、NEWb563 S1、NEWb563 S10 各保留6000 controls ×4人 = **24,000 actor-controls**。ground 的四个连续120Hz物理子步、before/after control计数、episode/reset/termination字段逐项对应；末子步 named Pelvis position/quaternion 与同30Hz physical root bitwise 相等。native XY速度来自同末子步 `body_link_lin_vel_w`，不是位置差分、COM alias或reference速度。参考target／velocity来自已保存计划的 `age+1`，复用原全6000相位重建。完整采样发生在reset前，跌倒末帧与下一reset首帧都保留。

## 指标与空值

- `V_actual−V_reference` 的范数及沿参考运动方向／左侧横向分量；position误差同为actual−reference，沿方向为正代表在前，负代表落后。
- 实际root `+X` 的水平投影 heading 与参考**速度方向**的wrapped差；AMP原任务输入是未来XY点，不含独立目标yaw，不能冒称yaw命令跟踪。
- 参考速度 **`<.1m/s`** 时，运动方向和相关投影/角度显式undefined；`.1m/s`等号属于defined。实际root水平heading投影norm `<=1e-12` 时也显式undefined，避免将跌倒时的奇异朝向造作角度。
- 速度向量变化的norm（m/s）和wrapped heading变化（rad）是相邻控制末端差，**不是加速度或延迟**。只在同episode且非reset使用；首帧/reset差分为null。低速方向无法用于方向差，但速度差仍保留。
- 原idle定义仍是合法同episode root XY位移速度`<.1m/s`。各组报告all、phase1–15、terminated、reset首帧、真实peer/static/any/no contact及idle的响应分布；每项另列undefined数量。undefined行始终保留在组分母，不删除低速、跌倒或reset。

输出全 **72,000行** CSV，包含原frame/control/episode/phase、reset/fall/contact/idle标记、native/reference XY速度和全部指标；undefined数值为空。JSON各指标的有限值计数与undefined数合起来等于该组完整actor-controls。完整phase分组之和每组24,000。原XY误差不平移、不对齐到其它phase，不搜索lag或挑选较好滞后。

这只能描述参考变化、身体响应和偏离的观察关系。缺少独立保存的actions/task observation及reset前置root速度，不能识别真实action→response延迟、证明原因或宣告模型优越。它不添加success/promotion门，也不修改common50候选适配、185源码、训练或默认模型。

## 审阅后执行

在仓库根目录、含 NumPy 的 CPU Python中：

```bash
CUDA_VISIBLE_DEVICES='' python docs/style_improvement_20261004/reference_response_CPU_v1/diagnose_response.py
```

生成本目录 `actual_diagnostic.json` 和 `all_actor_control_rows.csv`，已存在时拒绝覆盖。准备阶段只验证小合成标量/数组接口与源码，不读取捕获或权重。
