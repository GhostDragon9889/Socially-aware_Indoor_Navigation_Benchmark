# Guidance 20→40：完整 200 秒真实导航对照

同一固定 controller、Diffusion、共同训练配置值与 reconstruction fixture；新序列化 metadata 仅增加未启用的 expert_sampling=None。部署 argv 只改变 guidance_weight 20→40（另有输出路径）；不宣称后续工作树全部源码相同或重复运行位级一致。保留全部终止帧、跌倒及 episode reset，冻结原 validation reference/特征/采样协议。完整每-run source/config/runtime 字典在 comparison.json。

| 指标 | guidance20 | guidance40 |
|---|---:|---:|
| 训练 transitions | 8192000 | 8192000 |
| 跌倒 | 0 | 1 |
| 路线顶点 | 21 | 29 |
| 四顶点 cycles累计 | 4 | 6 |
| inclusive 分层姿态+速度 MMD² | 0.07238060315284697 | 0.0863525797735131 |
| 物理速度分布 W1 (m/s) | 0.3533679939881452 | 0.21723250509033898 |
| 位移速度 <0.1m/s 的有效 interval比例 | 0.24349891648608102 | 0.05772035840800167 |
| 直立敏感性保留比例（非主指标） | 0.9995415902650442 | 0.9989997916232548 |
| 实际静态接触 actor-frames | 0 | 0 |

Cycles 为各 actor 累计顶点数整除4再求和；reset 与首次终止的原始时刻均保留，不以该累计数替代无中断 episode 完成率。

## 停滞与命令关系

仅同 episode rootXY差分速度 <0.1m/s 为 idle；不合并短暂速度波动。global_goal_xy 是 timed A* lookahead endpoint，最终路线顶点另看 route_vertex_xy。

| actor | guidance20 idle | guidance40 idle | 20最长(s) | 40最长(s) | 40段 .5s raw位移(m) | 40执行target误差(m) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 58.11% | 2.55% | 38.267 | 0.967 | 0.1058 | 0.1302 |
| 1 | 5.03% | 8.15% | 1.767 | 13.933 | 0.0447 | 0.0487 |
| 2 | 15.10% | 7.90% | 5.233 | 1.333 | 0.0813 | 0.1209 |
| 3 | 19.15% | 4.48% | 8.600 | 1.467 | 0.1294 | 0.1280 |

每个最长段包含 native 同瞬时 root速度、执行reference速度与误差、原始 .5s/5s 计划的位移/向lookahead投影、双支持和 root点至AABB几何距离。它们定位命令与物理响应关系，不建立 terrain RMS 因果；几何点距离不是全身体碰撞距离。

## 真正足力/COM通道

两份 normal-force sensor 均由独立 CPU audit 重算四子步聚合、COM和时钟；历史 neutral-stance probe=False 不改。AMASS 真 COM/接触/摩擦仍未知。真实接触条件下的 link原点速度代理可按同定义比较，但不能叫接触点滑动；另有 frozen style 的近地 toe差分代理，二者不混用。

![同冻结协议分层风格和物理速度分布](comparison.png)

来源：本目录 ground_support_actual_verification.json 校验实际运行前70文件归档，未用后续工作树宣称全源码不变。style实际输出在 frozen_style/evaluation.json；reference SHA保持 `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。单个 seed / fixture 的控制部署对照不是论文八场景或风格验收。

## 跌倒 / reset 原始时序

Actor 3 在 source frame 187（t=6.2667s）终止，Pelvis 0.40617m；旧 episode 1 的数据先完整捕获，下一 frame 才 reset 为 episode 2。终止帧 peer 接触对象 []，此前 .5s peak peer法向力 0.00N；只是同时间关系，不据此单因归因。

完整 6000 帧均保留。Guide20 有效同-episode interval 23996；guide40 23995，唯一跨-reset interval 被明确排除，终止帧仍计入 inclusive 风格。Guide20 run dictionary 与先前唯一输出逐项相等，reference byte-identical；见 [终止与来源核验](terminal_and_source_verification.json)。

实际 peer接触共有3个连续 pair事件、13个 unordered pair-frames（26个 actor-frames），发生于72.733s、88.467–88.767s及131.067–131.100s；首次跌倒6.267s不与这些事件重合。不能把此次跌倒解释成已测 peer碰撞造成。
