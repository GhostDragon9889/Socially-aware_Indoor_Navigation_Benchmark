# 新 ORCA Maps 1000：独立 CPU 数据 QC

基于实际 closed0 generation receipt，独立 reader 在 CPU 完成一次全批 QC（`21.692 s` / exit0）。没有重新生成 RVO2 数据、加载模型、读取 sealed34 行人测试结果或修改旧输入。[实际执行记录](actual_QC_process.json) / [完整审计](actual_QC.json)

| 项目 | 实际核验结果 |
|---|---:|
| unique 非空布局 / episode | 1000 / 1000 |
| train / validation / test 布局 | 800 / 100 / 100 |
| 各 episode 保存帧数 | 101，`t=0..10 s`，10Hz |
| 全部 saved actor-frames | 957,884 |
| 全部合法 history31 / future52 候选窗口 | 180,196 |
| 完整条件窗口 | 171,746 |
| train / validation / test 条件窗口 | 137,689 / 17,359 / 16,698 |
| 完整文件 SHA 检查 | 3003 文件 / 36,973,374 B |
| unique 已记录 seed / 已保留拒绝 attempts | 2158 / 158 |

全部 NPZ 使用 `allow_pickle=False`；所有字段无 object、有限且满足声明的 shape / dtype。检查了每个文件 SHA、非空 polygon、scene/layout/episode/seed 的分割唯一性、真实时间格点、每条窗口的 history/future 边界、yaw 来源和轨迹速度上限。此处新 ORCA `test` split 仅做完整性/几何审计，没有用于模型训练或选择。

所有 180,196 合法候选的 15m 真实邻居与 `.25 m²` crop/obstacle 相交条件均独立重算，保留窗口与源索引完整一致。171,746 个保留窗口均在**真实 crop 栅格采样点**找到 occupied 值；先使用 crop 的间隔8格点作为正见证，无见证的 284 个窗口再检查完整224×224格点。全局 map/geometry 另按明示的间隔16格点核验 world 坐标和 positive-y row 约定；没有将它写成全像素重新生成。

所有保存的 10Hz actor/frame 均重新计算真实圆盘表面间距及 signed circle/polygon 间距。最小 pair gap 为 `-9.8095382e-5 m`、最小障碍 gap 为 `-2.9064615e-5 m`，负值保留；原 `-1e-4 m` 容差以下的 overlap count 都为零，不能声称数学上严格无穿透。一个独立已知圆盘/方形 fixture 同时验证了正 pair gap 与内部负 signed obstacle gap。

原 producer 的 **60Hz** QA 逐场景与实际日志、manifest、固定源和 closed0 receipt 绑定。发布的 NPZ 没有保存60Hz states，因此本审计没有也不能独立重算中间60Hz位置或声称排除采样之间的短暂碰撞。相关限制与原始 QA 数值保留在完整报告中。

新 manifest：1,553,588 B / `97e1d639…`；独立 reader：19,942 B / `01dd9f38…`；实际完整报告：2,225,114 B / `41a02951…`。该数据是本地 map-only 扩展，4–15 agents、非空 maps、障碍上限20；没有 ORCA-Interact，没有人体姿态/力矩，不是作者 NavIsaacLab 数据，也不证明新 Diffusion 或物理导航效果。
