# ORCA-Interact：实际可用数据

官方 TRACE 固定源码生成的 1,000 个无地图场景已实际转换、完成全量 CPU 检查并封存。它们是本地重新生成的数据，作者训练数据的场景实例仍未知。

| 项目 | 实际结果 |
| --- | --- |
| 场景分割 | train 800／validation 100／test 100 |
| 可用窗口 | 150,439／19,237／19,692 |
| 无有效交互窗口的场景 | 49／7／6，全部保留 |
| 行人实例／帧 | 10,608／1,060,800 |
| 帧时刻 | 100 帧，0.0 至 9.9 秒，10 Hz |
| 最大真实邻居数 | 19 |
| 实际半径 | 全部 0.4 m |
| 文件 | 1,000 个 episode 加 manifest，均 0444 |

数据位于 `data/diffusion_amp/orca_interact_usable_1000_v1`。转换在原始 `initPos/initVelocity` 的真实零时刻与官方记录时间戳上插值；无地图掩码为 false。yaw 是速度方向代理，静止零方向另有有效性掩码，不能当作真实人体朝向。缺失地图路径已在 staged 接口中与地图编码器隔离。

## 实际证据

- [转换进程记录](conversion_actual_launch.json)：实际进程正常退出。
- [全量 CPU 结果](../orca_interact_usable_QC_v1/actual_QC.json)：SHA256 `14d8cdb4…`，三个分割均实际构造 19 邻居批次。
- [CPU 进程闭合](../orca_interact_usable_QC_v1/actual_CPU_closure.json)：实际 exit0，未启动 SDK、GPU 或项目神经网络。
- [完整只读封存](usable_root_seal.json)：SHA256 `e5848610…`，按 QC 核对全部 1,001 文件后设置只读。
- [接口源码与转换说明](../orca_interact_usable_staging_v1/README.md)：冻结时的 prepared 状态是历史记录，实际转换完成状态以本页和进程记录为准。

尚未训练 Maps＋Interact 混合模型；当前只确认数据格式和 CPU 加载接口。均匀拼接两个语料会按窗口数产生约 52.2% Interact，并非显式 50:50 采样。当前数据、yaw 代理和本地 Maps 生成参数不能证明作者数据等价或论文效果。

[TRACE/PACE 附录 A.1](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)把无地图场景的全 0.5 栅格送入地图编码器。已冻结 staged 接口中的“跳过无地图编码器”属于本地适配消融。后续混合训练以作者的未知栅格编码作为默认，数据中真实的无地图标记仍保留；训练模式、模型输入和恢复身份须明确区分这两种处理。
