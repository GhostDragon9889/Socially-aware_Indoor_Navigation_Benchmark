# 外部视角回放的独立来源与时钟核查

对象：`native_warehouse_recorded_replay_30s_v2`。**1,149 项只读 CPU 检查通过**，原始 result/NPZ/MP4/53 个 USD 层和历史 V6 source171 归档均吻合；五个 renderer 声明模块逐项与历史归档比较，不用后来 V7 当前树替代运行来源。

450 个 frame records 全部对齐源 step0..898、stride2、30Hz post-control timestamp、源 reset/episode。所有 frame timeline0→0，声明 physics/policy0；实际 ffprobe 独立确认 **450 帧、15FPS、30s、1280×960**。源是已经录制的 900 控制步、0 跌倒、[2,2,2,2] 顶点物理数据，没有新增物理或策略验收。

复用资源 agent 已完成的 450 PNG/MP4 banner 全量审计，绑定其 reader SHA 与实际报告 SHA，没有重复解码全部视频。这些媒体完整性门禁不检查室内行人是否可见。

**视觉交付未通过**：独立查看 `frame_00225.png`（源15.033s）确认为仓库外部俯视屋顶，行人被遮挡。因此 report 的 status 为 `source_clock_media_contracts_passed_visual_delivery_rejected_roof_occlusion`，没有把 renderer `passed` 提升为可见步态或论文风格效果通过。

室内新相机的 fresh output 应独立验收，此旧回放、屋顶遮挡和历史源码证据完整保留。本审计没有启动 Kit/GPU、修改场景几何/visibility 或任何生产文件。

详见 [actual_replay30_v2_epoch_and_clock.json](actual_replay30_v2_epoch_and_clock.json)；资源 agent 的全量媒体报告为 [actual_replay30_v2.json](../native_recorded_replay_reader_v1/actual_replay30_v2.json)。
