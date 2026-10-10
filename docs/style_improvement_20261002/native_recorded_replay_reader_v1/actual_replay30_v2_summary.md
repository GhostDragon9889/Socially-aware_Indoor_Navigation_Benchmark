# Native Warehouse 外部镜头回放：编码通过，视觉交付未通过

实际450帧/15FPS/30秒MP4、全部450PNG永久source-time横幅、全部450编码横幅crop decode通过；源result/NPZ及53个USD层逐SHA不变，Root弱层1/session0、physics0/policy0/timeline0。source171V6 renderer，历史physical source158，与后续174V7分开。

首/中/尾人工view_image均是屋顶，无可见行人。因此编码/资源合同PASS不是行人视觉演示PASS。本次视频及此前两个0帧失败全部保留。详见 [actual_replay30_v2.json](actual_replay30_v2.json) 与 [summary](actual_replay30_v2_summary.json)。

[CPU roof bounds](actual_roof_CPU_bounds.json)：72个roof/ceiling Boundables。拟室内camera(-6.75,-8.95,3.2)的XY上方唯一屋顶Mesh `/Root/SM_CeilingA_6X34/SM_CeilingA_6X6`，bbox z=8.999999436..8.999999665 m，camera确实在其下。此为静态BBox证明，不保证新镜头无遮挡/四演员同时可见。未隐藏屋顶或改原USD，室内回放仍待实际生成与人工视觉审核。
