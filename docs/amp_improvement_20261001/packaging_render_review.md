# 最终源码包与渲染工件核对

本轮发布规则只修改 `MANIFEST.in`，新增 5 条具体文件包含规则；业务代码、wheel 包发现、SDK 环境、历史资源和权重保持原状态。核对使用 CPU，未启动 Kit 或 GPU。

## 源码发行包

- 实际通过 `setuptools.build_meta.build_sdist` 构建：`/tmp/protomotions-final-packaging-20261001/protomotions-3.1.0.tar.gz`，914 个文件，41220849 字节。
- SHA256：`8ae1b94d53ddac9f29cb95e3ca397b3b226701901b13e38295b08ade00571b2f`。
- `application` 配置/资源/预检、`trajectories.py`、`reference.py`、`contact_monitor.py` 已由原有 `recursive-include CrowdSim *.py` 覆盖。补入资源获取、clean guidance 基准、录制状态回放脚本及两个当前指南。
- 必需源码与当前工作区逐文件字节一致；遗漏文件 0，权重、SMPL 参数/资产、大型场景/纹理/HDR、状态 NPZ 与视频混入 0。
- 5 项打包测试通过；从真实源码包解压后，当前 YAML 与纯配置入口可用，未导入 Torch、Isaac Lab/Sim 或 Omni。默认外置权重缺失会在 SDK 启动前明确报错。
- CrowdSim 继续采用源码包/源码检出工作流；core wheel 只发布 ProtoMotions。本文没有重新构建或验证 core wheel。历史证据链接指向完整工作区，未将所有报告与工件装入源码包。
- `replay_paper_pedestrians.py` 是 `recorded_pose_visualization`：展示已记录姿态；不会作为新控制器运行、接触复现或新的零跌倒验证。

核对详情见 [packaging_structure_verification.json](packaging_structure_verification.json)、[packaging_tests.xml](packaging_tests.xml)。复核工具：[verify_source_distribution.py](verify_source_distribution.py)。源码包中的指南是构建时快照；之后若指南更新，正式发布前应再次构建。

## 实际渲染结果

| 运行 | 视频 | 实际跌倒 | 最低 pelvis 高度 | 跌倒时刻 |
|---|---|---:|---:|---:|
| S1 | 960×720、15 FPS、450 帧、30 s | 1 | 0.434713 m | actor 2、16.200000 s |
| S2 | 960×720、15 FPS、450 帧、30 s | 1 | 0.411048 m | actor 2、17.833333 s |

MP4 经 `ffprobe -count_frames` 实际解码；PNG 为 control index 0…898、每 2 步一次。result、900×4 状态、termination 与 episode 变更、450 帧视频互相一致。`status=passed` 表示有界运行与工件输出完成，不能解释为无跌倒。

既有单样本、physics-only、4 人 200 秒运行记录 0 次跌倒，可保留为对照配置；它与这两次新鲜渲染仿真的结果并存。两次渲染均跌倒一次，因此不将 S2 提升为新的稳定性参考，也不将 S1 渲染声称为零跌倒演示。当前数据不能确定渲染是否导致跌倒，也不能证明 S2 的普遍优劣或论文精确复现。

逐运行来源 SHA、ffprobe、终止帧与验收边界见 [render_capture_verification.json](render_capture_verification.json)。
