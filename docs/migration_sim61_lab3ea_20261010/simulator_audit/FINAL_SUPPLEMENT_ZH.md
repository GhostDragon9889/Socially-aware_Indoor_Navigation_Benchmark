# 仿真迁移补充审查事实

本文件补充初始逐文件 `coverage.json` / `AUDIT_ZH.md`，不覆盖历史失败日志。

## 真实 SDK 断点与修复

- 新版 USD importer 的 root default `Physics=physics` 保留并行 hinge 结构；对应 `PhysX` 子层已具有正确 D6 关节。root 的实际 N1 装载曾只有 23 个控制维度，不能用虚构 69 个名称解决。
- 机器人 `UsdFileCfg` 和 USD body-path 检查一致选择 `Physics=physx`；临时解析使用 session layer，不改原 USD 字节。
- 实际 importer 输出 CPU 解析：generic 69 Revolute / 0 D6；physx 23 D6 / 69 个未锁旋转轴 / 24 个 body，原文件 SHA 不变。
- `Articulation.num_joints` 与公开 `joint_names` 仍按实际 DOF 表示，官方后端 shared_metatype 使用 `dof_count` / `dof_names`。没有新建假想 `find_dofs` 或名称扩张；按实时 69 DOF 验证 semantic ordering，维度不足立即拒绝。
- 新增 `test_isaaclab61_physx_variants.py` 六项实际 pxr USD / SMPL configuration CPU 测试通过，详见 `../d6_variant_fix/API_MATRIX_ZH.md`。CPU schema 证明不替代活体 PhysX 测试。

## Upstream reference 兼容边界

`data/pretrained_models/upstream_reference_registry.json` 为两个 NavIsaacLab 旧分支 MaskedMimic 参考配置登记原文件、SHA 和明确 unsupported 理由；当前上游重构后不能反序列化其旧 class，也已不接受旧资产配置字段。加载器在 pickle 执行前给出可操作拒绝。未造 alias，未修改权重、侧配置或封存文件。

登记仅覆盖 `masked_mimic/smpl` 与 `motion_tracker/smpl-terrains`，没有封禁当前 TRACE/PACER 或本地训练模型。真实 registry + modern sidecar + config utility 选择集：20 passed。

## 项目运行环境

独立 overlay/安装边界审查另见 `../install_metadata_audit/AUDIT_ZH.md`。最终项目 metadata SHA 为 `69f89568881108ce518ac8d8c2a9d880684e18a97f9594712714bb672708b750`。

## 物理证据由主任务归档

主任务已报告新 SDK 的 N1 真实 60 control steps / 0 falls / process exit 0，以及五项 CUDA 验证通过。当前完整四场景 1800 controls 和 RTX 视频仍由主任务采集。这里不把 CPU、短程、旧 SDK 录像或软件导入成功解释为长程可靠性/论文指标达标。

没有修改 SDK 源码或安装二进制，没有提交或推送。本 agent 的仿真代码与运行环境代码已冻结供主任务长程采样。
