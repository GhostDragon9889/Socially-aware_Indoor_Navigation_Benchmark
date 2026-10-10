# 源码包 v3 最终兼容修正快照

核对时间：`2026-10-02T01:03:58.863179+08:00`（UTC+8）；归档修改时间：`2026-10-01T16:58:03.031185+00:00`（UTC）。本任务仅重建和 CPU 核验，未修改发布规则、业务代码、SDK 环境或权重。

- 归档：`/tmp/protomotions-final-packaging-20261002-v3/protomotions-3.1.0.tar.gz`。
- 915 个文件、41228307 字节。
- SHA256：`75720909e90a6485d4322116ee11545454e4ab129c703e1ab9b4bd38f409dc9a`。
- 24 个必需路径均存在；源码与测试逐一匹配构建时当前字节；训练入口、配置入口和时钟兼容测试分别匹配父任务冻结的 SHA。
- 两个主中文指南在归档后更新；其归档/当前 SHA 与不一致字段在 JSON 中明确保存。仅这两个已授权、仍在编写的指南接受后续文字更新，其他源码/配置/测试差异仍会失败。状态为 `passed_code_snapshot_with_later_guide_updates`，没有宣称所有当前文档也完全一致。
- 保留 SDK 6.0.0.1 缺少可选 PhysX clock API 时的 `None`/unobserved 与 SDK 原因；没有以推测时钟替代。这里只核对所发布源码，不新增 GPU/Kit 执行结论。
- 遗漏文件 0；权重、SMPL 二进制和大型外部资源混入 0。
- 解包配置入口 Torch/Kit imports 为 `[]`。确定性评估选项存在，CUBLAS 预检只读并拒绝非法设置；默认外置权重缺失会在 SDK 启动前失败。
- 5 项打包测试、核验脚本 Ruff 通过；未重新执行完整训练、物理仿真或视频采集。

v1/v2 归档及 JSON/测试/hash 证据完整保留。v3 代表上述时间的源码与指南快照。正式发行如需包含最终指南文字，可以在指南冻结后再次构建。

详情：[packaging_structure_verification_v3.json](packaging_structure_verification_v3.json)、[packaging_v3_tests.xml](packaging_v3_tests.xml)、[packaging_sdist_v3_build.log](packaging_sdist_v3_build.log)。
