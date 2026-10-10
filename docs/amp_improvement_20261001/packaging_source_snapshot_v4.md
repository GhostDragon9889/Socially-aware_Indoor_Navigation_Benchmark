# 源码包 v4 最终源码与指南快照

核对时间：`2026-10-02T01:25:05.293024+08:00`（UTC+8）。本任务仅执行CPU构建和核验；未修改业务代码、发布规则、环境或权重，未启动Kit/GPU。

- 归档：`/tmp/protomotions-final-packaging-20261002-v4/protomotions-3.1.0.tar.gz`。
- 文件数：915；大小：41230624字节。
- SHA256：`9f8c7e769c705c1ed22794076cd76968090b7ee13c913fe812b448b37af325fb`。
- 最终75项runtime源码均存在于包内，逐项与当前文件和 `final_runtime_source.json` 的大小/SHA完全一致；遗漏0，字节差0。
- 原24个必需路径、两个最终中文指南与 `FULL_PIPELINE.md` 也逐字节匹配；此版本没有指南后续编辑例外。
- `FULL_PIPELINE` 已将1,886项回归标为较早完成阶段，明确当前463项/20文件是focused核心测试；最新文字已包含。
- 归档包含SDK可选native-clock兼容修正和未使用import清理；v1/v2/v3及其hash/报告均保留。
- 5项打包测试通过；纯配置入口Torch/Kit imports为 `[]`，CUBLAS预检只读且拒绝非法设置。外部权重缺失会在SDK启动前失败。
- 权重、受许可SMPL二进制、大型外部资源混入0。Core wheel包范围保持原设计，CrowdSim为源码包工作流。

源码归档与选定核心测试另见 [final_source_test_verification.md](final_source_test_verification.md)：实际463项通过、20个文件、0失败/错误/跳过。它不是全仓库、Kit或新的物理稳定性结论。

详情：[packaging_structure_verification_v4.json](packaging_structure_verification_v4.json)、[packaging_v4_tests.xml](packaging_v4_tests.xml)、[packaging_sdist_v4_build.log](packaging_sdist_v4_build.log)。
