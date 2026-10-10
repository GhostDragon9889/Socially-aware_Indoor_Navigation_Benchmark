# 源码包 v2 最终代码快照

核对时间：`2026-10-02T00:50:08.537085+08:00`（UTC+8）。本轮仅重新构建与 CPU 核验；`MANIFEST.in` 无需再修改。业务源码、环境、权重和历史工件均未改动。

- 新归档：`/tmp/protomotions-final-packaging-20261002-v2/protomotions-3.1.0.tar.gz`。
- 文件数：915；大小：41227601 字节。
- SHA256：`c70b9b6990f3870d77c3e1dddc4c0fe9bbd224dfb1e349fc0324512fbf23a884`。
- 24 个必需文件逐一匹配当前源码字节，含确定性评估 CLI/CUBLAS 预检、新评估测试、回放 v2、SMPL 可视化与日志修正。遗漏 0；权重、SMPL 二进制及大型外部资源混入 0。
- 解包运行纯配置入口：Torch/Kit imports 为 `[]`，`--deterministic-evaluation` 可见，CUBLAS 预检不会修改环境且拒绝非法值，默认外置权重缺失在 SDK 启动前明确失败。
- 5 项打包测试通过，验证脚本 Ruff 通过；没有启动 Kit/GPU，也没有再次执行完整训练或物理评估。

上一版归档保留在 `/tmp/protomotions-final-packaging-20261001/`，旧 `packaging_structure_verification.json` 与视频核验不覆盖。新归档是上述时间的源码与指南快照；后续主指南文字更新尚不属于这个 SHA，正式发布时可再次构建。

详情：[packaging_structure_verification_v2.json](packaging_structure_verification_v2.json)、[packaging_v2_tests.xml](packaging_v2_tests.xml)、[packaging_sdist_v2_build.log](packaging_sdist_v2_build.log)。
