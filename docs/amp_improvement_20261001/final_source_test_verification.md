# 最终源码与选定核心测试核对

核对时间：`2026-10-02T01:19:20.941118+08:00`（UTC+8）。本任务只读既有源码/归档/日志，未运行测试、Kit 或 GPU，未修改业务代码或发布规则。

## 当前源码快照

- `final_runtime_source.json` 的75个条目均与当前文件及 tar 中字节、大小、SHA256完全一致；重复、遗漏、额外条目均为0。
- 归档大小：223599 字节；SHA256：`7969f0e2320f7ba84bac4b4b212c1e5b3cd9ee88a9faa26aeb3a5da991227e6c`。
- 归档不包含权重或 SMPL 受许可资源。JSON保存全部75个文件的声明/当前/tar SHA及逐项校验。

## 实际测试与静态检查

- `final_core_tests.log`：`463 passed, 44 warnings in 13.57s`。
- JUnit实际463个 testcase、20个测试模块，失败0、错误0、跳过0；这些测试文件均在75项快照内。
- `final_ruff.log` 为 `All checks passed!`；75项快照包含71个Python文件。
- 44项warning分别是43项 `torch.jit.script` 弃用提醒与1项非法转义 SyntaxWarning，不是失败或跳过。

## 源码包 v3 的边界

[v3源码包报告](packaging_source_snapshot_v3.md)仍是有效历史快照；其归档与当前75项逐路径比较，只有下面的文件不同：

- `protomotions/simulator/isaaclab/utils/scene.py`：v3 SHA `9fe48faff55ef90e7f7c4b0ce52692443e0fb927b28643ffdf95ec356e83d491`；当前 SHA `b721a0e2d4f4b5529e191182a5bb152619dde65de36d400627c92ed95731d6f1`。

差异来自父任务删除未使用的 `ISAAC_NUCLEUS_DIR` import。v3不能声称与当前源码逐字节一致；[v4最终源码包](packaging_source_snapshot_v4.md)已纳入该清理及最终指南，全部75项runtime源码和24项必需路径与当前文件字节一致。

这里验证的是选定20个核心测试文件，并非全仓库测试通过，也不代表75个源文件全部被执行。没有新增物理稳定性、接触、模型训练或渲染一致性结论。历史200秒运行和录制回放仍使用各自来源证据。

详情：[final_source_test_verification.json](final_source_test_verification.json)。
