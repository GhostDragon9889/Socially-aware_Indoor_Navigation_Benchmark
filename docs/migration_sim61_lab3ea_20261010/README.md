# IsaacLab3EA / IsaacSim6.1 迁移证据

本目录为小型、可复核的源码、配置、依赖与实际执行证据；未提交或推送。最终结论见[总报告](../ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)。部分子任务报告是阶段快照，其当时统计和失败保持原身份；不能把不同阶段数字相加。

- [最终本地交付与未解决项](final_delivery_receipt.json) / [最终分发验证](packaging/LATEST_VERIFICATION.md)
- [逐Python文件/行/定义/import/SHA](sdk_audit/extended_scope/all_python_summary.json)
- [SDK差异、API矩阵和原始GDB退出定位](sdk_audit/README.md)
- [当前4,293测试节点逐项结果](functional_audit/functional_test_matrix_latest.json)
- [当前完整CPU XML](functional_audit/current_clean_sim61_final.xml) / [实际CUDA XML](runtime/gpu_five_actual.xml)
- [354历史节点原清单](functional_audit/historical_nodes.txt) / [原全量CPU第二轮失败栈](functional_audit/original/full_cpu_02_failures.json.gz)
- [生产文件/函数未覆盖行](functional_audit/functional_file_coverage_matrix_final.json) / [测量版本与当前版本对应](functional_audit/coverage_source_version_match.json)
- [项目依赖overlay](runtime/ENVIRONMENT_ZH.md) / [安装与分发](packaging/PACKAGE_AUDIT_ZH.md)
- [四场景真实物理结果](physics/) / [AMASS训练实际退出](training/amp12_sim61_execution.json)
- [checkpoint恢复界限](training/amp12_sim61_checkpoint_roundtrip.json) / [RTX实际解码](render/SecondFloor_three_frames_02_decode.json)
- [干净参考源码40文件补丁回合验证](sdk_audit/reference_patch_roundtrip.json)
- [复制与原文件SHA索引](evidence_copy_index.json)

`.gz` 是原始JSON/XML/log的可逆压缩；解压所得字节须与复制索引的source_sha256相同。大数组、数据集、模型、视频、完整源码副本不进入此文档目录；索引记载本机原始证据位置，核心运行不会依赖本次临时审查目录。
