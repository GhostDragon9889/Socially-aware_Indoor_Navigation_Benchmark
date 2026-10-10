# Native runtime source V11

实际合并后的 selected source freeze，183 个文件。父 V10 的 181 个文件中，五个变化为这次授权的 settings/trainer/checkpoint_contract/resources 与 runner fixture 测试；其余 **176 个逐位不变**。新增 helper 与 profile 两个文件。没有增加其他选定来源。

- `native_pedestrian_source_v11.json`：183 文件 SHA/字节数及父来源关联。
- `native_pedestrian_source_v11.tar.gz`：相对路径排序的 USTAR；gzip/tar mtime=0、uid/gid=0、空 uname/gname、mode0644、仅 regular 文件。两次独立构建逐位一致，归档中的183 payload 已重新逐项验证。
- `actual_merge_receipt.json`：六个实际合并目标匹配冻结 proposal，另有 fixture 最终字节；181 个父文件的全部 before/after SHA、字节和变更标记。
- `parent_archive_validation.json`：父 V10 tar 的实际 SHA/大小以及全部181 payload 与父 manifest 的一致性。
- `protected_fingerprints.json`：重新读取旧 isolated931 全部源码与六个冻结输入，校验原 SHA/大小。六个输入合计2,307,757,289字节，包括公共 PACER、diffusion checkpoint、AMASS manifest/train/validation/test；stat tuple 也保持原值。数据只做 hash，不反序列化。

`runner_reward_fixture_fix_v1/after_final.py` 和 `manifest_v2.json` 保存 root 的最终 fixture 源码与两次实际测试证据：第一次补训练上下文后仍因 transition AST 位置报两项 StopIteration；最终 ast.walk 修复后两项通过。原 before.py、after.py、manifest.json 与两份 XML 保留历史原字节。

此冻结过程未修改生产、旧 isolated931、SDK、数据或权重，也未启动 GPU/Kit。来源封包验证不证明物理运行质量；merged CPU 验证与后续实验由 root 执行。
