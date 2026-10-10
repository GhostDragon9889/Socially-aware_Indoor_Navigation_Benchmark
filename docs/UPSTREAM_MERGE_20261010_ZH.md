# 上游完整合并（2026-10-10）

本次将 NVlabs/ProtoMotions 的 `main` 提交
`7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1` 合并到本项目。
私有仓库保留 `upstream/protomotions-main` 原始分支及其 74 个提交，最终合并提交同时保留
私有仓库原 `main` 和原始上游提交作为父提交。

## 文件合并

逐项核对上游 1,017 个 Git 文件，补回之前发布范围省略的 235 个文件。
已有 Isaac Sim 6 / IsaacLab 3 迁移代码和 Diffusion＋AMP 实现继续保留。
同名文件以当前实现为合并结果；上游原版可从原始分支或上游父提交读取。

目录映射为：`protomotions/` → `src/protomotions/`，`deployment/` → `src/deployment/`，
`usd_convert/` → `tools/asset_conversion/`，`pyroki/` → `tools/retargeting/`。
兼容链接保留原导入和资源路径；Dockerfile、示例、模型卡、权重配置和示例数据均已纳入。
每个文件的上游路径、合并路径、Git blob SHA 和处理结果记录在
`configs/upstream/protomotions_merge_20261010.json`。

旧 `external_source_assets.json` 中 47 条资产记录已重新核对：8 条路径及 OID 对应固定上游；
38 条路径不在该提交中，另 1 条路径存在但 OID 不同。后 39 条标记为来源未核实，不能使用上游下载冒充恢复。
这些记录的修正不影响本次 1,017 个真实上游文件的完整映射。

## LFS 下载

按用户选择，保留上游原始 LFS 引用，不将约 5.3 GB 的 LFS 实体重新上传到私有仓库。
223 个上游 LFS 文件中，已有源码发布包含部分已实体化的小型资产；本次补回 158 个 LFS 指针。
`.lfsconfig` 的下载地址指向原上游，上传地址指向本私有仓库；不会向原上游上传资源。
自训权重、AMASS 完整数据、外部场景和视频仍按本地资源管理。

按需克隆和下载：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git
cd Socially-aware_Indoor_Navigation_Benchmark
git lfs install --local
git lfs pull --include='src/protomotions/data/assets/**'
# 需要上游预训练模型时：
git lfs pull --include='data/pretrained_models/**'
```

Git LFS 引用在新机器上需要下载为实体，不能直接用于模拟器。
资源许可证和模型卡沿用上游原文；上游资源不等于论文作者的定制权重。
新增自行训练的 LFS 资源应配置其对应下载来源，不能假设上游服务存有本地生成的对象。

## 验证与本地历史

```bash
python scripts/verify_upstream_integration.py
git log -1 --format='%H%n%P%n%s'
git branch -vv
```

合并前的全部暂存状态另存为本地 `codex/local-before-upstream-merge-20261010`，
原 `.git`、历史对象及 LFS 缓存保留。
当前 `main` 将跟踪私有仓库的 `benchmark/main`；`origin` 继续提供原上游读取来源。
原始本地实验和训练资源保留在工作目录，不因切换提交而删除。
本次验证覆盖源码布局、上游文件完整性、Git/LFS 引用和 CPU 检查，未重新运行训练或视频录制。
