# 作者公开行人模型资源下载验收

日期：2026-10-01。来源为第一作者 [leele-bing/ProtoMotions](https://github.com/leele-bing/ProtoMotions/tree/8702e7919d33d7f264682a28c5c120934c8cc267)，固定版本 `8702e7919d33d7f264682a28c5c120934c8cc267`。

## 已完成

两套目录各六个文件已完整保存，全部 12 个文件合计 **362,205,589 B**。每个文件发布前均检查精确字节数、对应 Git blob 或 Git LFS SHA256；发布后再次独立读取最终文件并重算 SHA256，12/12 PASS。

| 模型 | 权重字节数 | 权重 SHA256 | 本地目录 |
|---|---:|---|---|
| MaskedMimic SMPL | 229,160,269 | `3147039996be020447d50738da9360878ca76561995b1442b8c22730e319231d` | `data/pretrained_models/paper_references/navisaaclab_public/masked_mimic/smpl/` |
| SMPL terrain tracker | 132,661,223 | `ff5ffd0dbaea9532c5d44ebdc43e3c0aa5e8c5671d5603bd862a1d9a07c0174b` | `data/pretrained_models/paper_references/navisaaclab_public/motion_tracker/smpl-terrains/` |

各目录包含 `last.ckpt`、`experiment_config.py`、`resolved_configs.pt`、`resolved_configs.yaml`、`resolved_configs_inference.pt`、`resolved_configs_inference.yaml`。正式目录 API 的文件名与 Git blob SHA 固定在 `author_public_masked_mimic_directory.json`、`author_public_smpl_terrains_directory.json`；各文件的具体来源、大小、哈希及重试历史见 `navisaaclab_public_download_manifest.json`。

## 下载与发布约束

第一批限制每文件 250 MiB、总量 450 MiB、600 秒；557.04 秒内完成两份权重及六个 sidecar，四个 sidecar 因网络连接超时未完成，停止该批网络请求。父代理随后确认用户下载授权并另给四个缺失配置文件独立 300 秒期限，每文件 1 MiB、总量 4 MiB。三个缺失指针由 GitHub MCP `fetch_blob` 正式接口取回，重算 Git blob SHA1 与已固定目录一致；四个 body 在 158.30 秒内完成，共传输 170,668 B。原批次的记录与期限均保留。

正文通过 GitHub 官方 `media.githubusercontent.com/media/...` 取回。临时文件通过精确大小与 SHA256 校验后 fsync、原子替换发布，并 fsync 目标目录；现有不同内容的目标文件拒绝覆盖。两批 curl 统计正文传输合计 362,176,894 B，另外两个非 LFS 的配置文件来自 Git blob API。

## 使用边界

作者的公开 CrowdSim README 将此分支描述为 SFM + MaskedMimic，另有 motion tracking 比较。这里补齐的是该公开实现的参考资源。论文 IV-A 定制 trajectory diffusion、IV-B SMPL navigation AMP、2,848 clip 精确子集的公开性结论见 `paper_publication_sources.md`；这两份模型不足以证明论文训练流程已完整复现。

资源位于独立 `paper_references` 目录，没有覆盖当前 v3 模型、执行历史配置、启动 SDK 或改变环境。下载验收证明文件完整性与固定来源；其网络层名、配置 API 和当前项目运行兼容性尚未由本次下载任务验证。
