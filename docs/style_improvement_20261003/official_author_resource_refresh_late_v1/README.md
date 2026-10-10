# 作者公开资源晚间增量核查

匿名 HTTPS 元数据请求时间：2026-10-03 UTC 15:07:26–15:09:24（Asia/Shanghai 同日 23:07:26–23:09:24）；此前本轮已用网页工具读取论文及公开页面。本轮比较今天 UTC 02:58–03:02 的[既有核查](../public_resource_refresh_v1/report.md)，保留实际请求 URL、时刻、状态、响应大小和 SHA256：[请求清单](fetch_manifest.json)、[LFS 小元数据清单](lfs_fetch_manifest.json)、[机器结果](summary.json)、[真实 CPU 闭合记录](actual_CPU_closure.json)。

## 结论

本次成功读取的作者公开入口中，未识别到新增、且能确认匹配论文的定制 AMP/Diffusion 权重、作者精确 2,848 条 AMASS 清单、八场景可重放清单或可执行指标协议。该结论限定于以下入口，不表示私有、未链接或未检索资源不存在。[项目页](https://broln7.github.io/NavIsaacLab-web/)、[官方仓库](https://github.com/yh83305/NavIsaaclab2.0)、[论文版本页](https://arxiv.org/abs/2606.26265)

| 已知作者入口 | 当前 HEAD | 与此前核查比较 |
|---|---|---|
| `yh83305/NavIsaaclab2.0/main` | `d742f0c5b421bbc281e2112e7e50427e0336c16a` | 未变；完整 820 条树记录逐项相同，未截断；releases/tags 为空。 |
| `BRoln7/NavIsaaclab2.0/main` | `d742f0c5b421bbc281e2112e7e50427e0336c16a` | 未变；本轮只刷新此 HEAD。 |
| `leele-bing/ProtoMotions/main` | `8702e7919d33d7f264682a28c5c120934c8cc267` | 未变；当前完整树 928 条、未截断，releases 为空。 |
| `BRoln7/NavIsaacLab-web/master` | `ae80e301166f6d16a1b23d692017021797cce27f` | 未变；完整 52 条树记录逐项相同；live 页面字节相同。 |
| `BRoln7/IsaacLab_AMP_rl-games/main` | `0207003d8117f9a283a29aa88c4514e414845c0d` | 与 2026-10-01 已知通用 AMP 仓库相同；releases 为空。 |
| `BRoln7/IsaacLab-socnav/main` | `b08178bf2461ea3f67741e1ff344516bba1ce47f` | 与 2026-10-01 已知作者仓库相同；releases 为空。 |

HEAD、日期和完整请求地址均见保存的 API 响应。[官方 main](https://api.github.com/repos/yh83305/NavIsaaclab2.0/branches/main)、[官方 releases](https://api.github.com/repos/yh83305/NavIsaaclab2.0/releases?per_page=100)、[第一作者 main](https://api.github.com/repos/leele-bing/ProtoMotions/branches/main)、[网站 master](https://api.github.com/repos/BRoln7/NavIsaacLab-web/branches/master)

官方 `FULL_PIPELINE.md`、`CrowdSim/README.md`、`CrowdSim/humanoid/README.md` 与此前保存字节完全相同，仍明确需要 MaskedMimic。live 项目页的项目链接仍指向 `yh83305/NavIsaaclab2.0`，没有新独立模型/数据/评估下载入口。arXiv 当前 submission history 仍只列 v1。[公开 pipeline](https://github.com/yh83305/NavIsaaclab2.0/blob/main/FULL_PIPELINE.md)、[CrowdSim 说明](https://github.com/yh83305/NavIsaaclab2.0/blob/main/CrowdSim/README.md)、[arXiv](https://arxiv.org/abs/2606.26265)

## LFS 与现有权重的身份

只读取第一作者仓库的 `.gitattributes`、完整目录和两份 134 B 的 Git blob pointer，未请求 `media.githubusercontent.com` 或权重正文。目录仍列 MaskedMimic/各类 motion tracker；下列既有指针身份与旧记录一致，不能据此标成作者定制 navigation AMP。[LFS 声明](https://github.com/leele-bing/ProtoMotions/blob/main/.gitattributes)、[pretrained 目录](https://github.com/leele-bing/ProtoMotions/tree/main/data/pretrained_models)

| 既有类型 | pointer 的 payload size / SHA256 |
|---|---|
| MaskedMimic SMPL | 229,160,269 B；`3147039996be020447d50738da9360878ca76561995b1442b8c22730e319231d` |
| SMPL terrain motion tracker | 132,661,223 B；`ff5ffd0dbaea9532c5d44ebdc43e3c0aa5e8c5671d5603bd862a1d9a07c0174b` |

这些大小和 SHA 是远端 LFS pointer 声明，本轮没有验证远端大型 payload 的可下载性或重新哈希本地权重。官方仓库根 `.gitattributes` 返回 404，完整树也没有该文件；不将这个 404 推论为 LFS 大型对象缺失。官方 `LICENSE` 仍为 Apache 2.0；这是代码许可文档，不能扩张为 AMASS/SMPL/外部模型资源授权。[官方 LICENSE](https://github.com/yh83305/NavIsaaclab2.0/blob/main/LICENSE)

## 复现仍需区分

数量相同的本地 2,848 段不能证明作者 clip ID/裁剪时间对应；八场景名称、初始状态/目标/seed、SFM 与 Diffusion 配对配置和 Table II/III 的具体分母/聚合/MMD 算法仍没有在本次已核查增量入口中获得可重放合同。论文公开的方法与本地补充的复现协议须继续分别标注。[论文 IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)、[此前具体缺项列表](../public_resource_refresh_v1/report.md)

网页工具对项目 live 页面/API 有临时访问错误，已通过小型匿名 HTTPS 成功核对上述元数据。初次 sandbox DNS 失败保留于 `first_fetch_manifest_dns_failure.json`；随后实际 CPU 两阶段均闭合 0，共 23 请求、22 个 HTTP 200、1 个上述 404。未使用凭据、执行远端代码、构建 NN/启动 Kit/GPU、下载模型对象、读取封存 34 测试输出、修改生产/default/旧证据。
