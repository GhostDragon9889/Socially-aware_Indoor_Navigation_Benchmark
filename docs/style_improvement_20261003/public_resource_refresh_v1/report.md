# NavIsaacLab 作者公开资源增量核查

核查日期：2026-10-03，Asia/Shanghai。匿名公开 HTTPS 请求从 UTC 02:58 开始，各请求时间、URL、HTTP 状态、响应大小和 SHA256 见 [fetch_manifest.json](fetch_manifest.json)。本轮只新增此报告目录中的文本和元数据；未使用 GPU、启动 Kit、改变环境或运行/训练源码，未下载权重正文，也未执行网络内容。

## 结论与增量

在本次成功核查的公开入口中，**未发现相较 2026-10-01 新增、且可确认匹配论文的作者定制 trajectory diffusion 配置/权重、SMPL navigation AMP 配置/权重、作者 2,848 条 AMASS 精确清单或可逐项重放的八场景评估协议**。官方仓库、作者 NavIsaacLab fork、第一作者 ProtoMotions fork 和项目网站的已核查 HEAD 均与旧报告相同。这是有范围的查询结果，不表示其他未检索入口或私有资源不存在。[官方仓库](https://github.com/yh83305/NavIsaaclab2.0)、[项目页](https://broln7.github.io/NavIsaacLab-web/)、[旧核查记录](../../resource_completion_20261001/paper_publication_sources.json)

本次新增了三项可复核证据：公开 Git refs 与 API HEAD 相同；只读参考仓库的全部 709 个 tracked blobs 与公开完整树逐项 SHA 相同；Hugging Face 的 8 项名称/作者索引全部 HTTP 200 且返回空，包括旧核查请求失败的 `leele-bing` models 和 `BRoln7` datasets。[Git refs 结果](official_git_remote_refs.json)、[机器可读结论](summary.json)、[请求证据](fetch_manifest.json)

## 实际核查范围

共保存 30 次匿名 HTTPS GET 的响应，全部 HTTP 200；另执行一次禁用凭据助手、全局/系统 Git 配置和交互提示的 `git ls-remote --heads --tags`。30 份响应的实际大小与 SHA256 均通过本地复核。[证据校验](evidence_verification.json)

| 入口 | 2026-10-03 实际结果 |
|---|---|
| `yh83305/NavIsaaclab2.0` | `main=d742f0c5b421bbc281e2112e7e50427e0336c16a`；最新 commit 日期 `2026-09-16T15:53:55Z`；9 commits；branches 仅 `main`；tags、releases、全部 issues、全部 pull requests 均空。完整递归树 820 项、未截断，709 blobs。 |
| 只读 `/project/EXP/NavIsaaclab2.0` | HEAD 与公开 `main` 相同；working tree clean；709 blobs 的路径及 SHA 全部相同。本轮未 fetch、checkout 或改写参考仓库。 |
| `BRoln7/NavIsaaclab2.0` | 本轮核查 `main` HEAD，仍为同一 `d742…c16a`。未重复扫描该 fork 的其他发布面。 |
| `leele-bing/ProtoMotions` | 本轮核查 `main=8702e7919d33d7f264682a28c5c120934c8cc267` 及 releases，后者空。未重复全树审计。 |
| `BRoln7/NavIsaacLab-web` | `master=ae80e301166f6d16a1b23d692017021797cce27f`；52 项递归树、未截断；live 项目页面与此 commit 的 `index.html` 字节相同。树中为网页、图片、视频及 `static/pdfs/sample.pdf` 等文件，没有模型包或动作清单路径。 |
| 作者 live 项目页/项目列表 | 项目相关活动链接仍为同一 arXiv 论文和官方仓库；未出现单独的模型、数据集或评估协议下载链接。项目模板链接不作为作者资源。 |
| arXiv abstract 与 HTML | 当前公开 submission history 只列 `v1`，日期 `2026-06-24T18:08:19Z`；本轮阅读 `2606.26265v1` 的 IV-C 和对应表格。 |
| GitHub repository search | `NavIsaacLab fork:true` 返回 4 项，`incomplete_results=false`；仍为 official repo、作者网站、作者 fork、`GhostDragon9889` 第三方 fork。未将第三方 fork 认定为作者发布。 |
| Hugging Face | `NavIsaacLab` models、`navisaaclab` datasets、`NavIsaacLab` spaces；`leele-bing`/`BRoln7`/`yh83305` models；`leele-bing`/`BRoln7` datasets，共 8 个指定索引，全为空。空索引仅约束这些查询。 |

上述元数据的直接公开接口：[main branches](https://api.github.com/repos/yh83305/NavIsaaclab2.0/branches?per_page=100)、[commits](https://api.github.com/repos/yh83305/NavIsaaclab2.0/commits?per_page=100)、[完整树](https://api.github.com/repos/yh83305/NavIsaaclab2.0/git/trees/main?recursive=1)、[releases](https://api.github.com/repos/yh83305/NavIsaaclab2.0/releases?per_page=100)、[tags](https://api.github.com/repos/yh83305/NavIsaaclab2.0/tags?per_page=100)、[issues](https://api.github.com/repos/yh83305/NavIsaaclab2.0/issues?state=all&per_page=100)、[pull requests](https://api.github.com/repos/yh83305/NavIsaaclab2.0/pulls?state=all&per_page=100)、[第一作者 main](https://api.github.com/repos/leele-bing/ProtoMotions/branches/main)、[项目网站树](https://api.github.com/repos/BRoln7/NavIsaacLab-web/git/trees/master?recursive=1)、[arXiv 版本页](https://arxiv.org/abs/2606.26265)。其余每个请求的完整 URL 保留在 manifest，页面活动链接另存为 `*_links.json`。

## 资源匹配边界

公开 `FULL_PIPELINE.md` 仍列外部 MaskedMimic checkpoint 和 robot PPO checkpoint；`CrowdSim/README.md` 明确恢复 MaskedMimic，`CrowdSim/humanoid/README.md` 的 avoidance diagnostic 也明确使用 MaskedMimic。这些文字支持“公开 pipeline 包含 SFM/MaskedMimic 和机器人 PPO”，不构成作者定制 navigation AMP 权重已发布的证据。[固定 commit pipeline](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/FULL_PIPELINE.md)、[CrowdSim 说明](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/README.md)、[humanoid 诊断说明](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/humanoid/README.md)

完整树中有 4 份 `protomotions/agents/amp/` 通用源码；没有 `.ckpt/.pth/.pt` 路径，没有名称含 `diffus` 的路径。7 个 `.npy/.npz` 路径是 examples 下的少量动作示例/LFS pointer，并非已识别的作者 2,848 清单。在固定 HEAD 的 tracked 文本中检索完整词 `diffusion`、独立计数 `2848/2,848`、`Motion-MMD` 和八场景/200 s 协议特征，均未匹配；SVG 和 notebooks 未纳入此文本检索。路径或关键词不匹配本身不是“不存在”的证明，本结论同时依赖入口说明、发布索引和与旧审查相同的 blob 内容。[路径清单](targeted_path_inventory.json)、[实际查询与返回状态](targeted_text_queries.json)

当前本地实验应继续区分：公共 TRACE/PACER baseline、通用 ProtoMotions AMP、MaskedMimic、作者定制 Diffusion+AMP，以及本地重新训练的候选方法。当前本地选出的 2,848 段只能称本地子集；没有作者 IDs/裁剪区间对应证据，不能称已恢复作者原始 2,848 段。已有下载 manifest 和本地 CPU 证据记录的公共 PACER checkpoint SHA256 为 `bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615`；它不应改标为作者定制 AMP。本轮未重新读取或哈希该大型权重，新 prior 的训练血缘和方法行为仍由主任务独立核验。[已存下载记录](../../resource_completion_20261001/pacer_download_manifest.json)、[已存 CPU 记录](../public_pacer_zero_action_review_v1/concept_cpu_proof.json)

## 论文评估仍缺少的可重放信息

论文 IV-C 明确写出：2,848 条导航相关 AMASS 动作；tracking 失败阈值为每步 mean rigid-body position error 超过 0.5 m；导航比较使用相同 AMP controller，8 场景、每场 3–15 人、200 s；多样性评估采用 15 keypoints、30 Hz、root-centering、heading alignment、scale normalization 和 K=50/100。以下是将这些实验具体重放时仍需明确的输入和计算合同，属于本次核查后的缺项判断。[论文 IV-C 与 Table II/III](https://arxiv.org/html/2606.26265v1#S4.SS3)

| 评估项 | 已核查材料中未识别到的具体值/文件 |
|---|---|
| AMASS 与 tracking 集 | 每条源 clip ID、裁剪时间区间、标签来源与筛选规则；训练和评估集合的对应关系；同一源动作重复段处理、预处理版本与重定向映射。仅数量相同无法建立身份对应。 |
| Tracking rollout | 每条动作的初始 root/joint pose 与 velocity、reset 与失败后样本处理、参考时钟及仿真/控制采样关系；参与比较的 rigid-body 名称、位置/旋转坐标系；mean/max 的 body、time、clip 聚合顺序及失败片段是否纳入误差汇总。 |
| 八场景输入 | 精确场景 IDs、资产版本、occupancy/collision 配置；每场具体人数；开始位置、目标/waypoint、seed、任务更新及 reset 序列。当前公开 scene YAML 名称不能自动建立与论文八场景的对应。 |
| SFM 与 Diffusion 配对 | 固定 AMP checkpoint 与匹配推理配置身份；SFM 参数；diffusion checkpoint、采样/引导参数和更新时钟；两变体是否共享同一初始化、goal 序列与随机种子。 |
| Table II 指标 | success/fall/map collision/agent collision 的判定阈值、持续时间和终止优先级；百分比的分母与事件/episode/agent 计数方式；path error 的几何/时间对应、采样权重、失败及 reset 时刻处理；重复次数和原始逐场结果。200 s 不足以确定这些量。 |
| 15-keypoint 转换 | 完整 15 个关节名称/索引及 AMASS、SMPL、IsaacSim Animated People 的映射；重采样插值、heading、尺度的数值定义和无效帧处理。文中列举的身体部位不是完整跨系统映射文件。 |
| Table III 计算 | AMASS clustering 的特征、样本集、算法/seed 与中心；coverage 判定阈值和 entropy 归一化/分箱；`Dist.` 列的明确公式；Motion-MMD 的特征/时间窗、kernel、bandwidth、估计量与样本数量；GoTo baseline 的 IsaacSim/角色版本、命令轨迹与速度分布。 |

上述缺项不应被本地实现的默认值补写成作者协议。本地补充的评估应标为明确版本的复现合同或改进合同，并保存其实际配置、输入清单和逐条结果。

## 边界与访问情况

- 本轮没有重新审计 Isaac Sim/Isaac Lab 全部依赖，也没有验证权重可加载、训练质量或 GPU 性能。
- 未扫描第三方 fork 内容、未公开/未链接仓库、私有发布、GitHub wiki、其他个人仓库或其他搜索关键词；Hugging Face 未认证且仅限记录的名称/作者索引。
- 网页工具最初访问项目页返回 Internal Error；直接只读 HTTPS 后取得 HTTP 200 内容，且与固定网站 commit 的 `index.html` 相同，因此项目页最终核验成功。此临时工具失败不作为“资源缺失”证据。
- 所有本轮正文结论由 [summary.json](summary.json)、公开响应和固定 HEAD 文本快照支持；冻结的生产/runtime source 未修改。
