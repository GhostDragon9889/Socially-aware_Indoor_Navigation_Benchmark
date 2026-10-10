# NavIsaacLab 论文行人资源的公开发布核查

核查日期：2026-10-01。只读访问作者官网、官方 GitHub/API 和公开 HF 索引；本代理未下载模型正文、未改变环境或业务代码、未启动 Kit。原始公开接口状态和小型官方源码已保存在本目录。

## 结论

在下述**实际核验成功的公开源**中，尚未找到可确认对应论文 IV-A 的定制 trajectory diffusion 实现/配置/权重、IV-B 的 SMPL navigation AMP actor/critic/discriminator 与匹配配置，或 IV-C 的 **2,848 条 AMASS 动作准确清单**。这是有范围的查询结果，不能推断作者从未发布；网络失败、私有仓库、未公开链接和未扫描分支均不算不存在。

论文描述的是 history/neighbors/map 条件 U-Net 轨迹生成，再由目标条件 SMPL AMP 跟踪，动作先验来自特定 AMASS 子集。通用 AMP、MaskedMimic 和 TRACE/PACER 公共 baseline 的下载成功，不能单独证明严格论文复现。[论文 v1](https://arxiv.org/abs/2606.26265)

## 1. 官方仓库与本地参考对应

作者项目列表和项目页面的 Code 链接均指向 `yh83305/NavIsaaclab2.0`；本地 `/project/EXP/NavIsaaclab2.0` 的 origin 与此一致，HEAD `d742f0c5b421bbc281e2112e7e50427e0336c16a`，与公开 `main` 匹配。[作者项目列表](https://broln7.github.io/projects/)、[项目页](https://broln7.github.io/NavIsaacLab-web/)、[官方仓库](https://github.com/yh83305/NavIsaaclab2.0)

| 核验源 | 实际范围与结果 |
|---|---|
| `yh83305/NavIsaaclab2.0` | API branches 仅 `main`；9 commits；git heads/tags 无 tag；releases 与全部 issues 空；完整 tree 820 项，无截断。公开入口为 SFM + MaskedMimic；内嵌通用 ProtoMotions AMP 代码存在。 |
| `BRoln7/NavIsaaclab2.0` | 作者 fork 的 `main` 同一 commit；完整 tree 同为 820 项，releases 空；git heads/tags 仅 `main`。 |
| `leele-bing/ProtoMotions` | 第一作者 fork，`main=8702e7919d33d7f264682a28c5c120934c8cc267`；仅 main、releases 空，完整 tree 928 项。README 明确 SFM + MaskedMimic；公开模型目录是 tracking/MaskedMimic。 |
| `BRoln7/NavIsaacLab-web` | master 完整 tree 52 项：网页、图片、视频、样例 PDF；未包含模型包或 2,848 清单。当前 live 页面链接已独立核验；部分 GitHub HTML 是旧缓存。 |
| `BRoln7/IsaacLab-socnav` | main 完整 tree 1,408 项，21 个分支已列出；main 未识别定制 SMPL/diffusion 资源。其余 20 个上游 feature 分支未做内容扫描。 |
| `BRoln7/IsaacLab_AMP_rl-games` | 仅 main `0207003d8117f9a283a29aa88c4514e414845c0d`，amp_rlg 源码子树 58 项；是通用 humanoid AMP。release/issues API 未成功，不能标为空。 |

主要证据：[公开 main 分支](https://api.github.com/repos/yh83305/NavIsaaclab2.0/branches)、[发布记录](https://api.github.com/repos/yh83305/NavIsaaclab2.0/releases?per_page=10)、[全部 issues](https://api.github.com/repos/yh83305/NavIsaaclab2.0/issues?state=all)、[官方 pipeline 说明](https://github.com/yh83305/NavIsaaclab2.0/blob/main/FULL_PIPELINE.md)、[第一作者 CrowdSim 说明](https://github.com/leele-bing/ProtoMotions/tree/main/CrowdSim)、[作者网页完整树](https://api.github.com/repos/BRoln7/NavIsaacLab-web/git/trees/master?recursive=1)。

GitHub repository search `NavIsaacLab fork:true` 返回 4 个仓库，其中第三方 `GhostDragon9889` fork 不属于作者确认发布源，未作为论文权重依据。HF 成功索引：models 搜索 NavIsaacLab/NavIsaac、datasets 搜索 navisaaclab、spaces 搜索 NavIsaacLab 均空；BRoln7/yh83305 models 和 leele-bing datasets 作者索引也空。leele-bing models、BRoln7 datasets 请求失败，仅登记访问限制。[GitHub repository search](https://api.github.com/search/repositories?q=NavIsaacLab+fork:true&per_page=100)、[HF models](https://huggingface.co/api/models?search=NavIsaacLab&limit=100)、[HF datasets](https://huggingface.co/api/datasets?search=navisaaclab)、[HF spaces](https://huggingface.co/api/spaces?search=NavIsaacLab&limit=100)

## 2. 确实公开、可独立保管的相关资源

### TRACE：论文引用的轨迹 diffusion 基线

官方当前 main：`594b2af0824c5853e1a1884e167cd5ed097dd81c`。作者提供的 [TRACE 权重 zip](https://www.dropbox.com/scl/fi/vq89o8nmjibyo70nn3ikr/trace.zip?dl=0&rlkey=xvzqpbf6cn76ryifdoa1pdlvl&st=st33d7yy) 提取位置为 `ckpt/trace/`，变体 `orca_mixed/`、`nusc_eth_ucy_mixed/`，使用 `iter40000` key；模型许可证为 CC-BY-NC-SA-4.0，原始开发栈 Python3.8/Torch1.10.2/CUDA11.1。[官方 README](https://github.com/nv-tlabs/trace/blob/main/README.md)

源码定位规则是递归选择唯一含 `iter40000` 的 `*.ckpt`，同时读取目录或其父目录 `config.json`。父代理已完成官方下载并逐项验证 zip，包大小 **1,024,117,268 B**，SHA256 `8178b7b1b03f93eaa87ddba91399fc7d1b6462f99749ac61a5d424923086aa79`；实际还包括第三个 `nusc_only` 变体。完整文件位置为 `data/pretrained_models/paper_references/trace/ckpt/trace/{orca_mixed,nusc_eth_ucy_mixed,nusc_only}/iter40000.ckpt`，分别 **376,736,841 / 376,987,721 / 376,987,721 B**，每份匹配 `config.json`，并保留 LICENSE。此处为读取父代理 manifest 确认，研究代理未读取权重正文。见 [已下载资源清单](trace_download_manifest.json)。[checkpoint 查找函数](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/experiment_utils.py)

训练配置已明确为 `configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json` 与 `diffuser_mixed_nusc_mlp_hist_grid_map_drop_10.json`。这些预训练对象是原 TRACE 数据/任务，需要当前 CrowdSim 的 history、map、邻居、目标与时间采样适配；尚无作者声明它就是 NavIsaacLab 定制 diffusion。[ORCA 训练配置](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json)

### PACER：论文引用的 SMPL AMP 基线

官方当前 main：`baeb6241f03fc9d78370a64e3c2dcef9d830d97f`。正式脚本提供以下链接；两个 Drive 公开预览均确认原文件名 `Humanoid.pth`，文件正文/尺寸由父代理另行下载核验，本代理没有读取。[官方下载脚本](https://github.com/nv-tlabs/pacer/blob/main/download_data.sh)

| 变体 | 正式下载 | 项目要求的目标位置 | 配置 |
|---|---|---|---|
| no_shape | [公开权重](https://drive.google.com/uc?id=1srmXSH8c-4Recy1g4jgWCNzZhDQ7mWfH) | `output/release/pacer_no_shape/Humanoid.pth` | `pacer/data/cfg/pacer_no_shape.yaml` |
| shape | [公开权重](https://drive.google.com/uc?id=1CF8mA9L3cAAOdDk6qtwDnAOTrQI8mien) | `output/release/pacer/Humanoid.pth` | `pacer/data/cfg/pacer.yaml` |

no_shape 匹配任务 `HumanoidPedestrianTerrain` 与训练网络 `pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml`，原栈 Python3.8/PyTorch CUDA11.6/IsaacGym，SMPL 参数需单独获得。仓库 LICENSE 为 CC-BY-NC-SA-4.0。[PACER README](https://github.com/nv-tlabs/pacer)、[LICENSE](https://github.com/nv-tlabs/pacer/blob/main/LICENSE)

#### no_shape 预期输入/动作合同（由源码推导，待正文核对）

| 项 | 原源码合同 |
|---|---|
| Self state | 24 个 SMPL rigid bodies 的 max coords，去 root translation/height：357 维；无 shape/weight 特征。 |
| Task state | 10 个间隔0.5s的局部轨迹样点：20维；head 朝向下4m×4m、32×32高度采样：1,024维。 |
| Actor raw observation | 357 + 20 + 1,024 = **1,401维**，需要原 running mean/std。 |
| Task encoder / actor | task MLP `[512,256]`，拼 self 后 **613维**；actor/critic `[2048,1024]`；连续输出 **69维**。 |
| 控制 | 23 body ×3 joint DOF，30Hz PD；offset+scale 映射，knee scale5，hand/toe targets 特殊归零。`has_dof_subset` 不减少 actor 输出。 |
| AMP / checkpoint | 10帧历史；discriminator `[1024,512]`。预期 model 层名前缀 `a2c_network`，子层 actor_mlp/critic_mlp/_task_mlp/mu/sigma/_disc_mlp；这些名称仍需与实际 checkpoint tensor keys 对照。 |

证据：[环境配置](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/pacer_no_shape.yaml)、[self/PD 定义](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid.py)、[task/height 观测](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py)、[分离任务编码器](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_network_sept_builder.py)、[网络配置](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml)。

### 作者另一个 AMP fork：公开权重存在，任务不匹配

其 `amp_env_cfg.py` 明确 **105维观测、28维动作、2帧AMP历史（210维）、AMP_HUMANOID/amp_humanoid2、amp_humanoid_run.npy**，README 测试栈为 Sim4.5/Lab2.0。不是论文 SMPL 导航合同。[作者配置](https://github.com/BRoln7/IsaacLab_AMP_rl-games/blob/0207003d8117f9a283a29aa88c4514e414845c0d/amp_rlg/tasks/amp/amp_env_cfg.py)、[README](https://raw.githubusercontent.com/BRoln7/IsaacLab_AMP_rl-games/main/README.md)

| 公开文件 | API声明尺寸 | 正式路径 |
|---|---:|---|
| run.pth | 24,308,314 B | [assets/pretrained/run.pth](https://raw.githubusercontent.com/BRoln7/IsaacLab_AMP_rl-games/main/assets/pretrained/run.pth) |
| jog.pth | 24,308,314 B | [assets/pretrained/jog.pth](https://raw.githubusercontent.com/BRoln7/IsaacLab_AMP_rl-games/main/assets/pretrained/jog.pth) |
| walk.pth | 24,308,056 B | [assets/pretrained/walk.pth](https://raw.githubusercontent.com/BRoln7/IsaacLab_AMP_rl-games/main/assets/pretrained/walk.pth) |

尺寸来自目录 API，不是读取文件正文。[公开权重目录元数据](https://api.github.com/repos/BRoln7/IsaacLab_AMP_rl-games/contents/assets/pretrained)

### 第一作者 ProtoMotions fork 的 SMPL 模型

公开 Git LFS 指针确认 MaskedMimic SMPL `last.ckpt` payload **229,160,269 B**，SHA256 `3147039996be020447d50738da9360878ca76561995b1442b8c22730e319231d`；SMPL-terrain tracking `last.ckpt` **132,661,223 B**，SHA256 `ff5ffd0dbaea9532c5d44ebdc43e3c0aa5e8c5671d5603bd862a1d9a07c0174b`。Git tree 中134B是指针大小，不能当模型体积。后续用户授权下载后，两份权重及各自五个 sidecar 已全部下载并按固定版本指针校验，见 `navisaaclab_public_download_manifest.json` 与 `navisaaclab_public_download_summary.md`。[MaskedMimic 指针路径](https://github.com/leele-bing/ProtoMotions/blob/8702e7919d33d7f264682a28c5c120934c8cc267/data/pretrained_models/masked_mimic/smpl/last.ckpt)、[terrain tracker 指针路径](https://github.com/leele-bing/ProtoMotions/blob/8702e7919d33d7f264682a28c5c120934c8cc267/data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt)

当前 MaskedMimic 配置是 latent64 的条件 VAE/Transformer prior，输入有 masked targets、mask/time、history 和 previous actions。与 PACER 的 raw1,401维 AMP 网络不同；即使两者动作都是69维，也不能直接重命名/装载替换。[作者 MM 配置](https://github.com/leele-bing/ProtoMotions/blob/8702e7919d33d7f264682a28c5c120934c8cc267/data/pretrained_models/masked_mimic/smpl/experiment_config.py)

## 3. 严格论文复现仍需的明确资源

1. **IV-A**：作者适配后的 diffusion 源码、checkpoint 和完整训练/推理配置，包括输入归一化、历史/邻居/地图编码、denoise schedule、目标引导与5秒输出采样协议。
2. **IV-B**：作者 SMPL 导航 AMP actor/critic/discriminator、normalizer，以及完全匹配的 obs/body/DOF/PD/奖励/时间协议；通用 AMP 源码不足以确认这些合同。
3. **IV-C**：2,848 条源 clip ID、截取区间、过滤参数、采样权重和预处理版本。PACER 的 sample standing motion、通用 AMASS 转换/随机 subset 脚本，不能证明得到论文准确子集。

第一作者曾在 PACER #2 讨论将其迁移到 IsaacSim，但可见页面只有问题，没有公布兼容权重或迁移结论。[官方 issue](https://github.com/nv-tlabs/pacer/issues/2)

## 4. 验证边界与证据文件

- API 的空列表仅在 HTTP200 且正文确为 `[]` 时登记；timeout/EOF/cache miss 全部单独记录。
- 未执行 authenticated code search，未访问私有/受限仓库，未检查未链接云盘目录，未联系作者。
- 原只读研究保存公开小型源码、配置、许可证与索引；后续用户授权的独立下载补齐作者公开 MM/tracker 及 sidecar。本代理只验证文件大小、Git blob 与 SHA256，没有对 checkpoint 做 torch tensor inspection 或仿真，未执行历史配置。父代理的 TRACE/PACER 下载与验证结果使用其独立资源清单。
- 结构化结果：`paper_publication_sources.json`；逐次接口证据：该 JSON 的 evidence_files/endpoint_attempts；小型原始源码：`official_sources/`，带 SHA256 的 `retrieved_sources_manifest.json`。
