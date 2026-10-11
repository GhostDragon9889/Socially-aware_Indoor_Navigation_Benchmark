> **2026-10-11 迁移入口：** 当前默认为 IsaacLab 3.0 EA（包17.0.2）／IsaacSim 6.1.0.0，项目 `.venv` 指向独立 SDK overlay。请先阅读[迁移报告](ISAACLAB3_EA_SIM61_MIGRATION_20261011_ZH.md)。下列旧 SDK 版本、缺包诊断、训练/评估和视频结果是原日期的历史快照，不能当作新环境验收。新 SDK 依赖缺口已在 overlay 修复；本轮真实运行与剩余边界以迁移报告为准。

# GitHub 源码仓库（2026-10-10）

项目及工作区名称：`Socially-aware_Indoor_Navigation_Benchmark`。
Python 安装工具将该名称规范化为 `socially-aware-indoor-navigation-benchmark`；二者是同一发行包。
已有 `protomotions`、`CrowdSim` 导入名及控制台入口继续保留。

## 发布状态

已创建私有仓库 [GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark](https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark)，默认分支为 `main`。
2026-10-10 已通过 GitHub 插件发布源码，首个完整源码提交为
[`51d58f2`](https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark/commit/51d58f2e81ac74cc2e3b2a3701e01f88c294d332)。
插件回读确认 3,779 个文件及完整 Git tree SHA 与本地发布快照一致。
独立发布副本的 `main` 与远端对齐，准备阶段的本地提交保留在 `codex/local-source-snapshot`。
原项目保留其原 `origin`，另加 `benchmark` 指向本仓库；发布副本的 `origin` 指向本仓库。

## 发布内容

GitHub 提供核心算法、运行和评估入口、配置、导航地图、测试、环境锁、VS Code 配置、说明及许可。
首个源码发布未包含 AMASS/SMPL 私有数据、自训权重和外部场景。后续 `pedestrians_20261010` 私有 LFS 发布加入最新选用权重、训练状态/参数、EVT-EXP 机器人与四场景依赖，详见[资源说明](../ASSET_RELEASE.md)。原始动作、SMPL 参数、SDK 和视频仍独立准备。上游公开历史与资源引用已随合并纳入。
GitHub 中的结果索引和报告描述之前的本地实验，资源路径和校验值保持原记录，不能代表克隆后已经具备这些资源。

首个提交由明确的文件清单构建；清单保存在 `configs/publication/source_manifest.json`。
忽略规则阻止本地凭据目录、下载资源、缓存和生成产物进入后续普通 `git add`。
原始 Git 元数据保留在当前项目的 `.git/` 中，包括原提交、引用、索引和 LFS 对象。
源码发布使用独立的 `output/github_publication_20261010/source_checkout/Socially-aware_Indoor_Navigation_Benchmark/`。
未对当前分支或历史执行替换、重置、清理或强制推送。查看原历史：

```bash
git log -5 --oneline
```

## 克隆后的准备

1. 使用 Python 3.12。仅查看项目状态不需要 SDK：

   ```bash
   PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 python -m social_navigation status
   ```

2. 查看上游固定来源；需要时恢复缺失的参考仓库：

   ```bash
   python scripts/reference_repositories.py
   python scripts/reference_repositories.py --clone
   ```

   该脚本核对已有目录的来源与提交；不会覆盖不匹配的本地工作。
   固定提交和许可证位置见 `configs/reference_repositories.json`。
   获取参考源码不等于获得权重或授权数据。

3. 按 [环境说明](../env/isaaclab3_ea_sim61/README.md) 安装已固定的 IsaacLab/Isaac Sim 环境。
   环境快照应应用到 IsaacLab 工作区；将 `.venv` 链接到准备好的环境，并安装本项目。

4. 按 [数据集位置](DATASET_LOCATIONS_ZH.md)、[AMASS 准备](AMASS_PREPARATION_ZH.md)、
   [资源审查](NAVISAACLAB_RESOURCES_AND_REPRODUCTION_20261001_ZH.md) 准备获授权的资源。
   权重获取入口包括 `scripts/fetch_pacer_reference.py`；本地微调配方见
   [AMASS 微调](AMASS_FINETUNE_ZH.md) 与 [四场景说明](FOUR_SCENE_SPEED_20261010_ZH.md)。

5. 本地既有资源仍可通过 `ProtoMotions` 和原带空格目录的兼容链接访问。
   新机器需重新生成资源预检、SDK 来源校验和原生执行证据；本机封存的绝对路径、SDK 清理校验及实验快照不能直接作为新机器的运行证明。
   先执行 `preflight`，再逐个运行原生任务；缺少资源时保持失败状态，不绕过检查。

GitHub 的轻量 CPU 检查仅验证源码布局、命令入口，不运行 SDK、训练或视频质量验收。
本项目仍未证明完成论文全部协议或达到论文宣称指标。

## 上游完整合并与 LFS

原上游 1,017 个文件和 74 个提交现已合入，之前省略的 235 个文件已补齐。
此前 47 条 USD/USDA 记录中，8 条路径及 OID 匹配该上游提交，已恢复为原始 LFS 引用；另外 38 条路径不在上游提交中，1 条路径存在但 OID 不同。这 39 条记录的来源尚未核实，不能当作已恢复的上游资源。下载来源由 `.lfsconfig` 固定到原上游。
普通 `git clone` 安装了 Git LFS 时可能下载全部上游资源；按需下载可使用
`GIT_LFS_SKIP_SMUDGE=1` 克隆，再执行指定路径的 `git lfs pull`。
详见 [上游合并记录](UPSTREAM_MERGE_20261010_ZH.md)。

## 后续 GitHub 推送

合并后的项目根目录 `main` 跟踪 `benchmark/main`。源码及上游引用可使用 `GIT_LFS_SKIP_PUSH=1 git push benchmark main`；含本项目新资源的提交必须先将对应 LFS 实体上传到 `benchmark` 并核验成功，再推送 Git 指针。私有资源的下载与完整性验证见[资源说明](../ASSET_RELEASE.md)，原上游 5.3 GB 继续按引用下载。

本次发布已通过插件完成，不依赖本机 `gh` 登录。以下为以后使用命令行发布的可选方式。

项目根目录已与私有仓库的 `main` 对齐，保留原 `.git` 和上游 `origin`。
下方工具从项目根目录推送到 `benchmark`，不会选择旧的发布副本。
完成 `gh auth login --hostname github.com --git-protocol https --web` 后，可运行：

```bash
python scripts/publish_github.py
```

工具只创建当前账号下指定名称的私有仓库、推送 `main`，并核对远端提交和私有状态。
工具只允许正常快进推送；若远端 `main` 已分叉则停止。其他分支不影响主分支推送，LFS 实体上传由单独的资源发布流程处理。实际成功记录写到本地
`output/github_publication_20261010/github_result.json`；缺少该记录不代表已经完成发布。
