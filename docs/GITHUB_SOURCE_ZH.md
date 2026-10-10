# GitHub 源码仓库（2026-10-10）

项目及工作区名称：`Socially-aware_Indoor_Navigation_Benchmark`。
Python 安装工具将该名称规范化为 `socially-aware-indoor-navigation-benchmark`；二者是同一发行包。
已有 `protomotions`、`CrowdSim` 导入名及控制台入口继续保留。

## 发布内容

GitHub 提供核心算法、运行和评估入口、配置、导航地图、测试、环境锁、VS Code 配置、说明及许可。
AMASS/SMPL 数据与人体模型、训练权重、外部场景、SDK、参考仓库、视频、原始实验数据和旧 Git/LFS 历史保留在本地。
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

3. 按 [环境说明](../env/isaaclab3_sim6/README.md) 安装已固定的 IsaacLab/Isaac Sim 环境。
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

## 上游 LFS 源资产

发布检查发现原工作区有 47 个尚未实体化的上游 USD/USDA LFS 指针，约 195 MB 的真实内容尚未齐备。
本仓库记录其来源、大小和 SHA256，省略不能加载的指针文件；这不改变本地已验证的 SMPL 行人运行资源。
这些上游通用机器人/场景资产与 AMASS/SMPL 的获授权资源是分别准备的。

需要上游机器人示例时，先在独立的 ProtoMotions 上游克隆中获取 LFS 实体，
再执行 `python scripts/restore_source_assets.py --from-upstream /path/to/ProtoMotions`。
脚本逐项核对 SHA256 和大小，只写缺失文件或替换原 LFS 指针，不覆盖不同的现有资产。
确切资源列表及上游提交见 `configs/publication/external_source_assets.json`。

## GitHub 推送

独立发布目录已准备 `main` 源码提交，并保留 `upstream` 远程。
当前项目的原 `.git`、分支和远程保持；从项目根目录运行下方工具会使用该发布副本。
完成 `gh auth login --hostname github.com --git-protocol https --web` 后，可运行：

```bash
python scripts/publish_github.py
```

工具只创建当前账号下指定名称的私有仓库、推送 `main`，并核对远端提交和私有状态。
若同名仓库已有不同提交，则停止，不覆盖其历史。实际成功记录写到本地
`output/github_publication_20261010/github_result.json`；缺少该记录不代表已经完成发布。
