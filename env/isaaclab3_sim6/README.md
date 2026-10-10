> 2026-10-10：editable 项目改名为 `socially-aware-indoor-navigation-benchmark`，路径迁至
> `../Socially-aware_Indoor_Navigation_Benchmark`。manifest/lock 已同步；273 个锁定包的版本、
> 外部源码与 wheel 哈希保持不变。历史安装记录中的旧项目名保留作追溯。

# CrowdSim 的 IsaacLab 3 / Isaac Sim 6 环境

这是 Linux x86_64、Python 3.12 的专用环境配置，固定 IsaacLab 源码提交
`4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`、Isaac Sim `6.0.0.1`、Torch
`2.11.0+cu128`。源码 `VERSION` 为 `3.0.0`，该提交的 Python 包版本为
`isaaclab==12.0.0`；产品版本和包版本不可互相替代。

该提交属于 3.0 开发线，并不是 release tag。官方
[3.0 beta2](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2)
支持 Sim 6.0；当前 [3.0 EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
要求 Sim 6.1。为匹配本项目适配和用户要求，本配置保留精确源码提交，不自动升级到 EA。

`pyproject.toml` 和 `uv.lock` 是应应用到 **IsaacLab 源码工作区根目录**的完整快照。
请保留两个仓库的源码；它们将 ProtoMotions 作为 editable 路径，不能替代项目源码和
AMASS／SMPL／场景等资源。不要在本目录直接运行 `uv sync`，其中的 `source/` 路径
相对于 IsaacLab 工作区。

已装好指定 Isaac Sim、Python 3.12 `.venv` 后，应用记录的补充依赖和锁：

```bash
cd '/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark'

../IsaacLab-Protomotions/.venv/bin/python scripts/isaaclab3_environment.py apply \
  --lab-root ../IsaacLab-Protomotions --sync
```

命令验证 IsaacLab 提交和现有 Sim 包版本，备份目标目录原有 manifest/lock，再执行
锁定同步。它仅管理已存在的环境，不启动 Kit，不接受 EULA，不创建另一套 Isaac Sim。
只想写入配置时省略 `--sync`；此前配置保存在目标工作区
`.crowdsim-environment-backups/<时间戳>/`。

检查环境：

```bash
../IsaacLab-Protomotions/.venv/bin/python -I scripts/isaaclab3_environment.py check

cd ../IsaacLab-Protomotions
uv sync --offline --locked --extra isaacsim --dry-run
uv pip check --python .venv/bin/python
```

开发测试组 `crowdsim-tests` 默认安装，锁定 `pytest==9.0.3`；可直接使用目标解释器
`python -m pytest`。业务测试应从 ProtoMotions 仓库运行，按需设置源码路径。

## USD 提供者

独立 Python 只使用 `usd-exchange==3.0.0`，其自带的 `pxr` 为 **USD 26.8**。
工作区移除 `usd-core` 请求，并通过
[`exclude-dependencies`](https://docs.astral.sh/uv/reference/settings/#exclude-dependencies)
阻止它由传递依赖再次进入。SDK 官方
[包说明](https://pypi.org/project/usd-exchange/3.0.0/)
明确禁止两个包共装，因为它们覆盖相同文件。

从双提供者环境迁移时，同步使用 `--reinstall-package usd-exchange`：卸载 usd-core
会删除共享文件，必须随后恢复唯一提供者的文件，不能仅卸载 usd-core。

`check` 在新独立进程中验证核心版本、唯一提供者、234 个相关 RECORD 哈希、USD/Sdf
文件读写及两个转换器导入。**它不是 Kit 进程的版本断言。** 本机真实 Kit 启动已加载
自身的 **USD 25.11**，因此不要在启动 Kit 后断言 `Usd.GetVersion()==(0,26,8)`。
Kit 物理闭环和转换器的运行验收仍应独立执行。

## 明确保留的覆盖及回滚证据

当前工作区采用固定 Newton Git 提交和 schemas 覆盖，并用 coverage 7.6.1 修复 Numba
的 Tracer 接口。`uv pip check` 仍报告三项元数据差异：Newton 1.4.0.dev0 / schemas 0.5.0 /
coverage 7.6.1 与 Sim 声明不同。这些覆盖写入 manifest 和 lock；导入成功不能替代
真实仿真验收。

本次修复前的完整 manifest、lock、计划和修复后验证位于
`docs/migration_20261001/environment/`。旧快照记录了已证实的双 USD 冲突，仅供对比与
恢复现场；恢复旧配置会重新引入该冲突，不能将它作为正常运行配置。

锁定 wheel 均记录 hash，Newton 固定 commit；Python 和本项目 editable 源码仍需一并
保留。此前 USD 修复阶段只移除 usd-core、重装同版 usd-exchange，并增加
pytest、iniconfig、pluggy 三个纯 Python 测试工具，未升级核心包。

后续全面修正增加 ProtoMotions 的 `crowd` extra，并离线刷新锁中 editable 项目的
依赖元数据。273个锁定包的版本完全不变；实际 `uv sync --offline --locked --extra isaacsim`
仅重建同版 `protomotions==3.1.0` 的 editable 安装元数据，未重装或升级 SDK/Torch。
同步后再次 dry-run 无待变更、独立 USD 检查通过，证据见
[`environment_metadata_sync.json`](../../docs/completion_20261001/environment_metadata_sync.json)。
