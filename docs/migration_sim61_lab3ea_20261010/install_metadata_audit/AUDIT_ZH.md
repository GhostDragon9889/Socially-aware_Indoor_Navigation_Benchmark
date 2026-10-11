# 安装元数据、隔离运行环境与默认文档审查

审查连续日期：2026-10-10 至 2026-10-11。没有提交、推送或修改两个 SDK。

## 最终源码身份

根 `pyproject.toml` SHA256：`69f89568881108ce518ac8d8c2a9d880684e18a97f9594712714bb672708b750`。

新 SDK：IsaacLab v3.0.0-EA / package 17.0.2 / source ae37b028ea415c91ea2bc32609efcd759ed2b974，Isaac Sim 6.1.0.0，Python 3.12.13。

旧 SDK profile 的 manifest、uv.lock 与 profile 内原 SHA 仍一致；新 SDK 的对应文件也未因根依赖修正而改变。运行环境 receipt 对根项目 manifest 的身份校验是严格的，因此需由主任务在最终源码稳定后重新构建并校验，不能手工改 SHA 来伪造一致。

## 发现和处理的实际问题

1. 根 extra `isaaclab` 原本只声明未发布于 PyPI 的 `isaaclab==17.0.2`。真实索引解析失败。增加两个 extra 对应的官方 Git 子目录与固定提交，不把 SDK 完整依赖误认为单个 Python 包。
2. 旧 `isaaclab6` 未加入所有后端的互斥规则，补齐六种后端全部 15 对冲突。
3. `requires-python>=3.12` 导致 uv 对未支持的未来 Python 3.14 分支解析失败。明确当前支持 `>=3.12,<3.13`，uv 平台限定已验证的 Linux x86_64。
4. Sim 6.1 间接依赖 `tinyobjloader==2.0.0rc13` 在默认预发布策略下被拒绝。根 `isaaclab` extra 明确声明此必需版本，未启用全局预发布。
5. Genesis 要求 MoviePy 2，而核心范围原来 `<2`。现有两个视频写入入口已有 1/2 接口兼容分支，核心允许 `<3`；当前独立 SDK overlay 仍锁定 MoviePy 1.0.3。
6. Genesis 旧 WandB 0.15.12 通过 pathtools 构建调用 Python 3.12 已删除的 `imp`，真实 dry-run 失败。该 extra 改为当前已验证的 WandB 0.23.0。
7. 历史 Newton 1.0 extra 使用 `[examples]`，可能引入额外 USD 提供者。改为 headless `[sim]`，并说明它与 SDK 内置 Newton 1.5.2 / Warp 1.16 的独立环境边界。
8. 原 bootstrap 可把冻结旧 SDK manifest 中的 `project[isaaclab]` 重新解释为新版 extra；现在旧 profile fresh install / apply 提前明确拒绝，仅复用匹配的已安装旧 SDK。
9. 原新 profile `install` 可能覆盖已存在 SDK，现提前拒绝并指引 `prepare-runtime`，现有 SDK 字节保持不变。

## 真实验证

- `uv_portable_platform_result.json`：官方 Git 双固定提交、默认预发布策略、六种互斥后端 universal `uv lock --dry-run` 实际 exit 0；命令额外指定 NVIDIA 官方索引。没有安装模拟器或宣称这些额外后端已通过物理测试。
- `uv_actual_source_final_result.json`：实际已安装 SDK 源元数据作为本地路径的独立初步 dry-run 实际 exit 0；该运行使用 allow 预发布策略，后续便携来源验证已在默认策略通过。
- `install_guards_final.xml`：27 passed，8 项实际 CLI/metadata/原 SHA 契约及 19 项独立运行环境测试。
- CLI 实际验证：旧 fresh install / apply 拒绝且目标目录不存在；现有 SDK 安装拒绝且所有 sentinel 字节保持；引号、空格、Unicode 路径可用。
- `moviepy2/actual_writer_result.json`：隔离 MoviePy 2.2.1 环境调用实际 CrowdSim 写视频函数，H264 12 帧、1 秒，ffprobe 帧数及 ffmpeg 完整解码通过。仅证明该写视频 API，不代表 Genesis 动力学或将 MoviePy 2 安装进新 SDK。
- 先前 overlay 32 精确包、116 default/core requirements 和 169 USD RECORD 项通过；dm-control 1.0.44 对 MuJoCo 3.11 的实际 CPU stepping/contact/reset/copy 通过。

## 修改文件范围

项目隔离运行环境：`src/social_navigation/runtime_environment.py`、`env/isaaclab3_ea_sim61/requirements.overlay.lock`、`runtime_constraints_receipt.txt`、`tests/test_runtime_environment_overlay.py`。

最终安装边界：根 `pyproject.toml`、`scripts/bootstrap_reproduction.py`、`scripts/isaaclab3_environment.py`、`tests/test_install_profile_contracts.py`。

默认文档：新 profile README、中英四种 guides、根中英 README、CLONE_REPRODUCTION、FULL_PIPELINE、默认 Sphinx index/install/migration 入口、`requirements_isaaclab.txt` 和新 machine-readable dependency inventory。

没有全局替换 changelog 或历史视频、配置、锁和实验记录。新 SDK 的长程场景、视频和论文指标由主任务独立采集，旧 Sim 6.0 证据保留其环境身份。
