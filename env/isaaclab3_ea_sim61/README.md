# IsaacLab 3.0 EA / Isaac Sim 6.1 运行环境

默认配置固定 Linux x86_64、Python3.12.13、Lab 源提交 `ae37b028ea415c91ea2bc32609efcd759ed2b974`（产品3.0.0，源码包17.0.2）、Sim6.1.0.0、Torch2.11.0+cu128、Warp1.16.0。`profile.json` 校验 SDK manifest、uv.lock 和附加包锁的 SHA256。

推荐使用外部 SDK 的 Python：

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root /absolute/path/to/IsaacLab
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
```

附加包下载到 `.runtime/external_sdk_packages`，不创建项目 Python 虚拟环境、不改动外部 SDK、不重复安装仿真和数值库。可用 `--python` 指定已安装 Lab 的 Python 或二进制 Sim launcher。编辑器配置自动生成在 `.runtime/external_sdk.code-workspace`。完整说明见[外部 SDK](../../docs/guides/external_sdk.zh-CN.md)。

新机器执行 `python3 scripts/bootstrap_reproduction.py install`，在 `.runtime/IsaacLab61/.venv` 完整安装受控 SDK，随后使用同一项目入口。已有 SDK 受保护，请用 `configure` 复用。附加包锁更新时添加 `--rebuild-addons`，旧包目录保留。启动新 Kit 前自行阅读并同意 NVIDIA EULA。

旧 `prepare-runtime --lab-root ...` 仍提供 `.runtime/isaaclab3_ea_sim61/.venv` 覆盖虚拟环境，供已有部署使用。它与直接外部解释器方式是两种不同选择。历史收据记录历史包锁和实测范围，不随当前锁更新改写。Sim6.1 不加载旧 Sim6.0 原生覆盖插件；旧 profile 及历史证据保留原身份。

本轮完整安装、外部解释器、克隆、预训练权重与 GPU 结果见[核验报告](../../docs/EXTERNAL_SDK_VERIFICATION_20261011_ZH.md)。

## English

Use `bootstrap_reproduction.py configure --lab-root /path/to/IsaacLab` to bind an external SDK. `run_external_python.py` launches that SDK interpreter directly; locked project packages live in `.runtime/external_sdk_packages`. Existing SDK files and project `.venv` are preserved. Use `bootstrap_reproduction.py check` for provider and requirement verification.

`bootstrap_reproduction.py install` performs a complete pinned SDK installation inside `.runtime/IsaacLab61`, then binds the project. `--python` selects another installed SDK interpreter; binary Sim launcher support requires Lab and package metadata in that environment and has not been exercised on this pip-only machine. Read and accept NVIDIA EULA before first Kit launch. The old overlay-venv command remains available for existing deployments.
