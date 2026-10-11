# 外部 SDK 与项目内完整安装

项目直接使用 IsaacLab3.0-EA/Isaac Sim6.1 的解释器。项目代码及附加 Python 包属于本仓库；SDK、USD、数据集和权重按配置或资源清单提供。现有 SDK 不会被配置命令改写。

## 方式一：配置外部 SDK

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root /project/EXP/IsaacLab
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
```

默认解释器为所选 SDK 的 `.venv/bin/python`。如果使用另一 Python3.12 环境或二进制 Sim，请加 `--python /absolute/path/to/python` 或 `--python /absolute/path/to/isaac-sim/python.sh`；该解释器必须已经安装匹配的 IsaacLab 源码包及 Sim6.1 元数据/依赖。配置不会把纯 Sim 安装自动升级为完整 Lab 环境；缺少包会在启动 Kit 前报错。二进制 `python.sh` 路径可配置，当前机器只有 pip SDK，尚未实测二进制发行版。

配置文件为 `.runtime/external_sdk.json`。附加包位于 `.runtime/external_sdk_packages`，按 `requirements.overlay.lock` 精确下载，禁止另装 Torch、NumPy、Warp、Isaac 或 USD 提供者。运行入口启动所选 SDK Python；其 `sys.executable` 和子进程均沿用该解释器。项目 `.venv` 可存在但无需创建。不要直接运行会重新解析 SDK 的 `uv run`，用上述入口运行项目。

解释器身份检查：

```bash
python3 scripts/run_external_python.py -c 'import sys; print(sys.executable)'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 scripts/run_external_python.py -m pytest -q
python3 scripts/run_external_python.py scripts/verify_pretrained_models.py --output output/pretrained_new.json
```

锁变更时，运行 `configure --lab-root ... --rebuild-addons`。新包目录先完成验证，旧目录保留为备份；SDK 不变。配置生成 `.runtime/external_sdk.code-workspace` 与环境文件。在 VS Code 中打开该 workspace，即可选用外部解释器、项目附加包与新的任务入口。原 `.vscode` 任务通过统一入口启动，不依赖 `.venv`。二进制发行版的调试仍须按其官方环境设置执行。

## 方式二：在项目内完整安装

Linux x86_64 主机安装 Git 和 uv 后运行：

```bash
python3 scripts/bootstrap_reproduction.py install
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
```

安装器克隆固定 Lab 提交、安装 Python3.12.13，并使用仓库内校验过 SHA256 的 SDK manifest/uv.lock 安装 Sim6.1.0.0 及扩展缓存；随后配置项目包。SDK 和其虚拟环境位于 `.runtime/IsaacLab61`。可加 `--lab-root .runtime/another_sdk` 新建另一个安装位置。已安装的 SDK 会被保护，复用请运行 `configure`。下载可使用本机 uv 缓存，磁盘仍需容纳完整 SDK。

首次启动新 Kit 前必须阅读并同意 [NVIDIA EULA](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html#running-isaac-sim)；同意后在运行终端设置 `export OMNI_KIT_ACCEPT_EULA=YES`。安装器不替用户接受协议。Python3.12、Linux GLIBC2.35+、GPU 和驱动要求见 [Sim6.1 安装](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)及 [Lab3EA 官方安装](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html)。官方也支持独立项目复用 Lab；这里的精确源提交用于保留已验证的源码身份，包17.0.2来自该源提交，不与官方分发 wheel 的版本号混淆。

## 克隆、资源与冻结输入

Git 仅保存可执行位，不保存只读权限。配置会先逐个验证 `configs/reproduction/frozen_inputs.json` 的大小及 SHA256，然后恢复冻结输入的只读位；任一内容不匹配会停止，不重写内容。`prepare-frozen` 可单独执行。换目录需重新 `configure`，不用复制旧 `.runtime`。

```bash
python3 scripts/run_external_python.py scripts/manage_release_assets.py fetch
python3 scripts/run_external_python.py scripts/manage_release_assets.py prepare --check-usd
```

已有资源可从本地发布缓存恢复；在线下载仍需要对应资源服务的权限。AMASS/SMPL 按许可另行获取，允许通过 `data/external_resources.json` 的资源 ID 绑定外部容器。源码、历史记录、模型 SHA 和场景几何不因换目录改写。详细步骤见 [克隆复现](../../CLONE_REPRODUCTION.md)。旧 `prepare-runtime` 覆盖虚拟环境仍保留，供已有部署使用；推荐新配置入口。

## 可选 ORCA 数据生成

需要 ORCA 专家数据时，安装系统 CMake 和 C++ 编译器后执行：

```bash
python3 scripts/run_external_python.py scripts/install_rvo2_external.py
git clone https://github.com/nv-tlabs/trace.git .runtime/dependencies/TRACE
git -C .runtime/dependencies/TRACE checkout --detach 594b2af0824c5853e1a1884e167cd5ed097dd81c
python3 scripts/run_external_python.py scripts/generate_orca_diffusion_data.py \
  --trace-source .runtime/dependencies/TRACE \
  --rvo2-source .runtime/dependencies/Python-RVO2 --output output/new_orca_corpus
```

构建使用所配置的 SDK Python，固定 RVO2 源提交；Cython、setuptools 仅进入临时构建目录。扩展及其 SHA256 收据保存到 `.runtime/rvo2_native`，运行入口验证后加载。不会向 SDK 安装构建包或扩展。已存在的扩展会被保留，脚本拒绝覆盖。生成器要求固定提交的独立 TRACE/RVO2 源码检出；上述命令将它们下载到项目依赖目录，也可显式指定已有检出。构建后重新运行 `configure` 可更新 VS Code 的可选扩展路径。

## 验证边界

模型文件清点覆盖指定目录全部文件，权重检查使用实际文件；当前模型严格加载与物理运行分别记录。历史 NavIsaacLab 分支的两组模型仍按 [注册表](../../data/pretrained_models/upstream_reference_registry.json)拒绝进入当前分支，不能因为张量可读便称其可运行。全项目行覆盖率、政策质量、多 GPU 和二进制 Sim 发行版测试需要独立证据，清点100%不能替代执行覆盖100%。本轮报告见 [外部 SDK 核验](../EXTERNAL_SDK_VERIFICATION_20261011_ZH.md)。
