# 安装与运行环境

项目仅支持两种运行环境设置方式，均使用 IsaacLab 3.0-EA / Isaac Sim 6.1 和 SDK 的 Python 3.12。

## 配置外部 SDK

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root ../IsaacLab
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
```

`--lab-root` 可以是相对路径，也可以是用户自行提供的位置。默认解释器为 SDK 的 `.venv/bin/python`；用 `--python` 选择其他已安装匹配 IsaacLab/Isaac Sim 的解释器或 `python.sh`。项目附加包由 uv 安装到 `.runtime/external_sdk_packages`，不修改 SDK，不建立第二个项目覆盖虚拟环境。包锁变更时使用 `--rebuild-addons`。编辑器使用生成的 `.runtime/external_sdk.code-workspace`。

## 通过 uv 安装项目内 SDK

Linux x86_64 主机准备 Git 和 uv 后执行：

```bash
python3 scripts/bootstrap_reproduction.py install
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
```

安装器使用固定 IsaacLab 源提交、受哈希校验的 `env/isaaclab3_ea_sim61/pyproject.toml` 和 `uv.lock`，通过 uv 安装 Python、SDK、Isaac Sim 及项目附加包。默认安装位置为 `.runtime/IsaacLab61`，可用 `--lab-root` 指定新目录。已有 SDK 受保护，复用请执行 `configure`。不要从根项目 metadata 重新解析或安装仿真后端。

首次启动新 Isaac Sim 前须自行阅读并同意 [NVIDIA EULA](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html#running-isaac-sim)，同意后设置 `OMNI_KIT_ACCEPT_EULA=YES`。安装器不替用户接受协议。硬件和系统要求见 [Isaac Sim 6.1 官方安装](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)和 [IsaacLab 3.0-EA 官方文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)。

## 资源与检查

```bash
python3 scripts/run_external_python.py scripts/manage_release_assets.py fetch
python3 scripts/run_external_python.py scripts/manage_release_assets.py prepare --check-usd
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
python3 scripts/run_external_python.py scripts/run_project_tests.py --output-dir output/tests_new
```

模型、USD、数据集和许可 SMPL/AMASS 资源与源码分开管理，见[克隆复现](../../CLONE_REPRODUCTION.md)。外部许可资源的位置写在被 Git 忽略的 `data/external_resources.json` 中；代码不嵌入机器目录。CPU 与 CUDA 测试分进程执行，缺少硬件或旧后端记为跳过，不能当作通过。

配置会验证 SDK 提供者和固定输入后再发布绑定；失败尝试回滚并保留诊断。持续 I/O 故障下，异常会注明保留的恢复快照路径，需排除故障后恢复。配置和短程训练通过不等于全部场景导航质量通过，见[验证状态](../validation.md)。
