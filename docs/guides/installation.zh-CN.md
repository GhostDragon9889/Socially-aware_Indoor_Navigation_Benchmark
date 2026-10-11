# 安装指南

[English](installation.en.md) · [文档目录](README.md)

默认环境为 **IsaacLab 3.0 EA / Isaac Sim 6.1.0.0**。完整资源、数据及运行步骤见[从 Git 克隆复现](../../CLONE_REPRODUCTION.md)。新机器从项目根目录运行：

```bash
uv python install 3.12.13
uv run --no-project --python 3.12.13 python scripts/bootstrap_reproduction.py install --profile isaaclab3_ea_sim61
export NAV_LAB="$PWD/.runtime/IsaacLab61"
export NAV_PY="$PWD/.runtime/isaaclab3_ea_sim61/.venv/bin/python"
export PYTHONPATH="$PWD/src:$PWD"
"$NAV_PY" -m social_navigation.runtime_environment check --lab-root "$NAV_LAB"
"$NAV_PY" scripts/manage_release_assets.py fetch
"$NAV_PY" scripts/manage_release_assets.py prepare --check-usd
```

已装固定 SDK 时，以实际路径代替 `NAV_LAB`，使用 `python3 scripts/bootstrap_reproduction.py prepare-runtime --profile isaaclab3_ea_sim61 --lab-root "$NAV_LAB"`。项目补充依赖安装到独立 overlay，不修改 SDK。只有匹配 SDK、锁与路径的环境才能复用；失败保留已选解释器。详见[运行环境说明](../../env/isaaclab3_ea_sim61/README.md)。

AMASS 原始动作和许可 SMPL 参数须另行准备。历史 Sim6.0 profile 使用 `--profile isaaclab3_sim6`；它的既有录像与指标不构成新 SDK 验证。录制仍须执行四场景质量门和完整视频解码。

安装问题：

- `pytetwild` 分支冲突：使用固定 SDK manifest/lock 的 Linux x86_64 环境，不在项目中执行 `uv sync --all-extras`。
- SDK 版本/源码/锁不同：安装器拒绝复用；选择固定 SDK，或在新的项目目录创建新的受控环境。
- pytest 自动加载系统 ROS 插件而缺 `lark`：清除外部 `PYTHONPATH`，CPU仓库测试使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。仅影响测试插件发现，不替换模拟器依赖。
- `usd-core` 与 `usd-exchange` 覆盖：运行只读 environment check；不要混装两个 USD 提供者。

纯 CPU 源码审查：`python3 scripts/audit_checkout.py`。无模拟器 PointGoal PPO 位于 `src/crowdsim_ppo/`；其结果不证明三维行人真实性。

封存的 `isaaclab3_sim6` 清单保持原 SHA，仍引用迁移前的项目 extra。fresh 安装或重新 apply 会在修改 SDK 前拒绝；只可复用匹配的已安装旧 SDK。新版 `install` 也保护已存在的 SDK，请用 `prepare-runtime` 复用。

下一步：[依赖](dependencies.zh-CN.md) · [配置](configuration.zh-CN.md) · [使用](usage.zh-CN.md)。
