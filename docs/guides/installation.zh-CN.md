# 安装指南

[English](installation.en.md) · [文档目录](README.md)

当前物理行人主流程的权威安装步骤见[从 Git 克隆复现](../../CLONE_REPRODUCTION.md)。使用项目内引导器安装固定 SDK，下载 Git LFS 资源并重建固定 AMASS 数据库，无需相邻目录布局或复制聊天工作区代码。

```bash
python3 scripts/bootstrap_reproduction.py install
export NAV_PY="$PWD/.runtime/IsaacLab/.venv/bin/python"
export PYTHONPATH="$PWD/src:$PWD"
"$NAV_PY" scripts/manage_release_assets.py fetch
"$NAV_PY" scripts/manage_release_assets.py prepare --check-usd
```

现有 SDK 环境可运行 `scripts/bootstrap_reproduction.py prepare-native`。软件版本、系统条件与已知覆盖见[依赖说明](dependencies.zh-CN.md)；训练、仿真和录制见[使用说明](usage.zh-CN.md)。完整 AMASS/SMPL 安装、上游网格 LFS 和 EULA 步骤均在克隆指南中。

纯 CPU 查看源码无需 SDK：`python3 scripts/audit_checkout.py`。独立 PointGoal PPO 使用 `configs/crowdsim_ppo/` 和 `src/crowdsim_ppo/`，它的 CPU 评测结果不代表完整三维行人仿真。
