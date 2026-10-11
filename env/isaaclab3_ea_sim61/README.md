# IsaacLab 3.0 EA / Isaac Sim 6.1 项目运行环境

## 中文

默认目标为 Linux x86_64、Python 3.12.13、IsaacLab 源码 `ae37b028ea415c91ea2bc32609efcd759ed2b974`（产品 3.0.0、包 17.0.2）与 `isaacsim==6.1.0.0`。`profile.json` 固定 SDK/提供者版本及 manifest、SDK lock、overlay lock 的 SHA。

### 复用已装固定 SDK

从项目根目录运行，路径可含空格：

```bash
export NAV_LAB=/absolute/path/to/IsaacLab
python3 scripts/bootstrap_reproduction.py prepare-runtime --profile isaaclab3_ea_sim61 --lab-root "$NAV_LAB"
export NAV_PY="$PWD/.runtime/isaaclab3_ea_sim61/.venv/bin/python"
"$NAV_PY" -m social_navigation.runtime_environment check --lab-root "$NAV_LAB"
"$NAV_PY" -I scripts/isaaclab3_environment.py --profile isaaclab3_ea_sim61 check
```

### 新机器

```bash
uv python install 3.12.13
uv run --no-project --python 3.12.13 python scripts/bootstrap_reproduction.py install --profile isaaclab3_ea_sim61
export NAV_LAB="$PWD/.runtime/IsaacLab61"
export NAV_PY="$PWD/.runtime/isaaclab3_ea_sim61/.venv/bin/python"
"$NAV_PY" -m social_navigation.runtime_environment check --lab-root "$NAV_LAB"
```

SDK 位于 `.runtime/IsaacLab61`；项目补充依赖位于独立 `.runtime/isaaclab3_ea_sim61/.venv`。安装器用 `.pth` 引用 SDK site-packages 与源码，本项目 `src/` 优先；不修改 SDK，不重复安装 Torch、NumPy、Warp、MuJoCo、Isaac 或 USD 提供者。

`install` 为新 SDK 应用 manifest/uv.lock；复用已有 SDK 使用 `prepare-runtime`。补充包按 `requirements.overlay.lock` 精确安装，先安装除 chumpy 外的包，再以 `--no-build-isolation --no-deps` 安装 chumpy。`runtime_constraints_receipt.txt` 是本机可见包收据，不是另一份 SDK 安装锁。

项目 runtime 包含 `runtime_receipt.json` 与安装日志。重复调用核对 SDK、锁、根目录身份及依赖；不匹配时保留环境并报错，失败不切换项目现有 `.venv`。bootstrap 仅在准备及验证成功后选择解释器。

本机已真实创建该 overlay，32 个精确包、116 项依赖、169 项 USD RECORD 校验通过。完整新 SDK 网络安装未在本轮重复下载；该软件/CPU检查不替代 GPU动力学和完整四场景视频验收。当前本机 `.venv` 已指向该 Sim6.1 runtime；物理训练、GPU恢复和退出状态见[增量复审报告](../../docs/ISAACLAB3_EA_SIM61_REAUDIT_20261011_ZH.md)。

Sim6.1 不加载旧 `env/isaaclab3_sim6/native_shutdown` ABI 库。旧 profile 保留供历史重放，已有视频保持原 Sim6.0 身份。资源和数据见[克隆复现](../../CLONE_REPRODUCTION.md)，旧分支配置限制见[参考注册表](../../data/pretrained_models/upstream_reference_registry.json)。

当前未筛选全量回归与真实 GPU 证据见[第二轮验证报告](../../docs/VERIFICATION_COMPLETION_20261011_ZH.md)：4645 项通过，98 项仍未通过，2 项旧 IsaacGym 测试跳过；环境安装检查通过不等于全部功能迁移完成。

## English

The default targets Linux x86_64, Python 3.12.13, IsaacLab source `ae37b028ea415c91ea2bc32609efcd759ed2b974` (product 3.0.0, package 17.0.2), and `isaacsim==6.1.0.0`. The profile checks source/provider identities and the SHA of the SDK manifest, SDK lock and overlay lock.

For a matching installed SDK, use `prepare-runtime` above. For a fresh machine, use `install --profile isaaclab3_ea_sim61`. SDK and project additions use separate virtual environments. The installer references SDK source/packages through `.pth`, prioritizing this checkout, without editing SDK files or duplicating binary providers.

All exact additions except chumpy install first; chumpy follows with `--no-build-isolation --no-deps`. The constraint receipt records observed packages and does not replace the SDK lock. `runtime_receipt.json` captures actual provider origins; reuse requires identical checkout, SDK and locks. Failure preserves the selected environment.

A real local overlay passed 32-package, 116-requirement and 169 USD RECORD checks. A complete fresh SDK download was not repeated. CPU/import validation does not certify GPU dynamics or full videos. The local root `.venv` now selects this Sim6.1 runtime; physical training, GPU restore and process-exit evidence are recorded in the [follow-up audit](../../docs/ISAACLAB3_EA_SIM61_REAUDIT_20261011_ZH.md).

Sim6.1 does not load the old Sim6.0 native ABI patch; historical recordings keep their original SDK identity. Continue with [clone reproduction](../../CLONE_REPRODUCTION.en.md). The [reference registry](../../data/pretrained_models/upstream_reference_registry.json) identifies two archived NavIsaacLab configs that cannot run with the current project branch.
