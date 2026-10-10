# NavDP／FLUX 与独立工具入口补充审查

日期：2026-10-01。对象为本仓库 NavDP／FLUX 模拟导航入口、`deployment/` 和 `usd_convert/`；对照固定 IsaacLab commit `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8` 的本地源码与已安装 Python 3.12 环境。只读审查，未构造 AppLauncher、启动 Kit 或加载不可信 checkpoint。本文件不重复 FLUX 服务端维度和未定义函数问题。

## B01 · P1：NavDP 模拟入口仍使用已不存在的 IsaacLab 命名空间

- 位置：`/project/EXP/SangforAndSaut/ProtoMotions/NavDP/eval_pointgoal_wheeled.py:2`；同类问题在 `eval_startgoal_wheeled.py:2` 及多个 teleop 入口。
- 触发：在当前 IsaacLab 12.0.0／Sim 6.0.0.1 环境执行上述入口，即使只请求 `--help`，也先执行 `from omni.isaac.lab.app import AppLauncher`。
- 实证：当前 `.venv/bin/python -I -B` 执行 `importlib.import_module('omni.isaac.lab.app')` 返回 `ModuleNotFoundError: No module named 'omni.isaac'`；同一命令导入 `isaaclab.app` 成功，来自 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/app/__init__.py`。没有调用 launcher 构造器。
- 后续旧路径还包括该入口 `:36` 的 `omni.isaac.lab.envs`、`:37` 的 `omni.isaac.lab.managers`、`:38` 的 `omni.isaac.lab_tasks.utils.wrappers.rsl_rl`。仅修复首条导入不代表完成迁移。
- 边界：这是在当前指定的新环境中的明确入口阻塞；原 NavDP 单独的旧环境可能支持这些旧路径，不能据此认定其原始环境无法运行。

## B02 · P1：三个 FLUX 评估入口强制加载不存在的 Isaac Sim 4.5 experience

- 位置：`/project/EXP/SangforAndSaut/ProtoMotions/FLUX/eval_pointgoal_wheeled.py:52`、`:53`；`FLUX/eval_nogoal_wheeled.py:27`、`:28`；`FLUX/eval_imagegoal_wheeled.py:27`、`:28`。
- 触发：执行默认 pointgoal／nogoal／imagegoal 评估，传入 `experience='/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit'`，无命令行配置或路径存在判断。
- 实证：硬编码绝对文件在本机不存在；固定 IsaacLab 工作区中同名 `apps/isaacsim_4_5/isaaclab.python.dyn.kit` 也不存在。当前 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/app/app_launcher.py:1155` 对不存在的绝对 experience 抛 `FileNotFoundError`；创建 `SimulationApp` 在后面的 `:1247`。
- 影响：启动流程在指定 experience 路径校验时阻塞。应通过当前 launcher 的 experience 选择机制或有效配置选择合适的 Sim 6 experience，不能把旧文件名直接搬入新目录后认为已兼容。
- 边界：结论仅针对这三个硬编码入口，未将其他已使用辅助函数或可选 experience 的动态／socialnav 入口归为同一故障；本次未实际启动任何 launcher。

## 独立工具检查与边界

- `usd_convert/convert_mjcf_to_usd.py:113` 传入的 `asset_path`、`usd_dir`、`force_usd_conversion`、`fix_base`、`self_collision`、`merge_mesh`、`collision_from_visuals` 均可在固定源码的 `MjcfConverterCfg`／基类中对齐。导入已使用 `isaaclab.sim.converters`，未发现旧构造字段导致的明确新增阻塞。
- `deployment/test_tracker_mujoco.py:152` 明确将仓库根加入 `sys.path`，以支持 `deployment.*` 导入；独立导出器使用当前 `protomotions.utils.export_utils`。本轮短审查未发现可证实的新增入口合同问题，不等同于完成全部模型导出与 MuJoCo 部署验收。
- 独立 USD 工具仍受依赖专项 D01 的 `usd-core`／`usd-exchange` 二进制覆盖影响；该问题另见 `dependencies.md`。MJCF converter 真正导入和生成 USD 的行为，需在解决环境后运行验收。
- 基线任务配置使用旧版本关节状态、仿真参数和传感器访问方式；本轮未启动环境，未将尚未进入真实分支的潜在 API 差异列为已验证故障。
