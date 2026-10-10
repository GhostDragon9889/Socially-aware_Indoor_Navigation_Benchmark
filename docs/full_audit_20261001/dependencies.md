# 依赖专项审计：2026-10-01

审查对象为当前 ProtoMotions 源码、CrowdSim、FLUX/NavDP 的全部依赖声明，以及 `/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv`。未安装、卸载或修改任何依赖，未启动 Isaac Sim；证据在同目录 `dependencies.json`。依赖满足、导入成功和 GPU 仿真兼容分别判断。

## 已确认的问题

### D01 · P1：两个 USD 包覆盖相同二进制，实际版本偏离声明

- 位置：`/project/EXP/SangforAndSaut/IsaacLab-Protomotions/pyproject.toml:63`、`:66`；`/project/EXP/SangforAndSaut/IsaacLab-Protomotions/uv.lock:3510`、`:3518`。
- 环境同时安装 `usd-core==25.11` 和 `usd-exchange==3.0.0`。两者的 RECORD 都声明 `pxr/Usd/_usd.so`，实际文件只匹配 usd-exchange 的 SHA-256，不匹配 usd-core。使用 `python -I` 导入，实际 `Usd.GetVersion()` 为 `(0, 26, 8)`，导入路径在当前 `.venv`。
- 传递来源：`isaacsim-core==6.0.0.1` 要求 `mujoco-usd-converter==0.2.0` 和 `urdf-usd-converter==0.1.2`，这两者均要求 `usd-exchange>=2.2.2`。只在顶层对架构设置互斥 marker，不能阻止这条传递依赖。
- 触发条件：独立 Python 的 USD 转换、资产审查或任何使用该 `.venv` 的 `pxr` 导入。重装/卸载任一提供者也可能覆盖或删掉另一个所需文件。
- 影响：声明中的 `usd-core>=25.11,<26.0` 并未约束实际加载的 USD 二进制。`uv pip check`、锁文件 dry-run 均未发现文件所有权冲突，因而“锁与环境一致”不足以证明 USD 安装正确。
- 边界：本次已证实版本和文件覆盖；未启动 Kit，不能据此断言所有 Isaac Sim 仿真都会崩溃。处理时必须同时考虑转换器与 Kit 的 USD 版本，不能简单卸载一个包后认为完成。

### D02 · P2：FLUX/NavDP 声明不能合并安装到当前环境

- 位置：`/project/EXP/SangforAndSaut/ProtoMotions/FLUX/requirements.txt:10`；`/project/EXP/SangforAndSaut/ProtoMotions/NavDP/requirements.txt:76`、`:113`、`:128`、`:169`、`:176`、`:208`。
- FLUX 固定 NumPy 1.26.4，当前工作区显式覆盖 NumPy 2.3.1；NavDP 固定 IsaacLab 2.0.2、NumPy 1.26.0、Pillow 11.0.0、rtree 1.4.0、sentry-sdk 1.43.0、Warp 1.6.1，与当前栈不一致。
- 当前可见包对照：FLUX 根声明 10 项中 6 项缺失、1 项版本不符；NavDP 根声明 220 项中 86 项缺失、101 项版本不符。这里只比较根 requirements，子目录各自的附加声明已列入证据清单，不能将数量解释成完整运行验收结果。
- 触发条件：直接在行人模拟 `.venv` 中执行 `pip/uv pip install -r FLUX/requirements.txt` 或 `NavDP/requirements.txt`。可能失败或改变已验证的仿真依赖。
- 边界：这些独立基线未被当前 ProtoMotions/CrowdSim 模拟入口自动安装；应维护独立环境，不是当前行人策略推理必需补装的所有包。

### D03 · P2：项目说明的安装与校验命令对应另一套不存在的打包配置

- 位置：`/project/EXP/SangforAndSaut/ProtoMotions/FULL_PIPELINE.md:55`、`:65`、`:71`、`:72`；`/project/EXP/SangforAndSaut/ProtoMotions/pyproject.toml:59`、`:149`。
- FULL_PIPELINE 声称目标 Sim 6.0.1 / Lab `v3.0.0-beta2`，而实际项目依赖为 Sim 6.0.0.1 / 固定 Lab commit 对应的包版本 12.0.0。
- 文中引用的 `constraints_isaaclab3.txt`、`docs/ISAAC_SIM6_LAB3_MIGRATION_ZH.md`、`CrowdSim/tools/check_isaaclab3.py`、`CrowdSim/tools/validate_isaaclab3.py` 均不存在。
- 文中 `pip install -e ".[test]"` 声称安装 `crowdsim-ppo`；当前项目没有 `test` extra，setuptools 仅发现 `protomotions`。安装后的 `python -I` 对 `crowdsim_ppo` 的 `find_spec` 返回 `None`；`src/crowdsim_ppo` 只能通过源码路径等额外配置使用。
- 触发条件：按此文档进行新机器安装、校验或安装模拟器之外的 PPO 参考包。至少校验命令和参考包导入不可执行。

### D04 · P2：当前环境的完整复现配置位于相邻工作区，且 CI 未覆盖 CrowdSim

- 位置：`/project/EXP/SangforAndSaut/IsaacLab-Protomotions/pyproject.toml:94`、`:285`、`:295`、`:348`；`/project/EXP/SangforAndSaut/ProtoMotions/.github/workflows/public-ci.yml:156`、`:173`、`:194`、`:232`。
- 当前有效锁在相邻 `IsaacLab-Protomotions/uv.lock`；包含修改后的 Linux x86_64 限制、仅 isaacsim extra、NumPy/coverage 覆盖、Chumpy 构建依赖和 TUNA 显式源。它将 ProtoMotions 作为 `../ProtoMotions` editable 路径，未记录此路径中的具体 Git commit或工作树内容。当前 ProtoMotions 依赖清单本身不包含完整 CrowdSim 所需 OpenCV、SMPL-X、Chumpy。
- 触发条件：只复制/克隆 ProtoMotions、只按通用安装文档安装，或丢失相邻 workspace 配置。不能复原此处已验证的 270 个环境包。
- 公共 CI 只运行 Python 3.10 / MuJoCo / CPU 的 `protomotions/tests/`，不执行 `CrowdSim/tests`、根 `tests` 或新 IsaacLab/Sim 栈，且各 job 受 `github.repository == 'NVlabs/ProtoMotions'` 限制。若本项目独立托管于其他仓库，这份 CI 会跳过。
- 边界：该限制在上游工作流中是明确的策略；本地扩展项目需要自己的环境与集成验收，不能把上游 CI 结果当作 CrowdSim 可运行证明。

## 当前已通过的检查与残余边界

- 隔离 Python (`-I`) 发现 270 个环境包。Python 3.12.13、IsaacLab 12.0.0、Isaac Sim 6.0.0.1、Torch 2.11.0+cu128、TensorDict 0.9.0、Lightning 2.5.6、MoviePy 1.0.3、NumPy 2.3.1。ProtoMotions core 和 isaaclab extra 的所有生效 requirement 都满足。
- Fabric、Lightning WandbLogger、SciPy Rotation、scikit-image、SMPL-X、OpenCV、ONNX Runtime、dm-control MJCF、CrowdSim 导航和 PPO 模块的轻量导入通过；Numba 实际 JIT 运算通过。
- Chumpy 0.70 在 Python 3.12 直接导入仍因删除的 `inspect.getargspec` 失败；调用 `CrowdSim/tools/smpl_mesh_visualizer.py:595` 的兼容层后，Chumpy 数组运算通过。当前人体可视化初始化在导入 SMPL-X/加载旧模型之前调用该层。其他单独使用 Chumpy 的脚本必须明确考虑这一边界。
- 当前 MoviePy 的 `ImageSequenceClip` 导入通过；`record.py` 两条分支现在使用相同导入，不再存在旧版顶层导入不兼容问题。实际视频编码沿用此前安装验收，不在本轮重复编码。
- `uv pip check` 仍报告 3 项明确覆盖：Isaac Sim 要求 Newton 1.2.0 / schemas 0.2.0 / coverage 7.4.4，实际安装 Newton 1.4.0.dev0 / schemas 0.5.0 / coverage 7.6.1。覆盖源自当前 workspace，不能按元数据直接宣称运行失败，也不能未经 Isaac Sim 验收视为已解决的兼容性。
- `uv sync --offline --locked --dry-run --extra isaacsim` 解析 271 个锁包、检查 270 个环境包、无待变更；由于只读限制，环境锁获取有警告。本次没有执行同步变更。
- 锁中 10 个预发行版本，包括固定 Newton/Warp，也包括 einops 0.9.0.dev0、PyOpenGL 4.0.0a6、onnxscript 0.7.3.dev20260916、tokenizers 0.23.0rc0 等。保留锁时版本被固定；重新生成锁时 `prerelease=allow` 和宽泛范围会扩大候选集合，不能认为每次重新解析得到相同环境。所有锁定 wheel 记录均具有 hash。
- 当前 ONNX Runtime 仅提供 Azure/CPU provider；ProtoMotions 行人推理使用 Torch，这不是它的阻塞项。只有要求 CUDA ONNX 推理时才需要另配 GPU provider。
- 非隔离 shell 继承 ROS Jazzy Python 路径，元数据额外可见 165 个 ROS 包；其中 generate-parameter-library-py 缺 typeguard，不属于这 270 个锁定环境包的问题。证据保留原始与隔离两套结果，避免把 ROS 包误计入新环境缺依赖。

建议先解决 D01 的 USD 提供者冲突并保存完整安装 profile，再完成 Isaac Sim 单环境、转换器及行人闭环验证。其余基线环境与模拟器分开维护。
