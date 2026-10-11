# NavDP 旧 namespace 与 Lab 3.0EA / Sim6.1 迁移矩阵

## 已完成源码修正

原始 NavDP 中 69 条 `omni.isaac.*` 导入语句映射为 22 个目标模块，目标安装源码路径 **69/69 可解析**。每条原导入文件 / 行号 / symbols / 新模块 / 实际目标源码路径见 `navdp_namespace_matrix.json`；摘要见 `navdp_matrix_summary.json`。

| 旧接口 | 目标接口 / 具体适配 | 验证范围 |
|---|---|---|
| omni.isaac.lab.* | isaaclab.* | 完整 AST + 目标源码路径 |
| lab_tasks.utils.wrappers.rsl_rl | isaaclab_rl.rsl_rl | 目标 RslRlVecEnvWrapper(env,clip_actions=None) 签名与已有 env-only 调用对应 |
| lab_tasks.manager_based.locomotion.velocity.mdp | isaaclab.envs.mdp | 目标已移除原任务目录；该文件实际8种基础函数 / cfg均在 envs.mdp，不做无效的仅 namespace 替换 |
| omni.isaac.core.utils.prims / numpy.rotations / types | isaacsim.core.utils.* | 目标6.1仍提供的 deprecated源码路径（保留其API；不宣称新 experimental API全面迁入） |
| 旧 XFormPrimView | 当前 isaacsim.core.prims.XFormPrim，保留旧局部alias | 目标 vector prim constructor prim_paths_expr / batch set_world_poses 源码一致 |
| 旧单体 XFormPrim | 当前 SingleXFormPrim，局部alias | 原scalar set_world_pose应匹配单体，不能用vector类假设等价 |
| self.sim.disable_contact_processing | 7入口显式 Kit `/physics/disableContactProcessing=true` | 目标 SimulationCfg 不再含此字段，SDK contact tests确认Kit设置；移除4个无效动态配置字段 |
| 默认物理 / actuators | NavDP4环境 cfg 显式 PhysxCfg、use_newton_actuators=False | 保留原 USD / implicit wheels 流程；未替换原训练结构 |

三处 `add_point_light`（FLUX、NavDP、x-navdp）用 `while create_prim(...).IsValid()` 不断创建有效 prim、不能退出；已改为查询 `get_prim_at_path` 后仅创建最终选中的一个 SphereLight。FLUX / x-navdp 对单体的误用亦修正。

NavDP / FLUX 的 `forward_batch` 原忽略标量方法中的 linear/angular/wheel limits，已与 scalar `forward` 对齐并检查 `(batch,2)` 输入。默认无限大上限保持原行为，有具体上限时不再被批处理忽略。两类的 CPU等价 / 超限回归已执行。

## 启动边界与可选依赖

新增 `third_party/navdp/runtime_compat.py` 使用纯标准库，在7个 standalone入口 **AppLauncher import 之前**核对 Python3.12、IsaacSim6.1及目标 SDK `isaaclab` 发行版本17.x（目标仓库 VERSION为3.0.0，发行包版本不是同一版本号）。随后查询可选模块是否存在，缺失时立即报出，不先启动Kit/GPU。

当前 overlay 有 open3d/cv2/imageio/requests/rsl_rl/torchvision/scipy/matplotlib；缺 **casadi、pathfinding、pynput**。前两项阻止原 NavDP MPC/地图 standalone，pynput另影响teleop。旧冻结 `requirements.txt` 仍留作归档隔离，其中 Lab2.0.2/Warp1.6.1/NumPy1.26不得覆盖主6.1环境。尚未安装这些可选baseline依赖，也未污染主 SDK 依赖。

运行时preflight只验证SDK家族与所需模块存在，不证明已加载Dingo USD、场景/占据地图、模型权重、CFM服务以及RTXcamera任务。

## CPU / 静态证据

- `vendor_navdp_cpu.xml`: 最终新增迁移 / vendor测试 **29 passed**。
- `final_newton_vendor_cpu.xml`: 最终 Newton所有邻近测试 + vendor **64 passed, 2 skipped**（2项CUDA专用明确跳过，其他均CPU运行）。
- 实际安装发行元数据6.1/17.x通过，无Kit/GPU加载；实际NavDP teleop入口执行至前置诊断，报告缺casadi/pathfinding/pynput，Kit import attempts=[]（navdp_entry_preflight.json）。
- 光源原方法执行：连续两个已占路径、第三路径空，验证有限查询、仅一次创建及正确world位置（外部USD边界注入，未运行GPU stage）。
- PACER被NumPy2移除的 np.int 两个索引改np.int64；trajectory的Torch dtype=np.int改torch.long，实际CPU constructor/reset生成0.5–1.5m/s有限轨迹。
- 所有69旧导入消除并进行AST回归。

## 仍未动态验证的具体任务

1. 7个 NavDP standalone GPU/RTX任务、Dingo drive asset、ContactSensor batches、RslRlWrapper PPO协议。
2. x-navdp分布式worker、AppLauncher(distributed=True)、完整训练/评估及权重。
3. Deprecated core utils/SingleXFormPrim 在真实 PhysxManager patched SimulationManager 上的stage pose/Fabric同步。
4. FLUX剩余旧 `self.sim.disable_contact_processing` 配置、custom Kit experience文件、视觉权重与完整基线启动，不能因基础bug修正声明全部已兼容。
5. 原作者NavDP额外场景map/采样点/模型服务；只在资源路径/输入齐备后可做端到端任务。

源码/API迁移和组件回归可审阅；这些尚未执行的任务必须与主项目已通过的 PhysX行人测试分别报告。本轮未提交或推送。


## NavDP 仍需的依赖：可安装性与隔离建议

以下为 2026-10-11 核查，不代表已经安装或验证完整 MPC/teleop：

| 当前缺模块 | 可安装性依据 | 仍需测试 |
|---|---|---|
| casadi | 官方发布 Python wheel / pip 安装；当前3.8.1采用 cp311-abi3，官方说明兼容Python3.12等。见[CasADi安装](https://web.casadi.org/get/)。 | 实际项目使用的 IPOPT/Opti solver创建、有限轨迹、速度/加速度约束与失效处理；当前并未执行 |
| pathfinding | 发布包有 py3-none-any wheel，可通过 pip 安装。见[维护者PyPI发布页](https://pypi.org/project/pathfinding/)。 | 地图障碍、无解路线、Grid重复搜索清理、返回坐标约定 |
| pynput（仅teleop） | 可通过 pip 安装；Linux keyboard后端依赖X或uinput，X需可用DISPLAY；Wayland/Xwayland有范围限制。见[官方平台限制](https://pynput.readthedocs.io/en/latest/limitations.html)。 | 真实桌面按键监听、事件权限、退出handler；无DISPLAY时安装包本身不足以提供交互 |

建议新增**独立、可选的 navdp_sim61 profile**，保留与主项目同一 SDK 核心版本/Python3.12，在独立 venv 中增加 casadi/pathfinding，teleop 再增加 pynput；通过实际回归后锁定候选版本。不能将旧 NavDP root requirements 的 IsaacLab2.0.2、Warp1.6.1、NumPy1.26 pins 覆盖到主6.1运行环境。本轮仅给出隔离方案，未创建/安装该profile，未修改主 runtime或overlay lock。

模型服务按本地 `third_party/navdp/README.md` 的 Python3.10流程，用**另一套 navdp_server_py310 环境**安装 `baselines/navdp/requirements.txt`；服务环境不加载 IsaacLab，通过 HTTP 与仿真通信。这样保留作者模型依赖与目标Sim6.1环境分别验证。服务端/客户端要核对端口（默认8888）、camera intrinsics、batch_size、reset和step接口，而不是仅检测HTTP端口打开。

## 权重、实景与交互资源快照

`navdp_resource_snapshot.json` 是仅 NavDP vendor子树的快照，不能据此推断主项目其他资产目录为空。

| NavDP 相对路径 / 资源 | 本轮实际状态 | 原始任务仍需 |
|---|---|---|
| `assets/robots/dingo.usd` | 存在 | 真实Sim6.1 articulation/wheel/camera加载回归 |
| `assets/scenes/cluttered_hard` | 目录存在 | 当前assets仅一个USD且它是Dingo；这里的episode文件不等于实际场景USD |
| 全 `assets` | 1 USD、120 NPY、2 PNG、3 GIF | Scene-N1各场景USD及引用Materials/SkyTexture等sidecar；不能由120条episode NPY替代 |
| `asset_scenes/cluttered_easy` | 不存在 | 部分入口默认旧路径需要显式正确 `--scene_dir`；通常以NavDP根目录运行 |
| `baselines/navdp/checkpoints` / `baselines/navdp/100.ckpt` | 均不存在 | 显式传入与模型结构匹配的checkpoint并验证keys/shape覆盖 |
| NavDP子树 .pth/.pt/.ckpt | 未发现候选 | 不将主项目AMP权重当作NavDP视觉导航权重 |

作者README列出的场景来源是 [InternRobotics Scene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1/tree/main/n1_eval_scenes)，需要按其布局保留USD外部引用。最新InternVLA-N1 checkpoint访问在作者README中通过表单申请；本次未申请/下载。`navdp_server.py` 默认checkpoint是作者机器 `/home/PJLAB/.../cross-waic-final4-125.ckpt`，本机不可用；需 `--checkpoint <实际文件>`。完整NavDP checkpoint是否包含视觉backbone参数须读取匹配state_dict验证，不能无证据要求额外独立DepthAnything权重；独立DepthAnything示例有自己的预训练文件要求。

目前未完成原生 NavDP 7个入口端到端动态测试的具体原因：前置依赖缺casadi/pathfinding（teleop另缺pynput）；vendor中无NavDP checkpoint及实际评估场景USD；HTTP模型服务未启动；teleop桌面后端未验证。此外，本轮GPU串行资源优先用于主行人场景与6.1 native shutdown崩溃定位。已完成的迁移矩阵/组件CPU测试不能填充以上缺项或证明完整任务通过。


## XYZW / ProxyArray 的进一步实际断点

进一步核对目标源码：deprecated `isaacsim.core.utils.numpy.rotations.euler_angles_to_quats` 仍返回 **WXYZ**，`isaaclab` root write和CameraCfg.OffsetCfg/CameraData/math改为 **XYZW**；deprecated core `XFormPrim` 的orientation仍为WXYZ。已做明确接口边界转换，而不是整体替换四元数顺序：

- 4种主NavDP reset、x-navdp两种reset：仅传入Lab root write的quat转XYZW；Goal/goal-camera core prim保留WXYZ。
- 7个主入口和x-navdp guidance/reward/metadata：Lab3数据已XYZW，送SciPy时去掉旧的二次换序，默认identity改XYZW。
- 主Dingo及x-navdp Dingo/GO2/G1配置：Camera offset常量由WXYZ转XYZW；原`convention="usd"`改目标支持的`"opengl"`（USD camera frame约定）；显式`update_latest_camera_pose=True`，使移动robot camera的pose观测跟随相机而非返回初始化pose。
- asset ProxyArray读边界显式 `.torch`：目标`.device`是Warp Device，不能直接当作torch device；scripted math需要Tensor。已修正所触及reset/observation/reward/metadata边界。
- 主NavDP `imagenav_reset` 在单goal prim写入时只传该环境一份quat，height按env_ids索引；原N>1时整批quat配一份position不匹配。
- 主NavDP stuck term按各env返回`[N]`bool Tensor、clone保存历史；原只取env0/返回一个bool，且NumPy历史可能借用不断更新的simulator内存，造成批量终止误判。

`vendor_navdp_quaternion_cpu.xml` 最新 **36 passed**，含4个实际reset源码函数的0/90度heading、多envgoal quat batch、env_ids、Proxy边界和独立停滞判断回归；29passed/64combined是之前的阶段收据，不是本轮新数量。

`navdp_live_sdk_cpu.json` 另使用**实际安装的 target SDK** ProxyArray、scripted math、deprecated core NumPy rotation utilities在CPU执行项目oracle与reset源码：core identity确为WXYZ `[1,0,0,0]`，Lab root写identity `[0,0,0,1]`、90度heading `[0,1,0]`正确；实际Warp Device经`.torch.device`用于Tensor运算，`kit_modules_loaded=[]`。边界scene/asset writer注入，不冒称Dingo/RTX已动态运行。

FLUX完整基线保留旧SDK隔离状态，其中同样存在旧reset/camera四元数顺序和SimCfg字段，不能因为共享CPU修正声明其已全面迁移；若要在6.1整套运行须依照同样接口矩阵独立迁移和模型/场景回归。x-navdp完整training/worker/render tasks依然未动态验证。


---

打包说明：本页保留审查当时的阶段结论。最新证据与复制范围见 [审计索引](README.md)；未随包复制的大清单、源码快照与完整日志见 [来源索引](SOURCE_INDEX.md)。项目源码位置以仓库相对路径表述；SDK 绝对路径仅记录被审查机器。
