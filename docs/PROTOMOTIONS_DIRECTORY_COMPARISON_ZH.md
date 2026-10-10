# 两份 ProtoMotions 目录差异审查

日期：2026-09-30。

- A：`/project/EXP/NavIsaaclab2.0/protomotions`
- B：`/project/EXP/SangforAndSaut/ProtoMotions/protomotions`
- A 所属仓库 HEAD：`d742f0c5b421bbc281e2112e7e50427e0336c16a`。
- B 所属仓库 HEAD：`7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1`。

比较对象为工作区文件，包含未提交内容。HEAD 仅作为定位信息；没有据提交时间认定两个仓库的祖先关系。主范围只覆盖指定的两个包目录，不覆盖根目录依赖清单、数据权重或完整 CrowdSim。下文引用 B 的 CrowdSim 仅用于解释集成影响。

## 1. 总体结论

A 保留旧版 MaskedMimic 专用 agent，以及嵌入 ProtoMotions 核心的 CrowdSim 网格/性能钩子。B 对 agent、模型状态、环境上下文、训练流程和 IsaacLab 后端作了较大重构。两者共享 SMPL 人体基础和稀疏目标控制思路，但不能通过整体覆盖包目录或只改导入名称完成迁移。

## 2. 全量文件统计

排除 `.git`、`__pycache__`、pytest/ruff/mypy 缓存；按相对路径及 SHA-256 比较。

- A：432 个文件，其中 Python 164 个。
- B：622 个文件，其中 Python 352 个。
- 同路径且内容相同：178 个。
- 同路径但内容不同：248 个。
- 仅 A 存在：6 个。
- 仅 B 存在：196 个。
- 158 个共同 Python 文件发生文本变化：其中 61 个 AST 完全相同，97 个 AST 不同。AST 不同还可能包含文档字符串、类型提示或未用导入变化，不能解释成 97 个已确认行为变化。
- tests 下 Python 文件由 6 个变成 126 个；数量包含 `__init__.py`，不是测试用例数，也不是测试通过数。

248 个修改文件按一级目录：agents 41、envs 49、simulator 25、components 11、utils 13、robot_configs 8、tests 6、data 91，另有 4 个包根文件。

## 3. 最重要的功能差异

### 3.1 MaskedMimic 从专用 agent 迁入通用监督学习框架

A 的五个专用文件位于 `agents/masked_mimic/`：`__init__.py`、`agent.py`、`config.py`、`model.py`、`utils.py`。B 中该目录不存在。

B 对应实现：

- `agents/supervised/agent.py:34`：`SupervisedAgent`。
- `agents/supervised/masked_mimic_model.py:41`：`MaskedMimicModel`。
- `agents/supervised/masked_mimic_config.py:103`：`MaskedMimicSupervisedAgentConfig`。
- `agents/common/supervision.py`：通用监督损失支持。

两版均有 prior、encoder、trunk 和 VAE；不能把这次重构理解为 B 删除了 MaskedMimic 算法。A 由 agent 管理并向输入注入 `vae_noise`；B 将这类状态纳入模型的 `RolloutStateSpec` 和 rollout context。B 的模型基类增加了初始化、reset、序列状态和模型损失接口。

**迁移影响**：旧配置/Pickle 中的 `protomotions.agents.masked_mimic.*` 类路径在 B 中不存在；需迁移类与配置 schema。网络参数能否复用还要核对 state_dict 键、形状和观测语义，不能只改 `_target_`。

### 3.2 AMP 改为可复用组件

A 把 AMP 大部分训练逻辑放在 `agents/amp/agent.py`。B 的该文件缩减为 `AMP(AMPAgentMixin, PPO)`，训练逻辑移入新增的 `agents/amp/component.py`；模型层增加 `AMPModelComponentsMixin`。

配置也有变化：A 的 `conditional_discriminator` 字段移除，B 增加 `use_disc_critic`。B 另增 PEFT、prior AMP、FSQ、离散潜变量、Muon 优化器和 fine_tuning 等模块。文件变短不代表 AMP 被删除。

### 3.3 IsaacLab 后端接口发生实质迁移

- A：`PhysxCfg` 从 `isaaclab.sim` 导入，`SimulationCfg` 使用 `physx=`。
- B：从 `isaaclab_physx.physics` 导入，使用 `physics=`。
- A 使用 `root_physx_view`；B 使用 `root_view` 并通过 `wp.to_torch(...)` 读取多项底层数据。
- `simulator/isaaclab/config.py`：A 默认 `w_last=False`（wxyz）；B 默认 `w_last=True`（xyzw）。这里是后端边界约定，不能推广成所有内部状态都变更顺序。
- B 的 `utils/simulator_imports.py` 增加 IsaacLab >=12.0.0 版本检查和固定提交提示；A 没有同样的门槛。不据此推断 A 的精确兼容版本范围。

**迁移影响**：直接访问 `_robot.data`、设置四元数、接入相机与 SMPL 网格的外部代码需同步适配。跳过版本检查不能解决这些 API 差异。

### 3.4 SMPL 资产加载：预制 USD 改为 MJCF 转换

- A `robot_configs/smpl.py` 同时指定 `mjcf/smpl_humanoid.xml`、`usd/smpl_humanoid.usda` 和固定 bodies prim 路径；A 的 IsaacLab scene 使用该 USD。
- B 删除后两个字段；在 `simulator/isaaclab/utils/scene.py:162` 调用 `convert_robot_mjcf_to_usd()`，再解析 articulation/body prim。
- B 新增 MJCF 转换、D6 workaround、关节执行器分组、碰撞烘焙、材质和 USD body path 模块。
- 两份 `data/assets/mjcf/smpl_humanoid.xml` 的 SHA-256 完全一致。
- 两份 `data/assets/usd/smpl_humanoid.usda` 都仍是相同的 131 字节 LFS 指针。

**影响**：A 的旧 USD 加载路径存在资产前置缺口；B 默认路径改为从已有 MJCF 生成 USD，需要对应新后端支持。物理骨架源文件相同，不意味着导入后的动力学和关节命名已经验证一致。

### 3.5 人体可视网格的挂接位置改变

A 的 `simulator/isaaclab/simulator.py` 内置：

- `_maybe_create_crowdsim_human_mesh()`：环境变量启用后导入 CrowdSim 的适配器。
- `_update_crowdsim_human_mesh()`：物理循环中、非 headless 渲染前更新网格。
- `inference_agent.py` 的 `--human-mesh`、`--hide-humanoid` 开关。

上述接口在 B 的 protomotions 包中被移除。A 的 BaseEnv 还有 `CROWDSIM_PROFILE_BASE_ENV` 计时钩子，B 移除了它。

范围外集成核对：B 项目在 `CrowdSim/world/builder.py:250` 创建人体网格，在 `CrowdSim/nav_manager.py` 的 step wrapper 物理执行后更新。因此 B 项目不是完全没有 SMPL 外观，而是功能移到了 CrowdSim 层；其原生 `protomotions.inference_agent` 不再提供 A 的两个 CLI 开关。

更新与渲染的时序发生变化；当前 B 适配器还直接读取底层数据、使用 wxyz 变换，应结合上面的 Warp/四元数契约专项验证。这里确认接口差异，未通过 GPU 运行认定具体画面错误。

### 3.6 已找到当前导入错误的来源

A 的 `utils/inference_utils.py:77` 有 `apply_backward_compatibility_fixes()`；B 删除了该函数。B 项目的 `CrowdSim/protomotions_runtime.py:122/132` 仍在导入和调用它。

**A 中该函数只有文档字符串和注释，没有实际配置迁移逻辑。**复制回来能恢复该名称，但不能证明旧检查点和新框架兼容。这是旧 CrowdSim 接口与 B 核心混用的明确证据。

### 3.7 环境上下文、reset 和观测接口升级

B 新增 `EnvContext.env_ids`，`BaseEnv._build_context(env_ids=...)` 和多个控制组件支持子集环境。`MaskedMimicControl.populate_context()` 会按 env_ids 切片 motion IDs、目标时刻、mask 和出生偏移；A 总是使用全部环境。

B 还增加场景表面/地形上下文、target 控制、最近表面观测、里程计扰动、跌倒恢复初始化及更完整的动作库骨架校验。`target_poses.py` 从固定 body 0 扩展为显式 `anchor_body_index`。

**迁移影响**：直接替换 `populate_context` 的导航钩子应处理子集 env_ids；否则可能发生批次维度或人体索引不匹配。是否触发取决于实际调用路径，尚未运行确认。

### 3.8 动作库与训练加载生命周期扩展

B 的 MotionLib 增加 fixed/live/restart 分片策略、分布式 shard 选择、目标状态时间查询、分段接触平滑；BaseEnv 增加 `install_motion_lib()`。

B 的 `BaseAgent.load()` 新增 `load_training_state` 参数，并有独立推理检查点保存/加载支持。训练调度、SLURM、Lepton、CLI、导出、分布式归一化及异常处理也发生扩展。

这些功能有助于训练和部署，但不等于 B 已完成论文的轨迹扩散、视觉感知或完整 NavIsaacLab 实验。

## 4. 资源差异不能当成模型结构变化

- A 有 138 个 LFS 指针；B 有 49 个。
- 89 个共同文件从 A 的指针变成 B 的实际内容：63 个 G1 网格、24 个 SMPL 网格、2 个 checkerboard 文件。
- 逐一核验：89 个文件的 SHA-256 和字节数均匹配 A 指针记录的原始对象。它们是同一资源的落地，并非替换了模型。
- 除 Python 和这 89 个资源外，共同文件的另一项修改是 `data/assets/mjcf/soma23_humanoid.xml`。
- B 独有 196 个文件中包含 120 个测试文件、46 个 agents 文件、12 个 simulator 文件等。完整相对路径见 CSV。

## 5. 选择与迁移建议

1. 若维护 A 的既有运行链路，应使用与 A 匹配的代码、配置、权重和后端；其 LFS 资产仍需补齐。不能仅凭旧接口存在就认定 A 已能运行。
2. 若以 B 为后续基础，应将 CrowdSim 适配至 B 的模型、资产、四元数和数据访问接口，保留 B 已落地的框架与资源改进。
3. 迁移顺序：配置类路径/schema → checkpoint 键与张量形状 → 后端版本与 MJCF 导入 → Warp/四元数边界 → 稀疏目标 env_ids → 人体网格更新与渲染 → 单人体闭环 → 多人体 reset。
4. 比较控制效果时固定 MotionLib、模型参数、控制频率、PD 参数、场景和随机种子。代码文件数量或新增测试数量不能证明动作更自然。

本次没有覆盖或修改两个代码目录、没有反序列化权重、没有启动模拟器、没有执行测试套件。完成的是全量文件/资源比对、共同变更 Python 的 AST 比对及关键执行路径审阅。

## 6. 可追溯证据

- [完整文件分类与哈希](dependency_audit/protomotions_directory_comparison.csv)
- [源码统一差异（A → B，含新增/删除文件）](dependency_audit/protomotions_source_comparison.diff)
