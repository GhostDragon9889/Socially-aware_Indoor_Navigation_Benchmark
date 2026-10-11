# 外部 IsaacLab 3.0 EA / Isaac Sim 6.1 核验

本轮实现并实测了两条运行路径：直接使用 `/project/EXP/IsaacLab` 的 Python，或者用配置脚本把完整 SDK 安装到项目目录。项目附加包单独保存，现有 SDK 不安装本项目，也不要求项目 `.venv`。实际使用 `data/pretrained_models` 中的权重完成模型检查、四场景短时物理、256 环境训练与恢复。

源码、配置与模型的清点可以覆盖所声明文件范围；这不等于每行、每分支、每功能都已执行，更不能证明没有任何 bug。下面分别记录清点、执行、物理和覆盖率，保留未验证项。

## 环境与使用

| 项目 | 实测值 |
| --- | --- |
| IsaacLab 源码 | `ae37b028ea415c91ea2bc32609efcd759ed2b974`，树 `27e5b57ad0928e0339e121a06be040ae234a99ed` |
| 产品 / 源码包版本 | `3.0.0` / `isaaclab==17.0.2` |
| Isaac Sim / Python | `6.1.0.0` / `3.12.13` |
| Torch / Warp / NumPy | `2.11.0+cu128` / `1.16.0` / `2.5.1` |
| GPU | 一张 RTX 4080 SUPER，16 GB，驱动 595.99.02 |
| 外部 SDK Python | `/project/EXP/IsaacLab/.venv/bin/python` |
| 新建完整 SDK | `.runtime/IsaacLab61_verified/.venv/bin/python`；首次新安装 `.runtime/IsaacLab61` 也完成物理启动 |
| 项目附加包 | `.runtime/external_sdk_packages`，33 个精确版本 |

```bash
# 已有外部 SDK
python3 scripts/bootstrap_reproduction.py configure --lab-root /project/EXP/IsaacLab
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status

# 新建完整 SDK，默认安装到 .runtime/IsaacLab61
python3 scripts/bootstrap_reproduction.py install
```

实际完整安装命令使用 `install --lab-root .runtime/IsaacLab61_verified`，真实终端退出码为 0。SDK 源码、锁和提供者均核验；配置检查 169 项 USD RECORD 文件及默认依赖闭合。SDK 环境及新的附加包目录与历史 overlay 分开。外部 SDK 的 `uv.lock` 原有本地差异不是本轮配置命令产生的；SDK 运行源码保持固定身份。

按照 [IsaacLab EA 官方安装](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html)和 [Isaac Sim 6.1 Python 安装](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)，采用 Python3.12、独立 SDK 与固定依赖。官方分发 wheel 的版本号与上述固定源码包版本分别记录。已阅读本地 Lab 安装文档及 IsaacLab-Protomotions 的运行方式。用户明确同意 NVIDIA EULA 后才设置新环境的 `OMNI_KIT_ACCEPT_EULA=YES`；安装器不代替用户接受协议。

打开生成的 `.runtime/external_sdk.code-workspace` 使用外部解释器；已移除会覆盖该设置的旧项目解释器、envFile 和终端 PYTHONPATH 默认值。更多参数、二进制发行版限制、资源和 ORCA 编译见[外部 SDK 指南](guides/external_sdk.zh-CN.md)。二进制 Sim `python.sh` 仅实现配置接口，本机没有该发行版，不能宣称实测通过。

## 发现与修复

| 问题 | 修复与验证 |
| --- | --- |
| 旧运行入口依赖项目 overlay 或开发机固定路径 | 增加直接 SDK 启动器、绑定收据与统一入口；子进程沿用 SDK `sys.executable`，任务和调试入口更新 |
| 新 SDK 缺少旧 SDK 偶然提供的 `tensorboardX` | 加入项目附加包锁，33 项依赖闭合；修正后从空安装位置完成完整 SDK 安装 |
| Ubuntu22.04 的 Python3.10 无 `tomllib` | 引导脚本和统一运行入口先切换 SDK Python；真实 Python3.10 配置、运行和附加包导入通过，错误指定 Python3.10 为 SDK 时直接拒绝，避免重启循环 |
| 调试入口只有当前 `sys.path`，子进程缺少附加包 | `--sdk-child` 同时准备子进程环境并核验当前解释器；新增真实进程及错误解释器回归 |
| Git clone 不保留研究输入只读位 | 先验证全部 1,338 个冻结文件的大小与 SHA256，再恢复权限；不改写内容 |
| 五个历史准入 JSON 只在本机、未进入 Git | 按原始字节补入仓库并登记冻结清单；不重新生成历史科学输入 |
| 项目或 SDK 的旧绝对路径、嵌套克隆重复前缀 | 显式搬迁已知前缀，保留未认识的外部路径；嵌套路径处理幂等，来源字节仍严格验证 |
| 路径变化使 episode 注册与当前来源不一致 | 新增独立外部 SDK 注册，原历史注册不变；搬迁克隆完成冷准入 |
| 普通依赖锁误按 TOML 解析 | 分别解析 package lock 与 uv/TOML 锁；克隆资源审查通过 |
| TRACE Lightning 回调的合法 `+inf` 被当成损坏模型 | 仅允许 min-mode 标量 `kth_value` 初始化哨兵；参数、优化器的其他 NaN/Inf 继续拒绝 |
| ORCA 需要另一解释器 | 使用 SDK Python 构建固定 RVO2；构建工具临时保存，扩展与收据留在项目，启动器核验 SHA256 |
| 原生工具 owner 固定旧开发机路径及权限参数 | 新 owner 绑定注册的项目路径和 SDK Python；保留完整历史命令推导，只替换解释器；13 项 V8 合约通过 |

搬迁克隆名含空格和中文，无项目 `.venv`。恢复发布缓存及绑定外部许可动作容器后，1,042 项已提交源码/配置、28 个所需资源检查通过；发布资源准备核验 467 条记录及 14 个 USD。在线下载的授权服务不是本次缓存恢复实测范围。变异测试移除搬迁条件的 `not` 后真实退出 1，恢复字节后退出 0，证明相关回归能发现该条件错误。

## 模型和真实训练

`verify_pretrained_models.py` 清点全部 **147 个文件**，计算大小和 SHA256；**25 个 `.pth`/`.ckpt`** 完成受限加载与张量检查。**12 个现行 ProtoMotions 模型**按保存配置构造网络并严格加载完整状态。三份 TRACE 检查点的 Lightning 哨兵单独列出。旧 NavIsaacLab 两组模型的归档、不支持状态保持不变；张量可读不能证明旧分支可在现行 SDK 运行。

训练从提供目录的公开 PACER no-shape 权重暖启动，SHA256 为 `bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615`。

| 验证 | 实测结果 |
| --- | --- |
| 并行 PPO/AMP | 256 个环境 × 32 控制步 × 16 次更新，131,072 个真实物理转换 |
| 专家覆盖 | 2,848 个训练动作 ID 全部采到；验证/测试动作不用于梯度 |
| 网络更新 | actor、task encoder、critic、discriminator 均改变 |
| 终止/重置 | 1 次跌倒、1,189 次重置；训练正常关闭 |
| 恢复 | 六个物理状态字段恢复；根速度最大误差 `1.1920929e-7`，其余为 0 |
| 部分重置 | 反向、非连续的 128 个偶数环境匹配源状态，128 个奇数环境保持不变，空选择无操作 |
| 恢复更新 | update16→17，再完成 8,192 个物理转换，四个网络均更新，0 终止、45 次超时 |

这是单 GPU 同一 PhysX stage 中共享平面地形的 256 个批处理环境。没有验证多 GPU、256 张独立室内地图、训练收敛或论文效果。恢复没有保存 PhysX solver 缓存，不宣称逐位重现完整轨迹。最初仅运行两次更新的试验因未采遍动作集而被正确拒绝，失败记录保留。

训练证据：[训练](external_sdk_verification_20261011/artifacts/pacer_pretrained_train256_v2/result.json)、[恢复](external_sdk_verification_20261011/artifacts/pacer_resume256/result.json)、[模型清单](external_sdk_verification_20261011/artifacts/pretrained_final.json)。新训练产生的巨大权重、SDK 和动作容器没有再次提交。

补充运行了真实 Warehouse/NovaCarter 机器人 PPO：外部 SDK 下 4 个人形（所提供的 MaskedMimic SMPL 权重）与 4 个机器人槽位，完成 64 个机器人转换、4 次更新和保存，实际退出 0。恢复到 robot_step64 后继续到80，新增16个转换，31个模型张量改变、全部有限，Adam 的步骤均为5，实际退出0。首次启动下载官方 Sim6.1 机器人 USD 依赖耗时约10分钟；使用缓存的恢复运行约半分钟。相机、深度和人形视觉网格关闭；机器人物理状态重新初始化，不能称作完整物理状态或随机轨迹恢复，也未证明策略收敛。[机器人训练](external_sdk_verification_20261011/artifacts/crowdsim_robot4_train_observation.json)与[恢复](external_sdk_verification_20261011/artifacts/crowdsim_robot4_resume_observation.json)保留原始日志和检查点哈希。

## 四场景短时物理

均使用所提供目录的 TRACE/PACER 或本地 AMASS 微调权重，以及实际 PhysX 步进；父进程 `Popen.wait` 观察到退出 0，来源保持不变、清理成功。

| 场景 | 环境 | 行人数 | 控制步 / 模拟时间 | 跌倒 |
| --- | --- | --- | --- | --- |
| YOHO | 项目内新 SDK | 4 | 300 / 10 s | 0 |
| NewMetro | 搬迁克隆 + 外部 SDK | 4 | 60 / 2 s | 0 |
| SecondFloor | 外部 SDK | 1 | 60 / 2 s | 0 |
| ThirdFloor | 外部 SDK | 1 | 60 / 2 s | 0 |

这是兼容性与生命周期短测。它没有替代完整 60 秒姿态、接触、速度、停滞和视频验收，也没有修复或消除旧长测中上层场景速度偏低、过度制动的质量待办。单人上层场景不代表拥挤交互。完整结果和进程观察在[证据清单](external_sdk_verification_20261011/manifest.json)中。

## 静态范围、测试与覆盖率

静态清点包括 **2,632 个 Python 文件、644,133 行物理文本、24,979 个函数定义**，包含历史快照和第三方代码；全部 AST 解析通过，致命 Ruff 检查通过。它不是逐行语义证明。逐文件 SHA、行数、函数数保存在[清单](external_sdk_verification_20261011/artifacts/all_python_inventory_final.json)。

修复后的完整 CPU 回归真实退出 0：**4,996 项通过、9 项跳过、130 个子测试通过**。其中 7 个 CUDA 跳过项另行实测通过，合并为 **5,003 项通过、2 项旧 IsaacGym 跳过**；不是把子测试、重复回归或静态清点加到用例数。7 项 CUDA 回归通过；78 个注册 N/seed/planner 源码接口通过；11 个统一 CLI 的帮助接口通过；13 项原生观察器 V8 合约通过。这些不同口径分别记录，不相加伪装成覆盖率。

当前 CPU 父进程测量覆盖 470 个源码文件、55,278 条可执行语句和 17,028 个分支：语句覆盖 **70.93%**，分支覆盖 **59.17%**。coverage 综合值为 68.16%，不能误称纯行覆盖率。16,070 条语句和 6,953 个分支未覆盖；3,988 个函数范围中 993 个没有观测到函数体行。子进程、GPU、scripts 和归档快照不计入此百分比，虚拟编译文件警告原样保留。覆盖测量结束前核对了当前源码 SHA，未发生源码变更。

逐行 HTML 可从[离线报告压缩包](external_sdk_verification_20261011/artifacts/coverage_html.zip)解压后打开 index.html。

证据：[CPU XML](external_sdk_verification_20261011/artifacts/full_cpu_current_v2.xml)、[CUDA XML](external_sdk_verification_20261011/artifacts/gpu_final.xml)、[覆盖汇总](external_sdk_verification_20261011/artifacts/coverage_summary.json)、[逐文件缺失行和分支](external_sdk_verification_20261011/artifacts/production_file_coverage.json)、[逐函数范围](external_sdk_verification_20261011/artifacts/production_function_coverage.json)。

## 明确未完成的范围

当前仍不能声明每行、每分支、每功能全部执行或保证无任何 bug。二进制 Sim、多 GPU、旧 IsaacGym 的真实运行、所有归档实验、完整作者八场景协议、所有模型的策略质量、长期收敛和全部视频流程没有本轮完整实测。当前作用域与证据一一对应见[功能矩阵](external_sdk_verification_20261011/feature_matrix.json)。历史结果和拒绝门槛保留，短测成功不改写为完整验收。
