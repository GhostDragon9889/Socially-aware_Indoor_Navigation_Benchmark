# Core 训练、检查点与 SLURM 修复审查

日期：2026-10-01。参考 `/project/EXP/NavIsaaclab2.0` 内嵌 ProtoMotions，审查当前 core agents、训练入口及保存/恢复链。本轮不启动 Kit、不提交作业、不读取真实凭据、不安装或升级运行环境。

## 已修复的问题

| 旧审计/级别 | 触发条件及原因 | 当前修复和实证 |
| --- | --- | --- |
| F07 / P1 | 启用 W&B，集群没有现成凭据时，旧 launcher 读取本地 API key 或提示输入，将其拼入打印、落盘、上传的脚本。参考仓库同样存在这一行为。 | 删除凭据探测、读取及提示函数。生成脚本仅引用已有环境变量；集群 `.netrc` 仅在文件存在时只读挂载。假 key 回归确认脚本/命令无值，生成时拒绝任何 SSH 探测，`bash -n` 通过。 |
| F14 / P2 | 保存中断可破坏原地覆盖的 latest；加载历史/最佳模型却读取 latest 环境 sidecar，产生跨代模型和课程权重。参考仓库主模型、环境同样分开保存。 | 单个 checkpoint 同时包含原有 agent 状态与全部 task 环境状态；独立 generation ID、原子发布、上一份 `.previous`。故障注入和真实 SIGKILL 均保留完整旧文件。恢复 score/inference/previous 时读取各自内嵌状态，刻意放置 999 的错误 sidecar 也不混用。 |
| F15 / P2 | pip install 返回非零，原安装门仍发布 ready，其他 rank 开始训练；原参考入口也使用分号忽略安装结果。 | 只有 install 成功才发布 ready；失败标记保存退出码，所有 rank 退出；step/restart/node 分离标记。真实生成命令配假 pip 返回 42，rank 0/1 均返回 42、无 ready、未启动假训练程序。 |
| 新发现 / P2 | Core PPO 普通优势归一化的默认样本方差在 singleton 时为 NaN；EMA 标准差也会被污染，并进入后续迭代；actor/critic 诊断量同样有 NaN。参考 fork 的优势统计也有此问题。 | 改为总体方差/标准差，与 `combine_moments` 的合同一致。5 个新增 CPU 回归在修复前全部失败，修复后通过；覆盖普通/EMA、是否减均值、singleton 后更大批次，真实 Adam 更新、梯度、参数、优化器状态和日志均有限。 |

代码位置：

- 凭据与安装门：[train_slurm.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/train_slurm.py:715)、W&B 环境引用 807 行及只读挂载 865 行。
- 保存/恢复：[BaseAgent](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/agents/base_agent/agent.py:290)、`save` 480 行；通用原子文件工具 [checkpoint.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/utils/checkpoint.py:34)。
- 课程权重：[BaseEnv](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/envs/base_env/env.py:1802)。构建场景时从 CPU/mmap 读取小环境状态，避免将无用模型副本载入 GPU；传统非 ZIP 文件有正常读取兼容路径。
- 配置保存及 warmstart：[train_agent.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/train_agent.py:483)、943 行。机器读取的 JSON/Torch 配置原子写入；warmstart 仅加载模型，resume 加载优化器和环境状态。
- PPO 统计：[ppo/agent.py](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/agents/ppo/agent.py:772)、814/840 行及 actor/critic 统计。

## 新检查点合同与旧格式兼容

原有 `model`、优化器、epoch 等顶层键不变。新增 `_protomotions_checkpoint` 元数据为 `version=1`、`generation_id` 和 `environment_states[task_id]`。每个 `last`、`score_based`、inference 文件自包含，恢复时只使用所选文件的状态；这些名称的文件发布并非一个全局事务，但每份文件内部始终完整且代际匹配。

每个唯一 task 的一个 rank 提供冻结的 CPU 状态，避免对象集合反序列化远端 CUDA 设备张量。rank 0 将其合并到已知的其他动作分片历史中；A→B→A 回归确认 A 的课程状态不会因为 B 成为当前分片而丢失。模型/环境快照失败会在写入前向所有 rank 传播，high-score 标志一致性也在发布前确认。

旧文件没有 bundle 时，仍加载**所选 checkpoint 所在目录**的 `env_<task>.ckpt`，发出代际无法验证的提示。旧格式本身没有共同 ID，无法追溯确认其 model/env 同代；不能把兼容加载当作旧文件已修复。新 bundle 缺当前 task 时保留新 task 默认状态，不借用旧 sidecar；无效版本/字段在模型状态变更前被拒绝。

`.previous` 是完整旧代文件，可显式传给 `--checkpoint`；没有静默自动回滚。原子替换及目录 fsync 要求输出文件系统具备 POSIX 语义。进程 SIGKILL 可能留下隐藏临时文件，但不会将其作为已发布检查点。保留上一份训练、推理及最佳文件会增加磁盘使用。

## 验证与范围

- `training_checkpoint_slurm_tests.log/xml`：**309 passed，0 failed，22.02 秒**。涵盖 BaseAgent、BaseEnv、train_agent、PPO、AMP、ASE、PEFT、SLURM site/startup/watchdog。
- `checkpoint_gloo_tests.log/xml`：**真实两进程 CPU Gloo 2 passed**，包含多 task 收集及 rank 1 环境快照异常的双 rank 退出；未将 mock collective 误记为真实分布式验证。
- 最后 CPU mmap/原子旧文件 fsync 补充变更：`training_checkpoint_final_changes.log/xml`，**22 passed、2 deselected**；2 项 Gloo 已由前述真实进程验证。
- `core_ppo_singleton_before.log`：原统计实现的 **5 failed** 失效证据。
- 修改的 Python 文件 Ruff 检查通过。修改前逐文件备份和 SHA256 见 `before/dependencies_audit_before.json`。

测试入口使用 `../IsaacLab-Protomotions/.venv/bin/python`、`PYTHONPATH=src:.`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；真实 Gloo 使用 `GLOO_SOCKET_IFNAME=lo`。沙箱不能发现本机 loopback，真实两 rank 测试在工具自动批准的本地执行权限下完成。没有验证生产集群容器/凭据注入政策、NCCL 多 GPU、长期训练或存储服务器掉电行为。本报告不把这些边界记为 PASS。

## 依赖锁元数据收尾

root 添加了 `crowd` optional extra。先执行 `uv sync --offline --locked --extra isaacsim --dry-run`，准确发现 workspace 锁中 editable metadata 过期。仅执行 `uv lock --offline`，再次 dry-run **退出 0**；273 个 package 的版本集合完全不变，唯一计划操作是重建 ProtoMotions editable metadata。未执行 sync/install/upgrade。

`env/isaaclab3_sim6/uv.lock` 与 `profile.json` 已同步；锁 SHA256 为 `521ccbae937c4fe9c26f04430fec4a74809b7a57dd8d8ac8cb6949decd5eb45d`。原锁、profile 已备份，版本差异空集的证据见 `environment_lock_metadata_refresh.json`。该刷新不改变此前 Sim 6.0.0.1 / Lab 源码 `4ecd0b` / Py3.12 / 单 USD provider 的选择。

源码 SHA256 是本子任务封存时的快照；全仓最终文件清单应以 root 最终报告为准。
