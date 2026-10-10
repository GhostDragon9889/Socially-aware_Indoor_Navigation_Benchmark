# 真实 AMASS 回归与训练分析工具

## 数据接口验证

`tests/test_amass_pedestrian_corpus.py` 实际读取本地三份 AMASS mmap 包：训练 319/228、验证 22/20、测试 34/15 个选中/扩散合格动作。没有构造替代动作数据，也没有启动 IsaacSim。

专用 Python 3.12 环境执行 **11 passed，1.03 秒**。Ruff 检查通过。完整记录见 [amass_corpus_verification.json](amass_corpus_verification.json) 与 [JUnit](amass_corpus_tests.xml)。

| 合同 | 真实验证 |
| --- | --- |
| split 与元数据 | 三个 split 的全部 375 个动作 ID、源文件名、帧数、30Hz dt、数组 24 body/69 DOF 形状，以及源动作无交集 |
| diffusion 窗口 | 每 split 批量 2,048 个窗口，共逐点核对 509,952 个 5D 时刻：31 个历史/当前点与 52 个未来点，stride=3 控制帧；独立全局 clip 索引核对无跨片段 |
| 专家历史 | 每 split 批量 512；逐字段核对 t、t−1…t−9 真实状态，确认 t+1 关节位置，并产生有限的 195 维 AMP 特征 |
| 续采样 | 保存/受限载入真实 generator state，混合 reference 与 diffusion 调用后续输出和最终 RNG 状态逐位相同 |
| 动作逆映射 | CPU 编译执行工作树中真实 `AMPWorld.expert_batch` 方法；真实下一帧姿态和 MJCF limits，经冻结 hand/toe、pi/knee5 目标映射后，随机物理 DOF 名称排列仍保持 PD 往返一致 |

源文件 SHA-256 来自 manifest，本次没有重复哈希超过 10GB 的数据包。测试使用 `needs_lfs` 标记；缺本地许可数据时明确 skip。此验证不证明作者 2,848 动作清单或物理稳定性。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest \
  tests/test_amass_pedestrian_corpus.py -q \
  --junitxml=docs/diffusion_amp_20261001/amass_corpus_tests.xml
```

## 可重复训练分析

`scripts/summarize_paper_training.py` 从 `diffusion_training/training_result.json`、AMP `metrics.jsonl` 及可选已完成 `result.json` 导出三个标准 Matplotlib PNG、输入快照 SHA-256、JSON 摘要和中文报告。默认路径以脚本所在仓库为基准；已经从 `/tmp` 成功运行两次。

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  /project/EXP/SangforAndSaut/ProtoMotions/scripts/summarize_paper_training.py
```

其他训练输出可用 `--diffusion-result`、`--amp-metrics`、`--amp-result`、`--output-dir` 指定。训练过程中和结束后均可重跑。

最新生成快照见 [训练报告](training_analysis/report.md)。工具处理行为验证见 [training_analysis_tool_verification.json](training_analysis_tool_verification.json)。

- 逐行读取完整 JSONL；只忽略最后一个尚未写完的记录。其他完整损坏记录报文件与行号错误。
- 原始曲线不平滑。显式 `fall_resets` 与 `time_limit_resets` 独立画图；缺失字段保持未知。
- 当前 runner 恢复训练时重置进程累计 counters/elapsed/最低高度/平均 episode 时长，因此检测到重启边界后断开曲线。每 rollout 重置次数仅取同一进程内相邻计数差；不推断未知的首条基线。
- 已完成 `result.json` 可能滞后正在续写的 `metrics.jsonl`；输出明确记录其 update 是否一致。
- 图表不能判断收敛、稳定或论文一致。当前 diffusion 数据仍是单人 AMASS、无同步邻居、全可通行平面，并非八场景导航评估。

截至工具第二次验证快照（update 134，后续训练仍会推进）：AMP 日志记录 274,432 个累计物理 transitions；当前进程 901 次 reset，其中 12 次 fall、889 次 time limit，运行最低 pelvis 0.366734m，完成 episode 平均 9.93167 秒。这些是中间统计，不是最终训练结果。
