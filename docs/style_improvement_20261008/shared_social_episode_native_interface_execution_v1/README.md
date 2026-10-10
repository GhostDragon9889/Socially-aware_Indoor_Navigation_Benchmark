# 原生 DP/SFM 评估：完整源码接口实际检查

在本轮 episode2000 训练继续运行时，实际执行全部 78 个源码接口：
人数 N3–15、种子 7／19／42、规划器 Diffusion／SFM。
各子进程均通过原 CLI 的 `--check-sources` 分支执行，实际退出 0；
CPU 批次由 `subprocess.Popen.wait()` 记录真实退出码，未推断 SDK 成功。

另保留先执行的 N3／seed7 与 N15／seed42 两臂实际工具退出及完整 stdout／stderr。
检查覆盖真实 CLI 参数转发、人数对应的生命周期声明、种子转发、
原生物理驱动与初态采集的源码逆转换，以及原控制／评分公式保持。
10 份绑定来源在前后核验一致，Ruff 通过。

**本批只证明源码接口可执行。**
所有调用均未加载 Torch、候选权重或启动 SDK，不证明原生世界构造、释放、
初态配对、200秒身体运动或模型效果。
原生任务仍需本轮训练及独立评估全部实际完成后释放；
SDK／原始数组审计仍需真实工具终态响应，不生成替代工具 chunk。

完整资源登记仍是四地图、156个cohort、144准入及12拒绝，
不能将78个源码组合当成78个真实场景，作者八场景协议尚未恢复。
后续实际命令与准入见[原生登记](../shared_social_episode_native_preparation_v1/README.md)。

## 复现

选择尚不存在的输出目录。该命令只启动最多两个并行 CPU 源码检查子进程。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/check_shared_social_native_interfaces.py \
  --output /tmp/new_native_source_interfaces
```

证据：[真实执行记录](actual_CPU_execution.json)、[全部78个实际退出及来源](all78/result.json)、
[来源副本](source_snapshot_index.json)。
