# S1 真实动作前任务与专家速度诊断

[`diagnose_shared_social_task_context.py`](../../../scripts/diagnose_shared_social_task_context.py) 对已实际退出0、封存的 S1 最后一批训练数据进行只读 CPU 诊断。使用固定 AMASS2848 的真实训练退出记录、末批32×256控制、4096个专家窗口和上一轮实际覆盖记录；不加载模型 checkpoint、重采样或启动 SDK／CUDA。

## 实际检查与时间关系

- 最终8项合成 CPU 契约测试 `04d2a1`／0、Ruff `3931a1`／0。
- 最终真实 CPU 分析 `45d47a`→会话11179→`232796`／0；源码、报告、全部行 NPZ、日志及输入 SHA 见 `execution.json`。
- 初版真实读取 `cda998`→会话17544→`0dcb19`／1：配置比较因 tensor packet 的 tuple 与结果 JSON 的 list 不同而拒绝。修正为严格 JSON 文档比较，仍拒绝顺序或 bool／number 变化；失败日志保留。
- 修正后的第一次成功 `442320`→会话45077→`5af971`／0。最终版补充 COMMON／native 速度核对；两版全部 `measurement` 和 expert 描述相同。

任务前20维是动作前根位置相对、根 heading 坐标系下 **+.5,+1,…,+5秒的10个 XY 目标位置（米）**，不是即时速度命令。这与[NavIsaacLab IV-A／IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2)的目标轨迹控制方法相符；具体10点／局部坐标来自实际绑定的本地 PACER 观测接口，不能声称论文公开了完全相同的私有配置。

| 描述量 | 计算 | 解释 |
| --- | --- | --- |
| 根到首个未来目标的位移率 | `norm(p[0])/.5` | 包含当前根位置落后于目标的误差 |
| 未来目标段位移率 | `norm(p[k+1]-p[k])/.5`，9段 | 去掉当前根到首目标的追赶距离；是区间平均 proxy |
| 每行未来段中位位移率 | 上述9段中位数 | 一个控制行对应一个值；不是即时期望速度 |
| 当前身体 XY 速度 | raw AMP newest slot0，索引6:8 | self357 的213:215槽逐项一致；不使用标准化 RMS 值 |
| 专家窗口中位 XY 速度 | 真实参考 history10 的速度模中位数 | 与同批原 short-window low／fast 分类核对；没有将其配对到某一 policy task 行 |

例如，一个固定未来目标落后当前身体2米时，根到首目标的位移率为4m/s，未来9段却都为0；不能把前者解释为“命令高速行走”。单元测试覆盖此情况及带4米 lag 的1m/s未来轨迹。

## 全量来源核对

动作前 raw `self_obs/task_obs/amp_history` 与动作后诊断是不同时间。诊断中的 quaternion、heading、身体状态和 flags 属于 **POST／转换结果**，没有用它们旋转 PRE 目标。

- 全部 **8192** 行保留；116 reset actor-controls、3 terminated、113 truncated、1 physical fall、2非脚接触终止均不筛除。
- **7824** 个相邻同 episode 行的 PRE history 与前一步 POST history 逐项相同；首控制256行及112个跨 reset 边界独立记为无法邻接核对。
- 所有 history episode、整组 reset／episode+1、年龄及32个control／128个physics步的时钟连续通过。
- 全8192行 POST COMMON root velocity 与 native root link velocity 逐项一致；POST AMP XY 速度模与 native 的最大误差在报告中保留，容差 `1e-5 m/s`。只以 heading 旋转保持 XY 速度模核对，未用 POST 姿态替换 PRE 数据。
- `all_PRE_actor_controls.npz` 保留每行 actor／control／episode、所有位移率 proxy、当前根速度及相关 flags，输出没有 RNG／CUDA 状态变化。

真实 actor 分支来自原结果配置：0–127为 AMASS targets，128–255为 procedural targets。每组映射为 `[2g,2g+1,128+2g,129+2g]`，因此每组2个 AMASS＋2个 procedural，并非按连续4行拆成纯分支组。

## 实际末批结果

下表每个分支4096 actor-controls；“快／慢目标”按每行 **9段未来位移率的中位数** 判断。它们不是原专家 history10 类别、评价速度桶或论文作者标签。

| 量 | AMASS 目标分支 | 程序目标分支 |
| --- | ---: | ---: |
| 当前身体 XY 均速 | .542247m/s | 1.148517m/s |
| 当前身体速度 <.12m/s | 287／4096 | 65／4096 |
| 当前身体速度 ≥1.8m/s | 19／4096 | 607／4096 |
| 每行未来段中位位移率的平均 | .388641m/s | 1.502178m/s |
| 未来段中位位移率 <.12m/s | 2039／4096 | 154／4096 |
| 未来段中位位移率 ≥1.8m/s | 20／4096 | 1628／4096 |
| 快未来段且当前身体速度 <.12m/s | 1／20快目标行 | 21／1628快目标行 |

专家末批 history10 中位速度均值 **.578857m/s**；980／4096低速、46／4096高速，与上一轮未修改的窗口分类 low／fast membership 完全相同。专家当前速度均值.583260m/s、54行≥1.8m/s；这与history10的46行分母定义不同。

整体训练末批身体当前均速 **.845382m/s**，只有352／8192行速度<.12m/s。历史独立N4导航中80.46%低速指标是 **身体位置间隔、另一任务、另一时间范围**，不能与此 raw velocity 比例作同协议性能比较。这里显示任务上下文差异，尚不能证明专家先验不兼容或定位低速导航原因。

## 后续实验约束

保留固定2848／原专家 prior；先完成S2和严格准入的独立跟踪／风格／路线／物理评价。任务分支、速度时间尺度及场景差异应分别记录；下一轮先验或课程消融需重新登记并记录全程 per-ID/center/stratum ledger。不能把程序目标快行39.75%与专家高速短窗1.12%直接当作逐行供需缺口，或据此自动改成五类等权。

该诊断仅覆盖一个终态32步 batch，不是全程任务分布或独立质量成绩；reset时较旧的 AMP history 可包含参考初始化。作者动作ID、八场景清单及计数分母仍缺少，当前未证明达到／超越论文效果。

## 复现

输出目录必须不存在：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_task_context.py \
  --diagnostic-exit-receipt docs/style_improvement_20261005/shared_social_D_input_diagnostic_v1/execution_receipt.json \
  --coverage-exit-receipt docs/style_improvement_20261005/shared_social_expert_coverage_CPU_v1/execution.json \
  --output-directory output/diffusion_amp/new_task_context_CPU_diagnostic
```

读取遵守[PyTorch2.11 `torch.load`](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)的 CPU／`weights_only=True`／`mmap=True` 接口。读取的tensor文件是约1GB实际rollout，模型 checkpoint 未加载；所有已绑定源和输入在读取前后重新核对 SHA。
