# 已有接触力的论文启发严重度诊断

三组真实捕获的新增CPU统计已实际完成并封存。独立 peer／root 源码审阅 Required=[]，root 封存最终8830B入口后执行；真实工具059378退出0，原来源准入及全部计数／峰值核对通过。

`diagnose_force_severity.py` 复用只读封存的 `actual_tracking_contact_CPU_v1/diagnose_tracking_contact.py` admission；先核对该源码的完整字节 SHA，再执行其中的 CPU admission 函数，不调用其旧诊断 main。旧 comparison、producer、真正 closed0 的 GPU/CPU owner、raw audit、完整 NPZ 字节身份及原控制周期契约沿用原门槛。全部来源在统计前后核对，不读权重、不启动模型/SDK/GPU，不更改历史源码、报告或默认模型。

每份 OLD63 S1、NEWb563 S1、NEWb563 S10 捕获只有两个数组被加载：`peer_contact_max_force_n[6000,4,4]` 和原 `peer_contact`。六个无序 actor pair 取 `max(Fij,Fji)`，每 pair 保留 6,000 controls；全帧分母为 36,000 pair-controls，包括跌倒、重置及 idle。三档严格为 `F<=1N`、`1N<F<=50N`、`F>50N`；1N 属第一档，50N 属第二档。总计及逐 pair 均报告三档计数，并对全部 `F>1N` 给出 NumPy linear P50/P95/max；空集合计数为 0，分位数和条件最大值为 null。

读取时核对有限、非负 FP32 力、零自接触对角及 bool flags 与 `force>1N` 完全一致。无序 contact 总数必须回到原 0／70／279，最大力必须回到原 0／1288.4229736328125／1069.25244140625N，并与原 sealed audit/comparison 一致。该诊断不重新分析几何代理、接触 onset 或姿态，也不设立新晋升门槛。

## 50N 的来源与限制

[Trace and Pace 补充材料第 17 页](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf#page=17) 及 [固定 PACER commit 的实际终止代码](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py#L1392) 给出 50N。代码在 `enableEarlyTermination` 下将 `contact_body_ids` 置零，对剩余 body **净接触力向量先求和再求 norm**，判断严格 `>50N` 且 `progress_buf>1`；另有轨迹偏差、`disableCollision` 和 timeout 逻辑。[同 commit 配置](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/pacer.yaml#L48) 的排除 body 为 `R_Ankle/L_Ankle/R_Toe/L_Toe`。

本地保存量来自 `CrowdSim/humanoid/contact_monitor.py`：每个 30Hz control 的四个 120Hz 物理子步内，对所有 actor body-pair 法向力幅值取最大值，包括足部，排除 floor/self。它与 PACER 被 mask 后的 body 净力向量和不等价；封存的 control 最大值也无法恢复逐子步波形、接触部位、body 净向量或冲量。因此 50N 只用于论文启发的严重度诊断，不能称论文终止条件复现、impulse、因果结论或模型优越性。

## 审阅后执行

在仓库根目录使用含 NumPy 的 CPU Python，明确隐藏 CUDA：

```bash
CUDA_VISIBLE_DEVICES='' python docs/style_improvement_20261004/contact_force_severity_CPU_v1/diagnose_force_severity.py
```

输出仅本目录 `actual_diagnostic.json`；已存在时拒绝覆盖。源码准备阶段只运行小型标量接口检查，不读取真实 NPZ。

## 实际完整结果

每组均为36,000个无序pair-controls。条件P50／P95／max只在全部`F>1N`样本上计算，单位N。

| 捕获 | ≤1N | (1,50]N | >50N | >1N条件P50 | 条件P95 | 条件max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLD63 S1 | 36,000 | 0 | 0 | null | null | null |
| NEWb563 S1 | 35,930 | 22 | 48 | 57.73260 | 959.02307 | 1288.42297 |
| NEWb563 S10 | 35,721 | 77 | 202 | 73.33865 | 327.63305 | 1069.25244 |

S10的>50N频次和条件中位数较高，同时P95与峰值较低；频次与尾部强度有不同变化，不能只据峰值降低认定物理改善。原>1N总数仍为0／70／279，既有0／1／3跌倒等结果保持；所有失败、重置与idle控制均保留。上述是已观察捕获的统计，不是候选数的配对因果结论或论文collision rate。

[实际报告](actual_diagnostic.json)为50,908B，SHA256 `91d1c25c9c079d1bc9971d98598d4f23d3a9998beff7589a3fea221041a5a4cd`；[真实工具记录](actual_CPU_tool_receipt.json)保留closed0、源码及stdout身份。报告内CPU计时0.13840秒。root另以f73bf3核对全部六pair、各档加总与完整分母；report／stdout／receipt均只读封存。未运行模型、SDK、GPU或新仿真。原标量边界检查eb2acf、最终compile154ded及Ruff014658均退出0，准备检查与实际捕获统计分别报告。
