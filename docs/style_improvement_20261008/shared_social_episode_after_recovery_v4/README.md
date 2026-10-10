# S3-R完成后的episode候选：修复接口与CUDA时钟

## 当前进度

2026-10-08 02:48:35（Asia/Shanghai）：8人／256人训练与两项CPU审计均由owner记录实际退出0。
1／2次更新全部执行，32／16,384物理转换、终态清理及有限张量检查通过。
实际OS owner473700已进入2000更新，正式SDK训练PID487451。
2026-10-08 03:31:36宿主核对：两个进程命令、父子关系及owner启动身份保持，
159／2000更新、1,302,528转换日志完整连续且有限；训练仍在运行。
最近20次平均16.13秒／更新，剩余训练约8.25小时，动态估计结束11:46；不含独立评估且不是完成保证。
请以[owner状态](owner_state.json)、训练metrics和之后生成的`completion.json`为准；尚未有完整训练终态。
本次从新模型初始化，保持seed7和固定真实AMASS 2848，不转移S3-R权重或优化器状态。

## 已完成的验证

- 10项CPU测试、21个subtest实际退出0；修复前的类型来源拒绝仍然有效。
- 三个训练stage与独立评价的四项冷预检实际退出0，未导入Torch或启动SDK。
- [实际资源探针](reference_clock_probe_result.json)覆盖CPU／CUDA各8人、256人，实际退出0；
  AMASS重置张量与RNG保持，重规划不重新抽取episode路径，状态恢复通过。
- 30Hz原生偏移末端为5.0秒；超过五秒仍拒绝；CPU／GPU资源检查未执行网络训练或物理模拟。
- [实际工具退出记录](reference_clock_probe_execution.json)、[计划](owner_plan.json)、
  57份[执行源码快照](source_snapshot_index.json)和7项输入已固定；全部AMASS bank哈希重算保持。
- Ruff通过。v2和v3两次SDK失败及原输出保留，不复用失败目录。
- [外部冷核验](smokes_external_verification.json)实际退出0：两轮物理预检的实际退出、
  原登记参考配置、保存provenance、专家来源及只读输出一致，57来源／7输入保持；未启动Torch／SDK。
  首次核验脚本误选不存在的smoke256 CPU登记键而退出1，失败与原脚本保留；v2使用同配置的training256登记后通过。

## 具体修正

1. 专家采样器与收集器的私有模块依赖，显式从旧World绑定到已登记的episode World。
   保持原类／来源检查；`prepare_control`、`step`、`complete_transition`三个方法AST与原World完全相同。
2. 原CUDA float32 `150 / 30` 超过五秒一个ULP，触发严格参考查询失败。
   对程序参考原生偏移、episode年龄及状态验证的30Hz除法，显式使用float64后转回float32。
   两个私有模块分别有3处／1处精确可逆替换，派生源码哈希写入结果与权重provenance。
3. 使用单独的S3-R完成记录与独立评价准入，原S3连续训练证书仍缺失，未补造证书。
4. 新smoke8_v3输出目录保留两次失败；未使用的256人和pilot2000登记目录保持。

数值实现参照固定版本的[PyTorch 2.11 arange](https://docs.pytorch.org/docs/2.11/generated/torch.arange.html)
与[div](https://docs.pytorch.org/docs/2.11/generated/torch.div.html)；具体CPU／CUDA差异来自本机实际测量。
OS owner以[Python Popen.wait](https://docs.python.org/3.12/library/subprocess.html#subprocess.Popen.wait)
记录真实退出码，逐阶段串行执行，保留其他GPU进程。

## 完整执行顺序与预算

| 阶段 | 人数 | 更新数 | 每次控制步 | 物理转换总数 |
|---|---:|---:|---:|---:|
| smoke8 | 8 | 1 | 4 | 32 |
| smoke256 | 256 | 2 | 32 | 16,384 |
| pilot2000 | 256 | 2,000 | 32 | 16,384,000 |

每次SDK训练完成后执行原Pop-Art CPU审计。只有所有训练及CPU审计实际退出0，
才调用原独立actor、完整validation22跟踪／冻结风格、导航短测、200秒导航与CPU数组审计。
任何阶段失败都停止后续阶段并保存失败。权重预选`step_002000.pt`，不依据test34选择权重。

运行入口：

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/own_shared_social_episode_pipeline.py \
  --plan docs/style_improvement_20261008/shared_social_episode_after_recovery_v4/owner_plan.json \
  --check-plan
```

该计划已启动，不能重复启动进入现有输出。SDK结果、实际退出码和数值审计完成前，不接受终态权重。

只读进度查询：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/shared_social_episode_status.py
```

新增查询适配本轮私有训练入口，核对完整日志、进程命令、父子关系和owner启动时钟；
6项检查与7个subtest、Ruff及实际宿主查询退出0。它不加载权重、不启动或重启训练。
权重文件存在不等于写入完成或恢复准入，查询对此只报告未验证文件观察。
查询及文档均位于已固定的57来源／7输入之外；正式训练来源复核实际退出0。

独立[同预算比较观察器](../shared_social_episode_comparison_preparation_v1/README.md)PID521166已启动。
它等待正式训练、CPU审计及独立评价全部实际成功后自动执行原评分的CPU对比，
尚无比较结果；不启动第二轮GPU训练。

## 效果与论文对齐

本次修复确保登记方法可以执行；尚未证明学习或风格／路线／物理质量改善。
episode程序参考在重置时生成固定路径，AMASS未来31帧保持；私有RNG曝光与末端缓冲差异已声明。
作者2848动作ID、完整八场景／初态及风格评分私有细节仍不齐，不能声称精确作者配方或达到／超越论文。
默认模型保持。此前恢复训练12500及组合独立评价已完成，详情见[S3-R对比](../shared_social_S3R_comparison_v1/README.md)。
