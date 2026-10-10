# 原生N3–15独立风格入口：CPU准备与历史保持证据

更新：2026-10-06。S3固定2848真实AMASS训练仍由现有owner独占GPU。
本轮没有启动新的SDK或策略推理，没有新原生捕获、模型质量或论文排名结论。

## 实际完成

- 新入口：`scripts/evaluate_shared_social_native_style.py`；专用边界：
  `CrowdSim/humanoid/analysis/native_navigation_style.py`。
- 显式N3–15、同场景/SHA、两臂完整6000控制步和原生raw CPU外部退出记录。
  默认拒绝旧N4 fixture、短检查、缺失/非整数/失败退出、来源或曝光单位漂移。
- 原生全24身体和Pelvis四元数必须逐值匹配第四物理子步；完整episode/reset与raw分母相同。
  失败片段不筛除。首帧及reset边界没有合法前驱时，单独报告原有区间指标的排除数量，
  同时报出完整捕获和指标端点中的失败数量；不把“保留档案”写成“每帧都进入导数指标”。
- 最终53项CPU检查通过，2.27秒；N3/N15两项来源检查及Ruff退出0。
  条件突变：首个外部CPU退出判断`is int`→`is not int`，实际21失败/32通过，退出1；
  finally恢复原字节，正常套件全部通过。
- 旧S1 DP/SFM两份真实N4 cuboid捕获的样本、inclusive summary、所有45D姿态字段、
  所有90D总体字段及配对共同速度桶字段完全保持。使用原validation参考和原K50/K100中心，
  未重新拟合；未读取Test34。
- v2外部实际退出0：launch `b1b8eb`，terminal `acc542`。四个CPU子进程均退出0。
  23项S3来源guard保持，训练参数、AMASS清单、采样和默认模型未改变。

## 保留的失败

v1外部退出1：launch `2ecc72`，terminal `9e4585`。
它的53项检查、来源检查和45D/90D总体数值核对通过，但直接用Python对象与JSON比较共同桶，
在`metric_spec.speed_bin_lower_edges_m_s`遇到tuple/list类型差异。

v2只修正证据runner的JSON表示边界：要求内存类型tuple、档案类型list、边界值相同，
并对完整对象进行JSON序列化后严格比较所有字段。未使用数值容差，未改指标或参数。
`historical_common_bin_representation_review.json`保存实际类型/值和完整比较结果。
生产reader、CLI与测试未因这次失败更改；v1注册、源码、日志和突变全部保留。

## 来源与范围

直接复用固定原风格函数：15关键点、30Hz、Pelvis相对位置、heading对齐、每人固定链长尺度；
45D pose/K50/K100以及90D pose+velocity、同速度桶MMD²各自报告。
两臂共同桶使用相同reference索引和相同样本数。无支持的桶保留unknown，不补零。

依据[NavIsaacLab§IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)核对表示要求。
作者完整关键点选择、尺度公式、聚类/抽样和MMD实现仍未匹配，因此本地分数不是Table III分数。
历史回归明确只检查N4数学读取；未调用新的原生SDK资源准入，不能记为N3–15原生评价。

## 新原生数据的正式用法

先做CPU来源检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_style.py --actors 3 --check-sources
```

实际原生两臂必须先完成SDK运行、完整raw审计和只读数组封存，再由外部观察者记录每臂raw CPU退出。
该外部记录schema为`shared_social.actual_native_raw_CPU_audit_execution.v1`，要求字段：
`actual_exit_code`（严格整数0）、`terminal_tool_chunk`、`actors`、`control_steps`（6000）、
`planner`、`scene_configuration`和`audit`。身份行含绝对path、bytes、sha256。
新reader重新准入实际SDK/初态/权重/地图来源，不仅检查CPU报告的status。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python scripts/evaluate_shared_social_native_style.py \
  --actors <登记人数> --scene-config <登记场景>.json --scene-sha256 <登记SHA256> \
  --diffusion-audit-execution <DP的实际raw-CPU退出记录>.json \
  --sfm-audit-execution <SFM的实际raw-CPU退出记录>.json \
  --output <新风格报告>.json
```

CPU来源检查不加载参考数组或网络。正式风格分析可用Torch读取原held-out动作，CUDA仍禁止初始化，
SDK禁止导入；没有GPU推理或训练。初态比较默认要求真实原生World档案。
当前原生SDK、真实两臂初态、完整200秒、多seed和八场景协议仍未完成。

`root_observed_CPU_execution.json`、`CPU_evidence_seal.json`及单独的封存工具退出记录保存实际证据。
