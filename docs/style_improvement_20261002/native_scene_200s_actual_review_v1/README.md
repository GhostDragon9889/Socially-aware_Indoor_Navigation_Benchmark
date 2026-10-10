# 原生 Warehouse 四行人 200 秒：CPU 审计准备

当前状态：**prepared，实际终态审计待 root 通知**。登记不读取运行中的 `progress.jsonl` 或部分捕获，不生成终态 passed，不推测跌倒、路线、接触或 COM 结果。

目标：`output/diffusion_amp/native_warehouse_clearance08_200s_v1`。预期 6,000 控制步、4 人、200 秒；24,000 actor-controls、24,000 shared physics substeps、96,000 actor-substeps、384,000 足部观测，以及原生 live-context 的 4 起点 +128 freecell 共 132 条 all-hit 地板射线。

## 保留的原门禁

`frozen_v1.py`、`frozen_v2.py`、`frozen_routes.py` 是既有真实 30 秒审计所执行三份源码的**完整字节复制**；输入哈希保存在 `frozen_reader_inputs.json`。原函数已经按实际 T 计算，无需更改 30Hz/120Hz、>1N 力阈值、四子步、COM/quaternion/mass、全部失败/reset/episode、static/peer 力、地板/filter/rig、132 native all-hit 或路线循环公式。wrapper 仅绑定特定 source168、终态 6,000 步/4 人计数、部署预登记与时长说明。

source168/V4 manifest SHA `07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f`，归档 SHA `310b87c79851ac44ffe02efc282cb412bb31f27236340242e39f21724cadc9f0`。登记阶段逐一复核 168 个归档成员和当前源码；实际审计会再复核并要求与登记值一致。当前任务的 root 约定保持这些路径至实际审计完成；若之后合法修改 current tree，既有归档和历史审计仍是历史身份，本工具的新一次 current-exact 复跑应明确拒绝，不能把 current 漂移偷偷忽略。

Warehouse 沿用配置绑定的真实 v5 资源图及历史 producer snapshot，未将 CPU v8 proposal 替换成已运行场景。当前 source168 是 prelaunch bytes 证明，不冒称全部 168 文件都是已观测 GPU imports。

## 实际终态后执行

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python -B \
  docs/style_improvement_20261002/native_scene_200s_actual_review_v1/audit_native_200s.py \
  --output docs/style_improvement_20261002/native_scene_200s_actual_review_v1/actual_audit.json
```

目标不存在或未完整时硬失败；既有输出拒绝覆盖。原 native_capture_probe 标志与原 result/NPZ 不改。`result.status=passed` 仅表示执行完整；wrapper 沿用原审计按真实跌倒分别给出 bounded observation 或 stability failed，质量和作者八场景结论不由登记获得。

COM 与地板/接触/子步核验来自已保存真实物理通道，CPU 不重启或重测原 stage。native engine timestamp 与 App/context 构造计数未独立观测，保持未知；submitted counters 不改称 engine time。足部速度仍是 body-origin 代理，不是接触点真实滑移。
