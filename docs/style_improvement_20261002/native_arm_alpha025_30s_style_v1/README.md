# Native Warehouse：arm α=0.25 的同 900 帧风格对照

仅 CPU 后处理；未启动 Kit、GPU、训练或模型推理。候选是真实四人 30 秒记录，基线是真实 Native4 完整 200 秒记录的**前 900 个控制帧**，双方窗口相同。原始文件不改写，不创建截短的替代 rollout/result。

| 指标 | source168 基线前 900 帧 | source171 arm α=.25 全 900 帧 |
| --- | ---: | ---: |
| 跌倒 / 路线点总数 | 0 / 8 | 0 / 8 |
| inclusive 90D pose+velocity MMD² | 0.0752332421512 | 0.0732570545068 |
| 实际速度分布 W1 (m/s) | 0.255965591281 | 0.266752721405 |
| 实际 root 差分速度均值 (m/s) | 1.12821839009 | 1.14275766958 |
| 45D pose-only MMD² | 0.0372305498888 | 0.0347047152108 |
| K50 coverage / entropy | .34 / .583076291 | .34 / .593962170 |
| K100 coverage / entropy | .31 / .575572480 | .30 / .568922578 |
| K50 nearest-center mean | .157260634 | .148844314 |
| K100 nearest-center mean | .156575429 | .147446965 |

**短时 MMD 与最近中心距离略改善，速度 W1 和 K100 coverage/entropy 略退。** 不构成普遍动作改善或论文效果验收。该试验 0 跌倒不代表完整 200 秒稳定性，默认控制器不提升。

## 冻结指标与完整分母

[原 90D reader](../../../scripts/evaluate_pedestrian_style.py) 与原 module 的 SHA 82ee083… / b09fa58…逐项核对；c274348… heldout reference 的同一 validation selection、真实数组 SHA、30 Hz、root/heading/chain scale、速度 bins、min30、Gaussian σ=1、每 bin cap256 均不变。四个受支持 bins 保留；<.1 m/s、2–3 m/s、≥3 m/s 缺样本仍 unsupported，不重新分配或调阈值。

每份全部 **900×4=3,600 raw actor frames** 读取；原函数排除每人首帧，共 **3,596 同 episode interval endpoints**、**3,592 acceleration triplets**。没有跌倒、idle、outlier 过滤；primary 没有 upright 过滤。upright 仅原定义的 secondary 行。<.1 m/s 仅 1/3,596 interval，双方相同（按原速度 bins，不构造新停留标准）。

45D 仅复用冻结 `sequence_style_samples.local_pose`：15 keypoint，足为 Ankle body origin；使用已封存 AMASS-only K50/100 中心与 exact SHA，不重新 fit，也不让生成样本参与 fit。pose-only Gaussian biased MMD² σ=1 / cap2048；coverage/entropy 则对所有 3,596 合法 pose endpoints 分配。原 45D 200 秒报告及旧 90D 字典均未修改。作者 scale、KMeans 配置及 bandwidth 未公开，不能与 TableIII 数值直接相比。

## 原生测量与模型来源

复用并重新绑定 [891 项实际原生审核](../native_arm_alpha025_30s_actual_review_v1/README.md) 的产物，不重复执行或声称新 physics 验证。双方前 900 步均每人到达 2 个路线点、static/peer 接触为 0；双支持比例基线每人约 .148–.170，候选 .141–.162。原始首帧、力、COM 与接触门禁保留。toe/Ankle body-origin 速度是代理，不能直接当接触点滑移或生理步态真值。

baseline checkpoint SHA 62c4edc…、arm donor 28aed874…由原生审核全文件 hash；仅 24 个 arm raw means 混合，再统一 clamp/57-active mask，单次 PD 解码。相同观测下的非 arm raw 坐标保持 baseline；跨两条不同实际轨迹与观测不能宣称那些坐标逐位相同。

**运行 epoch 不同。** source168 → source171 不是完整源码单变量实验；5 个 common 文件变化和 3 个新增文件在已冻结原生审核中列出。旧 V5 分支的兼容 CPU 门禁不等于 whole-code exact 或唯一因果证明。此次对比不使用 current tree 替代历史运行 archive。两个实际 raw/result SHA、900 prefix arrays SHA、referential parent disjointness 均列在 JSON。

## 工件与复跑

- [evaluation.json](evaluation.json)：两份完整 inclusive / upright-secondary / 45D / speed-bin 字典、源 SHA、实际 prefix 边界与原生审核引用。
- [execution.log](execution.log)：实际 CPU 执行。
- [evaluate.py](evaluate.py)：新 docs-only 前缀适配器，调用现有冻结公式。
- [source_freeze.json](source_freeze.json)：本次工具与结果 SHA。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261002/native_arm_alpha025_30s_style_v1/evaluate.py
```

输出独占写入，已有报告存在会拒绝覆盖；复跑须另外建立明确版本目录。未读 34 个 sealed test captures，没有新的作者等价阈值或质量 pass。
