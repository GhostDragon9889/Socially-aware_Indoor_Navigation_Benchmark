# 原生 Warehouse：arm α=0.25 的实际 30 秒独立审核

## 实际结果

**891 项原生审核门禁通过；这里只验证这次 30 秒记录。** 原 `native_scene_sdk_review_v2` 的 raw / COM / force / floor / route 公式与门禁未修改。未启动 Kit、GPU、训练或模型推理。

| 项目 | 本次 arm α=0.25 | 原 Native4 完整 200 秒的前 900 步 |
| --- | --- | --- |
| 四人控制步 / 时间 | 900 / 30 秒 | 900 / 30 秒 |
| 跌倒 / 时间截断 | 0 / 0 | 0 / 0 |
| 每人到达路线点 | 2 / 2 / 2 / 2 | 2 / 2 / 2 / 2 |
| 完整路线循环 | 0 | 0 |
| 障碍接触 actor-control | 0 / 3,600 | 0 / 3,600 |
| 行人接触 unordered-pair-control | 0 / 5,400 | 0 / 5,400 |
| 最低 pelvis 高度 | 0.824024 m | 0.825075 m |

本次累计 3,600 个 shared physics 子步、14,400 个 actor 子步、57,600 个足部观测。所有 900 控制步和四个子步完整保留、valid；计数连续 0→3,600。episode / reset / termination / truncation 与物理记录逐字段相同，未删除失败样本。

四人实际路径距离为 34.416 / 34.051 / 33.555 / 34.957 m；基线同窗口为 34.085 / 33.902 / 32.680 / 34.569 m。每人平均 root–执行 target XY 误差为 0.1076 / 0.0978 / 0.1032 / 0.0975 m，基线为 0.1120 / 0.0969 / 0.1003 / 0.0996 m。这是同窗口描述统计，不能据此认定步态风格改善。

## 原生地面、足力及 COM

实际 live context / stage 身份、745 个静态碰撞体、唯一地面 `/__PaperMapProbeReferenceGround`、原始 Plane / decals 禁用清单与基线合同相同。四个初始 root 加 128 个 admitted free cell 的 **132 条 native all-hit rays** 均保留；地面 support 命中恰好一个，查询 submitted counter 0→0，原始 53 个 source layer 文件全 SHA 再核对。没有把未知 native engine timestamp 或 App 创建次数伪补为零。

四足 floor-only filters 顺序、1 N 原始力门槛、side support / 双支持 / 无支持、每个子步同瞬时 COM 与质量加权 COM 全部重算通过。COM 最大重构差 9.5367e−7 m，质量加权 COM 最大差 4.7684e−6 m；body masses 全部为正且等于运行 rig 声明。最终子步 body-origin 位置与 physical common-body 捕获逐字节相同。

原 `native_capture_probe_passed` 历史值未改。本独立报告只声明实际 30 秒捕获经审核。足部 body-origin 切向速度仍是代理；没有原生接触点滑移 / 摩擦或 AMASS 真实力、COM ground truth，相关质量结论不补造。

## 真实源身份与对照范围

本次实际 source171：manifest `8635b1367b1186f8aa6eeacacae317a9e70f6398c107e8b67bea8f818201bf9f`，archive `b03bd7875f61bcf5e90f2134df58efe44ed0d802d86e9b3d1fd63f6a412441b8`。运行后 171 项当前文件均与运行前 archive 字节完全相同，已封存 [post snapshot](source171_actual_post_snapshot.json)；后续 root 合并不会成为本次执行源。

基线是**真实 Native Warehouse4 完整 200 秒记录的前 900 个控制帧**，不是 cuboid 场景。原 source168 manifest `07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f` 与原完整审核 JSON / 五个 raw artifacts 均逐 SHA 绑定。

两次 rig、physics、scene、provider、contact、foot capture 合同逐对象相同。driver 参数差异只有 output、900/6000 evaluate_steps、901/6001 episode_steps；这两个时间上限均未在前 900 帧触发。actor 顶层 baseline SHA 为 `62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b`，arm donor SHA 为 `28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72`，完整文件再 hash 通过。明确混合 24 个 arm raw means，再统一 clip / 57-active mask；两 actor 分别使用各自冻结 RMS。嵌套 loader 的 `prepared_not_runtime_integrated` 是准备时元数据，本报告没有篡改它。

**整体源码不是单变量实验。** source168→171 有 5 个 common 文件变化及 3 个新增文件，完整路径见 JSON。原 V5 数学在新 dispatch 中的兼容证据另有 CPU 门禁；本实际审核不能把两 epoch 宣称完全相同代码或独立证明唯一因果。

## 原始 prefix 差异完整保留

physical / ground / reference-plans 的 **77 字段：26 byte-exact、51 不同**。独立 `physical_roots.npz` 的冗余 root 字段另计后为 78 字段、26 相同 / 52 不同；每次 redundant root 都与自己的 physical state root 逐字节相同。

最大 root position 分量差 1.1741004 m；最大 body-position 分量差 1.2700009 m。candidate 272 条 planner rows，baseline prefix 268 条，不能对不同形状使用 `allclose` 或偷偷截短 / 搜索对齐。所有字段的 shape / dtype / raw SHA / 相等判断及适用的最大差完整记录在 JSON。

本次 0 跌倒不等于 200 秒质量、八场景或论文 style 验收。该独立报告不计算 / 改写 MMD、coverage、既有 style 指标或模型选择。

## 产物及复现

- [actual_audit.json](actual_audit.json)：891 门禁、全部数组 SHA、source / registration / models、raw prefix 差异。
- [actual_summary.json](actual_summary.json)：同窗口描述、冗余 root closure、明确质量边界。
- [audit_actual30s.py](audit_actual30s.py)：复用三个原冻结 reader，不改变公式。
- [summarize_actual.py](summarize_actual.py)：只读上述结果及基线 prefix。首次执行后仅删除无用 `Path` import，实际执行前版本另外保留，数学逻辑未变。

输出使用独占写入保护，不覆盖已封存证据：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python -B \
docs/style_improvement_20261002/native_arm_alpha025_30s_actual_review_v1/audit_actual30s.py
```
