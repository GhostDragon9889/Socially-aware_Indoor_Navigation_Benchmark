# 策略调用前观测：真实 30 秒独立 CPU 审计

**最终结果使用 `actual_cuboid30_v2/report.json` 与 `native_and_moments.json`。** 原 V1 及 executed_v1 读者保留：它的三个未使用 outcome 字段采用旧名称而返回 null；V2 改用实际 `fall_resets/time_limit_resets/elapsed_s`，重新执行完整相同准入/统计/配对，结论未变。该读者不修改生产或权重，不启动模型/SDK/GPU。

## 实测结论

- 原生运行 `pre_action_baseline_cuboid_30s_v1` 实际900控制帧、4人、30秒、0跌倒、0时间上限reset，顶点到达 `[1,0,1,0]`。这不是200秒稳定性或论文效果的验收。
- 30份 NPZ/JSONL 记录完整，indices为0,30,…,870；每项实际文件SHA、数组dtype/shape/C-order bytes、moments组合SHA通过。**120个抽样 actorframes**，不能称3600帧均被观测。
- 地形后缀1024：122880个输入元素中2061个裁剪，1.67724609375%；全部负向，60/120抽样actorframes有裁剪。轨迹前缀20：2400个元素，零裁剪。逐人/逐样本及raw/z分布全部保留于JSON。
- 地形原始输入范围[-10,0]、未裁剪z最小−48.48595047。这里raw−10是 PACER式 `(center_height-measured_Z).clamp(-3,3)*5` 的相对障碍高度特征（本fixture2m高），不是−10m真实地面高程。它不能单独证明裁剪导致闲置、退步或跌倒。
- 实际保存的task RMS count8392000，30样本的mean/variance/count全SHA相同；地形标准差min/median/max=.02572649/.23973207/.37119487。未采样的其他调用不由这些快照直接观测。

## 默认行为与计量边界

历史 source168 `pure_actor_only_baseline_30s_v1` 与实际 source177 的57项driver参数仅output不同；physics/runtime numerics/dataset/scene/trainingConfig均相同，权重/资源完整SHA重算一致。三个capture的**全部字段**（physical_states/reference_plans/ground_support）dtype和C-order字节完全相等，没有用容差替代原数组比较。源码168与177的选定文件仍有明确差异，列表和两份完整archive验证保留；这不意味着一般 PhysX/Warp bitwise 保证，也不是完整执行导入账本。

源码177 AST和已封CPU合同证明：原输入clone/z计算在act前；原act只调用一次，发布在act返回后、world.step前。每次抽样有两次额外normalizer forward，零额外模型forward；**没有为真实运行添加模型forward计数器或原生时间戳**，所以模型调用数区分源码/CPU证据和运行实测。实际加载的新增两helper/script路径、大小、SHA与archive逐一绑定。

原冻版native force/COM读者保持数学门禁，仅用独立source177/archive/resource准入回调替换它的历史源码身份分支：204项实际CPU检查通过。900 controls、3600共享120Hz子步、14400 actor-substeps、57600足部观测，submitted counter0→3600；没有原生引擎时间观测。真实COM独立坐标重构最大误差9.94e−7m；static/peer接触记录均零，floor不混入上述通道。1N阈值下双支撑率32.4375%，无足支撑1.1111%。足部link原点切向速度仍是proxy，不是真接触点slip。

所有900帧、reset/terminated/truncated与所有规划事件均参与比较，不筛失败帧；本次实际无失败帧。30份样本零异常，读者负例证明重算SHA的坏clipmask仍拒绝、后续有效行保留、缺末份样本不可报完整、微小浮点/dtype差异保留。4项CPU tests与Ruff通过。

## 重跑

只能向新证据目录写，不能覆盖历史：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/pre_action_actual_observation_reader_v1/audit_actual.py \
  --output /tmp/pre_action_independent_review_new
```

完整补充读者的既有输出目录固定用于此次审计、重复执行会拒绝覆盖。`source_freeze.json`记录实际读者、报告、JUnit和文档SHA；生产177与其后新的训练版本分开，历史实际数据不改。
