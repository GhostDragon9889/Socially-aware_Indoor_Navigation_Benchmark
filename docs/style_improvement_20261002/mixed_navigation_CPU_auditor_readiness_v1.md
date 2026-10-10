# 混合模型导航：独立 CPU 审查器就绪

新工具：[audit_mixed_navigation_ground.py](audit_mixed_navigation_ground.py)。21项CPU测试与Ruff通过；step500、1000、2000三个真实sealed checkpoint的navigation preparation分别155项通过。没有读取新候选导航捕获、启动GPU/Kit或读取34测试输出。

完整源码／单测／工具归档身份：[就绪JSON](mixed_navigation_CPU_auditor_readiness_v1.json)。实际CPU审查工具独立封存，不冒充GPU135来源或navigation138启动归档。

## 准入与检查范围

- 固定138成员、manifest `285e4126…`／archive `a06eae3d…`，逐成员完整bytes/SHA、安全路径／regular member核验。与GPU135比较，134个业务／runtime成员相同，唯一既有差异为已声明CPU审核工具，加三个参数注册。固定原入口未引入新的AMP inference／deployment／DynBench provider。
- 模型精确匹配终点220项独立admission与selected500／1000／2000完整SHA及原训练budget。注册和actual argv逐键逐类型相等；与旧negative guide20 baseline仅config／resume／output不同，旧缺省near-horizon mass显式视为0。
- 原`audit_ground_support_capture.py` SHA `f851e7d5…`保持不变，全部foot rawforce／1N支持、trueCOM／正质量、四子步clock／episode/reset、peer四字段／最后子步physical投影检查复用；source-admission只替换已独立验证的新configuration身份，完整physical For AST仍为`1adbfe51…`。
- 新增全部6000帧plan时序核验：raw51与reference151、每actor最近plan、post-step相位、reference velocity／deviation／endpoint residual、每15控制步及reset／route-arrival强制joint replan；完整raw计划和每条failure/reset事件留档。路线索引每帧重算，按actor/episode计算完整4顶点循环，不将不同actor的零碎顶点合并算循环。
- 实际旧0F基线6000帧、以及13F旧root模型6000帧通过新增时序门禁；13个失败与17个初始/后续reset全部保留。缺plan／错target phase／改clock／伪arrival均被拒绝。

## 完整实际导航结果就绪后的命令

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
../IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261002/audit_mixed_navigation_ground.py \
--candidate-dir output/diffusion_amp/shared_mixed_curriculum_step2000_ground_support_200s_v1 \
--candidate-step 2000 \
--registered-preparation docs/style_improvement_20261002/mixed_curriculum_step2000_ground_support_200s_preparation_v1.json \
--source-manifest docs/style_improvement_20261002/mixed_curriculum_navigation_source_v1.json \
--checkpoint-admission docs/style_improvement_20261002/mixed_curriculum_terminal500_1000_2000_independent_audit_v1.json \
--output /tmp/mixed_step2000_actual_ground_audit.json
```

`--preparation-only`只审核固定源码、模型与注册合同，不读取捕获并明确`actual_native_capture_validated=false`。完整捕获必须是6000控制帧／24000actor帧／96000actor子步；不补零、筛掉跌倒或改变1N阈值。历史native probe的false保持；actual native capture完整性与效果验收分开，质量字段保持null。姿态风格评估与34测试由各自既有协议负责，不能从本完整性报告推断作者效果达成。

v1新增测试曾把21顶点旧基线误期望为5个循环；按actor真实[2,8,6,5]应为0+2+1+1=4，审查计算正确，仅测试预期修正。失败日志保留，v3为最终21项通过日志。
