# 30秒显式入口独立对照工具已准备

13项CPU回归、Ruff通过；真实旧negative200前900控制帧已只读核验。新30秒运行尚未产生，因此 candidate_comparison_executed=false、paired_bitwise_equal=null；没有输出新物理配对成功。

旧前30秒实际数据：900×4=3600控制actor帧，3600共享物理子步/14400actor子步，57600foot观测；0fall/0time-limit/0额外reset，到达[1,0,1,0]，static/peer均0，double-support .324375、no-support .0111111，最大COM组合残差1.90735e-6m。此为旧运行的短前缀，完整200秒仍是0fall/21vertices基线。

旧selected7份source snapshot、后续guide40的70member tar、当前135member tar及新entry5member tar均逐字节可恢复。70tar是guide40预启动源，不作为旧guide20的完整runtime来源证明。三版本完整物理控制For AST相同。旧result没有完整per-file代码attestation；各证据用途分开记录。

实际新capture完成后运行：

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python docs/paper_pedestrian_deployment/compare_physical_prefix.py \
  --candidate-dir output/diffusion_amp/paper_pedestrians_baseline_30s_v1 \
  --output docs/paper_pedestrian_deployment/actual30s_prefix_comparison_v1
```

输出必须全新。工具核对同一真实权重/Diffusion/配置/场景/manifest SHA及56旧→57新参数合同，只允许路径绝对化、输出及6000/6001→900/901预算；实际data逐dtype/rawSHA/bitwise对照，差异记录而不自动通过。ground原测量公式审计复用固定helper，只有source admission callback不同。

第一物理帧是控制后的body/DOF/root；原始plan保存起点XY，可以实际比较，但未捕获控制前GT pose/history，不能称已逐字验证这些不存在的字段。native engine clock未知，接触阈值严格>1N，tangent速度仍是body-origin proxy；无true slip/作者指标声明。
