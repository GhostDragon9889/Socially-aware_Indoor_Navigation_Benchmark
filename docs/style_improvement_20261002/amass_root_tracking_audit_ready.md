# AMASS root full22审计准备

新读取脚本 `audit_amass_root_tracking.py` 已准备；8项 synthetic CPU contract tests 与 Ruff通过。没有读取尚未完成的新训练或新full22结果，没有读取34条test。

复用冻结 `decompose_motion_tracking.py` 和原negative-only cached decomposition。只接受validation全22段、无clip或step裁切；初始physical state/history和参考poses/XY目标/phase/time要求dtype及数组bitwise相同。记录实际checkpoint SHA、两个源码tar逐member SHA及captured runtime-source绑定，且不把当前工作树冒认为历史运行源。

新附加诊断：root XY/yaw、heading-local位置/姿态、parent-local关节旋转、reference-speed bins、每次fall连续时段/真实source帧/相对reset时间及首fall前后1秒。所有22与fall帧仍留在主指标。success-only与两run共同success交集另报clip/frame两分母，只作描述性分母敏感性分析，不替换主指标或宣称对应论文.218/.343。

资源agent的官方style/MMD不在本脚本重算。新run完整passed后执行：

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/audit_amass_root_tracking.py \
  --candidate-result output/diffusion_amp/motion_tracking_amass_root_curriculum_full_validation_v1/result.json \
  --candidate-training-result output/diffusion_amp/amp_style_amass_root_curriculum_v1/result.json \
  --output docs/style_improvement_20261002/amass_root_curriculum_tracking_v1
```

计划产物：comparison.json、amass_root_decomposition.json、report.md。历史证据禁止覆盖；defaults已绑定negative-only1000和两份冻结源码manifest。最终脚本SHA/准备门禁见 `amass_root_tracking_audit_preparation_v2.json`。新训练/导航/作者复现的结果尚未由此脚本验证。
