# v7 四场景实际证据读者：已准备，原生结果待生成

入口：[audit_actual_four_scenes.py](../audit_actual_four_scenes.py)。最终登记为 [registration_v2/report.json](registration_v2/report.json)，其中 `native_v7_execution_observed=false`、场景列表为空、路线验证为 `null`。它不读取旧 v6 结果来填补缺失结果，也不启动 SDK/GPU。

此目录保存的 `probe_paper_scene_maps.py` 只是历史源码快照，SHA 为 `a538bf92ffb64b536a7005329f38c080bd1b8818114df58d9880d5506f0665d5`；读者不会执行它。生产者仍需由独占 GPU 的实际 probe 单独运行。

## 实际结果生成后

在仓库根目录执行，输出路径必须尚不存在：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
  ../IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261002/native_map_admission_v7/audit_actual_four_scenes.py \
  --native-output output/scene_maps/native_reference_four_scenes_clock_obstacles_v7 \
  --probe-snapshot docs/style_improvement_20261002/native_map_admission_v7/actual_reader_preparation_v1/probe_paper_scene_maps.py \
  --output docs/style_improvement_20261002/native_map_admission_v7/actual_four_scenes_v7_v1
```

必须完整审计 Office、Warehouse、Full_Warehouse、Hospital。任何缺失、失败、不同 probe 身份或合成 fixture 均返回非零并保存失败原因；先前实际通过的场景记录保留。只有四场景全通过才出现 `actual_four_scene_v7_CPU_evidence_audit_passed`。读者观测的是保存的真实原生资源记录，不是本次新启动仿真的结果。

`verify_v7` 重新计算 raw4/5/6 固定布局、canonical known5 完整覆盖、Office 缺 API 行的显式来源、单格 padding 世界裁剪、单选定地面/已知低障碍 all-hit CSR、全高度 CSR、所有 mask/count/path 和不推进的实际查询时钟。读者另外保留各 NPZ 数组的 dtype、shape 和 C-order bytes SHA，拒绝 object/nonfinite buffers。

可重复添加 `--scene-config <new_config.yaml>`。每份配置必须绑定本次实际审计的同一 preflight/result bytes；读者核验所有出生点到第一路点、各路点和 wrap 的 A*，不会把旧 Warehouse v5 配置当 v7 配置。未提供配置时路线验证保持 `null`。

单地面原生资源通过、CPU A* 通过和实际人体导航成功是三个独立结论。此读者不会声称人体碰撞、步态或导航通过。原有 v5 的 bytewise 回归和 v6 raw CPU proof 保持独立、原字节不变。
