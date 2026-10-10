# 原生SFM与固定本地资源队列：实际CPU准备

2026-10-05，本轮使用已准入v9的四个真实原生地图。新代码、源码快照、实际退出与全部数组保留在本目录。
完整方法、接口和剩余验证见[原生共享AMP接入文档](../../SHARED_SOCIAL_NATIVE_EVALUATION_ZH.md)。

## 实际结果

- [资源预登记](resource_cohort_registration.json)：四地图各N3–15，共52个案例；固定clearance/出生间距及选择算法。
- [资源CPU报告](resource_cohort_CPU.json)：工具`679e0a`实际退出0；48通过、4拒绝，2058段A*通过；源资源不变，未导入Torch。
  Hospital N12–15拒绝，完整错误行保留；不换场景、不降低1.6m最小间距。
- [预测预登记](CPU_prediction_registration.json)：48个已通过资源准入的配置，登记前没有SFM预测结果。
  登记同时保留上述四个资源拒绝，资源准入数不能充当导航成功分母。
- [实际CPU退出](CPU_prediction_execution.json)：子进程实际退出0，外层工具`8560e6`也实际退出0，源文件前后不变。
- [完整CPU预测报告](actual_CPU_snapshots_v1/CPU_preparation.json)：48个完整联合预测通过，每人51个XY knots，0至5秒。
  注册的world XY、零速度/heading与重复合成历史不是AMP测量。全部输入和输出档案保留。
  没有加载AMP/DP权重、启动SDK或CUDA、测量物理/风格表现；RNG不变。
- [数值与来源检查](tests_v3.log)：17 passed in4.60s，实际退出0；另与入口联合最终检查30项通过，见入口准备目录。
  首次pytest自动加载ROS插件失败，原始`tests_v1.log`保留；禁用无关插件后`tests_v2.log`16项和`tests_v3.log`17项通过。

## 重跑

资源队列脚本[prepare_real_cohort_v1.py](prepare_real_cohort_v1.py)拒绝覆盖本目录的已有登记/配置。
预测入口支持显式选择这些已发布配置及一个新的输出目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/prepare_shared_social_native_sfm.py \
  --scene-config docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/scenes/Warehouse_N03.json \
  --seed 7 --output /tmp/native_sfm_CPU_snapshot_new
```

此处成功仅指合成快照预测通过。原生SDK闭环、两臂初态配对、完整200秒和作者八场景效果仍未完成。
