# 本轮episode500：同瞬时球关节速度核验

CPU实际退出0；4项独立SciPy坐标测试／4个subtest及Ruff通过。
读取本轮500更新的32控制步×256人完整归档，共8192个实际物理状态。
同一归档中关节速度、刚体姿势和刚体角速度进行同瞬时比较；没有位置有限差分。

| 对照 | RMS rad/s | 最大绝对差 rad/s |
| --- | ---: | ---: |
| 全69维原生速度−子关节角速度 | 3.07215e-7 | 6.26505e-6 |
| AMP57维原生速度−子关节角速度 | 3.28659e-7 | 6.26505e-6 |
| AMP57维原生速度−父关节角速度 | 0.549358 | 13.771402 |

计算世界角速度差`omega_child - omega_parent`，分别用当前子／父刚体的逆旋转转换。
全部23关节、69维的子坐标差通过1e-5 rad/s门禁；AMP57逐值等于实际映射后的原生速度。
本轮这批实际状态支持现有child_joint坐标选择，不需要再次旋转或用旋转向量分量差分替代。
这只核验该资产与这批状态，不推广到任意资产或整个训练曝光。

完整原相位检查通过：7840条同episode PRE←POST连续关系、256首行及96重置边界独立保留。
全部8192行COMMON／native根速度逐值一致；POST AMP根部XY速率最大差4.13462e-7 m/s。
同批96个重置控制步和所有失败状态保持；没有挑选正常步来评分。

## 证据与复现

- [结果／完整来源及逐关节误差](result.json)
- [保存的全部子／父坐标误差](sameinstant_errors.npz)
- [实际执行及测试记录](actual_CPU_execution.json)
- [实现快照索引](source_snapshot_index.json)
- 同一归档的[位置旋转向量核验](../shared_social_episode500_joint_pose_CPU_v1/README.md)

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_intermediate_joint_velocity.py --update 500 --output /tmp/episode500_velocity_new
```

读取完整501行日志前缀，确认保存后训练继续推进；不据此认证2000更新终态。
归档及源码执行前后哈希保持，原归档没有复制或改权限。没有加载策略／优化器或启动CUDA／SDK。
当前模型的独立风格、路线及物理表现仍须在训练终态后评价。
