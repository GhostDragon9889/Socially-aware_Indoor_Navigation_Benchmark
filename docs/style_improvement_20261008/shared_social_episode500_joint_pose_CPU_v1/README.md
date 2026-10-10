# 本轮episode500：同瞬时关节位置与刚体旋转核验

CPU实际退出0、Ruff通过；复用已核验的8192个物理状态和同一归档的时钟／重置证明。
对每状态23个非根关节，共188416个旋转比较：

- 69维原生位置按每关节三维旋转向量计算`Exp(position)`。
- 相同瞬时的父子世界旋转给出`R_parent^-1 R_child`。
- 两者相对旋转的测地角均值 **7.72083e-8 rad**、最大 **6.26320e-7 rad**，通过1e-5 rad门禁。

本轮这批状态与旋转向量位置定义一致，未发现需要Euler转换的证据。
所有失败／重置控制步保留，没有修改当前位置、速度、资产或当前训练。
该核验不代表全训练曝光、任意资产的关节定义或动作风格质量。

## 公开资料与计算边界

[PhysX球关节文档](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#articulation-joints)
说明多轴球关节旋转采用分解四元数积分。该公开文档不代表本机二进制的精确提交；实际参数语义由上述捕获数据核验。
[SciPy旋转组合](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.__mul__.html)
约定`p*q`先应用q，再应用p；因此残差计算为`Exp(position)^-1 R_parent^-1 R_child`。
使用独立SciPy旋转运算，实际库版本保存在结果中；没有依赖原AMP编码器来构造旋转残差。

## 证据与复现

- [结果、逐关节误差和输入身份](result.json)
- [保存的全部旋转误差](rotation_errors.npz)
- [实际CPU退出](actual_CPU_execution.json)
- [源实现快照](source_snapshot_index.json)
- [宿主训练观察](training_host_observation.json)：06:04:00，789/2000、6463488转换，实际进程身份保持。
- [同批速度与相位核验](../shared_social_episode500_joint_velocity_CPU_v1/README.md)

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_intermediate_joint_pose.py --output /tmp/episode500_pose_new
```

输出目录必须不存在。原速度审计、实际归档及来源身份在执行前后核对。
未加载策略／优化器或启动CUDA／SDK；未恢复物理World，未重启训练，也未认证2000更新终态。
