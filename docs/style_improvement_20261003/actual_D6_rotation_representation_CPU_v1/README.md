# 已录制 IsaacLab 关节旋转：实际CPU核对

使用已闭合、封存的state-only匹配采样试验全22条validation capture，将同帧实际body世界旋转转为parent-relative局部旋转，再与已保存的common DOF比较。source MJCF和capture metadata的24body／69DOF名称、顺序严格匹配；保留初始帧、四条跟踪失败和全部后续帧。

实际CPU会话30134，启动块895da4、闭合块567123，退出0。没有Torch、模型加载、SDK、GPU或新物理步，没有读取封存34条测试输出。[闭合记录](actual_CPU_closedreceipt.json) · [全部结果与22条来源](actual_rotation_comparison.json) · [入口](check_recorded_rotations.py)。

| 解释同一组已录制DOF的方式 | 276,092个局部关节观测的平均角差 | 最大角差 |
| --- | ---: | ---: |
| 当前exp-map | 8.23e-8 rad | 7.99e-7 rad |
| intrinsic XYZ Euler（诊断） | .056553 rad | 1.223419 rad |
| extrinsic xyz Euler（诊断） | .055421 rad | 1.107925 rad |

已录制AMASS参考的exp-map核对最大角差3.37e-6 rad。当前DOF与实际PhysX body姿势的一致性达到这些float32记录的数值精度；保留当前表示。核对使用SciPy的XYZW输入和SO(3)测地角，等价四元数符号和2π别名不会造成假差异。

这份结果限定于当前rig、记录源和实际IsaacLab迁移接口。它不证明任意D6／MJCF配置、其他backend或完整动作质量相同，也不会修复已观察到的旋转跟踪误差。下一轮动作风格训练继续隔离检查判别器时序，不据猜测更换旋转表示或PD目标。
