# 地形 RMS：只读 CPU 诊断

**存在共同的地形方差稀释，尚不能解释新模型导航退步。** 未启动 Kit/CUDA，未改模型、归一化缓冲、生产代码或资源。

| 保存 checkpoint | Terrain std 中位数 | 保存 task count | 人工 0.2 m uniform 障碍探针 clip 比例 |
|---|---:|---:|---:|
| public PACER | 1.527620 | 61931520001 | 不作为实测导航输入 |
| selected negative1000 | 0.239732 | 8392000 | 29.59% |
| mixed state-only2000 | 0.170591 | 16584000 | 74.02% |
| mixed state-action2000 | 0.170591 | 16584000 | 74.02% |

三组训练保存统计均与“公开统计 count 截至 200000，之后地形输入恒为零”的解析式相符，最大 mean/variance 误差 <9.1e-16。后两模型地形 mean/variance 完全逐字节相同。因此 flat 训练确实缩小继承方差；此事实不是 state-action 与 mixed 差异的单独原因，selected 零摔倒基线也有同类变化。

## 保存姿态重建

每隔 30 个控制帧取一帧，200×4=800 actor 帧；读取实际最后物理子步的身体位置与完整 XYZW 旋转。根位置/旋转与 physical capture 逐字节匹配。采用原 head-centered 32×32 地形采样公式、真实 head yaw、已声明 cuboid 顶面以及已知零高度地板，重建 **post-step terrain1024**。没有原 pre-action task 观测，故不是当次运行在线 clipping 实测。

| 模型 | 重建坐标 clipping | 重建 actor 帧任意坐标 clipping | 非零地形坐标 clipping |
|---|---:|---:|---:|
| selected | 1.147% | 39.25% | 100% |
| mixed state-only2000 | 2.830% | 48.50% | 100% |
| mixed state-action2000 | 0.727% | 15.25% | 100% |

非零障碍高度在本次重建中全部饱和，但大多数地形坐标仍为零。不同模型走过的位置不同，不能将 clipping 频率差异直接归因不同 normalizer。人工 uniform 探针只显示灵敏度，不代表场景像素概率。原地形边界/碰撞几何正确性不在此诊断中重新验证。

## 下一实验建议

先显式记录少量真实 **pre-action** raw1024 与未裁剪 z 值，分地形像素和 actor 帧统计；若需机制消融，可只读加载相同 actor 并明确记录“公开地形统计推理替换”的诊断身份，或在全新等预算训练中固定地形 suffix 统计。不能静默替换 learned encoder 所对应的部署尺度，也不能凭此保证停滞改善。现 state-action 近时域目标本身较弱，且 mixed/state-action RMS 相同；单独 RMS 实验应作为次级解释性诊断，而不是默认推广修复。

详细证据：[audit.json](audit.json)；复跑工具：[diagnose_navigation_terrain_rms_v1.py](../diagnose_navigation_terrain_rms_v1.py)。
