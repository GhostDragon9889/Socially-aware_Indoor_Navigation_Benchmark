# PACER 低层 AMP 移植与验证

日期：2026-10-01。目标为当前 Isaac Sim 6 / IsaacLab 3 的独立 SMPL 行人链。
本报告覆盖 CPU 数学及网络推理；本子任务没有启动 Kit、训练 GPU、安装旧 SDK 或修改环境。

## 1. 来源与权重边界

公开参考是 [NVIDIA PACER](https://github.com/nv-tlabs/pacer/tree/baeb6241f03fc9d78370a64e3c2dcef9d830d97f)，
固定 revision `baeb6241f03fc9d78370a64e3c2dcef9d830d97f`。原文件及许可证在
`third_party/pacer`，本地副本的相关源文件哈希见 `pacer_cpu_verification.json`。
公开仓库声明 CC BY-NC-SA 4.0；原 SMPL / AMASS 模型和数据仍适用各自许可。

只加载已核实的公开 `pacer_no_shape/Humanoid.pth`：119,755,239 字节，SHA-256
`bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615`。
加载先验证 hash，再使用 `torch.load(weights_only=True)`；完整验证 actor、critic、
task encoder、discriminator、固定 log sigma 的 **23 个模型键**，以及 observation、
value、discriminator 三组均值/方差/count。错 hash、错维度、缺键、非有限值、负方差
均拒绝。没有 MaskedMimic 替代路径；未加载权重的 reference controller 不能执行 act。

这些是 PACER 参考权重。没有证据证明它们就是 NavIsaacLab 论文发布的精确训练产物。
新论文合同的训练使用显式创建的可学习网络和独立任务/鉴别器，不自动冒充作者权重。

## 2. 实际接口

模块在 `CrowdSim/humanoid/amp`，只依赖当前 Torch 与通用 XYZW 旋转工具。

```python
from CrowdSim.humanoid.amp import PacerAMPController, action_to_pd

controller = PacerAMPController.from_checkpoint(
    checkpoint, dof_limits_lower, dof_limits_upper,
    body_names=actual_body_names, dof_names=actual_semantic_dof_names,
    device=device,
)
raw = controller.act(state, future_xy, terrain, return_raw=True)
targets = controller.decode_action(raw)
# targets 直接交给 simulator 的物理 PD target 接口，不再经过别的策略 action decoder。
```

- `state`：完整 24 body / 69 DOF 物理状态；body rot 是单位 XYZW，位置/速度是世界坐标。
- `future_xy`：`[B,10,2或3]`，对应未来 +0.5、+1、…、+5 秒。默认绝对世界 XY；
  `trajectory_frame="local_heading"` 明确表示相对 root、按 root heading 表达的偏移。
- `terrain.height_at(xy)`：输入 `[B,N,2]`，返回同 device 的 `[B,N]` 世界高度。
  明确提供地形，禁止悄悄回退到平地。高度采样器如何表达真实地形由 world adapter 负责。
- `return_raw=True`：返回 canonical PACER 顺序、裁剪至 [-1,1] 的 69 维动作。
  默认返回按实际 `dof_names` 排列的 69 维物理 PD targets。推理默认 deterministic。
- `action_to_pd(raw,L,U,dof_names=...)`：独立解码接口，可由新训练 agent 复用。
- `PacerSMPLMapping`：要求实际 names 与 canonical names 是完整、唯一的双射，支持任意重排。
  D6 backend 的 `:1/:2` 名字需先由现有 joint-name map 转成 semantic `*_x/y/z`。

当前真实 `smpl_humanoid.xml` 的 DFS body 顺序和 69 个 semantic DOF 顺序均与 PACER
canonical 顺序完全一致。不会将 SMPL 参数模型的原始 24 joint 数组顺序当成此物理顺序。

## 3. 观测合同

| 输入块 | 维数 | 精确规则 |
|---|---:|---|
| self | 357 | 23 相对 body 位置 + 24 个 6D rotation + 24 线速度 + 24 角速度；全部按 root heading 表达；不含 root height |
| 未来轨迹 | 20 | 十个世界 XY 点减 root XY，再按 root heading inverse 旋转 |
| 地形 | 1024 | 32×32，范围 ±2m，`ij` 展平；采样中心是 Head，方向取 Head heading |
| raw actor obs | 1401 | self 后接 trajectory 后接 heightmap |
| 单帧 AMP | 195 | root 6D rotation/速度/角速度 + 19 joints 的 6D exp-map rotation 与原始 DOF velocity + R/L ankle、R/L wrist 相对位置 |
| AMP history | 1950 | 十帧 `[t,t-1,…,t-9]`，不包含额外动作输入 |

terrain 值是 `(root周围9点平均地面高度 - Head网格地面高度).clip(-3,3)*5`。
原代码的 root 小网格朝向使用 **quaternion Z/W projection** (`quat_apply_yaw`)，
Head 大网格使用 **投影 X 轴的 heading**；root 倾斜时两者不同，移植保留了这个区别。
四个被鉴别器排除的 joint 是左右 hand 和 toe，所以保留 19 个 joint。

公开 PACER 的单帧 AMP kernel 将 3 个 DOF 数值解释为 exponential map，不先猜测进行
Euler 转换。这里复现原公式；新训练的专家状态与实际 simulator DOF 表示必须统一，
不能混用转换器输出与物理状态的不同角度约定。

## 4. 可学习网络与 PD 合同

`PacerAMPNetwork` 是普通可学习 `nn.Module`，并公开以下方法：

- `eval_actor(normalized_1401)` → `mu[69], log_sigma[69]`。
- `eval_critic(normalized_1401)` → `value[1]`。
- `eval_disc(normalized_history)` → `logit[1]`。

共享 task encoder 为 1044→512 ReLU→256 ReLU。self+task latent 的 613 维输入分别
进入 actor 与 critic：613→2048 ReLU→1024 ReLU→69/1 linear。
公开 D：1950→1024 ReLU→512 ReLU→1 linear。公开固定 log sigma 为 -2.9。
新训练可明确选择可学习 sigma 与不同 discriminator 维数；加载公开 checkpoint 不允许改尺寸。

`RunningNormalizer` 保留 float64 buffers，推理将统计转为 float32，epsilon=1e-5、
clip±5；value inverse 也先 clip±5。forward 不隐式更新统计，新训练必须显式 `update()`；
更新使用 population variance，避免 singleton NaN。reference 推理与原 RMS 完全一致。

PD 使用调用者给出的真实上下限。官方 `bias_offset=False` 分支进入 `if not self._bias_offset`：
每个 3DOF joint 的 scale 是 `min(1.2*max(abs(lower),abs(upper)), pi)`，offset 为 0；
左右 knee 的 y scale 覆盖为 5；hand targets 归零，默认冻结 toe targets。
0.7×范围的非对称扩展属于另一分支，不适用于本 no_shape 配置。

## 5. CPU 证据与剩余物理边界

专用 Python 3.12 / Torch 2.11 下 **13 项回归通过**，耗时 1.19 秒；Ruff 通过。
直接提取经过审查的官方数值函数 AST，隔离 Gym 入口并提供通用 XYZW quaternion
primitive；没有安装 Isaac Gym，也没有执行上游环境或安装脚本。

验证覆盖 self、AMP、完整 task/地形、PD 分支、RMS、names 重排、错误权重拒绝、
真实权重 actor/critic/discriminator/PD 前向，以及可学习网络反向梯度。
同一真实权重下官方 actor/critic/discriminator 数值均逐位匹配；随机物理张量上的
self 和 AMP 最大绝对差均为 **0**。这是 CPU kernel parity，未把合成状态当作真实行走。

记录：[pacer_cpu_verification.json](pacer_cpu_verification.json)、
[pacer_cpu_tests.xml](pacer_cpu_tests.xml)。运行命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../IsaacLab-Protomotions/.venv/bin/python -m pytest \
  CrowdSim/tests/test_amp_pacer_contract.py -q
```

Gym 原任务与新 Lab world 都是 30Hz 控制，但物理配置存在真实差异：原任务 60Hz / 控制
间隔2 / substeps2，新默认 SMPL 为120Hz / decimation4；原资产阻尼0.01、最大角速度100，
当前资产阻尼0、上限1000；solver velocity iterations、depenetration limits 也不同。
原 SMPL 是运行时按参数生成的几何、质量和 limits，不能由 checkpoint 文件推断它与
当前固定 XML 完全相同。ground friction1/restitution0、PD gains/mass/limits 由 root 的
真实 PhysX 验证单独记录。本模块没有把公开参考权重的跨物理后端移植宣布为行走通过。
