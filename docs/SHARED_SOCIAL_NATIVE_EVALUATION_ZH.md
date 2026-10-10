# 共享AMP原生场景接入：CPU准备与剩余验证

更新：2026-10-06。目标环境为当前固定Isaac Sim6／Isaac Lab3开发环境。
S3仍用同一v7清单的2848条真实AMASS训练；本页没有新增训练片段或选取中途权重。

## 论文协议

[NavIsaacLab原文§IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)规定：
Diffusion与SFM使用同一个AMP全身控制器，在8个场景、每场景3–15人条件下采样200秒。
本地当前只核验了Office、Warehouse、Full_Warehouse、Hospital四个原生地图；
作者八场景任务清单、具体人数分配、原始计数分母与Path Error定义尚未恢复。
四地图的CPU配置与预测验证不等于作者完整协议，也不提供人体物理表现。

## 本轮完成

| 项目 | 实际证据 | 范围 |
| --- | --- | --- |
| 固定资源配置 | 52个登记案例，48通过、4拒绝；2058段A*通过；工具退出0 | 真实v9地图的CPU资源准备 |
| 原生SFM接口 | 48个登记配置的完整51帧预测通过；实际CPU子进程退出0 | 注册出生位置的合成零速度快照 |
| 原生评价入口 | 3人DP、15人SFM源码检查及严格S1/Pop-Art资源预检均退出0 | 没有启动SDK或加载策略网络 |
| 源码与数值回归 | 30项检查通过，5.16秒；Ruff退出0 | 数值、来源、输入与实例别名释放 |
| 原生初态捕获接线 | 最终79项相关检查通过，7.76秒；两臂源码/严格预检退出0；准入突变触发3项失败并恢复 | CPU准备；没有真实World初态或200秒结果 |
| 原生raw审计接线 | 最终62项CPU检查通过，2.40秒；四项N3/N15源码检查与Ruff退出0；准入突变触发21项失败并恢复 | 显式30/6000控制步；没有新增原生SDK捕获 |
| 原始指标回归 | 历史S1 DP/SFM两份真实6000步的base/body/response/support/joint字段完全保持 | 旧N4 cuboid数据的计算回归，不计作原生场景效果 |
| 原生独立风格接线 | 53项CPU检查、N3/N15来源预检和Ruff退出0；真实历史DP/SFM的所有45D/90D/共同桶字段保持 | 原validation参考/中心不变；没有新的原生质量结果 |
| 显式seed入口及矩阵 | 48项CPU检查、12项来源检查及两项S1兼容预检通过；17个子进程退出0，类型突变触发26项失败并恢复 | 本地三seed登记156组合、144准入/12拒绝；288次200秒两臂运行均未执行 |
| 显式seed原始数组／风格准入 | 195项检查、24项raw/6项style冷来源检查通过；33个子进程实际退出0，seed条件突变触发13项失败并恢复 | 六个测量内核及历史真实两臂所有字段保持；没有新的原生SDK/质量结果 |

资源登记在运行前写入，顺序为四地图各N3至N15，路线clearance=.8m、出生clearance≥3m、
最小root间距1.6m。Hospital N12／13／14／15的最小间距分别为1.56205／1.5／1.253／1.11803m，
因此拒绝；未降低阈值、换场景或重采样。完整拒绝行保存在资源报告中。

预测输入使用已登记绝对world XY、零速度／heading及重复的合成历史；A*产生十个稀疏目标。
这些输入不是物理观测。全部48个输入档案和raw预测档案已保存；CPU预测不统计导航成功、
摔倒、碰撞、路径误差或动作风格。前后RNG不变，CUDA未初始化。

## 代码结构

- [原生SFM桥接](../CrowdSim/humanoid/trajectory/native_social_force_runtime.py)：
  复用已核验[公开SFM源码](https://github.com/yh83305/NavIsaaclab2.0/blob/d742f0c5b421bbc281e2112e7e50427e0336c16a/CrowdSim/world/sfm.py)
  的四项力、联合积分及速度裁剪；从原有N4桥接的两个方法进行12项可逆接口替换。
  距离场来自已准入ROS标签0／205／254及米制EDT；unknown、occupied和域外禁止通行，
  不投影为虚构cuboid，不拟合地图坐标。完整N行及N−1个同行人必须来自同一快照。
- [原生driver接线](../scripts/native_shared_social_derivation.py)：
  在已有同AMP SFM driver上只更换两处planner引用；完整逆变换和共享helper AST检查通过。
  原控制、因果参考适配、实际重置、接触测量及capture计算保持原实现。
- [原生观测迁移](../CrowdSim/humanoid/runtime/native_social_observation_producer.py)：
  在只属于该入口的模块中接入已验证的TorchScript heading来源检查。
  原生query、绝对坐标、时钟、观测和reset计算保持原方法；实际原生SDK使用尚待验证。
- [原生释放迁移](../CrowdSim/humanoid/runtime/native_social_navigation_lifecycle.py)：
  N个peer view加1个floor view，仍为一个vectorized articulation和24个SDK sensor。
  清除调用者实例的provider query／绑定counter及facade/core别名循环。
  不修改已安装SDK文件或SDK类方法；CPU实例夹具不证明真实native view已释放。
- [原生统一入口](../scripts/evaluate_shared_social_native.py)：
  显式选择DP／SFM、N3–15、scene/SHA、已完成训练的外部退出记录与配对DP资源。
  Pop-Art权重由现有严格reader读取。该原入口固定seed7；显式seed版本见下方独立入口。
- [CPU预测入口](../scripts/prepare_shared_social_native_sfm.py)：
  保留每个输入档案、预测及失败行；发生预测失败时无替代输出，进程返回非零。
- [原生初态配对入口](../scripts/evaluate_shared_social_native_paired.py)与
  [初态采集/读取](../CrowdSim/humanoid/analysis/native_paired_initial.py)：
  两臂均在actor准入/设种之后、planner构造之前采集。保留原两次同刻COMMON/native、完整RNG及
  submitted-step检查；九处接缝只扩展人数与原生provider边界。新增地图标签、origin、bounds和原SDF，
  CPU35通道，一块已初始化CUDA再增加1通道。完整资源/出生/路线/来源相同方可配对；hook不额外推进物理。
  临时回调在finally中恢复，避免World/capture被保留在模块引用循环。
- [正式原生配对CLI](../scripts/compare_shared_social_native_initial_conditions.py)：
  默认拒绝合成CPU夹具和旧N4 schema，逐字段验证档案及来源。配对通过仍需要实际退出和独立raw审计。
- [原生raw审计CLI](../scripts/audit_shared_social_native.py)与
  [CPU测量内核](../CrowdSim/humanoid/analysis/native_navigation_audit.py)：
  显式N3–15与30/6000控制步，六个来源固定内核共43项可逆接口替换；原指标公式保持。
  人数轴独立于四物理子步、四元数分量及四脚部传感器。检查全部24身体、完整计划、初态首锚点、
  native时钟、场景/provider/live stage和释放声明；默认拒绝旧fixture退出记录。
  修正了不存在于driver参数中的`native_dynbench`假设，改用实际输出的原生场景证据准入。
- [原生独立风格CLI](../scripts/evaluate_shared_social_native_style.py)与
  [专用读取边界](../CrowdSim/humanoid/analysis/native_navigation_style.py)：
  完整6000步、同场景/人数的两臂外部raw CPU退出及实际SDK/初态/权重来源必须相符。
  原有15关键点/30Hz、root/heading/尺度、45D/90D和固定聚类中心计算直接复用；失败片段不筛除。
  明确区分完整捕获中的失败、合法区间端点中的失败、首帧和reset边界，不把导数范围写成全帧。
- [显式seed原生入口](../scripts/evaluate_shared_social_native_seeded.py)与
  [本地登记CLI](../scripts/register_shared_social_native_seeds.py)：
  必须显式给seed，原paired driver逐字保持；内外main各三处接口接缝均通过完整逆变换。
  独立seed版本raw/风格CPU接口见下方；原seed7的reader仍固定旧来源，不能混用结果。
- [显式seed原生raw审计](../scripts/audit_shared_social_native_seeded.py)与
  [同seed两臂风格评价](../scripts/evaluate_shared_social_native_seeded_style.py)：
  SDK退出／driver／初态／入口注册必须使用同一整数seed；风格两臂的raw退出／报告／内核／底层准入也必须匹配。
  新来源与schema在独立namespace中选择；原测量内核、区间抽取、固定参考和评分公式保持。

## 复现CPU准备

### 串行实际工具退出观察

[案例登记与冷准入](../scripts/prepare_shared_social_native_owner.py)及
[工具API owner](../scripts/shared_social_native_tool_owner.js)已完成
[修正后CPU验证](style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/README.md)：
42项pytest、13项V8契约、四地图登记/冷读取及Ruff通过；71份文件只读封存。
运行时全部命令与登记逐字段匹配；每步再检查来源和S3完整owner证书。
短检查/完整两臂、raw审计、两次真实初态比较、捕获封存、独立风格及汇总共21个串行步骤。
持续观察失败保存结果未知；失败SDK/raw退出保留原值，后续停止。

实际工具链当前在S3证书检查退出1，未启动SDK。
只有根实际确认cell581完整结束并发布外部证书后，才可在functions.exec中使用
[冷加载启动模板v2](style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/launch_Office_N03_seed19_after_S3_v2.js)。
该模板使用实际tools API，不是普通Node入口；执行结束后还需根记录外部cell完成。
缺少证书时不会建立原生输出目录。冷加载/SDK会话均只观察原handle，不自动重启。
准备检查不能代替新增SDK捕获或论文效果，作者八场景协议仍未恢复。

先检查源码，无需GPU：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native.py --actors 3 --planner diffusion --check-sources

PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native.py --actors 15 --planner sfm --check-sources
```

已发布的48个本地配置在[scenes](style_improvement_20261005/shared_social_native_SFM_preparation_v1/scenes)。
完整CPU资源登记、输入/预测、实际退出和来源档案见
[SFM准备记录](style_improvement_20261005/shared_social_native_SFM_preparation_v1/README.md)；
两臂严格权重/资源预检及释放来源检查见
[原生入口准备记录](style_improvement_20261005/shared_social_native_entry_preparation_v1/README.md)。
所有输出需要新路径；已封存结果不覆盖。

原生初态版本使用同样参数，替换为新入口即可先做源码检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_paired.py --actors 3 --planner diffusion --check-sources
```

真实SDK执行仍需显式scene/SHA、已完成训练的execution receipt、配对DP/SHA与新output子路径。
`--control-steps 30`只用于smoke；`6000`对应200秒，不能把短检查记为完整试验。
两臂完成后才能运行：

```bash
python scripts/compare_shared_social_native_initial_conditions.py \
  --left <DP输出>/paired_initial_condition.json \
  --right <SFM输出>/paired_initial_condition.json \
  --output <新配对报告路径>.json
```

[原生初态最终CPU证据](style_improvement_20261005/shared_social_native_initial_preparation_v3/README.md)
保留全部注册、真实地图通道、子进程退出、突变及恢复、v1输出路径拒绝和v2中间源码。
CPU档案只有地图，未拼接合成人体；最终相关79项检查包含旧N4、原生SFM/生命周期及新初态42项。

原生raw reader可先检查来源，不需要实际SDK结果：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_native.py \
  --actors 3 --control-steps 6000 --planner diffusion --check-sources
```

真实执行后，用外部观察者保存的新receipt审计，每臂分别运行：

```bash
python scripts/audit_shared_social_native.py \
  --actors <登记人数> --control-steps 6000 --planner <diffusion或sfm> \
  --execution-receipt <外部实际退出记录>.json \
  --scene-config <登记原生场景>.json --scene-sha256 <登记SHA256> \
  --output <新raw审计路径>.json
```

新receipt schema为`shared_social.actual_native_navigation_execution.v1`。
除实际整数退出码0与terminal tool chunk外，需绑定人数、控制步、planner、scene configuration、
result和三份raw数组身份；完整字段说明见
[raw CPU封存记录](style_improvement_20261005/shared_social_native_raw_audit_preparation_v2/README.md)。
reader不能自己证明SDK正常退出，必须保留外部工具观察。6000步N3为18000 actor-controls、
24000全局physics-substeps及72000 actor-substeps；这些分母明确分开。
88份文件已核验SHA并只读封存，另保留封存工具实际退出记录。v1的参数ABI失败及修正前源码保留。

原生风格入口先做来源检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_style.py --actors 3 --check-sources
```

实际两臂SDK、raw审计及数组只读封存完成后，每臂需外部CPU退出记录，schema为
`shared_social.actual_native_raw_CPU_audit_execution.v1`，绑定整数退出码0、terminal tool chunk、
人数、6000控制步、planner、scene configuration及audit身份。
完整正式命令、CPU/Torch范围、历史回归和92份封存证据见
[原生风格CPU记录](style_improvement_20261005/shared_social_native_style_preparation_v2/README.md)。
v1直接比较Python tuple与JSON list而失败；v2严格比较同一JSON表示的全部数值/字段，无数值容差，
没有改变指标、聚类中心或参考。历史N4数据始终只作为数值回归。

显式seed版本可单独检查来源，实际物理运行尚未执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_seeded.py \
  --actors 3 --planner diffusion --seed 19 --check-sources
```

[三seed CPU准备及封存](style_improvement_20261006/shared_social_native_seeded_preparation_v1/README.md)
保留完整156组合、17个实际子进程退出、48项检查、参数转发比较、突变/恢复及88份只读文件。
四地图各N3–15按seed7/19/42展开，144资源准入、12原Hospital拒绝；计划288次两臂200秒运行、
14904000 actor-controls，均不是实测。每组合要求同AMP/seed/场景的DP/SFM真实初态及RNG匹配。
只选择S3预登记终态`step_012500.pt`，实际训练退出和权重SHA仍待准入；两项S1预检只验证接口。
这些seed是本地扩展，作者八场景与seed协议仍未匹配。新seed版本raw/风格接口现已完成CPU准备：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_native_seeded.py \
  --actors 3 --control-steps 6000 --planner diffusion --seed 19 --check-sources

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_seeded_style.py --actors 3 --seed 19 --check-sources
```

[完整CPU记录与正式参数](style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/README.md)
保留33个子进程实际退出、条件突变与恢复、历史真实raw/全身风格回归及119份只读文件。
SDK receipt与raw-CPU receipt分别使用显式seed schema，旧reader拒绝新entry/schema；
每臂实际SDK退出、初态和完整数组仍需准入，不能用历史N4数据替代。
[实际分项计数](style_improvement_20261006/shared_social_native_seeded_admission_preparation_v1/test_inventory_correction.json)
为新接口95项＋原回归100项，总195项；封存README的97/98误写已显式纠正。

## 还未完成

1. S3终态训练、严格权重审计和独立质量评价。当前GPU由其外部owner独占；日志进度不代表实际正常退出。
2. 原生入口新鲜30步SDK smoke，实际Pop-Art策略推理、N3/N15接触结构与退出释放验证。
   完整World或producer未返回时的异常清理仍未验证；不能凭CPU检查认定迁移完成。
3. 原生两臂的真实初态/RNG捕获与匹配。N3–15来源/资源/地图版本及入口已完成CPU检查，尚未执行实际World采集。
4. 原生完整200秒、N3–15、多seed的全部原始physical/ground/plan审计，以及独立全身风格评价。
   原seed7的raw和独立风格reader均已完成CPU接线/历史数值回归，尚未审计新的实际原生捕获；
   显式seed入口、156组合登记及独立raw/风格接口已通过CPU准备，每个cohort的实际准入尚未执行。
5. 八场景与作者协议恢复，以及可比较的成功率、碰撞分母和Path Error定义。

已有S1/S2和新鲜N4 DP/SFM实测均未通过质量要求，当前没有达到／超越论文完整效果的证据。
继续以独立跟踪、动作风格和路线／物理评价判断候选，默认模型未晋升。
