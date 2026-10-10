# 显式seed原生raw／风格准入：CPU准备

2026-10-06。补齐显式seed入口的独立原始数据审计和同seed两臂风格评价。
原seed7入口、reader、物理driver、已安装SDK、固定2848真实AMASS和S3训练保持原来源与配方。

## 实际执行

- 外部CPU流程：launch `f60656`、terminal `b25c7b`，实际退出0；33个子进程全部退出0。
- 195项相关检查通过，14.61秒：新接口97项、原raw/style回归98项；Ruff退出0。
- 24项raw冷检查：N3/N15 × 30/6000控制步 × DP/SFM × seed7/19/42。
  6项style冷检查：N3/N15 × seed7/19/42。全部未导入Torch或SDK。
- 条件突变：`require_registered_seed`的`value == seed`→`value != seed`，实际13失败/40通过，退出1。
  只运行新raw测试，避免通过style的固定源码SHA检查间接发现突变；finally精确恢复原字节。
- 两份历史真实S1/N4数据的全部base/body/response/support/joint指标保持；两臂的样本、45D、90D、
  共同速度桶及固定参考保持。JSON tuple/list仅按原表示边界序列化，没有数值容差或参数改变。
  原数据是cuboid场景，只用于数值回归，没有执行新seed/native正式准入。
- 初次探索检查实际退出1（terminal `66f50e`）：106通过、2失败。
  新测试调用`targets_from_saved_plans`时颠倒state/plans参数；修正测试调用后通过，生产公式未改动。
- S3的23项来源guard保持；本轮未启动新GPU训练、SDK模拟或新人体捕获。

## 准入与代码边界

- 新raw reader在独立模块namespace中执行已固定的原reader字节。六个测量内核和派生证明保持，
  只替换新入口/schema来源选择，并补充seed准入；原reader全局与文件保持。
- seed必须为真实整数`[0,2**32)`，不接受bool、float、缺省值或其他seed。
  SDK退出记录、实际driver参数、初态参数及入口注册必须一致；入口来源、内外CLI派生证明也必须精确匹配。
- 新style reader要求两臂外部raw-CPU退出、报告、测量内核与底层SDK准入中的seed一致。
  实际SDK资源、初态、完整6000控制步、只读数组、相同AMP及地图仍由原边界完整验证。
- 全身样本抽取直接引用原函数；失败不筛除，区间端点/首帧/reset边界范围保留。
  固定validation参考、K50/K100中心和45D/90D/共同速度桶计算不重新拟合；不读Test34。
- CLI只对main进行4处raw／5处style可逆接口替换；原循环和数值调用保持。

## 命令

先做CPU来源检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/audit_shared_social_native_seeded.py \
  --actors 3 --control-steps 6000 --planner diffusion --seed 19 --check-sources

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_seeded_style.py --actors 3 --seed 19 --check-sources
```

真实SDK和外部退出记录完成后，每臂单独运行原始数组审计：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python scripts/audit_shared_social_native_seeded.py \
  --actors <登记人数> --control-steps 6000 --planner <diffusion或sfm> --seed <登记seed> \
  --execution-receipt <外部SDK实际退出记录>.json \
  --scene-config <登记场景>.json --scene-sha256 <登记SHA> --output <新raw报告>.json
```

SDK receipt schema为`shared_social.actual_native_seeded_navigation_execution.v1`，
保留原native receipt所有字段，并显式绑定`seed`；必须是外部观察的整数退出码0和terminal tool chunk。
reader不能自行证明SDK成功退出。

完成两臂raw审计、外部CPU退出记录和数组只读封存后运行风格评价：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python scripts/evaluate_shared_social_native_seeded_style.py \
  --actors <登记人数> --seed <同一登记seed> \
  --scene-config <登记场景>.json --scene-sha256 <登记SHA> \
  --diffusion-audit-execution <DP外部raw-CPU退出记录>.json \
  --sfm-audit-execution <SFM外部raw-CPU退出记录>.json --output <新style报告>.json
```

raw-CPU receipt schema为`shared_social.actual_native_seeded_raw_CPU_audit_execution.v1`，
绑定原owner字段及`seed`；raw报告schema为`shared_social.native_seeded_scene_registered_controls_CPU_audit.v1`。
旧seed7 reader仍拒绝新的entry/schema，旧结果不改写为新seed结果。

## 剩余验证

原[156组合预登记](../shared_social_native_seeded_preparation_v1/README.md)保持：144资源准入、12拒绝，
288次两臂200秒运行仍未执行。此次完成的是CPU接口；每个实际cohort的SDK/raw/style准入仍为pending。
S3仅评价预登记终态`step_012500.pt`，成功训练退出和最终权重身份尚未准入。
仍需真实World/RNG初态配对、SDK smoke/生命周期和完整独立评价。

这是本地三seed鲁棒性扩展。[论文§IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)的八场景任务、
人数分配、seed、碰撞分母与Path Error定义尚未匹配。当前共享AMP候选质量仍未通过，
没有达到/超越论文完整效果的证据，部署默认模型保持。

CPU完整来源快照、突变、实际子进程退出与历史回归均只读封存；
封存工具的实际退出记录单独发布，不把脚本自报当作外部工具退出。
