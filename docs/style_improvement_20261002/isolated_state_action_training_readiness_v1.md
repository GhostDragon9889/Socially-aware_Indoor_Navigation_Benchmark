# state-action训练隔离快照：CPU准入

**CPU预检／59个实际业务模块来源检查／157项guard与复制验证通过；未启动GPU训练。** 新目录：[bundle](../../output/isolated_training/mixed_state_action_v1)、[CPU机器记录](../../output/isolated_training/mixed_state_action_v1/CPU_readiness_v1.json)、[独立检查](isolated_state_action_source_CPU_verification_v1.json)。

复制CrowdSim、protomotions、scripts、configs、src和固定资产，共926个普通文件104,671,269B，未使用hardlink代码。全部已登记146源由原e924af…manifest／62b76a…tar精确还原，包括旧main、world、paper与simulator；其余代码／资产来自复制时的树，另外逐项SHA冻结，不伪称这些文件也属于旧146时期。state-action原profile SHA6963d342…保持副本，实际profile只把data_manifest／公开PACERcheckpoint改为绝对路径；训练参数语义完全相同。

完整source archive SHA3563a3ad…，manifest SHAdc4ab8d4…，launch guard SHA2ef7d33b…；[完整清单](../../output/isolated_training/mixed_state_action_v1/source_manifest.json)。code普通文件0444／目录0555，资产默认强制指向复制树。guard在Kit前安装，业务模块只能从复制树解析；缺失模块直接拒绝，不能回退editable current checkout。实际exec audit拒绝原checkout可执行路径。CPU正负检查包括原模块来源拒绝、missing-submodule拒绝、原文件exec拒绝和复制路径exec允许；GPU真实导入仍须4+4验证，退出时另写实际owned-module来源报告。

data和third_party是显式资源symlink；6个实际资源文件（manifest／3个AMASS包／PACER／Diff）已完整SHA＋stat绑定，每次launch重核。symlink不提供OS级read-only sandbox，0444也不防文件所有者主动改权限；本方案用于隔离并行正常业务编辑。训练期间依赖环境及这些资源仍需保持固定，不宣称C++/PhysX缓存逐位恢复。没有复制.git／当前output作为可执行fallback。

## root串行4＋4

```bash
bash output/isolated_training/mixed_state_action_v1/pilot4.sh
```

该脚本设置isolated cwd／PYTHONPATH／asset root、绝对runner/config/model/data/output与固定256env／32steps，fresh4updates=32,768transitions。进程退出后root将实际last.pt独占封存为 `output/diffusion_amp/amp_style_mixed_state_action_isolated_pilot_v1/step_4.pt`，并另存真实result／metrics。**封存不存在时恢复预检不成立。** 随后：

```bash
bash output/isolated_training/mixed_state_action_v1/resume4to8.sh
```

新增4updates，累计8／65,536transitions。随后需真实payload／模型2019-D／Adam／nested3RNG／实际来源审核，不能把CPU预检当训练成功。完整fresh2000脚本已准备但未执行；状态明确prepared。此profile仅history-action输入工程变体：公开actor/critic warmstart、随机2019-D与1950-RMS，不是作者private weights或paper-exact实现。当前source-only复制不证明效果提高。
