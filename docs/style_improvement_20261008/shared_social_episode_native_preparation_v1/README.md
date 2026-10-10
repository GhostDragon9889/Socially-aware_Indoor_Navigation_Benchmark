# episode2000候选的原生DP/SFM配对入口

## 方法与当前状态

本入口为正在训练的episode2000终态预登记原生场景配对评估。
保持同一AMP控制器，分别使用Diffusion和SFM生成目标，依据
[NavIsaacLab论文IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)的对比方法。
论文报告八场景、每场景3–15名行人、每次200秒；当前可核验的本地资源为四张不同地图，
作者八场景任务ID及初态尚未恢复，本登记不认证作者八场景协议等价。

实际训练仍在进行，未释放任何原生SDK任务。当前只有CPU资源、来源与命令准备完成。
[实际CPU记录](actual_CPU_execution.json)、[登记](registration.json)及9份
[来源快照](source_snapshot_index.json)只读保存。

## 完整资源登记

| 项目 | 数量或条件 |
|---|---|
| 不同地图 | Office、Warehouse、Full_Warehouse、Hospital |
| 人数 | 3–15 |
| seed | 7、19、42 |
| 登记cohort | 156 |
| 资源准入cohort | 144 |
| 资源拒绝cohort | 12：Hospital N12–15，每种3个seed |
| 计划完整200秒任务 | 288臂：每个准入cohort均DP／SFM |
| 固定AMP终态 | `shared_social_episode_reference_pilot2000_v1/step_002000.pt` |
| 固定Diffusion | 已登记的40k终态 |

完整继承旧原生资源登记，只有候选权重选择、训练凭据、输出目录与schema改变。
全部继承字段的逆转换与原登记完全相等；地图、case顺序、seed、拒绝原因、间距与路线不变。
两臂在规划器构造前捕获同一身体／地图／RNG初态，实际配对仍需SDK运行后验证。

## 真实退出与训练准入

旧入口仍要求旧S3会话证书。新入口使用单独模块命名空间执行原退出桥源码，
将来源和候选选择绑定到本轮真实完成记录；原旧S3函数及拒绝条件保持。
本轮三次训练、三项CPU审计和独立评估全部实际成功，来源／预算／终态／完整捕获核验通过后，
才能发布原生SDK退出记录。

SDK和CPU原始数组审计仍使用原工具退出协议，必须提供对应命令的实际终态工具响应。
非零退出保存为失败，不能由文件存在或进程消失推断成功。
原SDK入口、身体控制、场景构造、初态捕获、原始数组和风格评分实现保持。
私有模块使用固定Python3.12的
[ModuleType](https://docs.python.org/3.12/library/types.html#types.ModuleType)；未修改共享模块全局变量。

## 已执行检查

- 9项CPU检查及3个subtest实际退出0，Ruff通过。
- 登记生成实际退出0；Office N3／seed7和Warehouse N15／seed42来源预检实际退出0。
- [实际负面准入核对](CPU_negative_admission_checks.json)通过：Hospital N15拒绝保留，未完成训练不能释放任务。
- 准备过程中未启动Torch、SDK或第二轮GPU训练，新增实现位于当前57来源／7输入之外。
- 初版7项检查退出0；初版Ruff因未使用的json导入退出1，修正后的实际记录保持。

只读核对已登记案例：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/prepare_shared_social_episode_native.py \
  --mode source-preflight --case-id Office_N03_seed7
```

[Office计划](Office_N03_seed7_plan.json)与[Warehouse计划](Warehouse_N15_seed42_plan.json)
已包含两臂30步短测、两臂6000步完整评估、原始数组审计、两项实际初态比较和独立风格命令。
实际执行、原生数组与风格结果尚未完成；本次准备不是效果验证。
