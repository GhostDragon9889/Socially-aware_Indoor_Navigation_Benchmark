# 完整22段：结果解读

实际严格配对审计通过。22段、各11,982 poststep帧，所有初始physical state/AMP history和reference poses/XY目标/phase/time的dtype与数组bitwise一致；全体指标和失败片段保留。两run均0fall，没有需要剔除的fall时段。

| ALL22主指标 | negative-only1000 | AMASS-root1000 |
| --- | --- | --- |
| tracking成功 | 11/22 | 17/22 |
| 全body世界位置误差 | .300708m | .229298m（-23.75%） |
| 全body世界旋转误差 | .861454rad | .831882rad（-3.43%） |

位置误差19/22段改善，旋转11/22段改善；新增成功IDs为4、14、242、254、259、260，未丢失旧成功片段。

## root运动与局部姿态

| 同帧均值 | negative-only | AMASS-root | 相对变化 |
| --- | --- | --- | --- |
| root XY误差 | .228346m | .142768m | -37.48% |
| root wrapped yaw误差 | .600987rad | .551314rad | -8.27% |
| 各自root/heading局部body位置 | .119189m | .125401m | +5.21% |
| heading局部body旋转 | .573434rad | .561986rad | -2.00% |
| parent局部关节旋转 | .301823rad | .287741rad | -4.67% |
| 实际planar速度 | .777697m/s | .896613m/s | +15.29% |
| root高度绝对误差 | .020758m | .026651m | +28.39% |

改善最明显的是root平面运动；局部body位置稍退化，因此不能称为全身姿态全面变好。局部坐标诊断不替代世界坐标指标，不能把误差差值变成独立因果贡献。reference速度在0…2m/s的四个主要区间中世界位置误差均下降；2…3m/s只有4帧且退化，不可泛化，3m/s以上无支持帧。

## 分母敏感性

自身成功子集：negative11段/7,191帧的frame加权位置/旋转为.243243m/.597196rad；AMASS17段/9,902帧为.194611m/.646962rad，集合已改变，不能将该旋转上升解释为全体退化。

共同成功交集固定11段/7,191帧：frame加权位置.243243→.168929m，旋转.597196→.550762rad；clip等权也单独列于report.md/comparison.json。所有成功子集仅是描述性分析，不能用于筛掉失败、替代ALL22或证明与论文.218m/.343rad/64.6%相同协议、动作集合或分母。

## 来源与边界

真实actor SHA256 `dc9cb02bcae06bb54451e8117f16aae739a8e79857581dd36df68edff6be028e`。外部sealed snapshot、完成训练的last.pt以及之后创建的step_1000交付alias均652,092,533字节且SHA一致；交付copy不是新增训练，详见immutable_checkpoint_training_bindings及step_1000_delivery.json。

两个93/124 member源码tar和captured runtime源哈希逐项绑定通过。reader初始拒绝snapshot目录和缺step alias的历史记录保留于amass_root_tracking_audit_first_snapshot_admission.json；之后严格用真实三份同SHA文件确认来源。frozen reader副本仅用于源码溯源，执行原路径脚本的命令见准备说明。

该工程配方同时改变绝对source命令、合法reset相位/clip-end和clip外线性延拓；不能归因到单独tail或单独AMP因素。单seed、已多次使用的validation，不能证明作者复现、统计显著性或34条test性能。没有读取34条test，没有重算资源agent官方style/MMD。
