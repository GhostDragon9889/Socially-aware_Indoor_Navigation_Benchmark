# Maps + Interact 薄训练入口与时间轴核查

**状态：staged source；实际CPU两模式一步模型/Adam/RNG保存恢复通过，未运行fulltrainer main、完整训练或物理，未修改185/q935或原数据。**
本目录接入已生成的两域，保留现有五状态网络/DDPM/loss/优化器数学。它不是作者数据或作者完整网络实现。

## 时间轴结论

| 边界 | 实际源码 | 时间含义 |
| --- | --- | --- |
| 历史31帧 | amp_world.py:349；91个30Hz样本每3帧抽取 | 含当前t0，过去30个间隔=3秒；不是3.1秒 |
| 训练未来52帧 | paper.py:479–502；数据window future52 | t=.1..5.2秒，全52帧参与loss |
| 论文描述 | Nav §IV-A；TRACE/PACE §4 Implementation | 描述5秒未来、3秒历史/10Hz |
| 作者配置 | TRACE config:99,123–127 / datamodule:43–59 | 明确horizon52/future52/dt.1，数据请求5.2秒 |
| 对外导航计划 | paper.py:689–700 | 只取前50帧，补t0成为51个节点，t=0..5秒 |
| 10个目标点 | train_paper_amp.py:503；paper.py:613–617 | .5,1,…,5秒，对应未来数组4,9,…,49，无索引偏移 |
| 30Hz执行参考 | reference.py:178–224 | 51个10Hz节点插值/适配为151个30Hz节点，仍0..5秒 |

不能仅凭论文写“5秒”判定52帧错误。作者公开配置也为52；未找到作者解释额外2帧目的的来源，本报告不推断它们是padding。

本地直接Diffusion执行 task_trajectory() 查询 `plan_age/30 + [.5,…,5]`，
interpolate() 在旧计划末端clamp（train_paper_amp.py:593–619）。
age=0时最后目标距当前5秒；age=7/14时实际最后节点距当前4.7667/4.5333秒；
age=15的next-value旧计划查询为4.5秒，随后重规划重新变为5秒。
这是已声明的有限计划边界（同文件:315），不是新发现的作者bug。
重规划15controls=.5秒，本地做法与TRACE/PACE §3.3公开2秒闭环间隔不同；
Nav未披露其准确重规划间隔。当前不修改控制时间轴、目标或模型horizon。
mixed_amass_procedural训练源走current-phase future_targets，不能把上述直接Diffusionclamp泛化到该训练分支。

## 最小混合接口：已经写入本目录

- `mixed_corpus.py`：两个原loader保留split/episode/window过滤。Maps14槽补至19，补槽state=0/mask=False/id=-1；Interact原19槽不变。
- 只有一个union采样RNG；全局window索引确定域和域内索引，child读取explicit indices不消耗child采样RNG。保存/恢复union RNG。
- Train合法窗口137689 Maps +150439 Interact =288128，未加域权重；Interact概率52.2126%。validation36596/test36390，分别保留域计数。
- `train_mixed_diffusion.py`：原Interact trainer的薄数据/元数据/评估派生。两个manifest、域顺序、window counts、采样规则、19槽和encoding选择共同绑定phase/resume；纯Interact/Maps旧phase不能静默resume。
- 真实data map_condition_mask始终Maps=True、Interact=False。模型encoder availability由显式CLI选择，不能把encoder True改写成数据有地图。
- validation/test按域调用原loss/原50帧ADE/FDE与各域原collision函数。Interact obstacle指标保持None/N/A，Maps保持原固定.4m半径口径；不把None计成0或混成“总体障碍碰撞率”。

作者config同时使用orca_maps和orca_no_maps，UnifiedDataset DataLoader shuffle=True，不设置50:50权重。
这里按合法window有放回均匀抽样，保留原trainer API；它对应不人为加域权重，**不声称等同作者逐epoch无放回顺序**。
原Maps仍是本地4–15人/独立生成/conditional-window过滤，Interact是官方raw独立resample且只保留真实邻居窗口；
因此也不能声称恢复了作者相同训练分布或Nav私有语料。

## no-map编码：默认正式TRACE方法，另留本地消融

`--missing-map-encoding trace_unknown_raster`（默认）：
真实未知地图raster=.5，数据presence=False；encoder mask单独ones_like，仍编码这张未知地图。
原staged paper_interact.py:333–363已在map-dropout发生时**先把对应raster填.5再编码**；
没有encode后将dropout grid清零的分支。neighbor-dropout仍仅关闭邻居mask。
未知raster梯度是unknown-map条件的证据，**不是观察到真实障碍物的证据**。

`--missing-map-encoding skip_encoder`：
传真实presence mask，只有有地图行进入MapEncoder；全missing跳过encoder，
避免BN更新/参数grad/Adam decay。该分支是本地适配消融，不能称TRACE原始no-map方法。
旧已冻结stage和已有数据不改；两模式checkpoint/resume身份不兼容。
first encoder evidence按实际encoded availability分别记录map/neighbor，纯missing的skip map grad None合法；
不强制重抽样到“两域各一”或更改union比例。

## 可执行训练命令（仅准备，禁止自行GPU运行）

从仓库根目录，使用拥有Torch的独立训练环境：

```bash
python docs/style_improvement_20261003/maps_interact_mixed_training_staging_v1/train_mixed_diffusion.py \
  --maps-manifest data/diffusion_amp/orca_maps_expanded_1000_v1/manifest.json \
  --interact-manifest data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json \
  --missing-map-encoding trace_unknown_raster \
  --init-checkpoint output/diffusion_amp/diffusion_training/step_1000.ckpt \
  --output-dir output/diffusion_amp/social_diffusion_maps_interact_staged_v1 \
  --steps 1000 --batch-size 8 --seed 20261001 --device cuda
```

此fulltrainer训练命令未执行；future正式checkpoint SHA为空。
已按root授权在隐藏CUDA/4threads的SDK Python进程运行同一真实B2两域batch：
global indices[0,140659]，实际邻居[3,19]，data mapmask[True,False]。
默认TRACE实际编码2行，unknown标志[False,True]；skip只编码真实地图1行。
两模式各一次trainstep均finite、map/neighbor梯度非零；model/Adam及torch/NumPy/Python/union下次随机抽样保存恢复逐值相同；
错encoding/Interact manifest/sampling均拒绝。实际工具session93701、launch0258dc、completion793c9d，observed exit0，模型smoke7.3169秒。
只一次读取/校验root授权4b6初始模型；未DDPM采样、SDK/CUDA初始化或物理。
两个一步snapshot标记smoke_only且只有train身份，不能当fulltrainer正式成果或效果证据。
见[实际CPU模型记录](actual_CPU_model_smoke.json)和[实际进程闭合记录](actual_CPU_closedreceipt.json)。
停止额外smoke/suite；实际效果需要新模型的独立open-loop与闭环物理实测。

下一正式候选将接入独立转换后的官方Maps1000与Interact1000；当前入口的localMaps定义保持，便于区分数据变化。
TRACE公开预算为40k iterations、LR2e-4；正式计划按此登记，batch需结合当前GPU内存决定并明确披露偏差，不能把1000步接口命令称完整作者预算。

## 原始资料与可核查定位

- [NavIsaacLab原论文 §IV-A](https://arxiv.org/html/2606.26265v1#S4.SS1)：五状态/历史+邻居+map/MSE/5秒；不披露上述局部时间参数。
- [TRACE/PACE作者论文](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)：印刷页6 §3.3及Implementation Details；页13 A.1.3的unknown .5/no-map/dropout；页15 B.1的10个.5秒目标；页17的ORCA与resampling资料。
  原缓存PDF SHA dc539e42766e573a6ccdeb526857278f3e716255a99e251523d63aa3243869ec；本次web取PDF因12.7MB限制失败，依据此前保存的primary PDF/text。
- [作者训练config 固定594b](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json#L35)：两域源35–42；horizon99；history/future/dt123–127。
- [作者datamodule 固定594b](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/datasets/trajdata_datamodules.py#L43)：时间43–59；combined dataset74；shuffle85–97。

作者TRACE是state-and-action/8维表示，当前是Nav五状态改写；保留网络训练预算、EMA等其它差异，
不把本目录窄接口修正称为完整作者NN复现。
