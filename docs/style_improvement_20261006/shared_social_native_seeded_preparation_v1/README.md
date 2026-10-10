# 显式seed原生共享AMP：CPU准备与本地预登记

2026-10-06。保留已封存seed7入口，新增独立的
`scripts/evaluate_shared_social_native_seeded.py`、`native_seed_protocol.py`及登记CLI。
原固定2848 AMASS的S3训练未改动；本轮未启动SDK、策略推理或新人体捕获。

## 实际证据

- 外部CPU流程实际退出0：launch `cd3d4b`，terminal `6b72a5`；17个子进程全退出0。
- 最终48项相关测试通过，7.57秒；Ruff退出0。
- N3/N15 × seed7/19/42 × DP/SFM的12项冷源码检查通过，没有Torch或SDK导入。
- S1资源兼容预检：N3/DP/seed19与N15/SFM/seed42实际退出0。
  与已封存seed7预检的实际driver参数相比，仅seed和output不同；没有修改其他参数。
  S1预检只验证已完成权重家族的接口，未来登记矩阵仍选定S3终态，未替换为S1。
- 新入口的实际派生物理driver与原paired入口逐字相同；两个main各3处seed接口接缝，
  完整逆变换及其他helper AST保持。原采集点、原RNG处理、控制、SFM/DP和测量代码不变。
- 必需突变实验：`validate_seed`类型判断`is not int`→`is int`，实际26失败/22通过，退出1。
  finally恢复原字节，之后48项全部通过；突变文件标为TEST_ONLY。
- 原driver使用传入seed调用Torch初始化、AMASSCorpus和命令RNG；冻结AMP loader保留调用者RNG，
  不恢复训练RNG。上述为源码核查，真实多seed World/RNG/初态仍待实际捕获验证。
- S3的23项来源guard不变；不更改数据清单、训练配方或默认模型。

## 本地完整预登记

`registered_S3_native_matrix.json`保留原四地图各N3–15的52个资源案例，再按seed7/19/42展开：

| 项目 | 登记数量 |
| --- | ---: |
| 全部场景/人数/seed组合 | 156 |
| 资源准入，等待实际训练与SDK | 144 |
| 保留的资源拒绝 | 12 |
| 准入组合的两臂完整200秒运行 | 288（均未执行） |
| 拒绝资源对应的两臂空位 | 24 |
| 计划两臂actor-controls | 14,904,000（不是实测曝光） |

Hospital N12/13/14/15的原1.6m间距拒绝均保留三次；未生成替代场景、降低clearance或重采样。
48个实际准入配置/证明文件已重新散列，并重跑CPU原生场景预检；seed不改变地图/出生/路线配置。
每组合要求DP/SFM同一个AMP、同seed、相同完整初态/body/map/RNG；实际退出和初态匹配均为pending。
只选S3预登记的`step_012500.pt`，其实际SHA/成功训练退出尚未准入；不选择best或中途权重。

这些seed是本地鲁棒性扩展。[论文§IV-C](https://arxiv.org/html/2606.26265v1#S4.SS3)的八场景任务、
人数分配、seed及计数分母尚未匹配，四地图156组合不计为作者八场景协议。

## 复现CPU准备

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/evaluate_shared_social_native_seeded.py \
  --actors 3 --planner diffusion --seed 19 --check-sources
```

新入口必须显式给seed；范围为真实整数`[0,2**32)`，拒绝bool/float及不同于注册namespace的转发值。
资源预检仍需已完成训练receipt、scene/SHA、paired Diffusion/SHA和新output。
登记CLI只创建登记JSON，不创建未来SDK输出目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python scripts/register_shared_social_native_seeds.py \
  --output-root output/diffusion_amp/<新未来SDK目录> --output <新登记JSON路径>
```

## 尚未完成

新seed版本的独立raw/style准入仍需补齐。旧raw/style入口固定匹配旧seed7来源，
不能将新的entry伪装为旧入口；新登记文件明确保留该gate为未完成。
还需真实原生SDK smoke、多seed的World/初态/RNG配对和完整200秒独立评价。
完整World/producer构造失败的SDK清理路径仍未验证。所有模型质量/论文效果状态保持未达标。

根代理观察到的退出、完整source快照、条件突变和只读索引由封存记录保存；
封存工具本身的实际退出记录单独发布，不把脚本的自报状态当作外部退出证明。
