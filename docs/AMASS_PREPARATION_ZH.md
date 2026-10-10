# AMASS 数据准备记录

日期：2026-09-30。范围：按当前 ProtoMotions 的 SMPL train/test/validation 清单准备动作库。论文的 2,848 段导航动作精确清单尚未取得，以下通用清单不能标作该论文子集。

## 输出位置

全部数据位于 `/project/EXP/SangforAndSaut/Assets/motion/`：

```text
motion/
├── raw/                              # 原始 tar.bz2 保留；解压目录使用归档内部名称
├── processed/smpl/<dataset>/...      # 每段 .motion 与对应 .json 处理记录
├── amass_smpl_validation.pt          # CrowdSim 默认配置指向此位置
├── amass_smpl_test.pt
├── amass_smpl_train.pt
└── preparation/                      # 覆盖率、哈希、转换清单、加载日志、校验结果
```

三个清单均已完成解压、转换、打包和校验，无缺失路径、无转换失败、无跨清单重复路径。

| 数据集 | 动作数 | 帧数 | 实际时长（秒） | 大小 | 帧率分布 |
|---|---:|---:|---:|---:|---|
| validation | 345 | 307,046 | 9,892.113 | 646.60 MiB | 30 Hz × 307、50 Hz × 38 |
| test | 138 | 29,705 | 985.567 | 62.57 MiB | 30 Hz × 138 |
| train | 11,303 | 4,331,631 | 122,620.894 | 8.91 GiB | 30 Hz × 7385、50 Hz × 3918 |

合计 **11,786 段、4,668,382 帧**。19 个所需归档解压出 14,246 个文件、26,649,390,688 字节（约 24.82 GiB）；22 个原始归档全部保留。GRAB、SOMA、HUMAN4D 不在当前三个清单中，保持压缩状态。H36M/converted_train 预处理文件未作为 SMPL 数据转换。

默认 CrowdSim 配置已经指向本次生成的验证库，无需修改路径。数值校验通过不表示人体控制器、GPU 仿真或论文指标已经验收。

[完整汇总与 SHA-256](resource_audit/amass_summary.json)。原始清单、骨架及转换源码的哈希，以及逐动作源文件哈希，保存在 `Assets/motion/preparation/` 和每个 `.motion` 相邻的 `.json` 中。

## 处理方法

- 仅转换 `data/yaml_files/amass_smpl_{validation,test,train}.yaml` 中列出的动作，按各条 `sub_motions[0].timings` 裁剪。
- 原始 NPZ 解压到 `raw/<归档内部数据集名>/`；每个压缩包根目录的 LICENSE.txt 保存在各自数据集目录中。
- 使用当前 `convert_amass_to_proto.convert_amass_to_motion()`，固定 `smpl_humanoid.xml` 骨架、24 刚体和 69 DOF；无需 SMPL 参数 PKL 即可完成这个转换流程。
- 沿用原脚本的坐标变换、地面高度修正、速度计算以及高度/速度阈值接触检测。四元数为 xyzw，手部末端旋转沿用转换器的零值近似。
- 帧率来自 NPZ 并四舍五入。以 30 Hz 为目标，采用源帧率整数倍降采样；100 Hz 输入转为 50 Hz，不能把整个产物称为固定 30 Hz 数据。
- 使用仓库原生 MotionLib 打包、重新加载，并对所有浮点字段检查有限值，检查帧索引、时长、形状和四元数单位长度；每条动作检查起点、37.1% 时间点和终点插值。
- 每条动作记录源/输出 SHA-256、源码和清单哈希、源帧区间、实际 FPS、接触标签数量。没有凭空增加接触标签，也没有删除清单中的零接触片段。

## 已发现的元数据差异

测试集中 13 段的 YAML fps 为 59，而 NPZ 为约 59.9999886。沿用上游转换器，以 NPZ 四舍五入后的 60 Hz 处理；未用 YAML 的 59 Hz 改变播放速度。

验证集的 `MPI_HDM05/dg/HDM_dg_06_03_04_120_poses.motion` 为 25 帧、0.8 秒，按原接触阈值没有正接触标签。保留该动作以维持清单一致；整库接触标签并非全零。这是阈值输出，不能视为人工标注接触真值。

训练集中还有 18 段零正接触标签动作，总计 19 段。完整列表见 [零接触片段记录](resource_audit/amass_zero_contact_clips.json)；帧率元数据差异见 [差异记录](resource_audit/amass_fps_metadata_differences.json)。没有为了通过检查删除动作或人工补造接触标签。

## 环境与执行方式

本次建立隔离环境 `/tmp/navisaaclab-amass-venv-20260930`，复用已有 MuJoco 环境的数值计算包。未升级已有 Conda 环境，也不需要启动 IsaacSim。

| 包 | 本次版本 |
|---|---|
| Python | 3.9.25 |
| torch | 2.8.0+cu128 |
| numpy / scipy | 2.0.2 / 1.13.1 |
| mujoco / dm-control | 3.3.7 / 1.0.34 |
| PyYAML / easydict | 6.0.3 / 1.13 |
| typer / tqdm | 0.23.2 / 4.67.1 |

准备新数据目录时，从仓库根目录执行：

```bash
/tmp/navisaaclab-amass-venv-20260930/bin/python \
  data/scripts/prepare_amass_dataset.py \
  /project/EXP/SangforAndSaut/Assets/motion \
  --splits validation test train --workers 4
```

脚本不覆盖已存在的最终 .pt。对已有产物只重新校验：

```bash
/tmp/navisaaclab-amass-venv-20260930/bin/python \
  data/scripts/prepare_amass_dataset.py \
  /project/EXP/SangforAndSaut/Assets/motion \
  --splits validation test train --stage verify
```

临时环境可能被系统清理；可用兼容 Python 环境安装上述依赖后执行同一脚本。本次原始包及既有 H36M/converted_train 文件均保留。

## 与论文复现的边界

- 本步骤准备当前项目能加载的 SMPL 通用动作库。论文选取的 2,848 段行走、跑步、转向等动作的精确 ID、筛选和划分仍待补齐。
- 论文动作分布评测所需 15 关键点、统一 30 Hz 重采样和归一化，应在取得确切评测协议后另行生成；未将当前 24 刚体库混称为该评测数据。
- 尚未运行 GPU 仿真或训练人体控制器；SMPL 可视网格参数、权重接口和模拟器兼容性是另一步验收。
