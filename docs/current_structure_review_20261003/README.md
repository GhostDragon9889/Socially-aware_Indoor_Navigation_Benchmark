# 当前结构、入口、AMASS 隔离与源码发行复核

本轮仅执行 CPU 读取、配置/资源预检及 `/tmp` 隔离构建。没有启动 Torch、CUDA、Kit 或仿真，没有修改生产181文件、隔离训练931文件、权重、AMASS、USD或原工作区 `egg-info`。经 root 明确授权，仅补充 `MANIFEST.in` 的发行条目；主指南措辞修复由 root 完成。

## 结果

- 选定 cuboid、原生 Warehouse 四人、原生 Warehouse 十五人，以及文档中的方法默认、纯预检、README 训练配置，共 **6 项真实本地预检通过**。89 个本地 Markdown 链接全部存在。预检不创建输出目录，不验证 checkpoint 内部 tensor 或物理质量。[完整机器证据](actual_read_only_checks.json)、[指南修正后的快照与安装 metadata](final_document_and_packaging_inventory.json)。
- 当前 wheel 的 Python 包边界是 `protomotions`。`CrowdSim`、其 `scripts` 和 `configs` 是源码应用；`[crowd]` 是附加依赖组。独立 `src` 分发具有真实 `crowdsim-ppo-train/eval` entrypoint，用于无模拟器的参考 PPO。安装 metadata 的三个 ProtoMotions console entrypoint 与 `pyproject.toml` 一致。
- `CrowdSim.humanoid.simulate` 仍是 SFM/MaskedMimic；`scripts/simulate_paper_pedestrians.py --inference-only` 是显式 AMP/Diffusion 入口。两者职责没有混淆，当前指南能够指向存在的配置、检查点和资源。

## AMASS 2,848 的来源与隔离

v7 清单 SHA256：`68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。

| split | 片段 | 父动作 | 核对结果 |
| --- | ---: | ---: | --- |
| train | 2,848 | 1,853 | 523,272 个控制帧；每父源最多4段；控制帧区间无重叠 |
| validation | 22 | 22 | 完整 bundle 与原始319集清单逐值一致 |
| test | 34 | 34 | 完整 bundle 与原始319集清单逐值一致 |

三个 split 的父动作交集均为零；本地数据集命名空间也互不相交。训练含 KIT/CMU/Eyes Japan/BioMotionLab/ACCAD/EKUT/TotalCapture/TCD，验证为 HumanEva/SFU，测试为 Transitions/SSM。这里只读取清单、银行文件头和 stat，没有读取 sealed34 测试输出或重算9.5GB来源库 SHA。银行字节数与声明一致；既有完整内容/FK/child-joint速度证据保留于 [v7 验证](../amass_curation_20261002/verification_v7.json)。

2,848 是本地真实 AMASS 动作的非重叠派生片段，不是作者2,848原始 ID；同父源片段相关，命名空间隔离不证明真实人员身份或公开 PACER 预训练数据独立。清单明确 `actual_author_ids_known=false`，当前说明保持这一边界。

## 发现及处理

| 问题 | 证据与修复 | 状态 |
| --- | --- | --- |
| P2：基础训练示例覆写历史资源/输出 | 原 `FULL_PIPELINE.md:73–76` 使用 preparer/trainer 默认路径；`prepare_diffusion_amp_data.py:15`、`amass_corpus.py:87`、`train_paper_diffusion.py:72/102/133/155` 确实允许重写。root 改为新 manifest 与两个新 output 名称 | 文档已修；业务默认未改 |
| P2：新示例引用不会生成的固定 checkpoint | `train_paper_diffusion.py:133` 只写 `last.ckpt`；没有 `step_1000.ckpt` 生成逻辑。root 将后续 AMP 训练输入改为新目录实际生成的 `last.ckpt`。部署仍要求 immutable named snapshot | 已修 |
| P3：完成状态与历史门禁措辞过时 | README 的十五人 pending、结构索引的尚无 pre-action、FULL 的463旧门禁称 current 均与真实证据矛盾 | root 已修；先前快照保留 |
| P2，源码发行条件：当前新增入口未进入 sdist | 新薄部署、数据构建、风格评价、原生地图与 pre-action 等入口存在，但原 MANIFEST 与 SOURCES 缺少条目 | 已精确补充并实际构建验证 |

这些问题不表明本地训练模型、数据或原生地图被破坏。本轮没有重跑模型效果评估。统计保护训练/验证/导航的真实结果由独立实验报告更新，不能用本次入口检查代替质量验收。

## sdist 修复与真实门禁

`MANIFEST.in` 精确新增9个现有脚本与当前 `AMP_STYLE_PROGRESS_20261002_ZH.md`；现有 `recursive-include configs *.json *.yaml` 已覆盖所需小配置，没有扩大为权重/报告/data目录。新增脚本为 simulate、AMASS curation/velocity conversion、style evaluation、complete tracking、foot probe、scene map probe、pre-action probe、native scene preparation。

- before MANIFEST SHA：`ae91296cc8d52e1a9955c839ab9210bc2fce8e9248424f7c1b62910623c92ba1`。
- after MANIFEST SHA：`4fb72f0f528d3abeaf02c187531851031d43553c50eed9480c9053c9b9731463`。
- 实际包：`/tmp/protomotions-current-sdist-20261003-dpu5ow1a/dist/protomotions-3.1.0.tar.gz`，**41,615,046 bytes**，SHA `7ed528d2db96ae7a3e7e828a18be7617b5671edfaa6103907ca27e491d7c4bf4`。
- 1,006 个复制输入、1,014 个 tar 文件；26 个必需路径的 tar 内容与复制源 SHA 完全一致。权重、外部 AMASS/data/场景、SMPL body参数及受限 SMPL asset 均没有进入包。
- 原工作区 `protomotions.egg-info/SOURCES.txt` before/after SHA 不变；没有安装依赖。`build` frontend 未安装，但已安装且符合项目要求的 setuptools PEP517 backend 可直接构建，无下载。
- 从实际 tar 解包后的两项入口 `--help` 与 protected profile 加载通过；导入来源明确在解包树，未导入 Torch/NumPy/SDK。
- 三个新增 CPU 工具 Ruff 通过。首轮隔离复制工具遗漏自身需要的 `MANIFEST.in`，被必需入口检查捕获；已修复制工具，保留原失败日志，不将其归为生产代码问题。

详细路径、每个输入/成员清单、前后 hash 和限制见 [sdist 门禁](packaging/sdist_verification.json)、[真实解包入口](packaging/extracted_frontend_checks.json)、[build log](packaging/build.log)、[Ruff](packaging/ruff.log)。这是具时间与内容 hash 的源码快照；主指南后续结果更新不回写已建历史包。依赖安装、授权资源配置、tensor加载、GPU物理和视频质量仍是各自的验收项。
