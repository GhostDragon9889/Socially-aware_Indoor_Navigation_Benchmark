# ORCA-Interact：官方 raw 资源生成准备

本目录仅准备独立 CPU 数据生成，尚未生成1000场景、接入 ORCACorpus 或训练。既有 Maps、AMASS、冻结q935、生产185和默认模型保持原样。新输出已预留为空：`data/diffusion_amp/orca_interact_official_raw_1000_v1/no_maps`。

## 来源与配置

TRACE固定commit为 `594b2af0824c5853e1a1884e167cd5ed097dd81c`，本地working tree clean；登记的五份官方文件逐字节等于该commit。[官方ORCA README](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim/README.md#L29)给出Maps1000／max_agents10／max_obstacles20，以及no_maps1000／max_agents20／max_obstacles0。后者即这里补充的Interact来源。

[原 `gen_dataset.py`](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim/gen_dataset.py#L294)每场在1–20人间随机采样，不是每场固定20人；支持零障碍、10秒、名义10Hz保存和goal重采样。无seed或jobs参数，场景循环串行。薄delegate只在加载模块后设置 `numpy.random.seed(20261004)`，按登记argv解析并调用原函数，不改官方算法。关闭可视化，CPU库线程上限4，CUDA不可见。

官方原始时间记录首帧预期为1/60秒，最后一次间隔预期5/60秒；这些需生成退出后按真实 `trackTime` 核查，不能用 `--save_rate 10` 宣称整个序列严格10Hz。

[官方场景分割实现](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset_specific/orca_peds/orca_dataset.py#L37)用 `Random(0)` shuffle 800／100／100个train／val／test标签。[提前登记的1000场景分割](scene_splits.json)采用同一算法；raw目录本身不分train子目录。生成seed是新的，不能声称数据与作者文件相同。

官方[混合配置](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json#L34)列出Maps与no_maps；[DataLoader](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/datasets/trajdata_datamodules.py#L89)shuffle统一的agent-time样本索引，没有显式50:50子集采样器。1000＋1000场景不等于窗口混合比例50:50；真实窗口比例须在接入后统计。

## 当前本地生成器缺口

实际文件名为冻结q935的 `scripts/generate_orca_diffusion_data.py`，没有 `generate_orca_corpus.py` 入口。[来源登记](registration.json)绑定q manifest `2e14550a…`及其两个实际源码文件。

| 环节 | 当前限制／所需适配 |
|---|---|
| CLI与人数 | CLI拒绝零障碍，无max_agents参数；代理采样器只允许4–15人，场景人数按固定周期产生 |
| 空障碍 | 非空map强制；空几何的spawn距离和QA最小距离没有缺失值语义 |
| 场景身份 | generator与reader均强制不同几何layout hash；所有空场景的 `[]` 几何相同，不能伪造不同hash |
| 窗口 | 同时要求真实邻人和可见障碍面积≥.25m²；Interact需要明确的neighbor-only规则 |
| 无图条件 | TRACE的no-map输入两层均填.5；当前空障碍栅格公式是walkable1／obstacle0，需要明确missing-map语义 |
| raw时序与朝向 | raw为trackPos／trackVel／trackTime，尚无当前manifest及yaw；时间重采样、速度朝向代理和完整源记录仍需适配 |

因此本轮先补充官方raw资源。它不能直接传给现有 `train_orca_diffusion.py --manifest`，也不代表已补齐可训练的Interact混合数据。

## 可执行独立CPU命令

[机器可读命令](commands_CPU.json)包含完整官方argv、环境、source SHA及输出路径。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nav_orca_interact_matplotlib_v1 \
data/tools/orca_generator/.venv/bin/python \
docs/style_improvement_20261003/orca_interact_raw_preparation_v1/generate_official_raw.py
```

加 `--prepare-only` 只核来源与新空目录。该模式已经实际exit0；原本地与官方CLI的 `--help` 也实际exit0，未模拟任何场景。实际工具环境为Python3.12.13／NumPy2.2.6／Shapely2.1.1／Matplotlib3.10.5／tqdm4.67.1／pyrvo2版本元数据0.0.0，RVO2仓库commit `c2c46ba8…`。软件包版本元数据不证明与作者历史运行的二进制环境相同。

Root实际生成退出后，只需短核1000份scene/sim.npz、真实时间间隔、人数≤20、位置／速度／时间finite、无map文件、分割身份和未跨scene统计，再记录真实退出与文件身份。这里不产生未来完成、模型质量或作者复现结论。
