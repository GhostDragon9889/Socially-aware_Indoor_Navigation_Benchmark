# Isaac Lab 3 EA / Isaac Sim 6.1 独立项目依赖 overlay

## 实测运行环境

- Python：3.12.13。
- SDK：`/project/EXP/IsaacLab`，Isaac Lab 软件包 17.0.2，Isaac Sim 6.1.0.0。
- 原有 CUDA 数值运行库：Torch 2.11.0+cu128、NumPy 2.5.1、Warp 1.16.0，均从 SDK 的原环境导入。
- 项目 overlay：本目录 `.venv`，只补项目/测试依赖。没有替换 SDK 文件，没有重复安装 Torch、NumPy、USD 或 Isaac Sim。
- 项目覆盖版本：OmegaConf 2.3.0、Hydra 1.3.2、TensorDict 0.9.0、protobuf 6.33.6、Wandb 0.23.0、Lightning 2.5.6、dm-control 1.0.44（匹配 SDK MuJoCo 3.11）。
- OpenCV 4.14.0 使用 SDK 的 `opencv-python-headless-noffmpeg`；视频编码另由 imageio-ffmpeg 0.6.0 提供。

`requirements.overlay.lock` 锁定 32 个 overlay 包。`full-runtime-constraints.txt` 是所有实际可见发行版版本的状态收据，它不是普通网络解析器安装文件；重建时先用 SDK 官方流程安装，再用 overlay lock 分阶段 `--no-deps` 安装。uv 不会识别这里通过 `.pth` 提供的 SDK 依赖，把全部要求直接交给 uv 解析会尝试重新安装核心 SDK/CUDA 数值库。

## 重建

在已安装的兼容 SDK 上运行：

```bash
./create_overlay.sh /project/EXP/IsaacLab/.venv/bin/python /path/to/Socially-aware_Indoor_Navigation_Benchmark /path/to/new-runtime-overlay
```

本脚本拒绝覆盖已有目录。`chumpy==0.70` 的 setup.py 依赖未声明的 pip，因此先锁定安装 pip/wheel，再以 `--no-build-isolation` 构建该包。SMPL 的实际模型数据应使用项目已有受限加载器，不直接执行未知 pickle；这个依赖环境不包含 SMPL 或 AMASS 数据。

## 已执行验证

- `import_check.json`：20 个数值库、配置库、动作解析、AMP、Diffusion、评估模块真实导入全部成功，并记录各模块文件路径。
- `runtime_dependency_check.json`：32 个 overlay 包，98 条默认环境依赖要求均有满足版本的可见 provider。
- `tests/test_isaaclab61_runtime_contracts.py`：在该独立 overlay 下 19 项 CPU 单位测试通过。
- `dm_control_cpu_regression.py`：真实 MuJoCo CPU 模型编译、1,000 步重力碰撞、模型复制与 reset 通过，SDK MuJoCo 保持 3.11.0。原 dm-control 1.0.40 会在建模时出现缺失 `flex_bandwidth` 错误，已按官方 [1.0.44 发布说明](https://github.com/google-deepmind/dm_control/releases/tag/1.0.44) 对齐。
- 以上没有启动 Kit 或 GPU；Isaac 仿真/训练/录像结果在主迁移审查报告中单独记录。
