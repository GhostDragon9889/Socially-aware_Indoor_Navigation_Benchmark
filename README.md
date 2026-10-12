# Socially-aware Indoor Navigation Benchmark

[English](README.en.md)

基于 TRACE 轨迹扩散、PACER/AMP 物理行人和 PPO 的室内社会导航研究代码。默认运行栈为 IsaacLab 3.0-EA / Isaac Sim 6.1。

## 快速开始

已有 SDK：

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root ../IsaacLab
```

新安装：

```bash
python3 scripts/bootstrap_reproduction.py install
```

两个入口均通过 uv 管理所需包，项目运行使用所配置 SDK 的 Python：

```bash
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
```

## 文档

- [安装与环境](docs/guides/installation.zh-CN.md)
- [资源准备与克隆复现](CLONE_REPRODUCTION.md)
- [运行与训练](docs/guides/usage.zh-CN.md)
- [项目结构](docs/structure.md)
- [验证状态与已知缺口](docs/validation.md)
- [全量验证与缓存维护](docs/verification.md)

实现位于 `src/`，当前配置位于 `configs/`，入口位于 `scripts/`，测试位于 `tests/`。模型、数据集和 USD 属于资源；生成输出不进入源码仓库。未达到全功能、全源码覆盖或全部场景质量通过，详见验证状态。

本项目使用 [ProtoMotions](https://github.com/NVlabs/ProtoMotions)、[TRACE](https://github.com/nv-tlabs/trace) 和 [PACER](https://github.com/nv-tlabs/pacer) 的代码与资源。许可及第三方说明见 [LICENSE.md](LICENSE.md) 和 `legal/`；SMPL/AMASS 必须遵守各自许可。

ProtoMotions references: [Pretrained models](docs/source/getting_started/pretrained_models.rst) · [GPC and PEFT guide](docs/source/user_guide/gpc.rst).
Robot definitions and assets: `src/protomotions/data/assets/mjcf/`.
