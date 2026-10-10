# 真实 tracking 捕获进入冻结风格评估的验证

## 结果

已新增 `scripts/evaluate_pedestrian_style.py --run NAME=trackingdir` 分支。保留原 `physical_states.npz` 导航读取与所有风格公式，只在不存在该文件且存在完整 tracking result 时读取明确声明的逐动作 NPZ。源码已冻结。

71 项 focused CPU 回归全部通过，Ruff 通过。真实旧模型、新 500、新 1000 三个完整 22 条捕获均通过严格输入检查；每个 11,982 个 post-step 帧得到 11,960 个合法风格区间。22 个 initial reset 帧排除，21 个跨动作拼接差分排除；跌倒帧 80 / 0 / 455 全部保留。

## 输入与来源合同

- 不用 glob 搜索补缺，只读取 `result.sequences` 声明的 capture 路径；目录边界、精确 motion filename、SHA、数量、motion ID、source path 与 preflight 逐项核对。
- 必须是真正的 post-physics 30Hz 同参考帧协议。连续 reference_frame、timestamp、true bool metric/reset/fall masks、唯一单 clip episode、initial reset 和完整 clip 均验证；缺失、哈希改变、跳帧、重复 reset、伪 mask 和部分捕获拒绝。
- 只保留每条动作的 true post-step metric 帧；每条 clip 设置新的 episode ID 和 before-frame reset。导数不会跨动作、跨 reset；不会搜索时间偏移或过滤跌倒。
- 实际训练 source 来自 `controller_provenance.dataset`，与 oracle heldout 输入分开核对。训练 manifest SHA、bank 声明 SHA、split=train、clip 数、当前 controller SHA 都绑定真实文件。规范化 `parent_source_path` 和 `#frames` 父源后验证完整留出清单无训练父源交集。
- legacy parent-joint 与新 child-joint 参考速度合同显式保留；新模型不伪装仅替换权重的同初始条件评估。
- 任务显式标记 truefutureXY oracle low-level tracking；真实 foot-floor contact、static/peer contact 和 true COM 未捕获时为 null。

## 不改变旧分数的实证

旧 raw 和 timed_causal 导航真实 physical_states 捕获重新通过原读取分支。`generated`、`reference`、`inclusive_speed_conditioned`、`upright_subset`、`heldout_between_clip_group_calibration` 五组字段，与旧冻结报告逐对象 **完全相等**。`CrowdSim/humanoid/style_evaluation.py` 源 SHA 未变，仍为 `b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2`。

冻结参考文件仍是 SHA `c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`，三检查点评估输出的参考副本逐字节相同。新增 `reference_load_provenance` 独列记录加载模式，没有改写旧 reference 对象、采样 IDs 或特征 SHA。

新增 reader 前源码 SHA 为 `79c19e637dd6335e0b631dc6d7b161d2a2a9ce04231a5065f157c4c0903ed1b9`；冻结后的源码为 `82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f`。before 快照、全部源路径和测试 XML 见 [机器验证报告](tracking_style_input_review.json)。旧 normal regression 报告的源码 SHA 属于最后一个 tracking-only 缺失 contact 元数据修正前的中间快照；该修正只把 tracking 的未测 contact 声明改为 null，未触碰原 physical_states 分支或其得分字段。最终三个 tracking 报告和 71 项回归使用上述冻结后源码。

## 回归命令

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python -m pytest -q \
tests/test_tracking_style_input.py tests/test_pedestrian_style_evaluation.py
```

回归覆盖完整来源/阶段失败合同，以及初始帧含巨大跳跃、不同 clip 相隔千米、跌倒帧仍保留的非镜像数值案例。所有操作只在 CPU 上进行，未启动 Kit。
