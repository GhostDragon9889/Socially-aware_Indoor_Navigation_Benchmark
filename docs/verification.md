# 验证与缓存维护

先用受支持的两个入口之一配置外部 SDK 或安装 uv SDK，见[安装说明](guides/installation.zh-CN.md)。以下命令均使用所配置 SDK 的 Python，不创建第三种运行环境。验证工具集中在 `scripts/verification/`，缓存维护入口在 `scripts/maintenance/`，可复用实现放在 `src/social_navigation/`。

## 全量回归与语句覆盖

普通回归将 CPU 和 CUDA 测试放在不同进程，避免 CPU 契约测试受已初始化的 CUDA 状态影响：

```bash
python3 scripts/run_external_python.py scripts/run_project_tests.py \
  --output-dir output/tests_new
```

严格覆盖率执行器同时运行该回归流程，并检查执行前后的源码 SHA256：

```bash
python3 scripts/run_external_python.py scripts/verification/run_statement_coverage.py \
  --ignore-isaacgym --output-dir output/coverage_new
```

输出目录必须是新的。测试失败、数据缺失或源文件改变时保留失败诊断；任何未覆盖语句都会使默认命令返回非零。`--report-only` 只允许生成未达标报告，其状态仍为 `coverage_gap`；`--cpu-only` 会保留 CUDA 跳过项。

源码范围来自 Git 索引及未忽略的待提交文件，包括所有非 `tests/` 目录中的 Python 文件、手动工具、资源中的参考实现以及尚未执行的模块。仓库内源码软链接按解析后的文件去重，越界或失效源码链接直接报错。默认不排除后端，任何模式均不排除 `__main__` 分支或 `pragma: no cover`。分母使用 Python/coverage 的可执行语句定义；注释、空行和编译器消除的常量死代码不能称为已执行语句。

按本次明确的验收要求，示例传入 `--ignore-isaacgym`。该选项仅豁免 `src/protomotions/simulator/isaacgym/` 的独立源码与旧后端硬件测试，完整源身份和豁免理由仍记录在 `excluded_files`；不豁免共用代码、IsaacLab 或其他功能。默认不传该选项时仍统计全部后端。执行后更改豁免范围会导致范围稳定性检查失败。

动态代码归属使用 [coverage 文件跟踪插件接口](https://coverage.readthedocs.io/en/7.15.2/api_plugin.html)。原文件必须先通过执行前清单中的字节数和 SHA256 核验；虚拟执行帧的代码对象还必须与原文件重新编译的代码对象完全一致，只忽略文件名本身。字节码、常量、名称、限定名称、标志、异常表、行号表和原始行号都参与比对。只有符合这项证明的实际执行行才归属原文件；改变函数或偏移行号的代码仍独立列在 `virtual_code`。不能仅凭虚拟文件名推测原文件。

每个进程的成功归属证明保存在 `virtual_frame_proofs/`，包含源文件 SHA256、代码对象 SHA256、限定名称和首行位置。直接执行和通过证明的虚拟执行使用同一跟踪身份，避免合并数据时产生插件身份冲突。`coverage.json` 包含逐文件源身份、已执行语句、缺失语句、零命中文件和门禁状态；分支执行数据单独采集。语句覆盖率仅证明语句被执行，不证明全部语义、物理质量或训练收敛。

已固定的 uv SDK 包含 coverage。启动钩子及 coverage 包的只读链接只写入本次输出目录，不向外部 SDK 安装 `.pth` 文件。子进程退出必须保存执行数据，机制参见 [coverage 官方进程说明](https://coverage.readthedocs.io/en/7.15.2/subprocess.html)。

单独核验已有的合并数据：

```bash
python3 scripts/run_external_python.py scripts/verification/audit_statement_coverage.py \
  --data output/coverage_new/data/.coverage \
  --baseline output/coverage_new/source_before.json \
  --output output/coverage_new/rechecked.json --ignore-isaacgym --require-complete
```

## 缓存清理

先查看选择结果，再执行清理：

```bash
python3 scripts/run_external_python.py scripts/maintenance/clean_project_cache.py
python3 scripts/run_external_python.py scripts/maintenance/clean_project_cache.py \
  --apply --output output/cache_cleanup_new.json
```

在训练、仿真、安装及测试进程结束后执行。清理器只选择已知可重建目录，并在删除前复核项目根标记、目录身份和白名单；不跟随软链接。它保留模型、数据集、USD、SDK 扩展 `extscache`、活动项目依赖和 Git LFS 对象。旧依赖备份只有在有效活动依赖绑定存在时才可清理。`.git/lfs/cache` 与 `.git/lfs/tmp` 属于临时缓存；对象库不会被该命令修剪。

当前通过项、真实物理测试及未验证范围统一记录在[验证状态](validation.md)。不得将跳过项、失败记录、资源预检或短程训练计为全功能通过。

## 导航与诊断契约

下面的测试直接执行 A*、SFM、导航观测、奖励及重置代码；仿真写入接口由内存传输层替代，因此不属于新一轮 PhysX 验收。封存探针按原文件加载并复核 SHA256，执行地图坐标、地面归属、查询时钟与真实 OpenUSD 图层操作：

```bash
python3 scripts/run_external_python.py -m pytest -p no:cacheprovider -q \
  tests/test_navigation_manager_flow.py tests/test_navigation_observation_channels.py \
  tests/test_navigation_scene_obstacles.py tests/test_social_force_field_of_view.py \
  tests/test_social_force_diagnostics.py tests/test_frozen_probe_contracts.py \
  tests/test_navigation_rendering.py tests/test_repository_audit.py
```

封存 v6 探针没有后续版本的单一地面支撑证书；其契约测试记录原始版本限制，不将它用于当前场景认证，也不修改封存字节。

SFM 手动入口使用当前控制器与地图接口，可以生成 CPU 数值诊断图：

```bash
python3 scripts/run_external_python.py -m CrowdSim.world.sfm \
  --map src/CrowdSim/maps/World0.png --headless --steps 200 \
  --output output/sfm_diagnostic.png
```

输出明确记录 `physics_executed=false`。离线导航 MP4 优先使用 OpenCV；SDK 的 OpenCV 缺少编码器时使用已有 MoviePy/imageio 的 [FFmpeg 流式接口](https://github.com/imageio/imageio-ffmpeg#api)。回归会实际编码并解码视频，核对帧数、尺寸与图像变化；H.264 输出尺寸向上取偶数。离线视频通过不等于四场景实时相机全部通过。
