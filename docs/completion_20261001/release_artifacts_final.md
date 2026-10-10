# 发行工件验证：passed

仅离线安装到 /tmp target；现有环境未安装/升级包。

| 工件 | 大小（bytes） | SHA256 |
| --- | ---: | --- |
| protomotions-3.1.0-py3-none-any.whl | 40239296 | `8171f5c072cac39ee545f62855d24a7f81914870ff3be7b1451754a8af1b299c` |
| protomotions-3.1.0.tar.gz | 40942187 | `5f219268516bbb3676e67937df0c52dfbf0b5839539623c75e4deff7a6cad51c` |
| crowdsim_ppo-0.1.0-py3-none-any.whl | 31734 | `478d2071690cbcaa86f09062c45fddacde81c37edaa123ea747954444814b108` |
| crowdsim_ppo-0.1.0.tar.gz | 30034 | `537ccedbeff6c980a87acff502367bc402b65b056d4e2086866c06f75b9b3c66` |

## 检查

- passed: core_wheel: no licensed SMPL asset or external weight file
- passed: core_wheel: safe archive member names
- passed: core_wheel: shipped source matches frozen tree
- passed: core_sdist: no licensed SMPL asset or external weight file
- passed: core_sdist: safe archive member names
- passed: core_sdist: shipped source matches frozen tree
- passed: crowd_wheel: no licensed SMPL asset or external weight file
- passed: crowd_wheel: safe archive member names
- passed: crowd_wheel: shipped source matches frozen tree
- passed: crowd_sdist: no licensed SMPL asset or external weight file
- passed: crowd_sdist: safe archive member names
- passed: crowd_sdist: shipped source matches frozen tree
- passed: core_wheel: isolated package namespace
- passed: crowd_wheel: isolated package namespace
- passed: core_wheel: checkpoint helper delivered
- passed: core_wheel: correct distribution
- passed: core_sdist: integration/profile/check scripts delivered
- passed: sdists: generated build trees omitted
- passed: crowd_wheel: no ProtoMotions/Isaac hard dependency
- passed: crowd_wheel: correct console entries
- passed: crowd_wheel: bundled default JSON delivered
- passed: crowd_sdist: bundled default JSON delivered
- passed: crowd artifacts: default JSON agrees

## 临时 wheel 实际执行

```json
{
  "status": "passed",
  "python": "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python",
  "temporary_directory": "/tmp/protomotions-release-verification-8zaq6wn7",
  "cwd": "/tmp/protomotions-release-verification-8zaq6wn7/arbitrary-working-directory",
  "install": "uv pip install --offline --no-index --no-deps --target TEMP",
  "default": {
    "package": "/tmp/protomotions-release-verification-8zaq6wn7/installed/crowdsim_ppo/__init__.py",
    "default_config": "/tmp/protomotions-release-verification-8zaq6wn7/installed/crowdsim_ppo/configs/point_goal.json",
    "default_sha256": "81391e785455b582fbff03ccbceac73b7413b46a49ed74eecc4757d5e59a67d6",
    "num_envs": 64,
    "rollout_steps": 128,
    "simulator_import_guard": "passed"
  },
  "initial": {
    "step": 8192,
    "model_sha256": "4011125950d2bd1ae243aa873fd242b6803f82a4211131d2fae2385af82f5759",
    "model_parameters": 53893,
    "finite_tensor_count": 52,
    "optimizer_step_min": 128.0,
    "optimizer_step_max": 128.0,
    "curriculum_present": true
  },
  "resumed": {
    "step": 16384,
    "model_sha256": "3bdd48e8f2a4fac84f8e2835ced8dbafa0a6c4b5918c4fea263104f2c708d5ac",
    "model_parameters": 53893,
    "finite_tensor_count": 52,
    "optimizer_step_min": 256.0,
    "optimizer_step_max": 256.0,
    "curriculum_present": true
  },
  "evaluation": {
    "episodes": 2.0,
    "success_rate": 0.0,
    "collision_rate": 0.0,
    "mean_return": -1.788106918334961,
    "mean_episode_length": 250.0
  },
  "existing_environment_modified": false,
  "scope": "Wheel code isolated in /tmp; existing NumPy/Torch dependencies reused; no Kit, no GPU training"
}
```

短训练使用 wheel 自带默认 JSON，按完整 rollout 提交步数；请求 1 步实际为 8192 步，恢复后为 16384 步。此处验证安装/入口/保存恢复链，指标不代表策略质量。

## 再次运行

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/completion_20261001/verify_release_artifacts.py --artifacts /tmp/protomotions-completion-packaging --prefix release_artifacts_final
```

初验使用 `--preliminary` 将尚未重建的源文件/新脚本列为待重建；最终模式要求发行代码与冻结源码匹配。父任务主报告创建之前，不把它的缺失算作最终缺陷。
