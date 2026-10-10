# 发行工件验证：preliminary_pass_requires_rebuild

仅离线安装到 /tmp target；现有环境未安装/升级包。

| 工件 | 大小（bytes） | SHA256 |
| --- | ---: | --- |
| protomotions-3.1.0-py3-none-any.whl | 40237523 | `9bff784c38534334999f721b37aaf75bc9734fd67b43c18794fa081a8c8b0f39` |
| protomotions-3.1.0.tar.gz | 40501794 | `9869f6af8b1a8ab57a8660863944a3ab3b80ae9e13bdf258a555c5eaee3a0e13` |
| crowdsim_ppo-0.1.0-py3-none-any.whl | 31645 | `0e4e69065286583da989c101aaf4d00abf68aec0e7f2bcbf0f0c14c68bd79998` |
| crowdsim_ppo-0.1.0.tar.gz | 29956 | `f5b20509f6430267aa713d8ebfa00aea4ac26f4cb5c3e5bf76eaf409362b17a8` |

## 检查

- passed: core_wheel: no licensed SMPL asset or external weight file
- passed: core_wheel: safe archive member names
- rebuild_required: core_wheel: shipped source matches frozen tree — `['protomotions/components/pose_lib.py', 'protomotions/components/terrains/terrain.py', 'protomotions/components/terrains/terrain_utils.py', 'protomotions/simulator/base_simulator/record.py', 'protomotions/simulator/base_simulator/utils.py', 'protomotions/simulator/mujoco/config.py', 'protomotions/simulator/mujoco/simulator.py', 'protomotions/simulator/newton/config.py', 'protomotions/simulator/newton/simulator.py', 'protomotions/utils/checkpoint.py']`
- passed: core_sdist: no licensed SMPL asset or external weight file
- passed: core_sdist: safe archive member names
- rebuild_required: core_sdist: shipped source matches frozen tree — `['CrowdSim/README.md', 'CrowdSim/humanoid/README.md', 'CrowdSim/nav_manager.py', 'CrowdSim/ppo/ppo_policy.py', 'CrowdSim/ppo/train_ppo.py', 'CrowdSim/ppo/train_ppo_finetune.py', 'CrowdSim/pref/pref_train_reward.py', 'CrowdSim/pref/train_offline_reward.py', 'CrowdSim/scene_setup.py', 'CrowdSim/tests/test_preference_padding.py', 'CrowdSim/tests/test_training_lifecycle.py', 'FULL_PIPELINE.md', 'docs/ISAAC_SIM6_LAB3_MIGRATION_ZH.md', 'protomotions/components/pose_lib.py', 'protomotions/components/terrains/terrain.py', 'protomotions/components/terrains/terrain_utils.py', 'protomotions/simulator/base_simulator/record.py', 'protomotions/simulator/base_simulator/utils.py', 'protomotions/simulator/mujoco/config.py', 'protomotions/simulator/mujoco/simulator.py', 'protomotions/simulator/newton/config.py', 'protomotions/simulator/newton/simulator.py', 'protomotions/utils/checkpoint.py', 'scripts/audit_repository.py', 'src/crowdsim_ppo/ppo_policy.py', 'tests/test_ppo_finite_and_resume.py']`
- passed: crowd_wheel: no licensed SMPL asset or external weight file
- passed: crowd_wheel: safe archive member names
- rebuild_required: crowd_wheel: shipped source matches frozen tree — `['crowdsim_ppo/ppo_policy.py']`
- passed: crowd_sdist: no licensed SMPL asset or external weight file
- passed: crowd_sdist: safe archive member names
- rebuild_required: crowd_sdist: shipped source matches frozen tree — `['crowdsim_ppo/ppo_policy.py']`
- passed: core_wheel: isolated package namespace
- passed: crowd_wheel: isolated package namespace
- passed: core_wheel: checkpoint helper delivered
- passed: core_wheel: correct distribution
- rebuild_required: core_sdist: integration/profile/check scripts delivered — `['scripts/check_isaaclab_smpl_fk.py', 'scripts/check_motion_asset_consistency.py', 'docs/PROJECT_COMPLETION_20261001_ZH.md']`
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
  "python": "/project/EXP/SangforAndSaut/ProtoMotions/../IsaacLab-Protomotions/.venv/bin/python",
  "temporary_directory": "/tmp/protomotions-release-verification-02o07mpx",
  "cwd": "/tmp/protomotions-release-verification-02o07mpx/arbitrary-working-directory",
  "install": "uv pip install --offline --no-index --no-deps --target TEMP",
  "default": {
    "package": "/tmp/protomotions-release-verification-02o07mpx/installed/crowdsim_ppo/__init__.py",
    "default_config": "/tmp/protomotions-release-verification-02o07mpx/installed/crowdsim_ppo/configs/point_goal.json",
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
    "success_rate": 0.5,
    "collision_rate": 0.5,
    "mean_return": 6.081435918807983,
    "mean_episode_length": 38.5
  },
  "existing_environment_modified": false,
  "scope": "Wheel code isolated in /tmp; existing NumPy/Torch dependencies reused; no Kit, no GPU training"
}
```

短训练使用 wheel 自带默认 JSON，按完整 rollout 提交步数；请求 1 步实际为 8192 步，恢复后为 16384 步。此处验证安装/入口/保存恢复链，指标不代表策略质量。

## 再次运行

```bash
/project/EXP/SangforAndSaut/ProtoMotions/../IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/completion_20261001/verify_release_artifacts.py --artifacts /tmp/protomotions-completion-packaging --prefix release_artifacts_final
```

初验使用 `--preliminary` 将尚未重建的源文件/新脚本列为待重建；最终模式要求发行代码与冻结源码匹配。父任务主报告创建之前，不把它的缺失算作最终缺陷。
