# 发行工件验证：archives_passed_runtime_not_run

仅离线安装到 /tmp target；现有环境未安装/升级包。

| 工件 | 大小（bytes） | SHA256 |
| --- | ---: | --- |
| protomotions-3.1.0-py3-none-any.whl | 40239296 | `6131656b86c2aa4f14fa4734e7a8131726fc1f8f28cefa24194e0be4717059e1` |
| protomotions-3.1.0.tar.gz | 40607846 | `53a294bffd2bdb792a94ffdf5309a73535a760814a33b5c837ea00ebba580f11` |
| crowdsim_ppo-0.1.0-py3-none-any.whl | 31734 | `c263123252a05c9a783ca99c36fcb0a373a7bd2e2e122d472051dd6427ef5967` |
| crowdsim_ppo-0.1.0.tar.gz | 30032 | `f67b3e825da54e42667ad9e7aed924615765cd2dad55c8d4c10873591987610d` |

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
  "status": "not_run"
}
```

运行未完成或已跳过；没有将训练、恢复、评估记为通过。

## 再次运行

```bash
/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python /project/EXP/SangforAndSaut/ProtoMotions/docs/completion_20261001/verify_release_artifacts.py --artifacts /tmp/protomotions-completion-packaging --prefix release_artifacts_final
```

初验使用 `--preliminary` 将尚未重建的源文件/新脚本列为待重建；最终模式要求发行代码与冻结源码匹配。父任务主报告创建之前，不把它的缺失算作最终缺陷。
