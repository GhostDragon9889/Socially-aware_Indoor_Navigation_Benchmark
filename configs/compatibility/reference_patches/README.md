# Optional reference compatibility patches / 可选参考仓库兼容补丁

`third_party/` 下的公开参考仓库被主仓库忽略。这里保存其当前 Python 源码修改，不包含数据、权重或生成文件。`manifest.json` 固定原始 commit、补丁 SHA、每个文件的修改前后 SHA；包含本轮迁移及此前本地适配的完整差异。

The main repository ignores optional public reference checkouts. These source patches preserve their current Python adaptations, including earlier local changes. No data, weights, generated outputs, Git commits or network operations are included.

| Reference | Pinned base | Files |
|---|---|---:|
| NavDP | `bff5eb8857f50e082243fb455d312409daabc3aa` | 24 |
| FLUX | `21898204e68d97e06dadfc391acba3effd12847d` | 6 |
| PACER | `baeb6241f03fc9d78370a64e3c2dcef9d830d97f` | 10 |

在项目根目录，先准备对应固定版本的参考仓库：

```bash
python scripts/apply_reference_compat.py check
python scripts/apply_reference_compat.py apply
# 也可只处理一个参考仓库 / Select one optional checkout:
python scripts/apply_reference_compat.py check --repository navdp
```

`check` 仅核验，`apply` 先检查所有所选仓库再应用。已应用时重复执行不改文件；遇到不同 commit、混合修改、错误 SHA 或越界路径会拒绝并保留现有内容。原始数据或 IsaacGym/HTTP/桌面服务依赖仍需单独准备；补丁通过不表示完整可选基线已在 Sim6.1 实机验收。核心 Diffusion+AMP 使用 `src/` 中的实现，不依赖这些原始参考仓库来执行四场景流程。

`check` is read-only; `apply` validates every selected checkout first. Already patched checkouts remain unchanged. Wrong revisions, partial modifications, hash mismatches and paths outside the checkout are rejected. These patches do not certify complete legacy baseline execution on Sim6.1. Core four-scene Diffusion+AMP execution uses the project implementations under `src/`.
