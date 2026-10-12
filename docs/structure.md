# Repository layout

The layout separates implementation, maintained configuration, entry points, tests and externally supplied resources, following the separation of model/configuration/data/scripts in the [TRACE](https://github.com/nv-tlabs/trace) and [PACER](https://github.com/nv-tlabs/pacer) CVPR 2023 repositories. Their historical installation stacks are not used here.

| Directory | Purpose |
| --- | --- |
| `src/` | Canonical Python packages |
| `configs/` | Current scene, training and integrity configurations |
| `scripts/` | Setup, data preparation, training and evaluation entry points |
| `tests/` | Runtime contracts and regression tests |
| `env/isaaclab3_ea_sim61/` | The single pinned uv SDK profile |
| `docs/` | Maintained usage and API documentation |
| `resources/` | Resource manifests, publication payloads and required reference inputs |
| `data/` | Local datasets, assets and pretrained models |
| `output/` | New generated results, ignored by Git |
| `.runtime/` | SDK installations, configured binding and project additions, ignored by Git |

`resources/runtime_sources` contains only retained source/fixture dependencies used by current code and tests. It does not preserve the former multi-gigabyte report archive. Fixed input hashes are recalculated explicitly for path-only source migration; numerical physics acceptance thresholds are unchanged. Valid source compatibility links remain; dangling links and the retired overlay interpreter link are removed.

Maintained executable code uses project-relative paths or `project:`, `lab:` and `sim:` references resolved from runtime configuration. User-provided absolute paths may appear in ignored local bindings and generated receipts. Original byte-bound release evidence retains its recorded locations; those locations are provenance, not installation defaults.

Fresh shared Social AMP recipes live in `configs/training/social_amp/`. Fixed S3/episode owner replay, old four-scene archive replay, and past state-action checkpoint/evaluation receipts are retired. Their specialized replay tests are retired with those tools; PPO, AMP, reset/velocity, source integrity, native-map geometry and metric mathematics tests remain.

The original numerical fixtures in `resources/runtime_sources` are excluded from automatic source formatting/header insertion. Their explicit hash checks and mathematical equivalence tests remain active.
