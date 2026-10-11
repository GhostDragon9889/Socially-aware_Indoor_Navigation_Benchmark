# Installation

[中文](installation.zh-CN.md) · [Guide index](README.md)

The default is **IsaacLab 3.0 EA / Isaac Sim 6.1.0.0**. Follow [clone reproduction](../../CLONE_REPRODUCTION.en.md) for complete assets, data and execution. On a fresh machine, run from the project root:

```bash
uv python install 3.12.13
uv run --no-project --python 3.12.13 python scripts/bootstrap_reproduction.py install --profile isaaclab3_ea_sim61
export NAV_LAB="$PWD/.runtime/IsaacLab61"
export NAV_PY="$PWD/.runtime/isaaclab3_ea_sim61/.venv/bin/python"
export PYTHONPATH="$PWD/src:$PWD"
"$NAV_PY" -m social_navigation.runtime_environment check --lab-root "$NAV_LAB"
"$NAV_PY" scripts/manage_release_assets.py fetch
"$NAV_PY" scripts/manage_release_assets.py prepare --check-usd
```

For a matching installed SDK, set `NAV_LAB` to its real path and run `python3 scripts/bootstrap_reproduction.py prepare-runtime --profile isaaclab3_ea_sim61 --lab-root "$NAV_LAB"`. Project dependencies install in an isolated overlay without editing the SDK. Reuse requires matching SDK, locks and paths; failure preserves the selected interpreter. See [runtime instructions](../../env/isaaclab3_ea_sim61/README.md).

Prepare licensed raw AMASS and SMPL separately. `--profile isaaclab3_sim6` retains the historical Sim6.0 environment. Its recordings and metrics do not certify the new SDK. New recording runs still require scene quality gates and complete video decoding.

Troubleshooting:

- Unsatisfiable `pytetwild` platform branches: use the pinned SDK manifest/lock for Linux x86_64; avoid project `uv sync --all-extras`.
- SDK/source/lock mismatch: reuse is rejected. Choose the matching SDK or create a new controlled environment in a separate checkout.
- pytest discovers a system ROS plugin requiring `lark`: clear inherited `PYTHONPATH`; run CPU repository tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. This controls plugin discovery, not simulation dependencies.
- USD provider overlap: run the read-only environment check. Do not co-install `usd-core` and `usd-exchange`.

CPU source inspection: `python3 scripts/audit_checkout.py`. The independent PointGoal PPO code in `src/crowdsim_ppo/` does not establish realistic 3D pedestrian performance.

The frozen `isaaclab3_sim6` manifest retains its original hashes and references the pre-migration project extra. Fresh installation or reapplication is rejected before SDK changes; only a matching already installed legacy SDK can be reused. An existing SDK is also preserved by the new-profile `install` action: use `prepare-runtime` for reuse.

Next: [dependencies](dependencies.en.md) · [configuration](configuration.en.md) · [usage](usage.en.md).
