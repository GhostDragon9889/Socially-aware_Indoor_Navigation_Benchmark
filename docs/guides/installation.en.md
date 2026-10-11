# Installation

[中文](installation.zh-CN.md) · [Guide index](README.md)

The default is **Lab3EA/Sim6.1**. Configure an external SDK with `python3 scripts/bootstrap_reproduction.py configure --lab-root /absolute/path/to/IsaacLab`, verify with `bootstrap_reproduction.py check`, and run scripts through `scripts/run_external_python.py`. A full local installation is available with `bootstrap_reproduction.py install`. See [setup](external_sdk.zh-CN.md) for resources and editor configuration.

AMASS and SMPL separately. `--profile isaaclab3_sim6` retains the historical Sim6.0 environment. Its recordings and metrics do not certify the new SDK. New recording runs still require scene quality gates and complete video decoding.

Troubleshooting:

- Unsatisfiable `pytetwild` platform branches: use the pinned SDK manifest/lock for Linux x86_64; avoid project `uv sync --all-extras`.
- SDK/source/lock mismatch: reuse is rejected. Choose the matching SDK or create a new controlled environment in a separate checkout.
- pytest discovers a system ROS plugin requiring `lark`: clear inherited `PYTHONPATH`; run CPU repository tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. This controls plugin discovery, not simulation dependencies.
- USD provider overlap: run the read-only environment check. Do not co-install `usd-core` and `usd-exchange`.

CPU source inspection: `python3 scripts/audit_checkout.py`. The independent PointGoal PPO code in `src/crowdsim_ppo/` does not establish realistic 3D pedestrian performance.

The frozen `isaaclab3_sim6` manifest retains its original hashes and references the pre-migration project extra. Fresh installation or reapplication is rejected before SDK changes; only a matching already installed legacy SDK can be reused. An existing SDK is also preserved by the new-profile `install` action: use `prepare-runtime` for reuse.

Next: [dependencies](dependencies.en.md) · [configuration](configuration.en.md) · [usage](usage.en.md).
