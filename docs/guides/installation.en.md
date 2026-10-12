# Installation and runtime

The project supports exactly two setup methods for IsaacLab 3.0-EA / Isaac Sim 6.1. Both run the project with the SDK Python 3.12.

## Configure an external SDK

```bash
python3 scripts/bootstrap_reproduction.py configure --lab-root ../IsaacLab
python3 scripts/bootstrap_reproduction.py check
python3 scripts/run_external_python.py -m social_navigation status
```

Use `--python` for another matching SDK interpreter or binary Sim launcher. uv installs locked project additions under `.runtime/external_sdk_packages`; the external SDK is preserved. Use `--rebuild-addons` after changing the additions lock. Open `.runtime/external_sdk.code-workspace` for editor setup.

## Install the SDK with uv

```bash
python3 scripts/bootstrap_reproduction.py install
python3 scripts/bootstrap_reproduction.py check
```

Git and uv are required on Linux x86_64. The installer uses the pinned Lab source commit and hash-checked manifest/lock in `env/isaaclab3_ea_sim61` to install Python, Lab and Sim under `.runtime/IsaacLab61`. `--lab-root` selects a new destination; an existing SDK must be reused with `configure`. Read and accept the [NVIDIA EULA](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html#running-isaac-sim) before setting `OMNI_KIT_ACCEPT_EULA=YES` and launching Sim.

## Resources and verification

```bash
python3 scripts/run_external_python.py scripts/manage_release_assets.py fetch
python3 scripts/run_external_python.py scripts/manage_release_assets.py prepare --check-usd
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
python3 scripts/run_external_python.py scripts/run_project_tests.py --output-dir output/tests_new
```

Weights, USD scenes and licensed SMPL/AMASS data are separate resources. See [clone reproduction](../../CLONE_REPRODUCTION.en.md). Local resource locations belong in ignored configuration, not source code. Test skips remain visible; setup and short training checks do not establish full navigation quality. See [validation status](../validation.md). The detailed [Chinese setup guide](external_sdk.zh-CN.md) describes provider checks and recoverable publication.
