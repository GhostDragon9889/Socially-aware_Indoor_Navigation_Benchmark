# Runtime dependencies

Use only the external SDK configuration or pinned uv SDK installation described in [installation](installation.en.md). The SDK provides IsaacLab, Isaac Sim, Torch, NumPy, Warp and USD. Project additions are pinned by `env/isaaclab3_ea_sim61/requirements.overlay.lock` and installed with dependency resolution disabled. Root package extras support documentation, development and crowd helpers; they do not install a simulator.

Current versions, platform requirements and manifest/lock SHA256 values are recorded in `env/isaaclab3_ea_sim61/profile.json`. Local SDK bindings and datasets are not committed. Model and dataset provenance belongs in resource registries.
