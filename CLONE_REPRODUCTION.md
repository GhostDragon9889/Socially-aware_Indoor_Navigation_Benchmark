# Clone and reproduce

Clone this repository, then follow [installation](docs/guides/installation.en.md): configure an external IsaacLab/Isaac Sim SDK or install the pinned SDK with uv. Source code contains no workstation installation paths.

```bash
python3 scripts/run_external_python.py scripts/manage_release_assets.py fetch
python3 scripts/run_external_python.py scripts/manage_release_assets.py prepare --check-usd
python3 scripts/run_external_python.py scripts/verify_pretrained_models.py --output output/models.json
python3 scripts/run_external_python.py -m social_navigation preflight --scene YOHO
```

Resource services may require access authorization. Obtain licensed SMPL/AMASS assets separately. Configure external datasets in the ignored `data/external_resources.json`, using IDs from `configs/reproduction/resource_bindings.json`. Preserve model/asset SHA256 and geometry; do not substitute an unavailable dataset silently. New outputs belong in ignored `output/`.

Run `scripts/run_project_tests.py` through the SDK entry for CPU/CUDA isolation. Missing resources, hardware, and unresolved physical quality remain explicit failures or gaps; a clone/setup check does not certify the full research protocol.
