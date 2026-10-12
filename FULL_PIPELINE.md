# Training and evaluation

Set up one of the two supported [SDK environments](docs/guides/installation.en.md), prepare the [resources](CLONE_REPRODUCTION.en.md), then use the unified entry points in [usage](docs/guides/usage.en.md). Current scene profiles are in `configs/reproduction/four_scene`; training recipes are in `configs/diffusion_amp` and `configs/crowdsim`.

Record fresh outputs in `output/`. Keep checkpoint provenance, actual process exits and physical quality checks separate. The repository no longer ships dated experimental logs or obsolete training checkpoints. See [validation limits](docs/validation.md).
