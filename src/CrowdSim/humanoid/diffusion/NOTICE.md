# TRACE-derived inference and training code

Source: https://github.com/nv-tlabs/trace, commit `594b2af0824c5853e1a1884e167cd5ed097dd81c`. Derived low-level modules are subject to the included NVIDIA Source Code License-NC, not the root project licence. TRACE README separately describes pretrained weights under CC-BY-NC-SA-4.0. Original Diffuser/UNet attribution comments are retained in source excerpts.

Temporal/diffusion blocks are based on https://github.com/jannerm/diffuser and https://github.com/lucidrains/denoising-diffusion-pytorch, as attributed by TRACE; the bundled upstream Diffuser MIT notice is retained in LICENSE.diffuser. Map decoder U-Net blocks retain the upstream https://github.com/milesial/Pytorch-UNet attribution. The independent five-state application selects architecture details explicitly and does not load official TRACE weights.

Port changes: imports confined to this package; remove dataset/Lightning/guidance registration dependencies; CPU tensor.device and GPU map index device compatibility. Adapter-specific input validation and world/local transforms are new code. Full upstream model/checkpoint key contract is retained; the checkpoint has policy and EMA copies, each 406 entries.
