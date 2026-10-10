# Usable Interact interface — staged, awaiting root source review

The official raw1000 scenes are sealed and verified. This directory supplies an independent, inspectable adapter for interaction-only training. **Full conversion and training have not run.** Production185, q935, original raw trajectories and defaults are unchanged. These are locally generated official-algorithm data, not the NavIsaacLab authors' dataset.

## Five necessary interfaces

| Interface | Existing restriction | Staged implementation |
|---|---|---|
| Official raw → five-state episode | First recorded timestamp is1/60s, then float-clock intervals, with a short last interval; no yaw | `convert_official_raw.resample_raw` prepends the genuine `initPos`/`initVelocity` at t=0, interpolates actual positions and velocities at100 timestamps0..9.9, with no extrapolation. Yaw is `atan2` of current interpolated velocity; stationary yaw0 is explicitly a proxy and its validity mask is stored. |
| Manifest / scene split | `ORCACorpus.__init__` rejects any repeated geometry hash, even the real empty layout | `InteractionORCACorpus` accepts only the explicit interaction schema, no map and empty obstacles. All empty geometries share the same truthful hash. Scene/raw/episode IDs remain unique and the registered train800/val100/test100 split is retained; `val` is only renamed `validation` for the local API. |
| Windows / synchronized neighbors | `conditional_windows` requires at least.25m² visible obstacles; batching has only14 neighbor slots | Windows require a real current neighbor within15m and complete31-history/52-future support. Up to19 neighbors are retained for20-agent scenes. Single-agent/isolated scenes remain in the manifest with zero windows; no obstacle or interaction is fabricated. |
| Map conditioning | Filling the raster with.5 can still produce nonzero encoded features | The raster is unknown.5 in both channels and `map_condition_mask=False` is passed through the actual staged train-batch/context functions. The encoded map grid is explicitly zero. Maps present in a context retain their original grid. |
| Training evidence / offline metrics | First-gradient gate requires map gradients; metric assumes.4m discs and finite obstacle distances | The staged trainer requires real neighbor gradients and zero map gradients for batches without maps. It records available-map windows. ADE/FDE formulas are retained; neighbor gap uses the official agents' actual radii. Obstacle metrics are `null`/N/A. The original DDPM schedule, loss, Adam, architecture, sampling and guidance code remain otherwise unchanged. |

The actual pipeline is `train_interact_diffusion.py` → `InteractionORCACorpus` → staged `paper_interact.py`; the older alternative `diffusion/model.py` is not used. [source_derivation.json](source_derivation.json) records exact substitutions and full inverse bytes/AST to the current original paper and trainer sources.

## Small actual probes

[actual_positive_inputs.json](actual_positive_inputs.json) passed17 checks in an actual CPU process, exit0. It exercised analytic irregular-clock resampling, genuine t0, stationary-heading semantics, zero-window single-agent data, a full19-neighbor boundary, and **only three** real20-agent scenes, one per registered split. Their partial conversion was created under `/tmp` and removed. The actual staged tensor/context and encode-context functions were checked with explicit fake encoders; no neural-network constructor, optimizer, DDPM, checkpoint or SDK was executed. This is interface evidence, not training or physical quality evidence.

The preserved executed checker and the current checker differ only by two Ruff import-order suppression comments; their ASTs are identical. [actual_CPU_closure.json](actual_CPU_closure.json) records the actual process and source identities.

## Limits and next action

The current mask zeros grid features and map gradients. The map encoder still executes, so its BatchNorm running buffers can change in training mode. This is explicitly **not** a claim that later mapped inference remains identical after interaction-only training. Root should decide whether to protect those buffers before selecting a real training recipe.

After root source review, use the exact conversion command in [planned_commands.json](planned_commands.json) to create a fresh independent usable dataset. Then perform one whole-corpus QC for1000 scenes, true100-point uniform timing, exact split, finite states, actual variable radii, masks, and reported per-split eligible windows including zero-window scenes. No source can claim the actual window mixture before that conversion; official raw1000 per domain does not imply50:50 training windows. Mixing with the existing Maps corpus is a later explicit dataset/sampler decision, not implemented or claimed here.

Primary conventions are pinned to [TRACE official ORCA README](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim/README.md), [official generator](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/orca_sim/gen_dataset.py#L164-L190) and [ORCA dataset parser](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset_specific/orca_peds/orca_dataset.py). Explicit map-grid masking is a local compatibility adaptation; it is not asserted as the author's exact no-map implementation.
