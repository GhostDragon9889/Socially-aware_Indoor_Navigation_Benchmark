# Validation status

Verification date: 2026-10-12. Runtime: external IsaacLab 3.0-EA, Isaac Sim 6.1.0, Python 3.12.13 and PyTorch 2.11/CUDA 12.8 on one RTX 4080 SUPER. The project-local uv SDK also passed package/provider and USD read/write checks. Network certificate failures required the cached offline uv route; a new online download was not revalidated in this run.

The final reorganized tree passed 4,924 ordinary CPU/CUDA tests and 38 subtests, with zero failed tests. Two upstream IsaacGym tests skipped because that retired backend is not installed. CPU and CUDA phases both returned exit zero. Eighteen archive-dependent tests were retired with the deleted historical replay/SDK functionality, not counted as passes.

An intermediate full run exposed an intermittent CPU distributed-worker exit hang. The test now limits CPU threads, emits timeout diagnostics and enforces a bounded worker lifetime with cleanup; its seven PPO/distributed tests and the final complete suite passed. The failed run remains in ignored local diagnostics. Numerical and physical acceptance thresholds were not lowered.

Current runtime checks:

- All four current scene resource preflights passed; the current model/USD release passed hashes and local USD dependency closure.
- YOHO with the selected pretrained TRACE/AMP weights passed 120 controls with no falls, observed process exit zero and full application cleanup. This is a short screen.
- Fresh episode-reference training passed with 8 actors and one PPO/AMP update, producing 32 transitions.
- Fresh future81 training passed with 256 actors, 32 controls and one PPO/AMP update, producing 8,192 transitions. The final portable registration was exercised physically. Selected/untouched groups, reversed reset IDs, link/COM velocity correction, unchanged input tensors, checkpoint writing and full shutdown were checked.
- The actual 256-actor checkpoint passed frozen-actor inference, optimizer checkpoint loading and CUDA RNG restoration; deterministic actions matched exactly. Full physical World/collector resume is not established by this check.
- A fresh source-only checkout with Git LFS payloads omitted passed the standalone entry point and layout tests, with no dangling source links. Maintained Python syntax/name checks, shell parsing and private workstation-path checks passed.

Earlier bounded 60-second screens passed YOHO and SecondFloor; NewMetro and ThirdFloor still have unresolved physical quality/admission failures. Short training does not establish convergence or publication reproduction. Multi-GPU hardware validation is unavailable on this single-GPU host, as acknowledged by the user. The earlier source statement coverage measurement was 76.04%, not 100%; it was not recalculated for this cleanup.

Obsolete reports, logs, checkpoints, configurations and replay tools were removed. Working-tree size decreased from about 292 GiB to about 71 GiB including the retained local SDK, current weights, new validation outputs and Git/LFS cache. Unused LFS cache could not be remotely verified for pruning and was retained; Git history was not rewritten. Required numerical fixtures remain in `resources/runtime_sources`; retired historical runs remain available in Git history at `333bf4c3`.

Current raw validation outputs are local and ignored under `output/repo_cleanup/`. The preceding historical results are not a claim of full code coverage, bug-free operation, or all-scene physical quality.
