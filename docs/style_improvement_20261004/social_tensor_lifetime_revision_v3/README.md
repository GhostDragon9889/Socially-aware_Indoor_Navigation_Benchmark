# Owned tensor lifetime revision v3 — smoke only

Prepared source; ROOT and independent peer review completed. Files are sealed before SDK execution. Neither this revision nor a GDB debugger exit code establishes an SDK process exit of zero.

## Evidence and method

The parent is the sealed cleanup v2 smoke entry (`fd44f44dae78aea69321ca7f5031e07e2c6fc1ec432c0a807e5720f38403d661`). In the actual GDB run, physics checks completed, but the inferior received SIGSEGV while Python finalized a traceback: tensor plugin destruction, list/frame/traceback deallocation, OSError cleanup, and Python GC. The complete log is `../social_actual_runtime_v1/group_physics_gdb_v4.log`, lines 412–450. The debugger returning 0 does not convert this failure into a successful SDK exit.

[Isaac Sim 6 known issues](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/known_issues.html) identifies shutdown crashes when Python tensor API objects outlive their extension. The installed plain Python `RigidContactView` class permits weak references and exposes no public destroy/release/close API. SimulationView.invalidate retains the backend wrapper; global tensors.reset resets buffers and does not promise Python object disposal. Neither is used as a disposal substitute.

The new entry initializes partial-construction outer locals. After stop, it clears the returned native contact list and caller asset list **in place**, drops the final explicit asset loop alias and scene, and runs GC while tensor extensions remain available. It requires all weak references to the returned contact views to be dead before context clear. Context clear still checks owned singleton identity. A second GC after context cleanup releases any cycles made unreachable by manager/service teardown before App.close. Independent cleanup exceptions are recorded, and App.close is still attempted.

All physics/state/contact math, exact collision-filter probes, 64 physics substeps for the default two phases, seed, configuration, source pins, output snapshots and original catch boundary remain unchanged. Only the owned reference disposal sequence and its success flags are added. No native resets, private SDK writes, fast shutdown, os._exit, policy construction, model loading, or training occurs in source preparation.

## Scope and remaining limits

This smoke revision does not modify the sealed World/probe/training sources. Failure during helper construction can retain native objects never returned to this entry, and an active helper exception can hold its internal view aliases. Outer initialization and a zero-live-view guard do not prove all possible error paths safe. SDK process exit of zero must be observed independently; a report containing passed flags is insufficient.

`source_derivation.json` records exact literal replacements, complete inverse parent byte/AST equality, and installed source identity. Use a new independent output directory under ROOT's exclusive GPU schedule only after review and sealing.

## Source review

Author compile/inverse checks `a6ef20` and Ruff `964fed` returned0. ROOT read the changes/README/derivation in `c154af` and independently checked complete inverse bytes/AST/compile in `910512`. Independent peer found Required=[] and Optional=[], verified all eight literal forward/inverse replacements and unchanged helpers, and ran Ruff `ee20c8`; its stdlib chunk was not printed and is not invented here. Actual SDK evidence will be recorded separately.
