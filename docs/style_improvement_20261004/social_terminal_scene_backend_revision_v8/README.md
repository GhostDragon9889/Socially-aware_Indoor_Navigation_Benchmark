# Terminal scene backend release, revision v8

This independent source revision starts from sealed v7 `smoke_group_physics.py` (40359 bytes, SHA256 `fc0b43baeee2b98450161b773f308e1078b5c133547c8093ec6acca41f4c6176`). Original sources and installed SDK files remain unchanged.

## Observed failure and source inference

ROOT's real v11 process, session 25781 (`1f1ff2` → `1cf318`), exited 139 despite returned owned-manager normalization, Base cleanup, Core shutdown and Timeline-cache release; all eight owned contact wrappers were dead and cleanup errors were empty. Its sealed owner JSON SHA256 is `500eea6a6a3d174f11ddf5c721e2819420956243b38a12061159deefec4fcd98`.

The subsequent debugger session 23879 (`22ea63` → `cda736`) exited debugger 0 with inferior SIGSEGV during PhysX tensor/module finalization. The sealed owner JSON records the remaining native tensor owner as unidentified. This revision therefore treats the scene backend as a source-derived candidate owner, not a runtime-proven cause or a solved crash.

Installed pinned `PhysxSceneDataBackend` stores `_simulation_view` and optional `_rigid_body_view` (`physx_manager.py:158–171`). Warmup assigns the manager view to that backend (line 806). Manager `close` and `_invalidate_views` clear manager views without clearing these backend aliases (425–449, 813 onward). Public `PhysxManager.get_scene_data_backend` (347–350) returns the existing backend. Public `backend.simulation_view = None` uses the pinned setter, which clears both aliases.

## Exact added teardown boundary

The existing v7 `before_clear` manager normalization is retained. After the original strict cleared-context/Base/Lab guard, the helper verifies the canonical backend class, actual bound getter, property getter/setter and constructor origins against the existing pinned Lab file. It also verifies already loaded `omni.physics.tensors.api` (382927 bytes, SHA256 `b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903`), its two exact wrapper classes and public SimulationView validity-property origin. There are no additional SDK imports.

The helper reads only existing backend views. SimulationView must actually be present with the exact pinned class; missing views fail with recorded presence metadata and no invented successful count. RigidBodyView may be absent. Existing wrapper/native-backend type names are recorded; SimulationView's actual public boolean validity is recorded without requiring either value. The installed RigidBodyView has no public `is_valid` property, so its validity is explicitly unknown. The lazy `get_rigid_body_view` method is never called.

One or two weakrefs describe the wrappers actually present. The sole new SDK mutation is the **public** `backend.simulation_view = None`; both backend fields must then be None. No private backend field or SDK method is assigned. All helper strong aliases are cleared in `finally`; only the weakref tuple is returned after the unchanged Core/Timeline cleanup.

The original post-context GC is retained. Afterwards, the actual returned weakref tuple/count must be valid and every wrapper must be dead before App unload. New release/zero-live flags are required for the entry's passed status. These flags alone do not establish a real child exit 0; ROOT must observe the actual tool closure separately.

## Preserved scope and preparation status

The exact derivation contains ten literal changes. Nine original functions retain identical source bytes and ASTs, including the complete separated/overlap physics phases, force thresholds, observations, default targets and state capture. Changes are limited to terminal helper source/origin checks and backend release, retaining returned weakrefs around existing GC, and initialization/validation of the new metadata flags. The original Core, Timeline, manager-normalization and failure/cleanup paths are preserved.

The source construction, full inverse/AST comparison and stdlib compilation returned exit 0 (`7c6284`). Ruff `62f0b8` also returned exit 0. No candidate module, SDK, Torch, NN, model, dataset or array was imported/executed. No new physics, cleanup or process-success result exists for v8. Source remains mutable pending independent and ROOT review; runtime execution is owned by ROOT after sealing.
