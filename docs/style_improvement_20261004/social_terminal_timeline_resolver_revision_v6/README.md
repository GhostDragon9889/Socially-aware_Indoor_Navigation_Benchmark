# Terminal Timeline owned-manager resolver correction v6 — smoke only

Prepared source only; independent peer review, ROOT review/seal and actual SDK execution remain pending. This keeps v5's private, process-terminal Timeline cache workaround and changes only its owned-manager identity validation.

## Concrete v9 failure

ROOT's v9 executed the sealed v5 entry (`0f65e25e138a9d8d1e13ba84e58a960bef602fb8050272844f6a8b097da6103e`) and reported exit 139. All eight contact weak references were dead and owned `clear_instance` returned. The compatibility helper rejected its combined context/manager identity guard before Core shutdown or either Timeline cache release. The v9 report did not separate those three booleans; it is not evidence that an alternative SDK manager class was actually imported.

The installed `SimulationContext` assigns `self._physics.class_type` directly to `self.physics_manager` at line 174. Installed `PhysxCfg` declares `{DIR}.physx_manager:PhysxManager` at line 42. `configclass` expands `{DIR}` from the field's declaring package and wraps the canonical string in `ResolvableString`. Its attribute access delegates through `_resolve()` to `string_to_callable`, which uses `importlib.import_module`; initialized manager methods therefore operate on the canonical imported class while `sim.physics_manager` remains a string proxy. Comparing that proxy itself with the class using `is` cannot establish its target identity.

## Exact correction

The new helper keeps every v5 Lab/Core/base/context/extension whole-source pin and method-origin check. It adds exact whole-source pins for installed `string.py`, `configclass.py` and `physx_manager_cfg.py`, plus resolver/configclass function origins and exact resolver/config class module/qualname checks. It requires the exact cleared SimulationContext, its exact PhysxCfg and that config's same `class_type` proxy. The manager must be an exact `ResolvableString` containing only `isaaclab_physx.physics.physx_manager:PhysxManager`. It resolves only this owned proxy and requires the result to be the very same pinned canonical Lab PhysxManager class. No unknown SDK globals are enumerated or imported.

The report separates exact context type, cleared singleton and raw proxy/class identity, and adds the exact proxy/type/reference and resolved canonical identity. It records only booleans and source/module/qualname strings, never SDK object values or native handles. A cached traceback cannot keep the new proxy/resolved class aliases through this helper: both are initialized before construction and cleared in its existing finally block.

After that strict corrected identity guard, the existing behavior is byte-identical: callback/subscription/view checks, explicit Core `_shutdown()`, release of only Lab/Core `_timeline`, required cleanup success flags, original GC calls and App.close. This remains a **private SDK runtime compatibility workaround**, pinned to installed source and terminal to this fresh single-App process. Creating another App/context afterward is unsupported. Core's interface pointer remains untouched for its extension's repeated shutdown and release. World, normal-force factory, probe and training sources are unchanged.

`source_derivation.json` records three exact replacements and complete byte/AST inverse recovery to sealed v5, unchanged original functions, installed source identities and the actual v9 failure report identity. Preparation only reads, parses and compiles source; no SDK/NN/candidate import, archive/model loading or training is performed. Exit 139 elimination and successful repeated shutdown are not claimed.
