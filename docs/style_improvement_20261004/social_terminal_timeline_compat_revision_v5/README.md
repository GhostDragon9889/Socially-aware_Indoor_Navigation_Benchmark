# Terminal Timeline cache compatibility v5 — smoke only

Source reviewed by ROOT and an independent peer; files sealed before SDK execution. This is a narrowly scoped **private SDK runtime compatibility workaround**, not a public API. It changes no SDK files on disk.

## Concrete remaining crash

The sealed parent is helper-lifetime smoke v4 (`32ee5b234852be447762503a20c61895a7870efa8a968d1c55dd403300226597`). ROOT's actual session 92584 still exited 139 even though all eight contact weak references were dead. The second native GDB run (1539 / 00b7b2 → 17b581; stack read 8245b6) showed SIGSEGV in `omni.timeline._timeline` during native deallocation from class dictionary cleanup and Python finalization. A debugger exit of zero is not a successful inferior exit.

Installed Lab PhysxManager caches a native Timeline on its class at line 248 and never clears it in close. Installed Core SimulationManager also retains its class Timeline (line 57); its own shutdown clears callback subscriptions and physics state but leaves that Timeline cache. The global `omni.timeline.get_timeline_interface.timeline` is a separate ITimeline cache; it is not changed by this revision.

## Exact compatibility scope

After successful owned context clear, the new helper validates exact installed Lab/Core/base/context/extension whole-file SHA, imported class origins and bound-method source file/first line. It requires the entry's exact cleared SimulationContext and PhysxManager, empty Lab callback/subscription tables and invalidated manager views. It records only Timeline class/module names and whether Lab/Core hold the same Python object; native Timeline objects are never stored in helper locals or report values.

The helper calls Core `_shutdown()` once while its extension interface is still alive. It requires every default callback flag false, all relevant subscriptions None, empty callback and physics-scene tables, and no Core native simulation views. It then sets **only** Lab PhysxManager._timeline and Core SimulationManager._timeline to None. The original final GC and App.close follow.

Core Extension.on_shutdown will call `_shutdown()` again during App.close, then release the native simulation-manager interface. Installed false callback branches do not access Timeline; `_reset` operates on the still-live interface and clears remaining callback/physics state. Its native reset documentation describes clearing callbacks and scene lists. This supports the repeated-shutdown design from source; real runtime confirmation remains required. The interface pointer is deliberately retained until the extension releases it.

This workaround is terminal to this fresh, single-App process. Creating another App/context or enabling Core timeline callbacks afterward is unsupported. No event dispatcher, other physics interface, getter cache, native backend, class source file, failure exit code, fast_shutdown setting, physics calculation, 64-substep budget, collision gate or existing GC call is changed. Original errors are preserved; new cleanup flags are required for report status. SDK child exit zero must still be observed independently.

`source_derivation.json` contains exact literal replacements and complete inverse parent byte/AST equality, plus whole-file and bound-method identities. No SDK/NN/candidate import, data or weight load, physics execution or training occurs in source preparation. World, normal factory, probe and training files remain unchanged.

## Source review

Author inverse/compile803bbd and Ruff6b90db pass. ROOT fullhelper/README read5dc19c/fcf3f8, complete inverse/AST/compile9ee634, and primary Core/extension sourcea5d043/ec93ac pass. Independent aa71b4 confirms full inverse/function bytes/AST/SDKpins;3c4021 checks compiledmethod/decorator identities, Required=[]/Optional=[]. ROOT initially misread adjacent source outputs; single-file ec93ac confirmed Extension.on_shutdown is an ordinary instance method. Original False predicate and entry bytes remain unchanged. No SDK result is claimed in this source record.
