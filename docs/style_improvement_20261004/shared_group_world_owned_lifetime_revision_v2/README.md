# SharedGroupWorld owned native lifetime revision v2

Prepared source only. Peer review, ROOT review/seal and actual SDK checks remain pending. The sealed parent is `shared_group_world_staging_v1/shared_group_world.py`, 35,097 bytes, SHA `b608cae6cd8249e2c66d841813b4291b8bd229ca7beae45af625c33daf28fceb`.

## Ownership and release

World holds SDK objects through `sim`, `scene`, its four `assets`, `adapter.assets`, and the normal-force `bundle`. The bundle factory's wrappers/binding have their own native aliases; they require that factory's explicit owned release API. Current World snapshots, limits, histories, plans, pending packets and curriculum payloads are owned Torch values. They are retained.

`release_owned_native(normal_release_callback)` accepts the caller's source-verified `release_normal_force_bundle` function. A present bundle is passed once to that callback. The returned receipt must contain exactly `native_weakrefs` (a tuple of plain weak references), `native_view_count` (a nonnegative integer matching both the captured original bundle view count and this World's existing actor count), and `module_owned_aliases_released=True`. The receipt is stored for idempotent repeated release. It does **not** mean all referenced native objects are dead. If no successful bundle exists, the callback is not called and no receipt is invented.

Whether release succeeds or raises, World detaches only its bundle/assets/adapter/assets/sim/scene ownership fields, then clears the release helper's local aliases. An invalid bundle, missing callback or invalid receipt remains an error. The method performs no garbage collection, simulator stop/close/step or private SDK cache mutation. The caller finishes callbacks/sampling, saves failure evidence, captures its owned context before release, invokes this method, collects while SDK extensions remain live and separately checks weak reference survival before context/App teardown.

## Constructor frames and failures

Both constructors add optional `normal_release_callback=None`; successful existing calls remain valid. New probe/train entries must pass the source-verified new factory release function. `create_after_app` retains a manually allocated partial World before calling its original initializer so an initializer failure can release that object. `__init__` initializes only owned reference fields before its unchanged original body. On failure each attempts owned release, preserves the original exception with bare raise, and adds cleanup error type/message as a Python exception note if disposal also failed. Original exception type, message and cause are not replaced.

On both success and failure, constructor finally blocks clear local SDK aliases (`sim`, `scene`, `world`/`self`, and the create helper's caller-App alias). Clearing fields and local aliases prevents those helper frames from retaining these owned references if an external caught exception caches a traceback. This is not a claim that arbitrary deeper SDK or the unchanged adapter constructor frames are empty; concrete surviving owners would need their own reviewed revision.

## Preserved behavior and explicit identity migration

All original configuration, source PINS, 120 Hz/4-substep clock, physical observations, PD conversion, reward/termination, group reset and control methods remain unchanged. The full original initializer statement body is retained inside the lifetime wrapper, and the create helper's validation/configuration/context/scene/reset statements are retained; its final constructor call now retains the partial object. Exact byte/AST inverse recovery to the sealed parent and unchanged method/function identities are recorded in `source_derivation.json`.

This new World has its own source identity. Original factory, collector, expert and probe pins still refer to the sealed parent and cannot accept it by pretending it has the old SHA. New revisions must explicitly pin this World. New probe additionally pins the new factory and passes its verified callback. New training must pin/import that new probe and migrated collector/expert; the existing cleanup training entry imports the original probe, so copying the cleanup probe alone would not change training's executed setup. The collector's three original World control method AST pins remain valid. Original sealed sources have not been changed.

Preparation only reads, parses and compiles source. No candidate import, SDK/NN execution, arrays, archives, weights, AMASS load or training occurs here. Native survival counts and successful process exit require ROOT's later actual SDK evidence.
