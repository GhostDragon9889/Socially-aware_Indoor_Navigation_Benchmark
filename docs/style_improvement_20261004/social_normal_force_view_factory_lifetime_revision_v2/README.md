# Normal-force factory owned lifetime revision v2

Source preparation only. The sealed 11,810-byte parent (`517afb01605e…`) is
unchanged. No candidate function, Torch, SDK, native view, array, weight or
training workload has been imported or executed for this revision.

## Contract

`create_normal_force_bundle(sim, scene, assets, group_actor_rows)` returns the
same three-tuple of per-actor views, layouts and body names. Its path, body,
contact-report, manager ownership, collision configuration, clock and 1/120 s
force contracts remain. The original factory body is inside one `try`, with
two declared ownership changes: direct slot identity checks replace a native
asset generator, and the returned tuple identity is recorded before returning.
Success preserves the live bundle. Failure drops this module's unreturned
wrappers and binding fields; it does not roll back SDK creation side effects.

Binding construction/checks and wrapper getters also clear their helper-frame
aliases in `finally`. Named-root comparisons use a direct loop rather than a
generator frame that could retain the last native asset on an exception.
Successful getter return values are retained by the caller; the frozen sampler
still performs its original immediate detach/clone.

`release_normal_force_bundle(bundle)` is a one-shot operation for the **exact
tuple returned by this module**. It rejects subsets, reconstructed bundles,
foreign wrapper classes, duplicate views, mixed bindings and repeated release.
It validates every wrapper/layout and creates every weak reference before
changing ownership. The binding retains the actual returned tuple, and release
checks object identity directly; it does not rely on an integer object ID that
could be reused. This creates a small tuple/wrapper/binding ownership cycle,
which explicit release breaks by clearing the wrappers and binding._bundle.
It does not query validity, invalidate native views, step,
stop, clear a context, collect garbage or edit private SDK caches.

The returned dictionary contains:

- `native_weakrefs`: tuple of `weakref.ref` objects for the original contact views.
- `native_view_count`: the number of views submitted for release.
- `module_owned_aliases_released`: true only after all module fields are cleared.

Wrappers become inert: `native` and `binding` are None, and subsequent getters
raise. The binding drops sim, scene, assets, roots, manager view, stage, resolver
and ownership metadata. Weak references do **not** prove that native objects
are dead. The caller must drop its other owned references and inspect `ref()`
after its own GC while the SDK interface is still live, before context/App
teardown. A release error propagates; it is not a successful cleanup receipt.

If creation fails before returning any bundle, the caller must not pass None
to the release function. The factory has already attempted its local cleanup.
This does not prove cleanup of a partially constructed World or foreign SDK
exception frames.

## Required propagation into future sources

The source derivation records the current owner files and their byte identities.
They are inspected, **not modified** here. Integration still requires:

- World owns `bundle`, `assets`, `adapter`, `sim` and `scene`; the adapter also
  owns its own `assets` tuple. Clearing only World.assets leaves that second owner.
- Constructor failure can leave `world=None` in the entry while a cached
  `__init__`/`create_after_app` frame retains partial `self.bundle`, assets and
  adapter. This requires a new World constructor/release contract; a factory
  release on the successful-World path cannot solve it.
- Collector.world and expert._world keep World reachable. Entry and helper
  world/sim/scene/asset aliases must also be dropped, including failure paths.
- Probe/training cleanup must retain detached failure data first, release the
  owned bundle/World while the native interface is live, then perform the
  separately reviewed context/App cleanup. No physics retry or failure filtering.

This module cannot clear caller aliases, SDK helper frames or SDK class caches.
The weakref result permits a future entry to detect surviving native views;
there is no claim that this source alone eliminates the exit-139 crash.

## Source verification and pending identity migration

The derivation records three independently reversible layers: four lifetime
changes, four exact tuple-pointer replacements, and one World source-identity
literal. Reversing all three restores the complete parent bytes and AST. Three
source helpers remain byte/AST exact; reversing the two declared ownership
fragments restores the exact original factory body. dt and all source pins
except World remain unchanged. The final source-only inverse/compile check
`863595` and Ruff `2061d7` exited0. No runtime or mock tests have been run.

The separately recorded `PINS['world']` identity is the 39,943-byte new World
lifetime revision (`e76182d2be5a…`). Its release integration and constructor
failure cleanup require the caller to pass this verified public release
function. World review/freeze and actual integration remain separate from
this source revision. These files remain writable pending independent review
and ROOT freeze; no SDK lifecycle success is claimed.
