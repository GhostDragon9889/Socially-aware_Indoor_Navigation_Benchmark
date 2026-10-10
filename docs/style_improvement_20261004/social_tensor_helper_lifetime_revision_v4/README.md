# Contact helper lifetime revision v4 — smoke only

Source reviewed by ROOT and an independent peer; files sealed before SDK execution. Actual SDK process exit of zero remains unverified.

## Concrete failure evidence

The parent is the sealed smoke v3 (`4d5fdba1122fd9670b6275c0f0ea41a7e5c5791f9252a43bcab94537795af4c2`), not the diagnostic entry. ROOT's diagnostic run, session 66636 / launch ffdc5d / completion 5060cb, exited **139**. Read 00bb1b identified exactly one retained contact wrapper: index 7, directly referenced by the completed `_contact_views` frame at its return line 133. That object is the helper's final loop `view` alias. Clearing the returned list in the caller releases seven wrappers but cannot clear this separate helper-frame alias.

The earlier native GDB stack ended in physics tensor destruction while Python collected exception/traceback frames during finalization. The diagnostic establishes the owning frame; it does not identify which optional import cached the original exception, and this revision does not claim such an origin.

## Minimal lifetime change

Only `_contact_views` changes. It initializes partial-construction aliases, executes every original helper statement inside try, and marks successful return immediately before returning the original `(views, layouts, manifest)`. Its finally block preserves the returned list on success, clears the unreturned partial list in place on failure, and always sets the helper's `view`, `physics_view`, `sim`, and `assets` aliases to None. This prevents a completed cached helper frame from keeping its last contact wrapper or caller's asset/context owners alive.

Every original import, path construction, contact-factory call, body/filter gate, manifest field, and return expression remains unchanged modulo indentation. Caller disposal, weak-reference zero-live guard, GC timing, singleton cleanup, App.close, original error reporting, physics calculations and both 32-substep phases are identical to parent v3.

No arbitrary exception globals are cleared. No private SDK backend is modified. No global native reset, runtime referrer scan, fast shutdown or os._exit is added. Source preparation does not import the candidate, load data/weights, run SDK, construct policies or train.

## Scope

Partial native views stored in the helper's list are cleared before propagating helper failure. A deeper API frame may still retain its own objects during an active exception traceback; this revision does not prove all SDK failure paths safely unload. A report with physics or cleanup flags remains insufficient: ROOT must observe the real SDK child exit zero. World, probe and training sources remain unchanged until that gate passes.

`source_derivation.json` records the single exact function replacement, complete inverse parent byte/AST equality, and original helper-body AST equality after removing only the success assignment and lifetime wrapper.

## Review

Author inverse/AST/compile9f711a and Ruff9a6f4a pass; ROOT complete inverse/AST/compile and full helper/README review7930ae pass. Independent peer732f01 and Ruff5e32bc pass, Required=[]/Optional=[]. SDK evidence is recorded separately.
