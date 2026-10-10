# Existing scene backend view referrer diagnostic

This independent diagnostic derives from sealed v8 (49961 bytes, SHA256 `a9da29022b7f514ffb6d4034f152190cd1cbd302601542a1418fcc764c207ba3`). It changes no sealed source or SDK file. It is prepared source only and remains writable pending ROOT review.

ROOT's actual v13 returned the public backend setter with both caches None, but the one existing SimulationView wrapper remained alive after GC; its reported validity was False. The original live-view guard failed. All eight contact wrappers were dead, and the earlier Base/Core/Timeline gates passed. This diagnostic has not run and does not identify an actual remaining owner in advance.

## Restricted query

Only weakref targets already captured and validated by the exact v8 helper are queried. The existing nonzero-live failure branch records structural metadata, then raises the **same original RuntimeError**. All original teardown flags, source gates, physical functions and failure paths are unchanged.

Each owner record contains its type module and qualname. Frames add code function/source/current line; exact dictionaries add only names/counts of string keys whose value is the target; lists and tuples add their lengths. Values, tensors, object reprs, frame locals and arbitrary dictionary contents are never serialized.

There is one additional, nonrecursive hop only for exact installed `Articulation`/`ArticulationData` type names or a dictionary with a target-valued `_physics_sim_view` key. That hop records who refers to the already identified owner, using the same structural metadata. Each level records its total owner count and limits persisted owner records to 64, with omitted counts explicit. Matching string-key names are limited to 64, also with their full count and omitted count explicit. No global native-object scan or arbitrary owner-field traversal occurs.

The query itself temporarily retains targets and owner lists. These are diagnostic references, not independent evidence of production ownership; diagnostic frame records are labeled when identifiable. Both helpers clear all strong target/owner/candidate/key/list/frame-reference aliases in `finally`. Metadata contains only strings, integers, booleans and lists/dictionaries of those values. Inspector failures report their exception type only, preserving the caller's original failure. No owner field is cleared, no SDK is imported, no new GC is called, and no shutdown guard is relaxed.

## Source-derived candidates

Installed `isaaclab_physx/assets/articulation/articulation.py:3859` assigns `self._physics_sim_view`; line 3880 constructs ArticulationData, whose `articulation_data.py:87` independently saves the same manager getter result. The articulation's STOP callback (3976–3980) clears only `_root_view`; AssetBase's destructor (114–117) clears callbacks. These are explicit **candidate** aliases whose actual owners require the target diagnostic. They are not justification to mutate arbitrary asset fields.

The tensor API `create_simulation_view` (`api.py:12–71`) returns a Python wrapper and has no Python module-level SimulationView registry in this code. The existing v8 helper already verifies both Core simulation-view caches are None. Shared native views for a stage do not establish who retains the surviving Python wrapper.

The source derivation records two exact replacements: insert the two metadata functions before `_run`, and call the inspector in the existing failed weakref gate. Eleven original functions, including the entire backend release helper and `main`, remain identical in bytes and AST. Final full reverse bytes/AST and stdlib compilation passed (`5991e2`, exit 0); final Ruff `ffdb61` exited 0. Initial Ruff `429a3c` found an unused exception alias; the prepared source now reports exception type via `sys.exception()` without binding/retaining that exception. No runtime diagnostic was executed. The one-hop SDK type classifier also requires identity with its already loaded module class attribute. All checks were source only; no candidate/SDK/Torch/NN/array/data/weight operation was executed. Actual diagnostics, native release and process exit remain ROOT-owned future observations.
