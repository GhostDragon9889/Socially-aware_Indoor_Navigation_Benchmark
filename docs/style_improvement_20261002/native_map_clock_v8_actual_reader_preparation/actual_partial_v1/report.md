# Actual V8 four-scene partial verification

Actual producer `20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a`; folder `output/scene_maps/native_reference_four_scenes_session_clock_v8`. Original artifacts are retained.

| Scene | Actual resource status | Original source RAM at query | Admission |
|---|---|---|---|
| Office | Failed | Changed | Rejected |
| Warehouse | Failed | Changed | Rejected |
| Hospital | Failed | Changed | Rejected |
| Full_Warehouse | Passed | Unchanged | Complete CPU gate passed |

`root_RAM_and_source_layers.json` independently matched the initial RAM hashes to the original USD canonical layer text. Office/Hospital query RAM matches exactly the original text with start=0/end=60; Warehouse matches exactly with end=60; Full_Warehouse remains unchanged at 0..100. All referenced source layer file hashes remain unchanged. This identifies the exact RAM metadata difference; it does not establish which asynchronous SDK operation wrote it.

The failed scenes had session edit-target=true, composed range 0..60 and timeline range 0..1. Their initial query-clock artifact was not saved because the V8 RAM guard failed earlier: no actual current-timeline value is inferred from the range. Full_Warehouse has actual submitted counter 2 and frozen timeline 0.

`Full_Warehouse_complete_admission.json` recomputes all 155,300 native known5 cells, full floor/height CSR, 12 low-obstacle cells, world crop, unknown preservation, exact source hashes and actual clock. The new N4 scene configuration in `configurations/` passed CPU initial/wrap A*, .8 m clearance, ≥3 m spawn-clearance and ≥1.6 m root separation. It is prepared configuration, not a physical navigation result. No configuration was generated for failed scenes.

V9 addresses layer ownership with an anonymous runtime root. This partial V8 report does not substitute for fresh V9 native execution.
