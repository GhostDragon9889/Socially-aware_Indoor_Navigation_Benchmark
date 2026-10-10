# Actual full Interact usable dataset QC

Passed the complete NumPy dataset and loader contract in an actual CPU process, exit0. [actual_QC.json](actual_QC.json) binds the real closed conversion owner, manifest, registered split, sealed original raw files and all1000 converted episode identities. This is usable input evidence, not training, style or physical quality evidence.

| Item | Actual result |
|---|---:|
| Scenes / independent episodes / files | 1000 / 1000 / 1001 |
| Train / validation / test scenes | 800 / 100 / 100 |
| Eligible interaction windows | 150439 / 19237 / 19692 |
| Retained zero-window scenes | 49 / 7 / 6 |
| Agents / agent-frames | 10608 / 1060800 |
| Per-episode time | 100 float64 points, t=0..9.9s, nominal dt=.1s |
| Actual radius range | .4.. .4m |
| Maximum real current neighbors | 19 |

Every episode has finite float32 five-state data, exact declared time grid, a truthful current-velocity heading proxy and matching `velocity_heading_valid`. Stationary heading0 is explicitly a proxy. Every t=0 XY/velocity and every per-agent radius matches its official raw initial state/properties. The raw support contains the full target interval. This QC did not rerun the converter or compare a second interpolation implementation.

True empty geometries share one geometry hash. There is no map or fabricated obstacle, and every scene has `map_condition_mask=False`. Unique original scene/raw/episode identities and the preassigned split were checked against the sealed source records. All1000 original raw SHA identities, readonly modes and nlink1 were rechecked; the auditor changed no raw or converted file modes.

Each split loaded its full eligible-window bank and produced a real B=2 NumPy batch: history[2,31,5], future[2,52,5], neighbors[2,19,31,5], raster[2,2,224,224] filled with.5, false map mask, and real current neighbors. Single-agent and isolated scenes remain present with zero eligible windows. The map-encoder BN/optimizer behavior belongs to the separately tested staged model adapter; no Torch/project model/weights/SDK/GPU or training was used in this whole-dataset QC.

The auditor's first attempt compared a root-seal record containing additional `mode`/`nlink` fields to a plain path/bytes/SHA record. It stopped before dataset scanning. [first_attempt_error.json](first_attempt_error.json) preserves that checker mistake; the final check compares identity fields and independently verifies actual modes/link counts. No dataset defect was found.

Root owns sealing the1001 converted files and any later model training. Mixed Maps/Interact sampling is still unregistered; the actual189368 total eligible Interact windows must not be described as a50:50 mixture with Maps.
