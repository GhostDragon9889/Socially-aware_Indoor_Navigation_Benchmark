# Production source V12

Source-only freeze completed with actual CPU exit 0. The selected set contains **185** files: V11's 183, with two approved reward-resume guard changes, plus one explicit PACER reward-factor profile and its canonical regression test. The other **181** V11 files are byte-identical.

| Artifact | SHA256 |
|---|---|
| [185-file manifest](native_pedestrian_source_v12.json) | `9ddbfe994bc692b344917796251780a08931d8d9846a1a19ae7476ba9753535c` |
| [Deterministic archive](native_pedestrian_source_v12.tar.gz) | `9a3cd41a6534966ec6e1bc1fd5b976aa1a9feb290522345e6712d562f98352ef` |
| [Protected fingerprints](protected_fingerprints.json) | `04073be26bde19158461ed2d8aab965f65a75cc76a5264b4b7ba281e0dca4abd` |

The archive was independently constructed twice within the freeze process, produced identical bytes, and all 185 regular members were reread and matched to source bytes. Gzip/tar timestamps, ownership, and member modes are normalized. The manifest, archive, and protected fingerprints are OS readonly.

The freeze bound the independently reviewed five-field training-resume reward guard and root's actual SDK-free public profile loader receipt. The new profile explicitly selects outer AMP weight `.5` with internal scale `2`, yielding coefficient `1`. Fresh math/RNG and evaluation behavior are unchanged by the guard. The new recipe is opt-in; this source freeze does not establish a locomotion improvement.

All **934** original isolated-training files remain readonly and match their earlier hashes. All six shared input fingerprints, totaling **2,307,757,289 bytes**, match. Data were hashed as opaque bytes; sealed 34 test outputs were not accessed. The selected default checkpoint still matches SHA `62c4edc1…`; its existing mode is **0644**, so the report correctly records `selected_default_OS_readonly: false`. Neither its bytes nor its permissions were changed.

Two preparatory script mistakes were preserved: [wrong directory name](first_attempt_failure.json) and [incorrect readonly assumption](second_attempt_failure.json). Both attempts exited 1 before creating the manifest/archive. Their script copies remain available. The final freeze exited 0 and did not change production sources, historical V11 evidence, the SDK, default model, or old training sources.

The independently reviewed **935-file training bundle** is a separate provenance domain. Its final v2 source/profile and actual cold loader evidence are linked from [the training audit](../training_style_root_cause_v1/README.md); it is not interchangeable with this 185-file production archive.
