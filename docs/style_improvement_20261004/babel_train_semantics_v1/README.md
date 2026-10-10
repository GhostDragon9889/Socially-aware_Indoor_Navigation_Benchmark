# BABEL train metadata overlay for fixed AMASS v7

This independent stdlib CLI preserves **all 2848 v7 train rows in original order**, including their original IDs, source/parent names, frame/time ranges, kinematic classes and existing label evidence. It writes a separate semantic metadata overlay. It does not alter v7, its bank, production185, labels, sampling, training or defaults, and does not recover the unpublished author 2848 list.

## Inputs and matching

The CLI pins the existing ZIP `/project/EXP/SangforAndSaut/Assets/babel_v1-0_release.zip` SHA `863d79b78f2d0225df74140b94cd61e06e04217726ae4ba1d2f50bad0d902428` and v7 manifest SHA `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`.

Only these annotation members are decoded, with exact size/SHA and ZIP CRC checks:

| Member under `babel_v1.0_release/` | Bytes | SHA256 |
|---|---:|---|
| `train.json` | 15101932 | `5da8185d4374670682cd6e3391d8635e4297ecebf5fe19394d28c2ca08030c6d` |
| `extra_train.json` | 7302453 | `1643abb3e8313f4828d706373563e82db4a713f44bba1103df80f28f7ecc5f30` |

It uses only the v7 **train** section and its 1853 conversion JSON sidecars. It does not decode BABEL val/test labels or load AMASS validation/test outputs, raw NPZ motions, motion banks, models, Torch, NN, SDK or GPU. The opaque whole-ZIP and manifest hashes identify the existing containers; they do not imply other splits were used. The incomplete local `babel_dense_and_extra_feats_labels.tar.gz` is not an input. Local ZIP identity is reproducible; no claim of a newly authenticated remote download is made.

Exactly one BABEL packaging prefix is removed from `feat_p`, e.g. `KIT/KIT/572/a.npz` becomes `KIT/572/a.npz`. That string must equal the original `conversion_sidecar.source` relative to `Assets/motion/raw/`. The output `.motion` basename is never used as an alias. Case, spaces, hyphens and underscores are not normalized. Ambiguous normalized paths within one member are rejected. Unknown parents remain in the overlay without invented annotations.

## Annotation and time contract

`parents` retains each complete original conversion record and each matched BABEL record, separately by train member. Singular `seq_ann/frame_ann` and plural `seq_anns/frame_anns` are preserved; every family/annotator label retains SID, record key, babel_lid, anntr_id, segment ID, original raw/proc/act_cat and start/end times. Raw records also preserve fields not interpreted by this CLI. Null/absent/empty/unknown structures and empty fields are reported. No category mapping, majority vote, synonym folding, replacement of style5 classes or inferred labels occurs.

The absolute BABEL time is `source_frame_slice[0] / source_fps + v7.parent_start/end_time_s`. The sidecar records the original raw FPS and slice. The existing conversion producer rounds FPS before choosing slice frame indices; both its slicing FPS and rounding delta are exposed in each alignment record. The earlier matched-parent inventory had only integral100/120/60/250Hz, with all raw slice starts0, so this convention has no offset ambiguity for those matches. It does not verify original pose/sample clocks. Processed 30/50Hz parent timing is checked against sidecar output FPS. The original half-open control/source frame ranges remain untouched. The coverage range ends at the **last sampled pose**, not the half-open stop frame.

Reversed, zero-length, negative, nonfinite, missing-time or missing-text frame labels stay visible but are excluded from timed evidence. A frame end exceeding its declared BABEL duration by more than .05s is retained as a consistency diagnostic and excluded, matching the earlier metadata join rule; this is not a verification of raw poses or annotation clocks. Missing act_cat alone does not erase nonempty human text. Sequence labels are explicitly not time assigned.

Valid frame intervals are sorted and unioned. Full coverage means **one gap-free merged interval contains the entire first-to-last sample time range** (endpoint comparison tolerance1e-8s); labels at just the two endpoints do not establish coverage. Coverage says nothing about one action occupying the full clip. Same exact intervals with different label sets and overlapping different label sets are retained as possible concurrent actions or annotator disagreement, not declared corruption and not resolved by voting.

## Output and execution status

The JSON contains source identities, summary counts, 1853 parent records and 2848 `train` rows. Each `original_v7_train_row` is the original row unchanged, with an explicit parent index and additional evidence/time links. Sources are checked again before publication. Output must be a new file inside this independent staging directory; exclusive creation prevents replacing an existing overlay.

The previous ad hoc metadata observation (`e374cd`, exit0/.650s; `813cdf`, exit0/.146s) found dense856 parents/1290 clips, extra1091/1644, union1093/1646 and844 complete interval-union-covered clips. Its command existed only as a tool cell, not as a saved CLI source. The new CLI was subsequently whole-source reviewed, independently reviewed with Required=[], and source/test frozen in actual tool `baff23`. Root ran it once: launch `eca9b5`, session68758, actual completion `a88240`, exit0. The CLI did not record elapsed execution time; no duration is inferred from polling.

Actual output [actual_train_semantic_overlay.json](actual_train_semantic_overlay.json): 37,531,278 bytes, SHA256 `09e16f40f6cefa283c6366589577ad0af1b2aa8123dc032af3f65893ab7889b1`. Whole-output check `296829` verifies all2848 original rows and ordering against v7,1853 parent mappings and all sequence/timed label references. Output, stdout and [minimal actual tool receipt](actual_CPU_tool_receipt.json) were sealed readonly/nlink1 in `06d312`. The actual output agrees with the historical counts:1093 matched parents,1646 matched clips,760 unmatched parents and1202 unmatched clips. Only844 clips have positive timed overlap; all844 also have complete continuous first-to-last sample coverage. The other802 matched clips do not gain timed coverage from their sequence-level labels.

One reversed frame interval is retained and excluded from valid timed union. Three parents have same-exact-span distinct label sets;231 parents contain1723 overlapping distinct label pairs. These are retained without choosing an annotator or asserting a single action. Missing sequence act_cat is preserved in51 dense and408 extra labels. This output is evidence for a future semantics-aware training experiment; no sampling, labels, model or physical behavior has been changed.

Before that historical audit was narrowed to train-only, the initial candidate scan also parsed BABEL val/extra_val/test JSON metadata. Those candidates were not used in the final train-only join. AMASS heldout outputs were never read. The new CLI has only the two explicit train-member read calls; it does not repeat that initial wider scan.

Executed command (exclusive creation rejects reuse of the existing output):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python docs/style_improvement_20261004/babel_train_semantics_v1/join_train_semantics.py --output docs/style_improvement_20261004/babel_train_semantics_v1/actual_train_semantic_overlay.json
```

Three small synthetic metadata tests check exact paths, nonzero raw-slice time origin, gap-free union, multi-annotator/null preservation and reversed intervals. They do not execute the CLI or inspect actual annotation/motion files.

Actual synthetic check: `e5037b`, exit0, 3 tests passed (unittest reported0.000s). Ruff: `d26416`, exit0, all checks passed. Only the application summary's distinction counters and documentation were added afterward; the tested path/interval/label functions and test file were unchanged before the unique actual main execution above. No synthetic tests were rerun without a remaining defect.
