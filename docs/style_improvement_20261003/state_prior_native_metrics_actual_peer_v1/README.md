# Actual native CPU metric peer QC

This review reads saved evidence from the completed native4 and native15 runs. It does not run controllers, native physics, renderers, feature extraction, MMD calculations, or reference fitting.

- `actual_six_metric_QC_v1.json`: 1,742 checks of six completed native4/old4/old15 metric phases. These include exact source epochs 183/168/174, actual CPU import ledgers, full interval populations, all speed buckets, fixed c274 reference and b25 centers.
- `actual_native4_native15_comparison_QC_v1.json`: 1,243 additional checks binding the completed native15 JSON representation revision, closed CPU phases, and both saved comparisons.
- `actual_comparison_summary_v1.json`: concise numerical summary with supporting record identities.

Native4 primary90 support consists of four buckets on each side, but the bucket sets differ. Native15 support is five old buckets versus four new buckets. Common supported bucket means are secondary; original primary aggregates are retained.

Native4 K50/K100 coverage improves from 0.34/0.31 to 0.58/0.47, while route vertices/cycles decline from 57/12 to 44/8. Native15 K50 coverage declines from 0.76 to 0.70 and K100 stays 0.61; vertices/cycles decline from 211/49 to 160/32. Lower saved MMD values do not establish increased diversity, naturalness, or overall improvement.

The first QC attempt incorrectly required an old historical raw audit to have readonly filesystem mode. It stopped before emitting a report. The actual old contract pins historical bytes and archived source, without imposing a new chmod requirement. `execution_history_v1.json` retains that diagnosis; current QC requires readonly mode for newly sealed evidence and exact hashes for historical artifacts. Production files and capture permissions were not changed.

No default promotion, bitwise planner RNG match, single-factor causal gain, sealed34 evaluation, or subjective media visibility claim is made.
