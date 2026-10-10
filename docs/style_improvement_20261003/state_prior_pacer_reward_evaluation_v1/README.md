# PACER reward terminal evaluation preparation

This directory prepares an evaluation of the isolated **935v2** reward trial.
It does not admit a future checkpoint or claim a quality improvement.
Production, historical registrations and previous results are unchanged.

## Fixed comparison

- Tracking uses all **22 validation motions / 11,982 metric controls**. Failed
  sequences remain in the complete success denominator; there is no clip limit,
  alignment correction or fall-based frame filtering.
- The original shared-obstacle navigation fixture uses **four actors / 6,000
  controls / 200 seconds**, with identical guidance, contact measurements,
  physical fall definition and deterministic seed.
- Primary style remains the original **90D pose/velocity MMD²**, frozen
  `c274348f` AMASS reference, sigma 1, original speed bins, minimum 30 and
  maximum 256 samples. Failed, reset and idle frames remain included.
  Unsupported bins remain unknown.
- Secondary pose diversity uses the saved AMASS-only **b25dffa6** 45D basis
  and its 50/100 centers. No refit or replacement of the primary metric occurs.

The inherited execution and quality criteria remain separate. Full22 tracking
and fixed-basis diversity reports are required. Navigation retains its minimum
of zero falls, at least 21 route vertices and inclusive primary MMD² at most
0.07238060315284697. These criteria do not automatically promote a default or
establish equivalence to the author's unpublished eight-scene protocol.

## Source and checkpoint admission

The bundle is `mixed_state_action_terrain_state_prior_pacer_reward_v2`, source
manifest `1fd9b5e1…`, with exactly **935** immutable regular source files.
The absolute reward profile has SHA `fa95b419…`. The strict original launcher,
full-network adapter and evaluator mathematics are retained.

`prepare_registration.py` first writes `registration_draft.json` after hashing
all owned sources and calling the actual SDK-free parsers from that bundle.
The draft explicitly lacks an independent reader freeze. Finalization requires
the actual new independent reader/helper source freeze and read-only files;
it then writes `registration.json`, binds its full SHA in the executor and
writes exact future commands. No candidate checkpoint is read during either
preparation step.

Actual evaluation requires the exclusive sealed `step_2000.pt`, its full SHA
and the new reader's actual `terminal2000/audit.json`. Admission retains all
old full-payload, D2019, coupled public state-prior/history-RMS, terrain policy,
2,000-update/16,384,000-transition and actual-import checks. It additionally
requires the new audit schema, source count 935 and exact typed reward fields:
scale 2.0, AMP weight 0.5, task weight 0.5, `backwards_penalty`, epsilon 0.0001.

Tracking and navigation output to `pacer_reward_tracking2000_v1` and
`pacer_reward_nav2000_v1`. Their matching import reports belong to the new
bundle. Root executes the GPU commands serially after training closes and the
terminal payload is admitted. The SHA placeholder is deliberately invalid.

`evaluation_evidence.py` checks actual complete captures, arguments, committed
owned-import events, all 935 source hashes and the unchanged physical control
loop. The original helper mathematics and four-actor protocol remain intact.
Large shared banks receive link/stat checks here; the launcher independently
checks resource payloads before simulation.

## CPU proof and independent review

The SDK-free parser ledger records actual new-bundle imports and finalization.
The small contract tests check the five reward fields, old/partial admissions,
complete argv, SDK-free provenance and unchanged command/metric ASTs. Fixture
data stays in memory and is never written as a real checkpoint or admission.
Preparation and tests never launch GPU, Kit or any simulation process.

The independent implementation peer passed 40 source/protocol checks and
20 necessary CPU tests. Its in-memory reward-guard mutation caused one failure
with 19 tests passing; production and the reviewed source were never mutated.
The first finalization failed closed because the new reader freeze calls its
file list `files`; only the two consumer references were corrected, and the
failure record is retained. Final reader binding now passes.

`source_freeze_v1.json` freezes this prepared source and the independent peer
snapshot. Prepared registration alone proves neither terminal training
completion nor gait quality; actual results remain pending.
