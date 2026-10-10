# Fixed actor mixed curriculum: CPU component acceptance

Status: component frozen; **30 focused CPU tests, 128 actual v7 checks and Ruff passed**. No GPU, Kit or physics steps were run. This validates command sources and checkpoint contracts; it does not establish improved gait or navigation.

## Implemented scope

New module: `CrowdSim/humanoid/training/mixed_trajectories.py`. New unit tests: `tests/test_mixed_trajectories.py`. World/runner/config integration belongs to the separate CrowdSim owner.

- Even actor count, minimum two. The first half uses AMASS and the second half uses the existing procedural curriculum. These are fixed global assignments; actor observations receive no branch ID.
- Both groups receive exactly half of actual control transitions. Reset and plan counts remain measured independently. At 256 actors × 32 rollout steps × 2,000 updates, each group receives 8,192,000 transitions if that run completes.
- Real resets use a shallow corpus copy sharing the actual read-only CPU bank, with private CPU seed `seed+1`. Procedural reset/expert sampling retains the original corpus generator; procedural plans use their own seed `seed`. The mixed experiment does **not** retain the old full-batch reset RNG sequence.
- Reset sampling requests ten newest-first history frames and 31 future frames for AMASS; procedural resets retain the existing one-future-frame requirement. The five returned values preserve arbitrary requested global actor order.
- AMASS plans retain actual 151 native 30 Hz source positions, fixed reset translations and explicit linear command tails. Source clip end truncates only AMASS actors. No live-root reanchor or physical teleport is performed.
- Procedural plans retain original 51 knots. Stored 151-point buffers are linear upsampling; task and reward queries evaluate the **original 51-knot formula**, including nongrid procedural times. The procedural recipe must be explicitly provided; the integration recipe is smooth speed 2 m/s and acceleration 1.5 m/s².
- Resume requires exact fixed assignment, source/config, child states, global procedural source/reset/episode clocks, original canonical plans, and private RNGs. The original corpus RNG is separately runner-owned. Child RNG tensors must be moved explicitly to CPU by a caller loading a checkpoint onto CUDA.
- Newly reset actors may retain unused stale plan buffers only with the declared forced-replan age. Active actors require matching plan clocks and source/plan data. Configurations returned in state/description are independent copies; editing an exported state cannot alter the live expected source contract.

## Evidence

- [Focused test log](tests.log), [JUnit](tests.xml): 30 passed.
- [Actual v7 proof](actual_v7.json), [reproduction script](verify_actual_v7.py), [execution](actual_v7.log): 128 passed checks over the real mmap bank, all 2,848 selected segments retained, 256 actors, shuffled global IDs, 15 actual completed control callbacks, asynchronous resets, and 1,920 transitions per group. Source/history/next-frame data, procedural formula, native source plans, private RNG continuation, original-stream expert draws and restored next plans match exactly where reported.
- [Ruff](ruff.log): passed for module, tests and proof script.
- [Snapshot metadata](snapshot.json): exact current source hashes and evidence scope.

The bank SHA is the frozen manifest declaration; this small CPU proof checked actual file size and manifest SHA without rehashing the complete bank. Existing resource integrity evidence remains separate. No sealed 34-test outputs were read. The module provides 2D commands and no collision, floor or physical stability guarantee. Actual GPU pilot and full-run acceptance remain separate gates.
