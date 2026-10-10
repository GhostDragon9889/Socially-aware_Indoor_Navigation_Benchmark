# Available intermediate checkpoint bytes

This independent standard-library observer writes only its separate history
directory. It does not alter training, deserialize tensors, execute pickle
GLOBAL/REDUCE, load a model, or start a GPU/SDK. The preparation checks below are historical. Root has now saved the actual
15000 checkpoint and started the independent CPU watcher; tensors and model
effects remain unvalidated.

The registered local header is the root dictionary from
`orca_official_mixed_training_staging_v2/training_state.py:checkpoint_payload`:
exact integer `step == optimizer_updates`, `microsteps == 2*step`,
`training_windows == 400*step`, and `microsteps_in_current_update == 0`.
The schema and raw/EMA checkpoint kind must also match. Nested Adam `step`
values and memoized root keys are handled by a small passive opcode stack;
tensor/storage contents remain opaque and unvalidated.

Only observed complete 15000, 16000, ..., 39000 headers are archived. An
independent temporary byte copy determines the filename, with SHA/size checks
against the opened source before/after copying and the current source path.
Atomic source replacement discards only this observer's temporary and retries.
Each readonly single-link `observed_step_N.ckpt` has an exclusive JSON record.
An existing step with different bytes fails; existing evidence is not replaced.
The formal output directory is always prohibited as a destination. The append
journal refuses symlinks and multiple hard links.

Unchanged published generations are skipped using dev/inode/size/mtime/ctime
only as an I/O cache, **not** as a new SHA or tensor validation claim. A changed
generation receives the complete byte checks. Polling defaults to 15 seconds.
Unavailable `/proc/PID` is recorded as a visibility limitation and never stops
observation or proves exit. Seeing header 40000 stops without saving it and
records missing history steps; it never establishes formal training completion.
Deadline records missing steps and returns 2 if any are missing. Root's real
training tool and terminal auditor remain the authority for the full 40000 run.

## Root commands and actual execution

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/nonterminal_checkpoint_archive_v1/archive_available.py --once

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/nonterminal_checkpoint_archive_v1/archive_available.py \
  --parent-pid 873975 --poll-seconds 15 --deadline-seconds 43200
```

Default source is the formal all-temporal `last.ckpt`; default destination is
`output/diffusion_amp/official_ORCA_all_temporal_available_history_v1`, separate
from the earlier observed 14000 copy. A single observation of 14000 skips it.
History is local diagnostic evidence: it cannot recover overwritten 9000/10000,
does not select a best checkpoint or use test results, and does not change the
fixed 40000 main comparison. For context, fixed author TRACE commit 594b2af's
[scripts/train.py:81](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/scripts/train.py#L81) retains top-5 validation EMA diffusion-loss checkpoints
every 3000 and additional 10000 snapshots; [README:59](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/README.md#L59) names evaluation 40000. Root checked these local primary-source files against the fixed Git blobs (1a2da1); web fixed-blob fetches were cache misses.
This observer is a local supplement, not a reconstruction of those checkpoints.


## Actual root execution

[Actual owner](actual_root_owner.json): single-copy session22948, launch39a3b4 → completiona8a3ff, exit0. The independent readonly/nlink1 15000 checkpoint has 372,121,656 bytes and SHA256 `d29a0968e9fdd4629f6f459cd440bbd212f252a804c28b7c7c12d91eb16d3f3a`; source-fd before/after, current path and copy agree. Passive scalar header reads 15000 optimizer updates, 30000 microsteps, 6000000 windows and zero incomplete microsteps. This is byte/header admission, not tensor, resume or quality validation.

Watcher session39421, launch7c41b6, latest genuine pollc48643 still running. Startup recognized the identical archived 15000 copy; actual stdout readbf0ed5 also confirms the unchanged-generation skip. Actual log readc942a0 confirms the watcher automatically copied the newly published 16000 generation and then skipped its unchanged generation. The 16000 copy is readonly/nlink1, 372,121,656 bytes, SHA256 `e982396bd06a87ee0036ca41d78d3137dba3fcc0ac4bf2d866cb171bf68c7a2e`; all opened/path/copy hashes agree. Log readac775e additionally confirms the 17000 generation: readonly/nlink1, 372,121,656 bytes, SHA256 `d79cc68974fe61d714201a6b3b33eb8a21aee77030bac71a6c3f6e5b79e76d22`, followed by its unchanged-generation skip. Log readfe2fa3 confirms the 18000 generation: readonly/nlink1, 372,121,656 bytes, SHA256 `c60f2db6c9d81a18ecfa0bcc9315eeff16216966da6b969452d722e1ba6f24b3`, followed by its unchanged-generation skip. Log read80ce07 confirms the 19000 generation: readonly/nlink1, 372,121,656 bytes, SHA256 `7e297b8b0535bb68cf79570eab71ad58601e3bc561a7fdedd2ec1443aacd4f70`, followed by its unchanged-generation skip. Actual log read84408a additionally confirms the20000/21000 generations, each readonly/nlink1 and372,121,656B: SHA256 `ce056ccf2763ab00d15a202c88f3cc0918cc8f907c06df2ec7b0c3a35263891e` and `d3d448ff7b88c563c0dd8d96cb14eb125b9052b9c86243807353e102cd577d97`. Their headers are20k/21k updates,40k/42k microsteps,8.0M/8.4M exposures. Actual immutable-record read1af260 confirms22k and23k copies, each372,121,656B/readonly/nlink1, SHA256 `9a656cd57f82f8122479ea6a8caee7ffdac6fe982f0f346a804dd8c1ed971b25` and `835e71406d9bf3a0d336e48a9a1bd6e66b36b1130e53d3b06db1ec827b28081c`. Headers show44k/46k microsteps and8.8M/9.2M exposures. These hashes come from the frozen watcher records, not a new root tensor load or checkpoint rehash. Actual immutable-record read3d7001 additionally confirms24k: readonly/nlink1,372,121,656B, SHA256 `f95ee386efc6238399ffb97dcb1de13e2289f4a2a98467390c6d6a760207b356`, header48k microsteps/9.6M exposures. No root tensor load or new checkpoint rehash. Actual immutable-record readc1ab61 additionally confirms25k:372,121,656B, nlink1, SHA256 `10f7bee2c36e5df882bc9b0155d1e99b9820e88af8ee177d5f8854952ea07e7e`, header50k microsteps/10M exposures. This is a recorded watcher SHA, not a new root checkpoint rehash/tensor load. Actual immutable-record read006afe additionally confirms26k:372,121,656B, nlink1, SHA256 `bddd36af143e0b1d8cc0f0df5c81248dfccc42b0ad4ba987eb1d1b46a5c9a8ac`, header52k microsteps/10.4M exposures. This is a watcher-recorded SHA, not a new root checkpoint rehash or tensor load. Actual immutable-record read733a8e additionally confirms27k:372,121,656B, nlink1, SHA256 `4ed0f79debf2f72231f6b0ccac08b0605815a72dbd322868e53ee031c0abd1fb`, header54k microsteps/10.8M exposures. This is a watcher-recorded SHA, not a new root checkpoint rehash or tensor load. Actual immutable-record readcb7c0f additionally confirms28k:372,121,656B, nlink1, SHA256 `7994512506be440a54daa2dd70060a94ada13d371e1b48c912b10293c6b00afa`, header56k microsteps/11.2M exposures. This is a watcher-recorded SHA, not a new root checkpoint rehash or tensor load. Later29k…35k observations are recorded below; future36k…39k copies are not yet claimed. The watcher only writes the separate history directory and stdout log; it never starts GPU work or establishes formal completion. Its journal/stdout remain mutable while running.

Formal training session50502 was independently polled 4a95bd and remains running. `/proc/873975` is invisible in this exec sandbox; that observation does not prove process exit. Root rehashed all185 original sources again (78d479), unchanged. Six synthetic tests, installed Ruff and the final independent source review passed; the script/tests and synthetic check record are now readonly.

The fixed 40000 main comparison remains unchanged. Any later history selection must be labelled exploratory, fix candidate inventory, EMA/noise seeds and validation-only ranking before model execution, and preserve Maps/Interact and tail-batch sample weights. Terminal test results and the 34 held-out AMASS clips are not selection inputs.

## Bounded CPU checks

Six small synthetic ZIP/pickle unittest cases passed (tool `57540f`, exit 0,
0.039 seconds). They cover root-vs-Adam/memo/opaque callables, malformed headers,
source replacement and retry, conflicting existing bytes, 14000/40000 plus
unavailable PID and partial deadline, and generation-cache behavior. They also
check protected destination aliases and journal symlink/hard-link refusal.
These are byte/archive interface tests, not real model validation.

Source compile passed (`d697c3`). Initial SDK Python Ruff invocation lacked the
Ruff package (`f9e02c`, exit 1); root then ran the installed Ruff binary once
successfully (`0727cb`, exit 0). No package was installed or test suite repeated.

Root observed immutable29k archive metadata in actual tool `c29422`: watcher copy at `2026-10-04T00:35:24.133254+00:00`,372121656 bytes, SHA `5f2c6f1ea8c7e1b34143897ed0d0aad7945442c50f3b218cd7913a958e27bf50`, ordinary single-link copy. Header29k/58k microsteps/11.6M exposures. Root read metadata only; no tensor load or model validation. Training50502 remained open in6ba233, watcher39421 remained open in35b384 at00:34:15UTC; these observations preceded the00:37:45UTC log read279690 and do not assert a terminal exit.

Root actual read92cc8e observed the immutable30k metadata: copied at2026-10-04T00:53:57.281906+00:00,372121656 bytes, SHA `4344c9e5a4f9ec3a465f3c07d611d16d6a9803904e3c755489758c5772419cb8`, single-link stable copy; header30k/60k/12M/currentmicro0. Root did not tensor-load/rehash this checkpoint. At00:58:42UTC training3f27df/50502 and watcher8efbb0/39421 were still open; this is not terminal evidence.

Root actual readb3d056 observed the immutable31k metadata: copied at2026-10-04T01:12:30.402365+00:00,372121656 bytes, SHA `2b8391d29c052dc51ade13923549069ef29680437a49015cc45f9b6d8c98cf01`, single-link stable copy; header31k/62k/12.4M/currentmicro0. Root did not tensor-load/rehash this checkpoint. At01:16:55UTC training6a798f/50502 remains open; this is not terminal evidence.

Root actual immutable metadata reads672d2e andda718f confirm32k and33k independent372121656B copies: SHA `a7ed0b3544c6cdbdb4a9a882d8ae9d16d1fffa6334a87e5ecd4c9d4381494e53` and `b5e9c5767ca62054399fe75f4c44c5174874f16bdc2e5216a49b14ce0ca0bd2a`, watcher times01:31:03.540386UTC and01:49:36.699942UTC. Headers32k/64k/12.8M and33k/66k/13.2M, currentmicro0; no root tensor load or checkpoint rehash. Actual50502 pollc85b00 remains live at01:58:28UTC, not terminal evidence.

Root actual metadata readd3c27d confirms34k:372121656B, single-link copy SHA `c450c7cd06c652db25c73e9a572c8912a786c798b46f4bf8fdb2532edb8ca975`, copied02:08:09.849058UTC, header34k/68k/13.6M/currentmicro0. No root checkpoint rehash or tensor load. Actual50502 pollc58902 remains live at02:08:36UTC; no terminal claim.

Root actual immutable metadata read1d7369 confirms35k:372121656B, single-link copy SHA `df797d5947d47aff6bdf9c9b3cbfe7243f7282ca01616af3c04738ac16e818dd`, copied02:25:43.023987UTC, header35k/70k/14M/currentmicro0. No root checkpoint rehash or tensor load. Actual50502 poll2ae7ec remains live with the35k logf1d8c1 at02:27:20UTC; no terminal claim.

Root actual immutable metadata read86089c confirms36k:372121656B, single-link copy SHA `b245388ced5c53a09eb377d87ca62323f8f721b1bc01ffd256447524c8bc18ce`, copied02:43:16.181472UTC, header36k/72k/14.4M/currentmicro0. No root checkpoint rehash or tensor load. Actual50502 poll63ebb1 remains live with36k log17c385 at02:50:52UTC; watcher39421 also remains open in1874e1. No terminal claim.

Root actual immutable metadata read732980 confirms37k:372121656B, single-link copy SHA `c10d7a6595aad97fe9fa92f30f26ddd7a88da2051317abb65e890c6edc9e1cab`, copied03:01:49.308072UTC, header37k/74k/14.8M/currentmicro0. No root checkpoint rehash or tensor load. Actual50502 polla644bc remains live with37k log41d42f at03:07:50UTC; no terminal claim.

Root actual immutable metadata read93d670 confirms38k:372121656B, single-link copy SHA `afa9e895972509c2c9fd2cc7258b905803747a9d6730cd44496dfebe98227d11`, copied03:20:22.477572UTC, header38k/76k/15.2M/currentmicro0. No root checkpoint rehash or tensor load. Actual50502 pollce90db remains live at03:22:14UTC; watcher39421 pollba07a5 remains live at03:23:50UTC. Not terminal evidence.

Root actual immutable metadata readf9560c confirms39k:372121656B, single-link copy SHA `fbc1fe2c46a3a7ccb8eeb0af03951862d53106141c5110c1f1ab4bd71b8bbef5`, copied03:39:25.719743UTC, header39k/78k/15.6M/currentmicro0. No root checkpoint rehash or tensor load. Actual50502 poll7b2efa remains live with39k log910173 at03:39:45UTC; watcher39421 remains open in79b867. No terminal claim.

Watcher39421 actually closed0 in bc5843. Final immutable journal observation at03:58:10.803308UTC has header40000, missing_steps=[], without archiving40000 or claiming formal terminal. Root separately observed formal50502/f10f35 closed0 at03:59:40UTC. Actual metadatae64deb and all25 frozen JSON records confirm15k..39k copies; owner and observer logs are now readonly. No checkpoint rehash/tensor load was added for this closure.
