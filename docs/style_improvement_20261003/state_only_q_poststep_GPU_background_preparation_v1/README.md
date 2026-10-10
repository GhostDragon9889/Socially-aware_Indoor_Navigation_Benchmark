# Dynamic GPU background observation — prepared

Root reviewed `background_gpu.py` in tool80ce0a and the contract in0b2d96,
with no required finding. Only bounded stdlib parsing/compilation and Ruff were
executed: compile42863b exit0; final Ruff837f33 exit0. The initial SDK-venv Ruff
path failed with exit127 and remains recorded. Neither the helper nor a live or
synthetic GPU observation was executed during preparation.

`validate_observation(evidence)` is the pure stdlib ABI for the new poststep
reader. It returns complete GPU idleness (`not rows`); recognized video rows
still require `no_parallel_GPU_at_launch=False` and may establish only
`no_parallel_training_GPU_at_launch=True`. These are launch-time observations,
not an atomic GPU lock or a guarantee about future processes.

Root will execute the registered one-shot command immediately before the future
launch, when the training slot is available, and bind the exact helper record
and genuine observation to a new readonly root owner/reader binding. Every PID
is taken from the actual query. No historical observation can be used as a new
idle claim. Pending actual observation, query exit and runtime tool remain null.

Only the exact ToDesk video executable-path name/argv0, explicit true video flag,
numeric localPort and stable PPID1/Name/start identity are admitted. NVIDIA raw
CSV contains checked PID/name/memory/UUID only; no localPort value, command line,
command-line bytes or port-derived hash is persisted. A denied exe read remains
an explicit limitation; readable exe mismatch, disappearance, other errors or
unrecognized compute tasks reject admission. No background process is changed.

The initially reviewed reader hash in the contract is historical interface
evidence. Its pending environment repair is independent of this helper. Final
actual reader identity is supplied by root's new readonly binding; this packet
does not pin a mutable pending wrapper as the executed source.
