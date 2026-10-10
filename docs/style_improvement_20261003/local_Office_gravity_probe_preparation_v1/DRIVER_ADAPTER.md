# Future executable adapter

`driver_adapter.py` implements the injection recipe in the unchanged original
`plan.json`. Preparation is SDK free:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261003/local_Office_gravity_probe_preparation_v1/driver_adapter.py \
  --prepare-only
```

The root-owned future command is the same interpreter/adapter without
`--prepare-only`. The adapter fixes the registered three-actor10second command;
it accepts no actor/duration/controller overrides. Before actual execution it
checks the same actual terminal reader admission as the Warehouse comparison,
recomputes the full selected checkpoint hash and requires one readonly file.

It first calls `deployment.require_fresh_sdk_process()`, then reuses
`deployment.main(argv, driver_main=callback)`. The callback preserves the
existing evaluation actor factory, wraps only the supplied native scene
factory and delegates once to `deployment.DRIVER`'s `module.main(**options)`.
The production driver owns the only App/context and control loop; the adapter
does not create another App, world, view or policy loop. Current production185
bytes are checked before and after that call.

The sole cold fake-delegation test passed: the actor factory and return value
are identical, the existing native factory is wrapped once, and no tensor/SDK
module is imported. This is CPU source/ABI evidence and has not observed a real
Office humanoid. The expected future capture remains three actors/300controls/
1,200shared substeps/14,400foot observations. The independent Warehouse4/15
reader cannot read this short case; its future Office metadata adapter remains
explicitly separate.

The saved producer `deployment_entry.evaluation_actor_request.base_checkpoint`
contains **path and SHA only**. The future sealed file's complete byte count is
verified separately by actual terminal and file records. A future Office
reader must preserve that producer ABI and record the actual adapter invocation
and root process closure, without treating `--prepare-only` as a completed run.

This local probe makes no claim about the authors' eight scenes or200second
performance. The root still owns serial GPU scheduling after the current
Warehouse comparison.
