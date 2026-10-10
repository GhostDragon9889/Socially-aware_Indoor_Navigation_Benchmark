# Native recorded-pose replay: prepared CPU contract

The replay accepts the exact native scene recorded by a completed physical run through `--native-scene-visuals`. It verifies the result/states, admitted map, preflight, historical probe snapshot, source USD and its layers. The new helper composes the original global USD into the caller's existing render-only stage, keeps its authored world coordinates, disables authored physics in the session, and adds the same finite reference-floor geometry as a visual mesh without collision. It creates no `SimulationContext`, advances no physics and runs no policy.

## Evidence

- 29 focused CPU tests passed; Ruff passed. Native stage tests use explicit fake USD interfaces and do not demonstrate an SDK render.
- The old cuboid preflight dictionary equals the preserved before source exactly. The recorded-pose frame loop has exactly the same AST as before.
- The actual `native_warehouse_clearance08_30s_v2` source passed CPU preflight: 900 recorded control frames, four actors, 450 selected images at 15 fps. The first selected source pose is post-control at 1/30 s; the final selected pose is at 899/30 s. The declared video interval is 30 s.
- Source hashes/archive and test logs are in [source_freeze.json](source_freeze.json). Actual preflight is in [actual_native_preflight.json](actual_native_preflight.json).

## Command

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
../IsaacLab-Protomotions/.venv/bin/python scripts/replay_paper_pedestrians.py \
  --states output/diffusion_amp/native_warehouse_clearance08_30s_v2/physical_states.npz \
  --source-result output/diffusion_amp/native_warehouse_clearance08_30s_v2/result.json \
  --output output/diffusion_amp/native_warehouse_recorded_replay_30s_v1 \
  --steps 900 --capture-every-steps 2 --native-scene-visuals --preflight-only
```

The parent may render by removing `--preflight-only` and the CPU-only CUDA environment restriction, with an explicit suitable Warehouse camera via the existing camera flags. That actual SDK run is pending. Every rendered image retains the permanent “Recorded physical state replay” banner and its source timestamp; the app timeline stays frozen. Rendering does not validate contacts, support, controller quality, or longer navigation stability. Native resources with missing or mismatched recorded provenance are rejected before SDK startup. The default cuboid path requires no new native flag or resources.
