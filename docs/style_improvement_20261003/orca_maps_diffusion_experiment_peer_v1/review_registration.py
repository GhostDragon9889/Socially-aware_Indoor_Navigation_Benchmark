"""Two CPU checks: cross-corpus identities and the actual frozen CLI parser.

No model is deserialized; no training, corpus regeneration or SDK is started.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def record(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    registration_path = ROOT / "docs/style_improvement_20261003/orca_maps_diffusion_experiment_v1/registration.json"
    registration = json.loads(registration_path.read_text())
    assert record(registration_path)["sha256"] == "89060cfc42d6217bd8b6ca0ab760a67c00fe3d3b44956af0f0cf970a02ed0c4e"
    old_path = ROOT / "data/diffusion_amp/orca_social_pilot_v1/manifest.json"
    new_path = Path(registration["bindings"]["new_manifest"]["path"])
    old, new = (json.loads(path.read_text()) for path in (old_path, new_path))
    assert record(new_path) == registration["bindings"]["new_manifest"]
    datasets = []
    for dataset in (old, new):
        scenes = [scene for split in dataset["splits"].values() for scene in split["scenes"]]
        layouts = {scene["layout_sha256"] for scene in scenes}
        seeds = {episode["episode_seed"] for scene in scenes for episode in scene["episodes"]}
        assert len(layouts) == len(scenes)
        assert len(seeds) == sum(len(scene["episodes"]) for scene in scenes)
        datasets.append((scenes, layouts, seeds))
    assert len(datasets[0][0]) == 24 and len(datasets[1][0]) == 1000
    layout_overlap = sorted(datasets[0][1] & datasets[1][1])
    seed_overlap = sorted(datasets[0][2] & datasets[1][2])
    assert not layout_overlap and not seed_overlap
    train_entry = Path(registration["command"][1])
    assert record(train_entry) == registration["bindings"]["train_entry"]
    source = json.loads(Path(registration["bindings"]["source_manifest"]["path"]).read_text())
    bundle = Path(registration["bindings"]["source_manifest"]["path"]).parent
    owned = {row["path"]: row for row in source["files"]}
    for relative in ("code/scripts/train_orca_diffusion.py", "code/CrowdSim/humanoid/diffusion/paper.py"):
        observed = record(bundle / relative)
        assert (observed["bytes"], observed["sha256"]) == (owned[relative]["bytes"], owned[relative]["sha256"])
    specification = importlib.util.spec_from_file_location("actual_frozen_ORCA_train_parser", train_entry)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    args = module.parse_args(registration["command"][2:])
    assert args.resume is None and record(args.init_checkpoint) == registration["bindings"]["initial_model"]
    assert args.manifest == new_path and args.steps == 5000 and args.batch_size == 8 and args.learning_rate == 1e-4
    assert args.seed == 20261001 and args.device == "cuda" and args.cuda_memory_fraction == .15
    assert args.eval_interval == args.checkpoint_interval == 500 and args.sample_windows == 32
    assert args.output_dir == Path(registration["future_checkpoint"]["path"]).parent
    old_config = json.loads(Path(registration["bindings"]["old_latest_config"]["path"]).read_text())
    assert old_config["initialization"]["sha256"] == registration["bindings"]["initial_model"]["sha256"]
    assert old_config["initialization"]["model_step"] == 1000
    tree = ast.parse(train_entry.read_text())
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    fresh = [node for node in main_node.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == "args.resume is None"
             and any(ast.unparse(child) == "planner.trained_steps = 0" for child in node.body)]
    assert len(fresh) == 1
    start = next(node for node in main_node.body if isinstance(node, ast.Assign)
                 and ast.unparse(node) == "start_step = planner.trained_steps")
    loop = next(node for node in main_node.body if isinstance(node, ast.For)
                and ast.unparse(node.iter) == "range(args.steps)")
    assert fresh[0].lineno < start.lineno < loop.lineno
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    result = {"schema": "local_orca_maps.independent_CPU_registration_peer.v1", "status": "passed",
              "registration": record(registration_path), "reader": record(__file__),
              "old_manifest": record(old_path), "new_manifest": record(new_path),
              "cross_corpus": {"old_layouts": 24, "new_layouts": 1000, "old_episode_seeds": len(datasets[0][2]),
                               "new_episode_seeds": len(datasets[1][2]), "layout_sha256_intersection": layout_overlap,
                               "episode_seed_intersection": seed_overlap,
                               "scope": "Complete published old/new manifest identities; new canonical geometry already verified by independent all-data QC. Scene labels are local to each corpus."},
              "actual_parser": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "phase_count": {"initial_AMASS_model_step": 1000, "fresh_social_start_step": 0,
                              "additional_social_optimizer_updates": 5000, "final_social_step": 5000,
                              "evidence": "Actual frozen parser plus inspected AST reset before start_step and range(args.steps); no training execution claimed."},
              "same_initialization_and_frozen_model_source": True, "future_model_read": False,
              "GPU_used": False, "SDK_started": False, "training_executed": False,
              "old_model_on_new_validation_comparison_executed": False, "quality_effect_claimed": False}
    with (HERE / "actual_CPU_peer.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": "passed", "layout_intersection": len(layout_overlap),
                      "episode_seed_intersection": len(seed_overlap), "fresh_phase_steps": 5000,
                      "report": record(HERE / "actual_CPU_peer.json")}))


if __name__ == "__main__":
    main()
