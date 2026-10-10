"""Small source derivation and exact saved-sampling branch check, no SDK."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
NEW = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"


def record(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    old = json.loads((OLD / "source_manifest.json").read_text())
    new = json.loads((NEW / "source_manifest.json").read_text())
    before, after = {r["path"]: r for r in old["files"]}, {r["path"]: r for r in new["files"]}
    assert len(before) == 934 and len(after) == 935
    added = set(after) - set(before)
    assert added == {"code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml"}
    for path, declared in before.items():
        assert OLD.joinpath(path).read_bytes() == NEW.joinpath(path).read_bytes()
        assert declared["sha256"] == after[path]["sha256"] == record(NEW / path)["sha256"]
        assert OLD.joinpath(path).stat().st_ino != NEW.joinpath(path).stat().st_ino
    old_profile = yaml.safe_load((OLD / "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml").read_text())
    profile = yaml.safe_load((NEW / next(iter(added))).read_text())
    changed = [key for key in set(profile["training"]) | set(old_profile["training"])
               if profile["training"].get(key) != old_profile["training"].get(key)]
    assert changed == ["expert_sampling"]
    for key in set(old_profile) - {"training", "profile", "assumptions"}:
        assert old_profile[key] == profile[key], key
    sys.path.insert(0, str(NEW / "code"))
    from CrowdSim.humanoid.application.settings import normalize_expert_sampling_config
    runner = NEW / "code/scripts/train_paper_amp.py"
    tree = ast.parse(runner.read_text())
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "not args.evaluate_steps"
              and len(node.body) > 3 and isinstance(node.body[0], ast.ImportFrom)
              and node.body[0].module == "CrowdSim.humanoid.application.settings"]
    assert len(blocks) == 1
    # Execute the four exact original statements preceding physical-world setup.
    block = ast.Module(body=deepcopy(blocks[0].body[:4]), type_ignores=[])
    compiled = compile(ast.fix_missing_locations(block), str(runner), "exec")
    config = profile["training"]["expert_sampling"]
    cases = [("baseline_None_to_enabled_resume", True, config, None, True),
             ("enabled_to_None_resume", True, None, config, True),
             ("matching_new_resume", True, config, config, False),
             ("fresh_new_sampling", False, config, None, False)]
    results = []
    for name, resume, requested, saved, should_reject in cases:
        env = {"args": SimpleNamespace(resume=resume), "profile": {"training": {"expert_sampling": requested}},
               "trainer": SimpleNamespace(config=SimpleNamespace(expert_sampling=saved)),
               "normalize_expert_sampling_config": normalize_expert_sampling_config}
        rejected, message = False, None
        try:
            exec(compiled, env)
        except ValueError as error:
            rejected, message = True, str(error)
        assert rejected == should_reject, name
        results.append({"name": name, "rejected": rejected, "expected_rejection": should_reject, "message": message})
    report = {"schema": "stratified_prior.actual_source_and_resume_branch_CPU.v1", "status": "passed",
              "source_manifest": record(NEW / "source_manifest.json"), "baseline_manifest": record(OLD / "source_manifest.json"),
              "old934_bytes_exact_and_separate_inodes": True, "added_only": sorted(added),
              "training_config_delta_only": changed, "reward_model_PD_replay_task_and_nonmetadata_exact": True,
              "original_runner": record(runner), "executed_exact_guard_AST": ast.dump(block, include_attributes=False),
              "actual_guard_cases": results, "CPU_only": True, "GPU_used": False, "future_model_read": False,
              "limit": "Only the unchanged runner's sampling-configuration guard statements executed, not Kit setup or checkpoint NN load."}
    with (HERE / "actual_source_and_resume_branch_CPU.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "old_source_exact": 934, "new_profile": 1, "training_delta": changed,
                      "actual_saved_sampling_guard_cases": len(results), "GPU_used": False}))


if __name__ == "__main__":
    main()
