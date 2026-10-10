"""Independent small frozen-source/config/record peer; no data bank, NN or SDK."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tarfile
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_preparation_v1"
OLD = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_v1"
NEW = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"


def record(path):
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def main():
    freeze = json.loads((PREP / "source_freeze_v1.json").read_text())
    assert record(PREP / "source_freeze_v1.json")["sha256"] == "8882de2a73a519b7bb30b0dfcbf0429e1f059ce93a91533a9c8c7d3d076f4a8f"
    assert freeze["schema"] == "stratified_prior.independent_preparation_source_freeze.v1"
    for row in freeze["files"]:
        path = Path(row["path"])
        assert record(path) == row and not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert len(freeze["files"]) == 19
    old, new = (json.loads((p / "source_manifest.json").read_text()) for p in (OLD, NEW))
    before, after = ({r["path"]: r for r in d["files"]} for d in (old, new))
    added = set(after) - set(before)
    expected_added = "code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml"
    assert len(before) == 934 and len(after) == 935 and added == {expected_added}
    for name, declared in after.items():
        actual = record(NEW / name)
        assert actual["sha256"] == declared["sha256"] and actual["bytes"] == declared["bytes"]
        if name in before:
            assert NEW.joinpath(name).read_bytes() == OLD.joinpath(name).read_bytes()
            assert NEW.joinpath(name).stat().st_ino != OLD.joinpath(name).stat().st_ino
    archive_names = set(after) | {"source_manifest.json", "launch_isolated.py", "watch_snapshots.py", "commands.json"}
    with tarfile.open(freeze["actual_source_archive"]["path"], "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(archive_names) == 939
        assert {m.name for m in members} == archive_names
        for member in members:
            assert member.isfile() and archive.extractfile(member).read() == NEW.joinpath(member.name).read_bytes()
    old_config = yaml.safe_load((OLD / "code/configs/diffusion_amp/isolated_terrain_state_prior_absolute.yaml").read_text())
    new_config = yaml.safe_load((NEW / expected_added).read_text())
    old_training, new_training = deepcopy(old_config["training"]), deepcopy(new_config["training"])
    assert old_training.pop("expert_sampling", None) is None
    sampling = new_training.pop("expert_sampling")
    assert old_training == new_training
    assert set(old_config) == set(new_config)
    assert {k: v for k, v in old_config.items() if k not in {"training", "profile", "assumptions"}} == {
        k: v for k, v in new_config.items() if k not in {"training", "profile", "assumptions"}}
    assert sampling["target_weights"] == {"low_speed": .1, "fast": .05, "sideways": .1, "turning": .25, "moving": .5}
    old_commands = json.loads((ROOT / "docs/style_improvement_20261003/state_prior_training_preparation_v1/root_GPU_launch_commands.json").read_text())
    commands = json.loads((PREP / "root_GPU_launch_commands.json").read_text())
    for name in ("pilot4", "resume4to8", "fullfresh2000"):
        expected = [value.replace("mixed_state_action_terrain_state_prior_v1", "mixed_state_action_terrain_state_prior_stratified_v1")
                    .replace("isolated_terrain_state_prior_absolute.yaml", "isolated_terrain_state_prior_stratified_absolute.yaml")
                    .replace("amp_style_state_prior_isolated", "amp_style_state_prior_stratified_isolated")
                    for value in old_commands["commands"][name]]
        assert expected == commands["commands"][name], name
    assert "--resume" not in commands["commands"]["fullfresh2000"]
    readiness = json.loads((PREP / "actual_CPU_readiness.json").read_text())
    closure = json.loads((PREP / "actual_CPU_process_closed.json").read_text())
    ledger = json.loads((PREP / "actual_CPU_imports.json").read_text())
    assert readiness["status"] == "passed" and readiness["bank_clip_counts"] == {"train": 2848, "validation": 22, "test": 34}
    assert readiness["source_manifest_sha256"] == freeze["source_manifest"]["sha256"]
    assert closure["actual_train_bank_process"]["observed_exit_code"] == 0
    assert closure["actual_train_bank_process"]["report"] == record(PREP / "actual_CPU_readiness.json")
    assert ledger["status"] == "finalized" and ledger["phase"] == "passed"
    assert ledger["source_manifest_sha256"] == freeze["source_manifest"]["sha256"]
    for row in ledger["owned_modules"]:
        actual = record(Path(row["path"]))
        assert Path(row["path"]).is_relative_to(NEW / "code")
        assert actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"]
        assert row["execution_observed"] is True
    sys.path.insert(0, str(NEW / "code"))
    from CrowdSim.humanoid.application.settings import normalize_expert_sampling_config
    tree = ast.parse((NEW / "code/scripts/train_paper_amp.py").read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "not args.evaluate_steps"
                 and n.body and isinstance(n.body[0], ast.ImportFrom) and n.body[0].module == "CrowdSim.humanoid.application.settings")
    module = ast.Module(body=deepcopy(block.body[:4]), type_ignores=[])
    guard = compile(ast.fix_missing_locations(module), "original_runner_sampling_guard", "exec")
    results = []
    for resume, requested, saved, reject in [(True, sampling, None, True), (True, None, sampling, True),
                                            (True, sampling, sampling, False), (False, sampling, None, False)]:
        caught = False
        try:
            exec(guard, {"args": SimpleNamespace(resume=resume), "profile": {"training": {"expert_sampling": requested}},
                         "trainer": SimpleNamespace(config=SimpleNamespace(expert_sampling=saved)),
                         "normalize_expert_sampling_config": normalize_expert_sampling_config})
        except ValueError:
            caught = True
        assert caught == reject
        results.append(caught)
    assert not any(name.split(".")[0] in {"torch", "isaaclab", "isaacsim", "omni", "pxr", "warp"} for name in sys.modules)
    report = {"schema": "stratified_prior.independent_frozen_delta_CPU_peer.v1", "status": "passed_source_config_records_only",
              "freeze": record(PREP / "source_freeze_v1.json"), "old934_all_byte_exact_separate_inodes": True,
              "new_profile_only_training_expert_sampling_delta": True, "archive939_all_payloads_byte_exact": True,
              "prepared19_files_all_hashes_readonly_verified": True, "commands_match_baseline_after_only_source_profile_output_paths": True,
              "real_CPU_closure_and_owned_source_records_bound": True, "reexecuted_exact_original_sampling_guard_outcomes": results,
              "NN_or_bank_deserialized": False, "SDK_started": False, "GPU_used": False, "sealed34_outputs_read": False,
              "future_training_or_quality_proven": False}
    with (HERE / "actual_delta_peer_checks.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "source_files": 935, "archive_members": 939, "guard_cases": 4, "GPU_used": False}))


if __name__ == "__main__":
    main()
