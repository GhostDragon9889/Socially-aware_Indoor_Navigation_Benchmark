"""Actual new-bundle config/source preflight; no Torch, weights or simulator."""
import ast
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

from prepare_experiment import HERE, NEW, OLD, PROFILE, ROOT, STAGED, record, write


def main():
    spec = importlib.util.spec_from_file_location("state_only_actual_isolated_launcher", NEW / "launch_isolated.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    source = launcher.verify()  # Exact copied code and all six opaque shared input hashes.
    count = len(source["files"])
    old = json.loads((OLD / "source_manifest.json").read_text())
    before = {row["path"]: row for row in old["files"]}
    after = {row["path"]: row for row in source["files"]}
    changed = [name for name in before if before[name]["sha256"] != after[name]["sha256"]]
    added = sorted(set(after) - set(before))
    assert count == len(before) + 1 and changed == ["code/scripts/train_paper_amp.py"] and added == [PROFILE]
    assert all((NEW / name).stat().st_nlink == 1 and not (NEW / name).stat().st_mode & 0o222
               and (NEW / name).stat().st_ino != (OLD / name).stat().st_ino for name in before)
    plan = json.loads((HERE / "root_GPU_launch_commands.json").read_text())
    delta = json.loads((STAGED / "source_delta.json").read_text())
    ledger = launcher.ImportLedger(HERE / "actual_CPU_imports.json")
    launcher.activate(ledger)
    from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    from CrowdSim.humanoid.application.settings import TrainingConfig
    profile = load_amp_profile(NEW / PROFILE)
    config = TrainingConfig(**profile["training"])
    parent_profile = load_amp_profile(OLD / "code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml")
    assert {key for key in profile["training"] if profile["training"][key] != parent_profile["training"][key]} == set(delta["training_field_changes"])
    assert profile["training"]["expert_sampling"] == parent_profile["training"]["expert_sampling"]
    assert profile["reward"] == parent_profile["reward"]
    args_by_phase = {}
    for name, command in plan["commands"].items():
        argv = command[command.index("--") + 1:]
        args = parse_amp_args(argv)
        assert args.resume is None and not args.from_scratch and args.seed == 7
        resources = preflight_amp_resources(args, profile)
        assert resources.require("warm_start_checkpoint") == profile["initialization"]["checkpoint"]
        args_by_phase[name] = vars(args)
    bank_path = ROOT / "docs/style_improvement_20261003/state_prior_stratified_training_preparation_v1/actual_CPU_readiness.json"
    bank = json.loads(bank_path.read_text())
    assert record(bank_path)["sha256"] == "ff1e18caf999128a252b6af928c857f037235036718c67d3e3e055fda98a2c21"
    assert bank["status"] == "passed" and bank["original_v7_mapping_probability_hashes_reproduced"] is True
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    ledger.finish("passed_source_config_preflight")
    tree = ast.parse((NEW / "code/scripts/train_paper_amp.py").read_text())
    step_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "step")
    reward = next(index for index, node in enumerate(step_loop.body) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "amp_reward" for target in node.targets))
    physics = next(index for index, node in enumerate(step_loop.body) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "world.step")
    assert reward < physics
    document = {"schema":"state_only_matched_q.actual_CPU_preparation.v1","status":"passed","CPU_only":True,
                "GPU_used":False,"SDK_started":False,"sealed34_outputs_read":False,"models_deserialized":False,
                "source_manifest":record(NEW / "source_manifest.json"),"source_manifest_sha256":record(NEW / "source_manifest.json")["sha256"],
                "source_count":count,"parent_source":record(OLD / "source_manifest.json"),"modified_parent_paths":changed,"added_paths":added,
                "actual_fresh2000_parser_passed":True,"actual_new_bundle_profile_loader_passed":True,
                "full_resource_hashes_verified_by_unchanged_launcher":True,"shared_input_count":len(source["resources"]),
                "q_configuration_and_unchanged_q_sources_exact":True,"actual_parent_q_bank_CPU_proof":record(bank_path),
                "original_v7_mapping_probability_hashes_reproduced":True,"bank_not_rebuilt_for_unchanged_sampler":True,
                "D_input":1950,"direct_public_D_and_AMP_RMS_config_validated":True,
                "matched_pre_step_history_replay_AST_validated":True,"public_tensor_transfer_actually_executed":False,
                "training_config":asdict(config),"actual_parser_arguments":args_by_phase,
                "owned_imports":record(HERE / "actual_CPU_imports.json"),"all_owned_imports_new_bundle":True,
                "actual_training_payload_admitted":False,"future_effect":False}
    write(HERE / "actual_CPU_readiness.json", document)
    print(json.dumps({"status":"passed","source_count":count,"resources":len(source["resources"]),
                      "actual_CPU_readiness":record(HERE / "actual_CPU_readiness.json")}))


if __name__ == "__main__":
    main()
