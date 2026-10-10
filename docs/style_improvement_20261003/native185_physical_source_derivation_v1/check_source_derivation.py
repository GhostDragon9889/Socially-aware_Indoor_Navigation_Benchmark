"""Read only three frozen native source archives, no imports/models/tests."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INPUTS = [
    ("native168", "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v4.json", "07a34630e052a356aefa3755781bd4c5a7d1406d6a6dd83a0f0f3ec0de80b54f"),
    ("native174", "docs/style_improvement_20261002/native_runtime_epoch_v7/native_pedestrian_source_v7.json", "f9c007ec503236790766e768967a8b745b0dcde1a4a96dfd0b58986763ce403f"),
    ("native185", "docs/style_improvement_20261003/native_runtime_epoch_v12/native_pedestrian_source_v12.json", "9ddbfe994bc692b344917796251780a08931d8d9846a1a19ae7476ba9753535c"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}


def functions(data):
    text = data.decode()
    found = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef):
                    found[name] = {"AST": ast.dump(child, include_attributes=False),
                                   "source": ast.get_source_segment(text, child), "line": child.lineno}
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(text))
    return found


def main():
    banks, identities = {}, {}
    for epoch, relative, expected in INPUTS:
        path = ROOT / relative
        assert record(path)["sha256"] == expected
        manifest = json.loads(path.read_text())
        archive = Path(manifest["archive"]["path"])
        archive = archive if archive.is_absolute() else ROOT / archive
        assert record(archive)["sha256"] == manifest["archive"]["sha256"]
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            assert all(row.isfile() and not Path(row.name).is_absolute() and ".." not in Path(row.name).parts for row in members)
            data = {row.name: tar.extractfile(row).read() for row in members}
        names = set()
        for row in manifest["files"]:
            value = Path(row["path"])
            name = str(value.relative_to(ROOT)) if value.is_absolute() else row["path"]
            names.add(name)
            assert len(data[name]) == row["bytes"] and sha(data[name]) == row["sha256"]
        assert len(members) == len(data) == len(names) == manifest["source_count"]
        assert names == set(data)
        banks[epoch] = data
        identities[epoch] = {"manifest": record(path), "archive": record(archive), "members_all_verified": len(data)}
    current = banks["native185"]
    comparisons = {}
    important = ["CrowdSim/humanoid/amp/controller.py", "CrowdSim/humanoid/amp/observations.py",
                 "CrowdSim/humanoid/amp_world.py", "CrowdSim/humanoid/contact_monitor.py",
                 "CrowdSim/humanoid/application/deployment.py", "CrowdSim/humanoid/application/native_deployment.py",
                 "CrowdSim/humanoid/runtime/evaluation_actor.py", "CrowdSim/humanoid/runtime/dynbench_live.py",
                 "protomotions/simulator/isaaclab/config.py", "protomotions/simulator/isaaclab/simulator.py",
                 "protomotions/simulator/isaaclab/utils/scene.py"]
    important += [name for name in current if name.startswith("CrowdSim/humanoid/diffusion/") and name.endswith(".py")]
    selected = ["CrowdSim/humanoid/application/configuration.py", "CrowdSim/humanoid/application/resources.py",
                "CrowdSim/humanoid/application/settings.py", "CrowdSim/humanoid/runtime/amp_inference.py",
                "CrowdSim/humanoid/runtime/checkpoint_contract.py", "CrowdSim/humanoid/training/trainer.py",
                "scripts/train_paper_amp.py", "CrowdSim/humanoid/runtime/dynbench_scene.py",
                "CrowdSim/humanoid/runtime/native_grid_scene.py", "CrowdSim/humanoid/runtime/native_visual_scene.py"]
    for epoch in ("native168", "native174"):
        old = banks[epoch]
        same_files = sorted(name for name in old if name in current and old[name] == current[name])
        changed_files = sorted(name for name in old if name in current and old[name] != current[name])
        exact = []
        for name in important:
            assert old[name] == current[name], (epoch, name)
            exact.append({"path": name, "file_sha256": sha(current[name]), "all_file_bytes_exact": True,
                          "functions": sorted(functions(current[name]))})
        nodes = []
        for name in selected:
            before, after = functions(old[name]), functions(current[name])
            unchanged = sorted(key for key in before if key in after and before[key]["AST"] == after[key]["AST"])
            changed = sorted(key for key in before if key in after and before[key]["AST"] != after[key]["AST"])
            nodes.append({"path": name, "old_file_sha256": sha(old[name]), "current_file_sha256": sha(current[name]),
                          "unchanged_function_AST": unchanged, "changed_function_AST": changed,
                          "added_functions": sorted(set(after) - set(before)), "removed_functions": sorted(set(before) - set(after))})
        # Diff is exactly the two resume-guard lines, inside a training-only branch.
        driver = current["scripts/train_paper_amp.py"].decode()
        guard = "                from CrowdSim.humanoid.application.configuration import validate_resume_reward_config\n                validate_resume_reward_config(profile[\"training\"], trainer.config)\n"
        assert driver.count(guard) == 1
        assert driver.replace(guard, "").encode() == old["scripts/train_paper_amp.py"]
        comparisons[epoch] = {"same_file_count": len(same_files), "changed_file_count": len(changed_files),
                              "added_count": len(set(current) - set(old)), "removed_count": len(set(old) - set(current)),
                              "changed_files": changed_files, "added_files": sorted(set(current) - set(old)),
                              "selected_exact_physics_control_PD_contacts_planner_files": exact,
                              "selected_function_differences": nodes,
                              "driver_bytes_exact_after_removing_only_training_resume_reward_guard": True,
                              "guard_enclosing_predicate": "if not args.evaluate_steps",
                              "guard_not_executed_for_explicit_evaluate6000": True}
    report = {"schema": "native185.readonly_physical_source_derivation.v1", "status": "passed",
              "inputs": identities, "comparisons": comparisons, "reader": record(Path(__file__)),
              "specific_changes": {
                  "configuration": "New validate_resume_reward_config only; prior parser/determinism/source functions AST exact.",
                  "settings_resources_checkpoint_contract": "New protected terrain/public action-state-prior fields, config/provenance admission and from-scratch rejection.",
                  "actor_loader": "Only from_checkpoint changed: validate optional saved protected-terrain statistics and retain JSON metadata. FrozenAMPActor, constructor fork_rng, normalization/act/clamp57 and actor-only pruning already existed in both168/174 and remain AST exact.",
                  "training_trainer": "Protected terrain/statistics/state-prior initialization and save/load/update support changed; cannot claim complete trainer/source equality. PaperAMPNetwork constructor/policy and RunningNormalizer math remain AST exact.",
                  "native168_to185_scene_admission": "Native map v8/v9 added; DynBenchSceneProvider.admit_native_map, _full_height_admission and compose_native_visual_scene changed before174. Current174→185 scene implementation files are byte exact."},
              "models_loaded": False, "tests_run": False, "SDK_started": False, "GPU_used": False,
              "sealed34_outputs_read": False, "whole_source_equal_claimed": False,
              "bitwise_RNG_or_physics_or_actor_outputs_equal_claimed": False,
              "limitations": ["Archive/function identity proves source sameness, not equal scene/config/model/input tensors or runtime outputs.",
                              "A function's imports and called helpers can differ even when its AST is equal; changed dependency nodes are listed.",
                              "The two historic native epochs remain separate observations; current185 baseline and new Diffusion must be run afresh with their explicit common protocol."]}
    with (HERE / "source_derivation.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "verified_archive_members": 168 + 174 + 185,
                      "from168": [152, 16, 17], "from174": [165, 9, 11], "models_or_tests": False}))


if __name__ == "__main__":
    main()
