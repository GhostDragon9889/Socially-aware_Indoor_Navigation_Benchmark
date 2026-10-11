"""Complete current-source admission for a serial episode execution plan."""
from pathlib import Path

from social_navigation.artifact_identity import artifact_path, same_artifact
from . import episode_evaluation_admission as admission
from . import episode_serial_release as release
from ..runtime import grouped_source_migration as grouped

ROOT = release.ROOT


def source_inventory():
    from scripts import own_shared_social_episode_pipeline as owner
    from scripts import train_shared_social_episode_after_recovery as training
    from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation
    registration = admission.registration_document("current")
    current = registration["episode_reference_experiment"]
    paths = [Path(__file__), Path(owner.__file__), Path(training.__file__), Path(evaluation.__file__),
             Path(release.__file__), Path(owner.io.__file__), Path(release.consumers.__file__),
             Path(release.clock_grid.__file__), Path(evaluation.original.__file__), Path(admission.__file__),
             admission.CURRENT_TRAINING_REGISTRATION, admission.PLAN, admission.PROBE,
             ROOT / "scripts/audit_shared_social_popart.py", ROOT / "scripts/audit_shared_social_training.py",
             ROOT / "scripts/prepare_shared_social_episode_pipeline.py",
             Path(grouped.__file__), grouped.MANIFEST,
             ROOT / "src/protomotions/envs/base_env/env.py",
             ROOT / "src/protomotions/simulator/isaaclab/simulator.py",
             ROOT / "src/protomotions/simulator/isaaclab/utils/sdk61_owned_close.py"]
    pipeline = evaluation.original.prepare_pipeline(registration)
    paths.extend(ROOT / relative for relative in pipeline.SOURCE_PATHS)
    # Include the maintained migration/reader helpers used by those entries,
    # even when a frozen parent knows only its direct entry source list.
    paths.extend((ROOT / "src/CrowdSim/humanoid/runtime").glob("*.py"))
    paths.extend((ROOT / "src/CrowdSim/humanoid/application").glob("*.py"))
    paths.extend((ROOT / "src/CrowdSim/humanoid/analysis").glob("*.py"))
    paths.extend((ROOT / "src/social_navigation").glob("*.py"))
    for rows in (current["entry_sources"], current["setup_sources"],
                 current["training_binding"]["World_dependencies"], grouped.manifest()["dependencies"]):
        paths.extend(artifact_path(row["path"] if Path(row["path"]).is_absolute() else ROOT / row["path"])
                     for row in rows.values())
    return [release.record(path) for path in sorted(set(map(artifact_path, paths)))]


def input_paths(plan):
    return [artifact_path(plan[key]["path"]) for key in ("serial_release", "cold_preflight")] + [
        ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json",
        ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7_train.pt",
        *(artifact_path(row["profile"]["path"]) for row in release.stages())]


def require_inventory(rows, expected, label):
    if type(rows) is not list:
        raise ValueError(label + " inventory must be a list")
    actual_paths = [artifact_path(row["path"]) for row in rows]
    expected_paths = [artifact_path(row["path"]) for row in expected]
    if len(set(actual_paths)) != len(actual_paths) or set(actual_paths) != set(expected_paths):
        raise ValueError(label + " inventory is incomplete or duplicated")
    by_path = dict(zip(actual_paths, rows, strict=True))
    if any(not same_artifact(by_path[path], row) for path, row in zip(expected_paths, expected, strict=True)):
        raise ValueError(label + " inventory identity changed")


def validate_inventory(plan):
    require_inventory(plan["sources"], source_inventory(), "Current source")
    require_inventory(plan["inputs"], [release.record(p) for p in input_paths(plan)], "Current input")
    for key in ("serial_release", "cold_preflight"):
        if not any(same_artifact(row, plan[key]) for row in plan["inputs"]):
            raise ValueError("Unbound plan " + key)
    corpus = {"manifest_labeled_v7.json": "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca",
              "manifest_labeled_v7_train.pt": "46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0"}
    for row in plan["inputs"]:
        if Path(row["path"]).name in corpus and row["sha256"] != corpus[Path(row["path"]).name]:
            raise ValueError("Fixed real AMASS corpus changed")


def validate_preflight(plan, preflight):
    import json
    if preflight.get("schema") != "shared_social.episode_cold_preflight.sim61.v2":
        raise ValueError("Cold preflight schema differs from current execution")
    # Cold probes are only applicable to the exact reviewed source inventory.
    require_inventory(preflight.get("sources"), plan["sources"], "Cold preflight source")
    from scripts import train_shared_social_episode_after_recovery as training
    from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation
    from social_navigation.external_runtime import selected_python
    executable = str(selected_python(ROOT))
    serial = str(artifact_path(plan["serial_release"]["path"]))
    commands = [[executable, str(Path(training.__file__)), "--serial-release", serial,
                 "--stage", stage, "--check-sources"] for stage in ("smoke8", "smoke256", "pilot2000")]
    commands.append([executable, str(Path(evaluation.__file__)), "--serial-release", serial, "--check-sources"])
    for row, command in zip(preflight["steps"], commands, strict=True):
        if (row.get("command") != command or row.get("exit_observation") != "subprocess.run.returncode"
                or type(row.get("actual_exit_code")) is not int or row["actual_exit_code"] != 0):
            raise ValueError("Cold preflight command is not the registered entry")
        for key in ("stdout", "stderr"):
            if not same_artifact(release.record(artifact_path(row[key]["path"])), row[key]):
                raise ValueError("Cold preflight output changed")
        document = json.loads(artifact_path(row["stdout"]["path"]).read_text())
        if document.get("Torch_imported") is not False or document.get("SDK_started") is not False:
            raise ValueError("Cold preflight stdout declares a live runtime")
