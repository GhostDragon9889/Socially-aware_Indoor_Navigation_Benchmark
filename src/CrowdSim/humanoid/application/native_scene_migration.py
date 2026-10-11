"""Explicit probe snapshots for the unchanged, registered native scene cohort."""
import json

from protomotions.repository import repository_root
from social_navigation.artifact_identity import artifact_path, same_artifact
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record

ROOT = repository_root(__file__)
COHORT = ROOT / "docs/style_improvement_20261005/shared_social_native_SFM_preparation_v1/resource_cohort_CPU.json"
COHORT_SHA = "3385701a74fe120d53ba540402e5257e6748653b6112b102c4ac872b982b72ec"
DIRECTORY = ROOT / "configs/compatibility/native_seeded_sim61/scenes"
SNAPSHOT = {
    "path": "../../../../docs/style_improvement_20261002/native_map_clock_v9_proposal_v2/proposed/scripts/probe_paper_scene_maps.py",
    "sha256": "e47cd51d9f40cd0ead27595283fefab95871ea229c7029c5622b57391a01ad82",
}


def configuration(identity):
    """Admit a separately stored config; every original field stays identical."""
    if record(COHORT)["sha256"] != COHORT_SHA:
        raise ValueError("Original native cohort changed")
    matches = [row for row in json.loads(COHORT.read_text())["cases"]
               if row["status"] == "CPU_resource_admitted"
               and same_artifact(row["configuration"], identity, root=ROOT)]
    if len(matches) != 1:
        raise ValueError("Select one originally admitted native scene")
    source = artifact_path(identity["path"], root=ROOT)
    if not same_artifact(record(source), identity, root=ROOT):
        raise ValueError("Original native configuration changed")
    target = DIRECTORY / source.name
    before = record(target)
    if target.stat().st_mode & 0o222:
        raise ValueError("Derived native configuration must be readonly")
    original, derived = json.loads(source.read_text()), json.loads(target.read_text())
    if "probe_script_snapshot" in original or derived != {**original, "probe_script_snapshot": SNAPSHOT}:
        raise ValueError("Native migration changed original geometry or resource bindings")
    snapshot = record((DIRECTORY / SNAPSHOT["path"]).resolve())
    if snapshot["bytes"] != 107170 or snapshot["sha256"] != SNAPSHOT["sha256"]:
        raise ValueError("Registered native probe snapshot changed")
    if record(target) != before:
        raise ValueError("Derived native configuration changed during admission")
    return before, {"version": "native_scene_probe.sim61.v2", "historical": identity,
                    "current": before, "probe_snapshot": snapshot, "adapter": record(__file__),
                    "all_original_fields_unchanged": True, "SDK_executed": False}
