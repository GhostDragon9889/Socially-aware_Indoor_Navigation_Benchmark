"""Prepare all52 explicit local resource cases; retain every rejection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application.native_scene_preparation import prepare_configuration, publish_configuration  # noqa: E402
from CrowdSim.humanoid.runtime.dynbench_scene import file_record  # noqa: E402

HERE = Path(__file__).resolve().parent
MAPS = ROOT / "output/scene_maps/native_reference_four_scenes_anonymous_root_v9"
SCENES = ("Office", "Warehouse", "Full_Warehouse", "Hospital")


def main():
    selected = {scene: {role: file_record(MAPS / scene / filename) for role, filename in
                       (("preflight", "preflight.json"), ("native_map", "result.json"))} for scene in SCENES}
    sources = {name: file_record(ROOT / name) for name in (
        "CrowdSim/humanoid/application/native_scene_preparation.py",
        "CrowdSim/humanoid/application/native_deployment.py",
        "CrowdSim/humanoid/runtime/native_grid_scene.py",
        "CrowdSim/humanoid/runtime/dynbench_scene.py")}
    sources["entry"] = file_record(__file__)
    registration = {"schema": "shared_social.local_native_resource_cohort_registration.v1",
        "case_order": [[scene, count] for scene in SCENES for count in range(3, 16)],
        "resources": selected, "sources": sources, "route_clearance_m": .8,
        "minimum_spawn_clearance_m": 3., "minimum_spawn_separation_m": 1.6,
        "selection": "Unchanged deterministic largest4-neighbour/deepest/maximin/first ROS row-major ties",
        "author_eight_scene_protocol": False, "physical_navigation_validated": False}
    registration_path = HERE / "resource_cohort_registration.json"
    with registration_path.open("x") as stream:
        stream.write(json.dumps(registration, indent=2) + "\n")
    registration_path.chmod(0o444)
    output = HERE / "resource_cohort_CPU.json"
    with output.open("x") as stream:
        stream.write("{}\n")
    folder = HERE / "scenes"
    folder.mkdir(exist_ok=False)
    report = {"schema": "shared_social.local_native_resource_cohort_CPU.v1", "registration": file_record(registration_path),
              "status": "running", "cases": [], "SDK_started": False, "GPU_used": False,
              "human_navigation_validated": False, "author_eight_scene_protocol": False}
    for scene, count in registration["case_order"]:
        case = {"scene": scene, "actors": count, "status": "pending"}
        report["cases"].append(case)
        try:
            config, proof = prepare_configuration(selected[scene]["preflight"]["path"],
                selected[scene]["native_map"]["path"], actors=count, clearance_m=.8,
                spawn_clearance_m=3., minimum_spawn_separation_m=1.6)
            config_path, proof_path = folder / f"{scene}_N{count:02d}.json", folder / f"{scene}_N{count:02d}_proof.json"
            proof = publish_configuration(config, proof, config_path, proof_path)
            config_path.chmod(0o444)
            proof_path.chmod(0o444)
            case.update(status="CPU_resource_admitted", configuration=file_record(config_path),
                        proof=file_record(proof_path), Astar_segments=len(proof["route_paths"]),
                        minimum_spawn_separation_m=proof["selection"]["minimum_spawn_separation_m"])
        except Exception as error:
            case.update(status="rejected", error={"type": type(error).__name__, "message": str(error)},
                        clearance_or_spacing_reduced=False, replacement_scene_or_resampling=False)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"scene": scene, "actors": count, "status": case["status"]}), flush=True)
    report["source_resources_unchanged"] = all(file_record(row["path"]) == row for rows in selected.values() for row in rows.values())
    report["sources_unchanged"] = all(file_record(row["path"]) == row for row in sources.values())
    report["admitted_cases"] = sum(row["status"] == "CPU_resource_admitted" for row in report["cases"])
    report["rejected_cases"] = sum(row["status"] == "rejected" for row in report["cases"])
    report["Astar_segments"] = sum(row.get("Astar_segments", 0) for row in report["cases"])
    report["Torch_imported"] = "torch" in sys.modules
    report["status"] = "CPU_resource_case_review_complete" if (report["source_resources_unchanged"]
        and report["sources_unchanged"] and not report["Torch_imported"]) else "failed"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "CPU_resource_case_review_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
