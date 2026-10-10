"""Read actual v9 evidence on CPU after approved merge; pending is not PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record
PROBE_SHA256 = "e47cd51d9f40cd0ead27595283fefab95871ea229c7029c5622b57391a01ad82"

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
SCENES = ("Office", "Warehouse", "Full_Warehouse", "Hospital")
DEFAULT_NATIVE = ROOT / "output/scene_maps/native_reference_four_scenes_anonymous_root_v9"


def registration(native_output, probe_snapshot):
    proposal = ROOT / "docs/style_improvement_20261002/native_map_clock_v9_proposal_v2"
    freeze_record = file_record(proposal / "source_freeze.json")
    if freeze_record["sha256"] != "80b9762ed7f0b34c363a055d34f876bd52437fbf6f6248fc438a22e0bc99eedc":
        raise ValueError("Frozen V9 V2 proposal identity changed")
    freeze = json.loads(Path(freeze_record["path"]).read_text())
    for record in freeze["files"]:
        if file_record(record["path"]) != record:
            raise ValueError("Frozen V9 V2 proposal payload/evidence changed")
    manifest_record = file_record(proposal / "merge_manifest.json")
    if manifest_record["sha256"] != "62cee030f798e9644ed7237f39611f8c17b0f6771cee657bb7e91cda8bab359d":
        raise ValueError("V9 V2 approved bounded source manifest changed")
    manifest = json.loads(Path(manifest_record["path"]).read_text())
    probe = file_record(probe_snapshot)
    if probe["sha256"] != PROBE_SHA256:
        raise ValueError("Native V9 reader requires exact anonymous-root producer snapshot")
    expected = [row for row in manifest["files"] if row["relative_path"].startswith("CrowdSim/") or row["relative_path"] == "scripts/probe_paper_scene_maps.py"]
    actual = []
    for row in expected:
        target = ROOT / row["relative_path"]
        current = file_record(target) if target.exists() else None
        actual.append({"path":str(target),"current":current,"expected_sha256":row["after"]["sha256"],
                       "candidate_bytes_installed":current is not None and current["sha256"]==row["after"]["sha256"]})
    return {"schema":"native_four_scene_v9_CPU_reader_registration.v1","status":"prepared_actual_results_pending",
            "CPU_only":True,"SDK_started":False,"GPU_used_by_reader":False,"native_v9_execution_observed":False,
            "measured_map_admission":None,"route_validation":None,"native_output":str(Path(native_output).resolve()),
            "scenes":list(SCENES),"proposal_freeze":freeze_record,"source_manifest":manifest_record,
            "actual_checkout_consumers":actual,"all_required_candidate_bytes_installed":all(row["candidate_bytes_installed"] for row in actual),
            "producer_snapshot":probe,"reader":file_record(__file__)}


def array_inventory(record):
    """Persist array dtype/shape/byte hashes, rejecting nonfinite/object buffers."""
    import numpy as np
    if file_record(record["path"]) != record:
        raise ValueError("Array artifact changed: " + record["path"])
    rows = {}
    with np.load(record["path"], allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            if value.dtype.kind == "O" or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
                raise ValueError("Object/nonfinite native buffer: " + key)
            rows[key] = {"shape": list(value.shape), "dtype": value.dtype.str,
                         "sha256_C_order": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    return {"file": record, "arrays": rows}


def audit_scene(folder, probe_snapshot):
    from CrowdSim.humanoid.runtime.native_map_v9 import is_v9, verify_v9
    result_path = Path(folder) / "result.json"
    data = json.loads(result_path.read_text())
    if data.get("scene") not in SCENES:
        raise ValueError("Unexpected scene result")
    if not is_v9(data) or data.get("CPU_contract_fixture_only") is True:
        raise ValueError("Older/failed or synthetic producer artifacts cannot be relabelled actual v9")
    preflight = data["preflight"]
    provider = DynBenchSceneProvider.from_preflight(
        preflight["path"], expected_sha256=preflight["sha256"], probe_script_snapshot=probe_snapshot)
    record = file_record(result_path)
    provider.admit_native_map(result_path, expected_sha256=record["sha256"])
    verified = verify_v9(provider, data, admitted=True)
    arrays = {"native_omap": array_inventory(data["native_omap"]),
              "canonical_world_positions": array_inventory(data["coordinate_validation"]["canonical_world_positions_artifact"]),
              "floor_all_hit_CSR": array_inventory(data["floor_support"]["artifact"]),
              "full_height_band_CSR": array_inventory(data["full_height_band_queries"]["artifact"])}
    clocks = {"height_queries": data["full_height_band_queries"]["clock_before"],
              "height_query_artifact": data["full_height_band_queries"]["clock_artifact"],
              "initial_query_clock": data["actual_query_clock_initial"]}
    return {"scene": provider.scene_name, "actual_result": record, "preflight": preflight,
            "status": "actual_native_v9_resource_admitted_by_CPU_reader", "admission": verified,
            "arrays": arrays, "observed_clocks": clocks, "floor_ownership": data["floor_ownership"],
            "derived_export_grid": data["derived_export_grid"],
            "runtime_time_range_override": data["runtime_time_range_override"],
            "original_native_status_preserved": data["status"],
            "reader_started_native_execution": False, "human_navigation_success_claim": False}


def audit_routes(path, scene_rows):
    import numpy as np
    from CrowdSim.humanoid.application.native_deployment import make_native_scene_factory, preflight_native_scene
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    request = preflight_native_scene(path, actors=len(config["origins_xy"]))
    row = next((row for row in scene_rows if row["scene"] == config["scene"]), None)
    if row is None or request["native_map"] != row["actual_result"] or request["preflight"] != row["preflight"]:
        raise ValueError("Scene configuration must bind this exact newly audited actual artifact")
    grid = make_native_scene_factory(request)(request["configuration"], device="cpu")
    paths = []
    for actor, (origin, route) in enumerate(zip(config["origins_xy"], config["routes_xy"])):
        starts = [origin, *route]
        ends = [*route, route[0]]
        for segment, (start, end) in enumerate(zip(starts, ends)):
            points = grid.astar_path(start, end)
            if points is None or len(points) < 2 or not grid.is_walkable(points).all():
                raise ValueError(f"Actor {actor} initial/cycle route is disconnected at segment {segment}")
            paths.append({"actor": actor, "segment": segment, "vertices": len(points),
                          "length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum()),
                          "path_float64_C_order_sha256": hashlib.sha256(points.tobytes()).hexdigest()})
    return {"configuration": request["configuration_file"], "scene": config["scene"],
            "actors": len(config["origins_xy"]), "all_initial_and_wrap_Astar_passed": True,
            "paths": paths, "physical_navigation_success_claim": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-output", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--probe-snapshot", type=Path, default=ROOT / "docs/style_improvement_20261002/native_map_clock_v9_proposal_v2/proposed/scripts/probe_paper_scene_maps.py")
    parser.add_argument("--scene-config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be new; prior reports will not be overwritten")
    registered = registration(args.native_output, args.probe_snapshot)
    if args.register_only and args.scene_config:
        parser.error("Registration cannot claim scene-configuration route measurements")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"registration": registered, "CPU_only": True, "SDK_started": False, "GPU_used_by_reader": False,
              "native_v9_execution_observed": False, "status": "prepared_actual_results_pending",
              "scenes": [], "routes": None}
    exit_code = 0
    if not args.register_only:
        try:
            if not registered["all_required_candidate_bytes_installed"]:
                raise ValueError("Actual v9 audit requires approved candidate installation; pending registration is not runtime success")
            failures=[]
            for scene in SCENES:
                try:
                    row=audit_scene(args.native_output/scene,args.probe_snapshot)
                    if row["scene"]!=scene:
                        raise ValueError("Native output folder and source scene disagree")
                    report["scenes"].append(row)
                    report["native_v9_execution_observed"]=True
                except Exception as error:
                    path=args.native_output/scene/"result.json"
                    actual=json.loads(path.read_text()) if path.exists() else None
                    failed={"scene":scene,"status":"rejected_actual_or_missing_evidence",
                        "actual_result":file_record(path) if path.exists() else None,
                        "original_native_status_preserved":actual.get("status") if actual else None,
                        "error":{"type":type(error).__name__,"message":str(error)},"admitted":False}
                    if actual is not None:
                        failed["actual_initial_query_clock"]=actual.get("actual_query_clock_initial")
                        failed["runtime_stage_composition"]=actual.get("runtime_stage_composition")
                        failed["runtime_stage_attachment"]=actual.get("runtime_stage_attachment")
                        failed["runtime_time_range_override"]=actual.get("runtime_time_range_override")
                        report["native_v9_execution_observed"]|=actual.get("native_started") is True
                    report["scenes"].append(failed)
                    failures.append(scene)
            admitted=[row for row in report["scenes"] if row["status"]=="actual_native_v9_resource_admitted_by_CPU_reader"]
            report["routes"]=[audit_routes(path,admitted) for path in args.scene_config] or None
            report["failed_scenes"]=failures
            report["status"]="actual_four_scene_v9_CPU_evidence_audit_passed" if not failures else "actual_partial_v9_evidence_rejected"
            exit_code=int(bool(failures))
        except Exception as error:
            report["status"] = "actual_evidence_audit_failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            exit_code = 1
    report["reader"] = file_record(__file__)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "actual_scenes_audited": len(report["scenes"]),
                      "native_v9_execution_observed": report["native_v9_execution_observed"],
                      "route_validation": "pending" if report["routes"] is None else "CPU_Astar_only"}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
