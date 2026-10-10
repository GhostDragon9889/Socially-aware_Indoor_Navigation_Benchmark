"""Read actual v7 evidence on CPU; registration is never a measured result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record
from CrowdSim.humanoid.runtime.native_map_v7 import PROBE_SHA256, is_v7, verify_v7

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
SCENES = ("Office", "Warehouse", "Full_Warehouse", "Hospital")
DEFAULT_NATIVE = ROOT / "output/scene_maps/native_reference_four_scenes_clock_obstacles_v7"


def registration(native_output, probe_snapshot):
    freeze = json.loads((DOC / "source_freeze.json").read_text())
    # Historical v7 bytes are bound to the immutable source archive. Explicit
    # later dispatch versions need not equal that historical mutable checkout.
    for record in (freeze["source_archive"], freeze["producer_readiness"]):
        if file_record(record["path"]) != record:
            raise ValueError("Frozen v7 audit source changed: " + record["path"])
    with tarfile.open(freeze["source_archive"]["path"], "r:gz") as archive:
        for record in freeze["source_files"]:
            member = str(Path(record["path"]).relative_to(ROOT))
            info = archive.getmember(member)
            payload = archive.extractfile(info)
            if not info.isfile() or payload is None:
                raise ValueError("Frozen v7 source archive member is not a file: " + member)
            content = payload.read()
            if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("Frozen v7 archive bytes changed: " + member)
    probe = file_record(probe_snapshot)
    if probe["sha256"] != PROBE_SHA256:
        raise ValueError("Actual four-scene reader requires the exact v7 producer snapshot")
    return {"schema": "native_four_scene_v7_CPU_reader_registration.v1",
            "status": "prepared_actual_results_pending", "CPU_only": True,
            "SDK_started": False, "GPU_used_by_reader": False,
            "native_v7_execution_observed": False, "measured_map_admission": None,
            "route_validation": None, "native_output": str(Path(native_output).resolve()),
            "scenes": list(SCENES), "consumer_freeze": file_record(DOC / "source_freeze.json"),
            "consumer_sources": freeze["source_files"],
            "consumer_sources_verified_from_immutable_archive": True,
            "current_checkout_consumer_sources": [file_record(record["path"]) for record in freeze["source_files"]],
            "producer_snapshot": probe,
            "producer_readiness": freeze["producer_readiness"], "reader": file_record(__file__)}


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
    result_path = Path(folder) / "result.json"
    data = json.loads(result_path.read_text())
    if data.get("scene") not in SCENES:
        raise ValueError("Unexpected scene result")
    if not is_v7(data) or data.get("CPU_contract_fixture_only") is True:
        raise ValueError("Older/failed or synthetic producer artifacts cannot be relabelled actual v7")
    preflight = data["preflight"]
    provider = DynBenchSceneProvider.from_preflight(
        preflight["path"], expected_sha256=preflight["sha256"], probe_script_snapshot=probe_snapshot)
    record = file_record(result_path)
    provider.admit_native_map(result_path, expected_sha256=record["sha256"])
    verified = verify_v7(provider, data, admitted=True)
    arrays = {"native_omap": array_inventory(data["native_omap"]),
              "canonical_world_positions": array_inventory(data["coordinate_validation"]["canonical_world_positions_artifact"]),
              "floor_all_hit_CSR": array_inventory(data["floor_support"]["artifact"]),
              "full_height_band_CSR": array_inventory(data["full_height_band_queries"]["artifact"])}
    clocks = {"height_queries": data["full_height_band_queries"]["clock_before"],
              "height_query_artifact": data["full_height_band_queries"]["clock_artifact"],
              "initial_query_clock": data["actual_query_clock_initial"]}
    return {"scene": provider.scene_name, "actual_result": record, "preflight": preflight,
            "status": "actual_native_v7_resource_admitted_by_CPU_reader", "admission": verified,
            "arrays": arrays, "observed_clocks": clocks, "floor_ownership": data["floor_ownership"],
            "derived_export_grid": data["derived_export_grid"],
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
    parser.add_argument("--probe-snapshot", type=Path, default=DOC / "actual_reader_preparation_v1/probe_paper_scene_maps.py")
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
              "native_v7_execution_observed": False, "status": "prepared_actual_results_pending",
              "scenes": [], "routes": None}
    exit_code = 0
    if not args.register_only:
        try:
            for scene in SCENES:
                row = audit_scene(args.native_output / scene, args.probe_snapshot)
                if row["scene"] != scene:
                    raise ValueError("Native output folder and source scene disagree")
                report["scenes"].append(row)
                report["native_v7_execution_observed"] = True
            report["routes"] = [audit_routes(path, report["scenes"]) for path in args.scene_config] or None
            report["status"] = "actual_four_scene_v7_CPU_evidence_audit_passed"
            report["native_v7_execution_observed"] = True
        except Exception as error:
            report["status"] = "actual_evidence_audit_failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            exit_code = 1
    report["reader"] = file_record(__file__)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "actual_scenes_audited": len(report["scenes"]),
                      "native_v7_execution_observed": report["native_v7_execution_observed"],
                      "route_validation": "pending" if report["routes"] is None else "CPU_Astar_only"}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
