"""Real USD CPU session/transform proof; native Timeline remains untested here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Usd, UsdGeom

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[2]
SPEC = importlib.util.spec_from_file_location("usd_clock_candidate", DOC / "probe_paper_scene_maps_candidate_v8.py")
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


class TimelineContract:
    """Documented setter interface only; no real Kit timeline is instantiated."""
    def __init__(self, stage):
        self.stage = stage

    def set_start_time(self, value):
        self.stage.SetStartTimeCode(value * self.stage.GetTimeCodesPerSecond())

    def set_end_time(self, value):
        self.stage.SetEndTimeCode(value * self.stage.GetTimeCodesPerSecond())

    def set_auto_update(self, value):
        assert value is False

    def set_current_time(self, value):
        assert value == 0.

    def get_start_time(self):
        return self.stage.GetStartTimeCode() / self.stage.GetTimeCodesPerSecond()

    def get_end_time(self):
        return self.stage.GetEndTimeCode() / self.stage.GetTimeCodesPerSecond()


def transforms(stage):
    paths = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Xformable)]
    records = []
    for time in (Usd.TimeCode.Default(), Usd.TimeCode(0.)):
        cache = UsdGeom.XformCache(time)
        digest = hashlib.sha256()
        for prim in paths:
            value = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64)
            assert np.isfinite(value).all()
            digest.update(str(prim.GetPath()).encode() + b"\0")
            digest.update(value.tobytes(order="C"))
        records.append({"time": str(time), "xformable_prims": len(paths), "path_and_matrix_C_order_SHA": digest.hexdigest()})
    return records


def main():
    assert not any(name in sys.modules for name in ("omni", "isaacsim", "isaaclab", "torch"))
    rows = []
    for scene in candidate.SCENES:
        preflight_path = ROOT / "output/scene_maps/native_reference_four_scenes_clock_obstacles_v7" / scene / "preflight.json"
        preflight = json.loads(preflight_path.read_text())
        records = [candidate.file_record(row["path"]) for row in preflight["layers"]]
        assert records == preflight["layers"]
        stage = Usd.Stage.Open(preflight["source"]["path"])
        original_range = [stage.GetStartTimeCode(), stage.GetEndTimeCode(), stage.GetTimeCodesPerSecond()]
        root_text = stage.GetRootLayer().ExportToString()
        before = transforms(stage)
        stage.SetEditTarget(stage.GetSessionLayer())
        override = candidate.freeze_query_timeline_range_in_session(stage, TimelineContract(stage))
        after = transforms(stage)
        assert before == after
        assert root_text == stage.GetRootLayer().ExportToString()
        assert records == [candidate.file_record(row["path"]) for row in preflight["layers"]]
        rows.append({"scene": scene, "preflight": candidate.file_record(preflight_path), "source": preflight["source"],
                     "source_original_composed_range_start_end_tcps": original_range,
                     "session_composed_range_start_end_tcps": [stage.GetStartTimeCode(), stage.GetEndTimeCode(), stage.GetTimeCodesPerSecond()],
                     "session_override": override, "source_root_in_memory_unchanged": True,
                     "source_layers_fullSHA_unchanged": True, "source_layers": records,
                     "default_and_zero_time_transforms_bitwise_equal": True, "transforms": before,
                     "real_Kit_Timeline_instantiated": False, "native_clock_fix_PASS_claim": False})
    assert not any(name in sys.modules for name in ("omni", "isaacsim", "isaaclab", "torch"))
    report = {"schema": "dynbench_v8_candidate_actual_USD_CPU_session_proof.v1", "CPU_only": True,
              "SDK_started": False, "GPU_used": False, "standalone_Usd_version": list(Usd.GetVersion()),
              "native_v8_executed": False, "real_native_clock_fix_observed": False,
              "sources": rows, "reader": candidate.file_record(__file__),
              "candidate": candidate.file_record(DOC / "probe_paper_scene_maps_candidate_v8.py")}
    output = DOC / "actual_four_scene_USD_session_proof.json"
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"proof": candidate.file_record(output), "source_scenes": len(rows),
                      "all_source_bytes_and_default_t0_transforms_unchanged": True,
                      "native_v8_executed": False}))


if __name__ == "__main__":
    main()
