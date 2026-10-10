"""Real standalone USD plus fake Timeline contracts; no native clock PASS."""
import importlib.util
from pathlib import Path
import sys

import pytest
from pxr import Sdf, Usd

DOC = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("clock_candidate", DOC / "probe_paper_scene_maps_candidate_v8.py")
candidate = importlib.util.module_from_spec(SPEC)
# dataclasses inspects the importing module; this is a CPU source import only.
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


class TimelineContract:
    def __init__(self, stage):
        self.stage = stage
        self.calls = []
        self.current = stage.GetStartTimeCode() / stage.GetTimeCodesPerSecond()

    def set_start_time(self, value):
        self.calls.append(("start", value))
        self.stage.SetStartTimeCode(value * self.stage.GetTimeCodesPerSecond())

    def set_end_time(self, value):
        self.calls.append(("end", value))
        self.stage.SetEndTimeCode(value * self.stage.GetTimeCodesPerSecond())

    def set_auto_update(self, value):
        self.calls.append(("auto", value))

    def set_current_time(self, value):
        self.calls.append(("current", value))
        self.current = value

    def get_start_time(self):
        return self.stage.GetStartTimeCode() / self.stage.GetTimeCodesPerSecond()

    def get_end_time(self):
        return self.stage.GetEndTimeCode() / self.stage.GetTimeCodesPerSecond()


@pytest.mark.parametrize("start,end,tcps", [(-1., 0., 60.), (0., 100., 60.), (0., 0., 60.), (-1., 0., 24.)])
def test_actual_usd_session_range_changes_composed_range_only(start, end, tcps):
    root = Sdf.Layer.CreateAnonymous("contract_source.usda")
    root.startTimeCode, root.endTimeCode, root.timeCodesPerSecond = start, end, tcps
    stage = Usd.Stage.Open(root)
    root_text = root.ExportToString()
    stage.SetEditTarget(stage.GetSessionLayer())
    timeline = TimelineContract(stage)
    result = candidate.freeze_query_timeline_range_in_session(stage, timeline)
    assert stage.GetStartTimeCode() == stage.GetSessionLayer().startTimeCode == 0.
    assert stage.GetEndTimeCode() == stage.GetSessionLayer().endTimeCode == max(end, tcps)
    assert root.ExportToString() == root_text
    assert (root.startTimeCode, root.endTimeCode, root.timeCodesPerSecond) == (start, end, tcps)
    assert result["original_source_composed_range"] == {"start_time_code": start, "end_time_code": end, "time_codes_per_second": tcps}
    assert result["actual_query_clock_pass_claim"] is False
    assert timeline.calls == [("start", 0.), ("end", max(end, tcps) / tcps), ("auto", False), ("current", 0.)]


def test_root_edit_target_rejected_before_any_mutation():
    root = Sdf.Layer.CreateAnonymous("contract_source.usda")
    root.startTimeCode, root.endTimeCode, root.timeCodesPerSecond = -1., 0., 60.
    stage = Usd.Stage.Open(root)
    original = root.ExportToString()
    timeline = TimelineContract(stage)
    with pytest.raises(ValueError, match="session layer"):
        candidate.freeze_query_timeline_range_in_session(stage, timeline)
    assert root.ExportToString() == original
    assert not timeline.calls


def test_static_stage_composition_and_default_geometry_unchanged():
    stage = Usd.Stage.CreateInMemory()
    stage.SetStartTimeCode(-1.)
    stage.SetEndTimeCode(0.)
    stage.SetTimeCodesPerSecond(60.)
    from pxr import UsdGeom
    from pxr import Gf
    prim = UsdGeom.Xform.Define(stage, "/Root/Object")
    prim.AddTranslateOp().Set(Gf.Vec3d(-6., -4., 2.))
    before = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim.GetPrim())
    source = stage.GetRootLayer().ExportToString()
    stage.SetEditTarget(stage.GetSessionLayer())
    candidate.freeze_query_timeline_range_in_session(stage, TimelineContract(stage))
    after = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim.GetPrim())
    assert before == after
    assert source == stage.GetRootLayer().ExportToString()


def test_source_candidate_preserves_existing_observation_rejection():
    with pytest.raises(ValueError, match="nonnegative"):
        candidate.observed_query_clock(lambda: 2, lambda: -1 / 60)
    assert candidate.observed_query_clock(lambda: 2, lambda: 0.) == {"submitted_physics_step": 2, "timeline_s": 0.}
