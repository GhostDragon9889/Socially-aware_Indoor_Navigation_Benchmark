"""Real USD and declared fake SDK interfaces; no actual Kit/V9 pass is claimed."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

import pytest
from pxr import Gf, Usd, UsdGeom, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('clock_v9_CPU_candidate', ROOT / 'scripts/probe_paper_scene_maps.py')
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


def fixture(tmp_path, *, start=-1., end=0.):
    original = Usd.Stage.CreateInMemory()
    original.SetStartTimeCode(start)
    original.SetEndTimeCode(end)
    original.SetTimeCodesPerSecond(60.)
    original.SetFramesPerSecond(60.)
    UsdGeom.SetStageUpAxis(original, 'Z')
    UsdGeom.SetStageMetersPerUnit(original, 1.)
    prim = UsdGeom.Xform.Define(original, '/Root')
    prim.AddTranslateOp().Set(Gf.Vec3d(-6., 4., 2.))
    original.SetDefaultPrim(prim.GetPrim())
    path = tmp_path / 'source.usda'
    original.GetRootLayer().Export(str(path))
    before = path.read_bytes()
    stage, layer, info = candidate.compose_owned_anonymous_runtime_stage(path, Usd.Stage.CreateInMemory)
    return path, before, stage, layer, info


@pytest.mark.parametrize('start,end', [(-1., 0.), (0., 0.), (0., 100.)])
def test_real_USD_runtime_root_owns_range_and_copies_all_noninherited_stage_metadata(tmp_path, start, end):
    path, before, stage, original, info = fixture(tmp_path, start=start, end=end)
    assert stage.GetRootLayer().anonymous and stage.GetRootLayer().subLayerPaths == [str(path)]
    assert stage.GetStartTimeCode() == 0. and stage.GetEndTimeCode() == max(end, 60.)
    assert stage.GetTimeCodesPerSecond() == stage.GetFramesPerSecond() == 60.
    assert UsdGeom.GetStageUpAxis(stage) == 'Z' and UsdGeom.GetStageMetersPerUnit(stage) == 1.
    assert str(stage.GetDefaultPrim().GetPath()) == '/Root'
    preview = Usd.Stage.Open(original)
    for code in (Usd.TimeCode.Default(), Usd.TimeCode(0.)):
        a, b = UsdGeom.XformCache(code), UsdGeom.XformCache(code)
        assert a.GetLocalToWorldTransform(preview.GetPrimAtPath('/Root')) == b.GetLocalToWorldTransform(stage.GetPrimAtPath('/Root'))
    assert path.read_bytes() == before
    assert hashlib.sha256(original.ExportToString().encode()).hexdigest() == info['original_source_root_in_memory_sha256']
    assert (preview.GetStartTimeCode(), preview.GetEndTimeCode()) == (start, end)
    assert stage.GetEditTarget().GetLayer() == stage.GetSessionLayer()


def test_file_backed_root_factory_rejected(tmp_path):
    path, _, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='fresh owned anonymous'):
        candidate.compose_owned_anonymous_runtime_stage(path, lambda: Usd.Stage.Open(str(path)))


class TimelineContract:
    def __init__(self, stage):
        self.stage = stage
        self.pending = []

    def set_start_time(self, value):
        self.stage.SetStartTimeCode(value * self.stage.GetTimeCodesPerSecond())
        self.pending.append(('startTimeCode', value * self.stage.GetTimeCodesPerSecond()))

    def set_end_time(self, value):
        self.stage.SetEndTimeCode(value * self.stage.GetTimeCodesPerSecond())
        self.pending.append(('endTimeCode', value * self.stage.GetTimeCodesPerSecond()))

    def set_auto_update(self, value):
        assert value is False

    def set_current_time(self, value):
        assert value == 0.

    def get_start_time(self):
        return self.stage.GetStartTimeCode() / self.stage.GetTimeCodesPerSecond()

    def get_end_time(self):
        return self.stage.GetEndTimeCode() / self.stage.GetTimeCodesPerSecond()

    def simulate_async_current_root_range_write(self):
        for key, value in self.pending:
            setattr(self.stage.GetRootLayer(), key, value)


def test_actual_USD_async_root_metadata_write_stays_in_own_layer_and_protects_original(tmp_path):
    path, before, stage, original, info = fixture(tmp_path)
    timeline = TimelineContract(stage)
    record = candidate.freeze_owned_runtime_query_range(stage, timeline, original, info)
    timeline.simulate_async_current_root_range_write()
    assert path.read_bytes() == before
    assert original.startTimeCode == -1. and original.endTimeCode == 0.
    assert stage.GetRootLayer().startTimeCode == 0. and stage.GetRootLayer().endTimeCode == 60.
    assert hashlib.sha256(original.ExportToString().encode()).hexdigest() == record['original_source_root_in_memory_sha256']
    assert record['actual_query_clock_pass_claim'] is False


def test_original_source_RAM_change_is_not_accepted(tmp_path):
    _, _, stage, original, info = fixture(tmp_path)
    original.endTimeCode = 60.
    with pytest.raises(ValueError, match='source root RAM changed before'):
        candidate.freeze_owned_runtime_query_range(stage, TimelineContract(stage), original, info)


def test_observed_async_context_cache_ownership_and_submitted_counter_contract(tmp_path):
    _, _, stage, _, _ = fixture(tmp_path)
    cache = UsdUtils.StageCache.Get()
    cache.Insert(stage)
    class Context:
        active = None
        def get_stage(self):
            return self.active
    context = Context()
    class Sim:
        def get_physics_step_count(self):
            return 0
        def set_setting(self, name, value):
            assert name == '/app/player/playSimulations' and value is False
    class App:
        def update(self):
            context.active = stage
    result = candidate.await_owned_stage_attachment(App(), Sim(), stage, context, deadline=time.monotonic()+10)
    assert result['observed_context_owns_exact_runtime_stage'] is True
    assert result['app_updates_waiting_for_async_attachment'] == 1
    assert result['submitted_step_counter_before'] == result['submitted_step_counter_after'] == 0
    assert result['explicit_physics_steps_issued_by_wait'] == 0 and result['native_engine_time_observed'] is None
    assert result['manual_context_or_PhysX_attach_called'] is False


def test_attachment_wait_rejects_a_submitted_counter_change(tmp_path):
    _, _, stage, _, _ = fixture(tmp_path)
    UsdUtils.StageCache.Get().Insert(stage)
    class Sim:
        counter = 0
        def get_physics_step_count(self):
            return self.counter
        def set_setting(self, name, value):
            pass
    sim = Sim()
    class Context:
        def get_stage(self):
            return None
    class App:
        def update(self):
            sim.counter += 1
    with pytest.raises(ValueError, match='counter advanced'):
        candidate.await_owned_stage_attachment(App(), sim, stage, Context(), deadline=time.monotonic()+10)


def test_attachment_wait_rejects_missing_cache_ownership(tmp_path):
    _, _, stage, _, _ = fixture(tmp_path)
    class Sim:
        def get_physics_step_count(self):
            return 0
    with pytest.raises(ValueError, match='stage cache'):
        candidate.await_owned_stage_attachment(None, Sim(), stage, None, deadline=time.monotonic()+10)
