"""Protect the actual writer/renderer ownership boundary before full SDK exit."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from scripts.replay_paper_pedestrians import close_recorded_rgb


def fixtures(events, pending=False):
    rep = SimpleNamespace(
        BackendDispatch=SimpleNamespace(is_done_writing=lambda: not pending),
        orchestrator=SimpleNamespace(set_capture_on_play=lambda value: events.append(("capture", value))),
    )
    product = SimpleNamespace(destroy=lambda: events.append("product_destroy"))
    annotator = SimpleNamespace(detach=lambda products: events.append(("detach", products[0] is product)))
    timeline = SimpleNamespace(stop=lambda: events.append("stop"))
    app = SimpleNamespace(close=lambda **kwargs: events.append(("full_close", kwargs)))
    return app, (rep, product, annotator), timeline


def test_pending_writer_forbids_any_renderer_or_app_teardown():
    events = []
    app, capture, timeline = fixtures(events, pending=True)
    with pytest.raises(RuntimeError, match="Pending Replicator writes"):
        close_recorded_rgb(app, capture, timeline)
    assert events == []


def test_live_renderer_is_released_before_full_extension_shutdown(monkeypatch):
    events = []
    app, capture, timeline = fixtures(events)
    context = SimpleNamespace(get_stage=lambda: None, can_close_stage=lambda: True,
        close_stage=lambda: events.append("stage_close"))
    usd = ModuleType("omni.usd")
    usd.get_context = lambda: context
    usd.release_all_hydra_engines = lambda value: events.append(("hydra_release", value is context))
    omni = ModuleType("omni")
    omni.usd = usd
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    result = close_recorded_rgb(app, capture, timeline)
    assert events == [("capture", False), "stop", ("detach", True), "product_destroy",
        "stage_close", ("hydra_release", True),
        ("full_close", {"wait_for_replicator": False, "skip_cleanup": False})]
    assert result["app_close_returned"] and result["hydra_engines_released_before_app_shutdown"]
    assert result["skip_cleanup"] is False
