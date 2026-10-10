"""One CPU source/ABI test; fake delegation is never actual Office evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

SPEC = importlib.util.spec_from_file_location("baseline_Office_adapter_test", Path(__file__).with_name("driver_adapter.py"))
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_baseline_identity_no_newreward_admission_and_original_callback(monkeypatch):
    original_record = TOOL.record

    def selected_only(path):
        assert "pacer_reward_isolated_snapshots" not in str(path)
        assert "terminal2000/audit.json" not in str(path)
        assert "local_Office3_baseline62c4_gravity10s_v1" not in str(path)
        return original_record(path)

    monkeypatch.setattr(TOOL, "record", selected_only)
    plan, _ = TOOL.prepare_sources()
    assert plan["selected_checkpoint"]["sha256"] == "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
    assert plan["selected_checkpoint"]["bytes"] == 651850375
    assert plan["new09e_reward_terminal_schema_used_for62c4"] is False
    assert plan["profile"]["path"].endswith("amp_style_amass_backward_penalty.yaml")
    assert plan["expected_counts"]["controls"] == 300 and plan["expected_counts"]["actors"] == 3
    actor = object()
    result = object()
    calls = []
    provider = SimpleNamespace(scene_name="Office", verify_live_runtime=lambda **kwargs: kwargs)
    scene = SimpleNamespace(provider=provider)

    def factory(description, *, device):
        return scene

    def driver(**options):
        calls.append(options)
        assert options["evaluation_actor_factory"] is actor
        assert options["native_scene_factory"]({"scene": "Office"}, device="cuda:0") is scene
        return result

    options = {"evaluation_actor_factory": actor, "native_scene_factory": factory}
    assert TOOL.make_driver_callback(plan, driver)(**options) is result
    assert len(calls) == 1 and options["native_scene_factory"] is factory
    assert not any(name in {"torch", "isaacsim", "isaaclab"}
                   or name.startswith(("omni.", "isaacsim.", "isaaclab.")) for name in sys.modules)
