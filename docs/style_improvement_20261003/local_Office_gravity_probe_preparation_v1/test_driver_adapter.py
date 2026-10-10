"""One cold fake delegation; this test starts no actual SDK or driver."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

SPEC = importlib.util.spec_from_file_location("Office_driver_adapter_test", Path(__file__).with_name("driver_adapter.py"))
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_callback_forwards_unchanged_actor_and_wraps_only_existing_native_factory():
    calls = []
    returned = object()
    provider = SimpleNamespace(scene_name="Office", verify_live_runtime=lambda **kwargs: kwargs)
    scene = SimpleNamespace(provider=provider)
    def original_factory(description, *, device):
        return scene
    actor_factory = object()

    def existing_driver(**options):
        calls.append(options)
        assert options["evaluation_actor_factory"] is actor_factory
        assert options["native_scene_factory"] is not original_factory
        assert options["native_scene_factory"]({"scene": "Office"}, device="cuda:0") is scene
        return returned

    original = {"evaluation_actor_factory": actor_factory, "native_scene_factory": original_factory}
    assert TOOL.make_driver_callback(existing_driver)(**original) is returned
    assert len(calls) == 1 and original["native_scene_factory"] is original_factory
    assert provider._local_Office_gravity_observer_attached is True
    assert not any(name in {"torch", "isaacsim", "isaaclab"}
                   or name.startswith(("omni.", "isaacsim.", "isaaclab.")) for name in sys.modules)
