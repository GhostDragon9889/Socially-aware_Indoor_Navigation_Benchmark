"""Owned, byte-verified actor/RMS predicates shared by both Pop-Art formats."""
import ast
from types import ModuleType

from . import social_popart_training as training
from .social_evaluation_lifecycle import record

ACTOR = training.ROOT / "resources/runtime_sources/style_improvement_20261004/social_deployment_integration_v1/inference_adapter.py"
ACTOR_SHA = "b81f5c5670c7e3cb893dfc27b8632a211c82cdb4a39ff2ce88a2f2dcea90127e"
HELPER = training.ROOT / "resources/runtime_sources/style_improvement_20261004/social_value_adapter_staging_v1/value_adapter.py"
HELPER_SHA = "11d24ca5e598307e112d53fb3abd24c19141bdb359cdf8926629443598ca1304"


def make_actor():
    before = record(ACTOR), record(HELPER)
    if (before[0]["sha256"] != ACTOR_SHA or before[1]["sha256"] != HELPER_SHA
            or ACTOR.stat().st_mode & 0o222 or HELPER.stat().st_mode & 0o222):
        raise ValueError("Frozen actor/RMS predicate source changed")
    actor = ModuleType("_owned_popart_actor_snapshot")
    actor.__file__ = str(ACTOR)
    exec(compile(ACTOR.read_bytes(), str(ACTOR), "exec"), actor.__dict__)
    helper = ModuleType("_owned_popart_RMS_snapshot")
    helper.__file__ = str(HELPER)
    exec(compile(HELPER.read_bytes(), str(HELPER), "exec"), helper.__dict__)
    historical = record(training.NUMERICAL_PARENT)
    if historical["sha256"] != training.NUMERICAL_PARENT_SHA:
        raise ValueError("Historical RMS implementation changed")
    helper.PINS = {**helper.PINS, "normalizer": (str(training.NUMERICAL_PARENT),
                    historical["bytes"], historical["sha256"])}

    def owned_helper():
        if (record(ACTOR), record(HELPER)) != before:
            raise ValueError("Actor/RMS predicates changed after admission")
        current_path = training.ROOT / "CrowdSim/humanoid/training/trainer.py"
        if record(current_path)["sha256"] != training.CURRENT_SOURCES["CrowdSim/humanoid/training/trainer.py"]:
            raise ValueError("Current normalizer source changed")
        def normalizer(source):
            nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RunningNormalizer"]
            if len(nodes) != 1:
                raise ValueError("Require exactly one population normalizer")
            return ast.dump(nodes[0], include_attributes=False)
        if normalizer(current_path.read_text()) != normalizer(training.NUMERICAL_PARENT.read_text()):
            raise ValueError("Population normalizer behavior changed")
        helper._sources()
        return helper

    actor._helper = owned_helper
    return actor
