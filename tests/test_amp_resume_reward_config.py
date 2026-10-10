"""CPU reward-identity regression; no Torch or simulator imports."""
import ast
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from CrowdSim.humanoid.application.configuration import (
    ConfigurationError,
    validate_resume_reward_config,
)
from CrowdSim.humanoid.application.settings import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def test_matching_objective_preserves_saved_configuration():
    saved = TrainingConfig(amp_reward_weight=.5, task_reward_weight=.5)
    before = asdict(saved)
    validate_resume_reward_config(before, saved)
    assert asdict(saved) == before


def test_omitted_requested_and_legacy_saved_fields_use_canonical_defaults():
    saved = TrainingConfig(**{})
    validate_resume_reward_config({}, saved)
    validate_resume_reward_config({"amp_reward_scale": 2.}, saved)


@pytest.mark.parametrize("field,requested", [
    ("amp_reward_scale", 1.),
    ("amp_reward_weight", .5),
    ("task_reward_weight", .25),
    ("direction_mode", "backwards_penalty"),
    ("amp_probability_epsilon", .001),
])
def test_every_reward_change_rejects_resume_without_mutating_saved(field, requested):
    saved = TrainingConfig()
    before = asdict(saved)
    profile = {field: requested}
    with pytest.raises(ConfigurationError, match=f"Resume reward {field} changed.*fresh explicit experiment"):
        validate_resume_reward_config(profile, saved)
    assert asdict(saved) == before


def _actual_driver_guard():
    tree = ast.parse((ROOT / "scripts/train_paper_amp.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    resume = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "args.resume")
    matching = []
    for branch in ast.walk(resume):
        if not isinstance(branch, ast.If) or ast.unparse(branch.test) != "not args.evaluate_steps":
            continue
        if len(branch.body) >= 2 and isinstance(branch.body[0], ast.ImportFrom):
            call = branch.body[1]
            if isinstance(call, ast.Expr) and isinstance(call.value, ast.Call) and ast.unparse(call.value.func) == "validate_resume_reward_config":
                assert ast.unparse(call.value.args[0]) == "profile['training']"
                assert ast.unparse(call.value.args[1]) == "trainer.config"
                # Keep the actual enclosing predicates and actual import/call.
                matching.append(ast.If(test=resume.test, body=[ast.If(test=branch.test, body=branch.body[:2], orelse=[])], orelse=[]))
    assert len(matching) == 1
    return compile(ast.fix_missing_locations(ast.Module(body=matching, type_ignores=[])), "actual_driver_resume_guard", "exec")


@pytest.mark.parametrize("resume,evaluate_steps,rejects", [
    ("old_weight.pt", 0, True),
    ("old_weight.pt", 60, False),
    (None, 0, False),
])
def test_actual_driver_guard_rejects_only_training_resume(resume, evaluate_steps, rejects):
    namespace = {"args": SimpleNamespace(resume=resume, evaluate_steps=evaluate_steps),
                 "profile": {"training": {"amp_reward_weight": .5}},
                 "trainer": SimpleNamespace(config=TrainingConfig(amp_reward_weight=2.))}
    if rejects:
        with pytest.raises(ConfigurationError, match="amp_reward_weight.*fresh explicit experiment"):
            exec(_actual_driver_guard(), namespace)
    else:
        exec(_actual_driver_guard(), namespace)
