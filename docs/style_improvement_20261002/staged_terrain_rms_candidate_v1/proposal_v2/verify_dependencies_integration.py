"""Independent before/default CPU arithmetic and cold-import proof for staging."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

DOC = Path(__file__).resolve().parent
WORKSPACE = DOC.parents[3]


def record(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def without_new_policy(tree):
    """Remove precisely declared opt-in hooks; retain the complete old program."""
    class RemoveHooks(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "terrain_statistics" else node

        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and "terrain_statistics_policy" in ast.unparse(call.func.value)):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            targets = " ".join(ast.unparse(target) for target in node.targets)
            if "terrain_statistics_policy" in targets or targets == "terrain_reference_count":
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if "task_terrain_statistics_mode" in ast.unparse(node.test):
                return None
            return self.generic_visit(node)

    return RemoveHooks().visit(tree)


def same(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def run(output):
    from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, TrainingConfig
    checks = []

    def check(name, condition):
        if not condition:
            raise ValueError(name)
        checks.append(name)

    before = DOC / "before/CrowdSim/humanoid/training/trainer.py"
    staged = DOC / "proposed/CrowdSim/humanoid/training/trainer.py"
    check("complete legacy trainer AST unchanged after removing declared opt-in hooks",
          ast.dump(without_new_policy(ast.parse(staged.read_text())), include_attributes=False)
          == ast.dump(ast.parse(before.read_text()), include_attributes=False))
    namespace = {"__name__": "rms_frozen_before_trainer", "__package__": "CrowdSim.humanoid.training"}
    exec(compile(before.read_text(), str(before), "exec"), namespace)
    options = dict(self_dim=4, task_dim=7, amp_frame_dim=3,
                   task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8),
                   update_epochs=1, minibatch_size=4, discriminator_batch_size=4)
    old = namespace["PaperAMPTrainer"](TrainingConfig(**deepcopy(options)))
    new = PaperAMPTrainer(TrainingConfig(**deepcopy(options)))
    check("default initial model tensors bitwise before", same(old.model.state_dict(), new.model.state_dict()))
    own, task = torch.randn(2, 2, 4), torch.randn(2, 2, 7)
    old_action = old.act(own.flatten(0, 1), task.flatten(0, 1), deterministic=True)
    new_action = new.act(own.flatten(0, 1), task.flatten(0, 1), deterministic=True)
    check("default deterministic actions likelihood values bitwise before", same(old_action, new_action))
    decision = old.act(own.flatten(0, 1), task.flatten(0, 1))
    data = {key: decision[key].reshape(2, 2, -1).squeeze(-1)
            for key in ("actions", "applied_actions", "log_prob", "values")}
    data.update(self_obs=own, task_obs=task, amp_history=torch.randn(2, 2, 10, 3),
                next_values=torch.zeros(2, 2), rewards=torch.tensor([[.1, .2], [.3, .4]]),
                terminated=torch.zeros(2, 2, dtype=torch.bool), truncated=torch.zeros(2, 2, dtype=torch.bool))
    expert_history, expert_actions = torch.randn(4, 10, 3), torch.zeros(4, 69)
    rng = torch.get_rng_state()
    expected = old.update(data, expert_history, expert_actions)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = new.update(data, expert_history, expert_actions)
    check("actual default PPO discriminator metric scalars bitwise before", expected == actual)
    check("actual default updated model tensors bitwise before", same(old.model.state_dict(), new.model.state_dict()))
    check("actual default Adam tensors and counters bitwise before", same(old.optimizer.state_dict(), new.optimizer.state_dict()))
    check("actual default discriminator Adam bitwise before", same(old.discriminator_optimizer.state_dict(), new.discriminator_optimizer.state_dict()))
    for key in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
        check("actual default normalizer bitwise before: " + key, same(getattr(old, key).state_dict(), getattr(new, key).state_dict()))
    check("actual default CPU RNG consumption bitwise before", torch.equal(expected_rng, torch.get_rng_state()))
    package_root = Path(sys.modules["CrowdSim"].__file__).resolve().parent.parent
    code = """import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1]); sys.path.append(sys.argv[2])
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.application.resources import preflight_amp_resources, ResourceError
TrainingConfig(task_terrain_statistics_mode='preserve_declared_suffix').validate()
try:
    preflight_amp_resources(SimpleNamespace(seed=7,from_scratch=True,trajectory_source='diffusion'),
        {'training':{'task_terrain_statistics_mode':'preserve_declared_suffix'}})
except ResourceError as e:
    assert 'from-scratch' in str(e)
else: raise AssertionError('cold gate did not reject')
assert not any(name in sys.modules for name in ('torch','isaacsim','isaaclab','omni'))
print('cold_gate_without_Torch_or_SDK_PASS')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(package_root), os.environ["RMS_SOURCE_ROOT"]],
                            check=True, capture_output=True, text=True, env=os.environ.copy())
    check("protected from-scratch cold rejects without importing Torch or SDK",
          result.stdout.strip() == "cold_gate_without_Torch_or_SDK_PASS")
    report = {"schema": "staged_terrain_RMS_dependencies_CPU_integration_review.v2",
              "status": "CPU_before_default_arithmetic_and_cold_gate_passed",
              "checks_passed": len(checks), "checks": checks, "CPU_only": True,
              "CUDA_initialized": torch.cuda.is_initialized(), "production_mutated": False,
              "sources": {"before": record(before), "staged": record(staged), "reader": record(__file__)},
              "synthetic_observations_and_rollouts": True, "real_physics_or_GPU_claim": False,
              "CUDA_RNG_parity": "pending actual protected4+4 GPU pilot",
              "checkpoint_file_SHA_bitwise_claim": False,
              "scope": "The entire before trainer AST and an actual CPU optimizer update are compared. The added default config field changes metadata; no historical checkpoint file equality is claimed."}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
