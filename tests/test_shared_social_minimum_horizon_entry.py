from social_navigation.artifact_identity import artifact_path
"""CPU source/type/profile contracts; these fixtures do not simulate PhysX."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts import train_shared_social_minimum_horizon as entry


def test_World_derivation_is_exactly_reversible_to_original_physics_source():
    derivation = json.loads((entry.HERE / "source_derivation.json").read_text())
    original, candidate = (artifact_path(derivation[key]["path"]).read_bytes() for key in ("original_World", "candidate_World"))
    for edit in reversed(derivation["byte_edits"]):
        assert candidate.count(edit["after"].encode()) == 1
        candidate = candidate.replace(edit["after"].encode(), edit["before"].encode())
    assert candidate == original


@pytest.fixture
def prepared(tmp_path):
    name = "_shared_social_original_training"
    assert name not in sys.modules
    module, binding = entry.prepare_entry()
    rows = module._sources()
    probe_name = "_minimum_future81_contract_probe"
    probe = module._load(module.PINS["probe_setup"][0], probe_name, rows["probe_setup"])
    package = ModuleType(name)
    package.__path__ = [str(entry.ROOT / "CrowdSim/humanoid/training")]
    sys.modules[name] = package
    try:
        for key, filename in (("procedural", "trajectories"), ("amass", "amass_trajectories"), ("mixed", "mixed_trajectories")):
            probe._load(Path(binding["World_dependencies"][key]["path"]), name + "." + filename, binding["World_dependencies"][key])
        path = Path(__file__).with_name("test_amass_root_curriculum.py")
        spec = importlib.util.spec_from_file_location("minimum_runtime_CPU_data_fixture", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        corpus = helper.corpus(tmp_path, counts=(91, 169))
        base = sys.modules[name + ".mixed_trajectories"].MixedTrajectoryCurriculum(corpus, 4,
            procedural_config=sys.modules[name + ".trajectories"].PacerTrajectoryConfig())
        report = {"actual_import_origins": {}, "minimum_horizon_experiment": {}}
        records = probe._sources()
        candidate = entry.replace_fresh_curriculum(corpus, base, {"actors": 4, "seed": 7}, "cpu", probe, records, report, torch)
        yield module, probe, corpus, base, candidate, report
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + ".") or key == probe_name:
                del sys.modules[key]


def world_guard():
    world = next(node for node in ast.parse(entry.WORLD.read_bytes()).body if isinstance(node, ast.ClassDef)
                 and node.name == "SharedGroupWorld")
    init = next(node for node in world.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignments = {node.targets[0].id: node for node in ast.walk(init) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("curriculum_module", "origin")}
    type_guard = next(node for node in ast.walk(init) if isinstance(node, ast.If)
                      and "MinimumHorizonMixedCurriculum" in ast.unparse(node.test))
    fresh_guard = next(node for node in ast.walk(init) if isinstance(node, ast.If)
                      and "curriculum.amass.initialized" in ast.unparse(node.test))
    function = ast.parse("def guard(self, curriculum):\n    return True\n").body[0]
    function.body = [deepcopy(assignments[key]) for key in ("curriculum_module", "origin")]
    function.body += [deepcopy(type_guard), deepcopy(fresh_guard), ast.Return(ast.Constant(True))]
    namespace = {"sys": sys, "Path": Path, "PINS": {"minimum_horizon": (entry.CURRICULUM,)}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(entry.WORLD), "exec"), namespace)
    return namespace["guard"]


def test_actual_private_package_replacement_and_World_type_source_admission(prepared):
    _, _, corpus, original, candidate, report = prepared
    assert candidate.corpus is original.corpus is corpus
    assert type(candidate).__module__ == "_shared_social_original_training.minimum_horizon_curriculum"
    assert world_guard()(SimpleNamespace(num_envs=4, device=torch.device("cpu")), candidate)
    assert report["actual_import_origins"]["minimum_horizon_curriculum"]["path"] == str(entry.CURRICULUM)
    assert report["minimum_horizon_experiment"]["actual_reset_configuration"]["real_reset_future_frames"] == 81
    assert original.config["real_reset_future_frames"] == 31


@pytest.mark.parametrize("bad", ("original_type", "real_future", "AMASS_future", "actors", "device", "origin", "already_bound"))
def test_World_actual_admission_condition_rejects_wrong_type_source_or_configuration(prepared, monkeypatch, bad):
    _, _, _, original, candidate, _ = prepared
    self = SimpleNamespace(num_envs=4, device=torch.device("cpu"))
    if bad == "original_type":
        candidate = original
    elif bad == "real_future":
        candidate.config["real_reset_future_frames"] = 31
    elif bad == "AMASS_future":
        candidate.amass.config["future_frames"] = 31
    elif bad == "actors":
        self.num_envs = 8
    elif bad == "device":
        self.device = torch.device("cuda:0")
    elif bad == "origin":
        monkeypatch.setattr(sys.modules[type(candidate).__module__], "__file__", str(entry.canonical.__file__))
    else:
        candidate.amass.initialized[0] = True
    with pytest.raises(ValueError):
        world_guard()(self, candidate)


def test_replacing_a_used_curriculum_fails_before_reimport_or_RNG_reset(prepared):
    _, probe, corpus, original, _, report = prepared
    original.statistics["amass_reference_samples"] = 1
    rng = corpus.generator.get_state().clone()
    with pytest.raises(ValueError, match="unused fresh"):
        entry.replace_fresh_curriculum(corpus, original, {"actors": 4, "seed": 7}, "cpu", probe, probe._sources(), report, torch)
    assert torch.equal(corpus.generator.get_state(), rng)


@pytest.mark.parametrize("name", entry.PROFILE_SHAS)
def test_registered_profiles_retain_original_training_and_runtime_fields(name):
    module, _ = entry.prepare_entry()
    profile, _ = module._profile(entry.PROFILE_DIR / name)
    assert profile["runtime"]["actors"] == (8 if name == "smoke.json" else 256)
    assert profile["runtime"]["updates"] == (2000 if name == "training.json" else 1)
    assert profile["runtime"]["rollout_controls"] == (32 if name == "training.json" else 4)
    assert profile["training"]["update_epochs"] == 6
    assert profile["training"]["direction_mode"] == "backwards_penalty"



def test_identical_profile_at_unregistered_location_is_rejected(tmp_path):
    module, _ = entry.prepare_entry()
    foreign = tmp_path / "training.json"
    foreign.write_bytes((entry.PROFILE_DIR / "training.json").read_bytes())
    with pytest.raises(ValueError, match="exact registered"):
        module._profile(foreign)


def subprocess_code(code):
    result = subprocess.run([sys.executable, "-I", "-c", "import sys;sys.path.insert(0," + repr(str(entry.ROOT)) + ");" + code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_fresh_source_preflight_imports_neither_Torch_nor_SDK():
    report = json.loads(subprocess_code("from scripts import train_shared_social_minimum_horizon as e;e.main(['--check-sources'])"))
    assert report["Torch_imported"] is False and report["SDK_started"] is False
    assert report["binding"]["World"]["sha256"] == entry.WORLD_SHA
    assert report["entry_sources"]["collector"]["sha256"] == "b207ab872683b023529785498f745c4149bd44b3c116c6fd15e5fbbd72d153be"



def test_fresh_training_requires_explicit_output_before_sdk_start():
    code = "from scripts import train_shared_social_minimum_horizon as e;" + "\ntry:e.main(['--config',str(e.PROFILE_DIR/'smoke.json')])\nexcept SystemExit as error:assert error.code==2\nelse:raise AssertionError('missing output accepted')\nassert not any(name.split('.')[0] in {'torch','isaacsim','isaaclab','omni','pxr','warp'} for name in sys.modules)"
    subprocess_code(code)
