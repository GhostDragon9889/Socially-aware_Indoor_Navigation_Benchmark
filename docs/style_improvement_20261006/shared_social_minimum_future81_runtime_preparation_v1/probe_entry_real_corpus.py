"""Actual private entry/World source ABI over fixed2848 CPU tensors only."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
sys.path.insert(0, str(ROOT))
from scripts import train_shared_social_minimum_horizon as entry  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

def check(value, message):
    if not value:
        raise ValueError(message)


check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-hidden CUDA required')
output = Path(__file__).with_name('actual_entry_probe.json')
check(not output.exists(), 'Preserve old output')
module, binding = entry.prepare_entry()
check('torch' not in sys.modules, 'Cold entry preparation imported Torch')
rows = module._sources()
probe = module._load(module.PINS['probe_setup'][0], '_minimum_future81_actual_CPU_probe', rows['probe_setup'])
records = probe._sources()
check(records['World'] == binding['World'] and records['minimum_horizon_curriculum'] == binding['curriculum'], 'Caller probe source selection differs')
source = ROOT / 'docs/style_improvement_20261006/shared_social_minimum_future81_CPU_v1/actual_corpus_probe.json'
previous = json.loads(source.read_text())
inputs = [identity(source), *[row for row in previous['inputs'] if row['sha256'] in (
    '68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca',
    '46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0')]]
check(inputs[0]['sha256'] == 'd0f96612e4351f45374edc60cd825430c8f102a923540458c39d38682962f25f'
      and len(inputs) == 3 and all(identity(row['path']) == row for row in inputs), 'Fixed2848 evidence/data changed')
sources = [identity(Path(__file__)), *rows.values(), *records.values(), *binding['World_dependencies'].values()]
sources = {row['path']: row for row in sources}
import torch  # noqa: E402
from CrowdSim.humanoid.amass_corpus import AMASSCorpus  # noqa: E402
check(not torch.cuda.is_initialized(), 'CPU tensor probe initialized CUDA')
rng = torch.random.get_rng_state().clone()
world = probe._load(entry.WORLD, '_minimum_future81_actual_CPU_World', records['World'])
dependencies = world._dependencies()
check(world.DT == 1 / 120 and world.SUBSTEPS == 4 and world.HISTORY == 10 and world.REPLAN == 15,
      'Original control/history/replanning constants changed')
check(callable(dependencies.action_to_pd), 'Original named PD dependency not loaded')
package = ModuleType('_shared_social_original_training')
package.__path__ = [str(ROOT / 'CrowdSim/humanoid/training')]
check(package.__name__ not in sys.modules, 'Private source package already imported')
sys.modules[package.__name__] = package
for key, filename in (('procedural', 'trajectories'), ('amass', 'amass_trajectories'), ('mixed', 'mixed_trajectories')):
    row = binding['World_dependencies'][key]
    probe._load(Path(row['path']), package.__name__ + '.' + filename, row)
corpus = AMASSCorpus(ROOT / 'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
                     device='cpu', seed=7, dof_velocity_frame='child_joint')
original = sys.modules[package.__name__ + '.mixed_trajectories'].MixedTrajectoryCurriculum(corpus, 256,
    procedural_config=sys.modules[package.__name__ + '.trajectories'].PacerTrajectoryConfig(), seed=7)
report = {'actual_import_origins': {}, 'minimum_horizon_experiment': {}}
candidate = entry.replace_fresh_curriculum(corpus, original, {'actors': 256, 'seed': 7}, 'cpu', probe, records, report, torch)
check(candidate.corpus is original.corpus is corpus and corpus.ids.tolist() == list(range(2848)), 'Source ownership/IDs changed')
# Execute the actual pre-World type/source/config/fresh admission statements.
# This does not construct the physical World or simulate any body/contact.
init = world.SharedGroupWorld.__init__
tree = ast.parse(entry.WORLD.read_bytes())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SharedGroupWorld')
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
assignments = {node.targets[0].id: node for node in ast.walk(method) if isinstance(node, ast.Assign)
    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('curriculum_module', 'origin')}
conditions = [next(node for node in ast.walk(method) if isinstance(node, ast.If) and text in ast.unparse(node.test))
              for text in ('MinimumHorizonMixedCurriculum', 'curriculum.amass.initialized')]
statements = [deepcopy(assignments[key]) for key in ('curriculum_module', 'origin')] + [deepcopy(node) for node in conditions]
scope = {**init.__globals__, 'self': SimpleNamespace(num_envs=256, device=torch.device('cpu')), 'curriculum': candidate}
exec(compile(ast.Module(body=statements, type_ignores=[]), str(entry.WORLD), 'exec'), scope)
reference, _, _, ids, frames = candidate.sample_reference(torch.arange(256))
remaining = corpus.data['motion_num_frames'][ids[:128]] - 1 - frames[:128]
check(bool((remaining >= 81).all()), 'Private entry reset formula lacks81 real future frames')
check(not bool(original.amass.initialized.any()) and not any(original.statistics.values()), 'Unused baseline holder was advanced')
check(torch.equal(torch.random.get_rng_state(), rng) and not torch.cuda.is_initialized(), 'Global RNG/CUDA changed')
check(not any(name.split('.')[0] in {'isaacsim', 'isaaclab', 'omni', 'pxr', 'warp'} for name in sys.modules), 'SDK imported')
check(all(identity(row['path']) == row for row in inputs + list(sources.values())), 'Source/data changed during entry probe')
result = {'schema': 'shared_social.actual_minimum_future81_private_entry_CPU_probe.v1', 'status': 'passed_CPU_entry_interfaces',
    'inputs': inputs, 'sources': list(sources.values()), 'actors': 256, 'clips_retained': 2848, 'seed': 7,
    'binding': binding, 'actual_source_selection': records, 'actual_minimum_horizon_report': report,
    'World_dependency_namespace_loaded': True, 'original_PD_two_function_dependency_loaded': True,
    'actual_World_type_source_config_fresh_condition_passed': True,
    'original_physics_constants': {'physics_hz': 120, 'control_substeps': 4, 'history': 10, 'replan_controls': 15},
    'minimum_actual_sampled_remaining_frames': int(remaining.min()), 'unused_original_holder_advanced': False,
    'global_RNG_consumed': False, 'CUDA_initialized': False, 'NN_constructed': False, 'SDK_started': False,
    'actual_physical_World_created': False, 'training_executed': False, 'paper_effect_achieved': False,
    'scope': 'Actual private-package source/type and CPU reset interfaces; no SDK physics or locomotion evidence'}
with output.open('x') as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'result': identity(output), 'sources_verified': len(sources), 'clips_retained': 2848,
    'minimum_actual_sampled_remaining_frames': int(remaining.min()), 'SDK_started': False}))
