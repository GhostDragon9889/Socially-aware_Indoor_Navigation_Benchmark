"""Serial execution cannot admit omitted or mismatched current sources."""
from copy import deepcopy
from pathlib import Path

import pytest

from CrowdSim.humanoid.application import episode_pipeline_plan as plans
from CrowdSim.humanoid.runtime import social_observation_producer as observations
from CrowdSim.humanoid.runtime import evaluation_source_migration as migration


@pytest.fixture(scope='module')
def inventory():
    return plans.source_inventory()


def test_real_inventory_admits_complete_sources_and_rejects_every_omission(inventory):
    plans.require_inventory(inventory, inventory, 'current')
    assert len(inventory) > 30
    for index in range(len(inventory)):
        with pytest.raises(ValueError, match='incomplete'):
            plans.require_inventory(inventory[:index] + inventory[index + 1:], inventory, 'current')


@pytest.mark.parametrize('relative', [
    'src/protomotions/envs/base_env/env.py',
    'src/protomotions/simulator/isaaclab/simulator.py',
    'src/protomotions/simulator/isaaclab/utils/sdk61_owned_close.py',
])
def test_inventory_covers_production_shutdown_implementation(inventory, relative):
    expected = plans.release.record(plans.ROOT / relative)
    assert expected in inventory


@pytest.mark.parametrize('relative', [
    'src/protomotions/envs/base_env/env.py',
    'src/protomotions/simulator/isaaclab/simulator.py',
    'src/protomotions/simulator/isaaclab/utils/sdk61_owned_close.py',
])
def test_shutdown_source_change_invalidates_current_inventory(inventory, monkeypatch, relative):
    actual_record = plans.release.record

    def changed_record(path):
        row = actual_record(path)
        if Path(path) == plans.ROOT / relative:
            row['sha256'] = '0' * 64
        return row

    monkeypatch.setattr(plans.release, 'record', changed_record)
    with pytest.raises(ValueError, match='inventory identity changed'):
        plans.require_inventory(inventory, plans.source_inventory(), 'Current source')


@pytest.mark.parametrize('defect', ['empty', 'duplicate', 'sha', 'size', 'outside'])
def test_inventory_rejects_forged_sources(inventory, defect):
    altered = deepcopy(inventory)
    if defect == 'empty':
        altered = []
    elif defect == 'duplicate':
        altered.append(altered[0])
    elif defect == 'sha':
        altered[0]['sha256'] = '0' * 64
    elif defect == 'size':
        altered[0]['bytes'] += 1
    else:
        altered[0]['path'] = '/tmp/unregistered-source.py'
    with pytest.raises(ValueError, match='inventory'):
        plans.require_inventory(altered, inventory, 'current')


def test_cold_preflight_must_bind_the_admitted_inventory(inventory):
    with pytest.raises(ValueError, match='Cold preflight source'):
        plans.validate_preflight({'sources': inventory}, {'schema': 'shared_social.episode_cold_preflight.sim61.v2', 'steps': []})


@pytest.fixture
def cold_preflight(tmp_path):
    import json
    from scripts import train_shared_social_episode_after_recovery as training
    from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation
    serial = tmp_path / 'release.json'
    serial.write_text('{}')
    plan = {'sources': [], 'serial_release': plans.release.record(serial)}
    from social_navigation.external_runtime import selected_python
    executable = str(selected_python(plans.ROOT))
    commands = [[executable, str(Path(training.__file__)), '--serial-release', str(serial),
                 '--stage', stage, '--check-sources'] for stage in ('smoke8', 'smoke256', 'pilot2000')]
    commands.append([executable, str(Path(evaluation.__file__)), '--serial-release', str(serial), '--check-sources'])
    preflight = {'schema': 'shared_social.episode_cold_preflight.sim61.v2', 'sources': [], 'steps': []}
    for index, command in enumerate(commands):
        stdout, stderr = tmp_path / f'{index}.stdout', tmp_path / f'{index}.stderr'
        stdout.write_text(json.dumps({'Torch_imported': False, 'SDK_started': False}))
        stderr.write_text('')
        preflight['steps'].append({'command': command, 'exit_observation': 'subprocess.run.returncode',
                                  'actual_exit_code': 0, 'stdout': plans.release.record(stdout),
                                  'stderr': plans.release.record(stderr)})
    return plan, preflight


def test_cold_preflight_accepts_exact_commands_outputs_and_observed_exits(cold_preflight):
    plans.validate_preflight(*cold_preflight)


@pytest.mark.parametrize('defect', ['schema', 'command', 'observation', 'bool_exit', 'failed_exit',
                                  'missing_step', 'extra_step', 'stdout_tamper', 'stderr_tamper',
                                  'Torch_imported', 'SDK_started'])
def test_cold_preflight_rejects_forged_execution(cold_preflight, defect):
    import json
    plan, preflight = cold_preflight
    row = preflight['steps'][0]
    if defect == 'schema':
        preflight['schema'] = 'old'
    elif defect == 'command':
        row['command'].append('--skip-validation')
    elif defect == 'observation':
        row['exit_observation'] = 'inferred'
    elif defect in ('bool_exit', 'failed_exit'):
        row['actual_exit_code'] = False if defect == 'bool_exit' else 1
    elif defect == 'missing_step':
        preflight['steps'].pop()
    elif defect == 'extra_step':
        preflight['steps'].append(deepcopy(row))
    elif defect.endswith('_tamper'):
        Path(row[defect.removesuffix('_tamper')]['path']).write_text('changed')
    else:
        path = Path(row['stdout']['path'])
        document = json.loads(path.read_text())
        document[defect] = True
        path.write_text(json.dumps(document))
        row['stdout'] = plans.release.record(path)
    with pytest.raises(ValueError):
        plans.validate_preflight(plan, preflight)


def test_raw_observation_dependencies_bind_real_import_locations():
    sources = observations._sources()
    assert len(sources) == 9
    for name, pin in observations._parent.PINS.items():
        assert Path(pin[0]).is_absolute()
        assert Path(pin[0]).resolve() == Path(pin[0])
        assert sources[name]['path'] == pin[0]


def test_raw_observation_pin_migration_rejects_a_changed_original_declaration():
    from types import SimpleNamespace
    from ast import parse, Assign, literal_eval
    tree = parse(observations.PARENT.read_text())
    declaration = next(n for n in tree.body if isinstance(n, Assign) and n.targets[0].id == 'PINS')
    original = literal_eval(declaration.value)
    for bad in (original['world'][:3], (*original['world'][:3], True)):
        module = SimpleNamespace(PINS={**original, 'world': bad})
        with pytest.raises(ValueError, match='Unregistered raw observation'):
            migration.migrate_observation_pins(module)


def historical_simulator_namespace():
    """Use the sealed evaluator's declaration, not the mutable migration table."""
    import ast
    from types import SimpleNamespace

    path = plans.ROOT / 'docs/style_improvement_20261004/social_tracking_evaluation_staging_v1/evaluate_social_tracking.py'
    tree = ast.parse(path.read_text())
    declaration = next(node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'PINS' for target in node.targets))
    namespace = {'ROOT': plans.ROOT, 'STAGES': plans.ROOT / 'docs/style_improvement_20261004', 'Path': Path}
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(PINS={'simulator': namespace['PINS']['simulator']})


def test_simulator_admission_binds_the_complete_current_source():
    namespace = historical_simulator_namespace()
    historical = namespace.PINS['simulator']
    proof = migration.migrate_pins(namespace)
    current = migration.record(plans.ROOT / 'src/protomotions/simulator/isaaclab/simulator.py')

    assert namespace.PINS['simulator'] == (Path(current['path']), current['bytes'], current['sha256'])
    assert proof['changes']['simulator']['current'] == current
    assert proof['changes']['simulator']['historical']['bytes'] == historical[1]
    assert proof['changes']['simulator']['historical']['sha256'] == historical[2]
    assert proof['sealed_sources_modified'] is False
    assert proof['runtime_execution_verified'] is False


@pytest.mark.parametrize('field, changed', [('bytes', 1), ('sha256', '0' * 64)])
def test_simulator_admission_rejects_unregistered_current_source(monkeypatch, field, changed):
    namespace = historical_simulator_namespace()
    original = dict(namespace.PINS)
    actual_record = migration.record

    def changed_record(path):
        row = actual_record(path)
        if Path(path) == plans.ROOT / 'src/protomotions/simulator/isaaclab/simulator.py':
            row[field] = changed
        return row

    monkeypatch.setattr(migration, 'record', changed_record)
    with pytest.raises(ValueError, match='Registered evaluator dependency changed: simulator'):
        migration.migrate_pins(namespace)
    assert namespace.PINS == original


@pytest.mark.parametrize('index, changed', [(1, 1), (2, '0' * 64)])
def test_simulator_admission_rejects_changed_sealed_declaration(index, changed):
    namespace = historical_simulator_namespace()
    altered = list(namespace.PINS['simulator'])
    altered[index] = changed
    namespace.PINS['simulator'] = tuple(altered)

    with pytest.raises(ValueError, match='Unregistered evaluator dependency: simulator'):
        migration.migrate_pins(namespace)
