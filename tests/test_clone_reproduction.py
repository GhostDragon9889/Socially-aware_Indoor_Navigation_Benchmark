import hashlib
import json
from pathlib import Path
import pytest
from social_navigation import resource_bindings as bindings

def registry(tmp_path, rows):
    path = tmp_path/'configs/reproduction/resource_bindings.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schema':'social_navigation.resource_bindings.v1','bindings':rows}))
    bindings._bindings.cache_clear()

def test_exact_location_view_preserves_hash_and_prim_paths(tmp_path):
    registry(tmp_path,[{'source':'/old/scene.usda','target':'data/scene.usda'}])
    raw = json.dumps({'file':'/old/scene.usda','prim':'/World/Scene','sha256':'abc','geometry':[1,2,3]})
    path = tmp_path/'receipt.json'; path.write_text(raw)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    view = bindings.relocate_document(json.loads(raw), root=tmp_path)
    assert view['file']==str(tmp_path/'data/scene.usda')
    assert view['prim']=='/World/Scene' and view['geometry']==[1,2,3] and view['sha256']=='abc'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert bindings.resolve_path('/unregistered/scene.usda',root=tmp_path)==Path('/unregistered/scene.usda')

@pytest.mark.parametrize('target',['../outside','/outside'])
def test_binding_escape_rejected(tmp_path,target):
    registry(tmp_path,[{'source':'/old','target':target}])
    with pytest.raises(ValueError,match='inside'):
        bindings.resolve_path('/old',root=tmp_path)

def test_duplicate_identity_rejected(tmp_path):
    registry(tmp_path,[{'source':'/old','target':'a'},{'source':'/old','target':'b'}])
    with pytest.raises(ValueError,match='Duplicate'):
        bindings.resolve_path('/old',root=tmp_path)

def test_explicit_licensed_dataset_mount(tmp_path):
    registry(tmp_path,[{'source':'/old/train.pt','target':'data/train.pt','resource_id':'amass_train'}])
    (tmp_path/'data').mkdir()
    (tmp_path/'data/external_resources.json').write_text(json.dumps({'amass_train':'/dataset/train.pt'}))
    assert bindings.resolve_path('/old/train.pt',root=tmp_path)==Path('/dataset/train.pt')

def test_private_lfs_fetch_detects_origin(monkeypatch,tmp_path):
    from social_navigation.asset_release import fetch
    calls=[]
    def output(command,**kw):
        return 'origin\n' if command[-1]=='remote' else 'https://github.com/GhostDragon9889/Socially-aware_Indoor_Navigation_Benchmark.git\n'
    monkeypatch.setattr('subprocess.check_output',output)
    monkeypatch.setattr('subprocess.run',lambda command,**kw:calls.append(command))
    fetch(tmp_path)
    assert 'origin' in calls[0]
    assert any('resources/pedestrian_visuals/**' in part for part in calls[0])


def test_unavailable_driver_fails_before_kit(monkeypatch):
    from social_navigation.sdk_paths import require_cuda_driver
    class Driver:
        def cuInit(self, flags):
            return 100
    monkeypatch.setattr('ctypes.CDLL',lambda name:Driver())
    with pytest.raises(RuntimeError,match='Kit was not started'):
        require_cuda_driver()


def test_sdk_literal_relocation_is_fully_reversible(monkeypatch,tmp_path):
    import CrowdSim.humanoid.runtime.relocated_terminal as terminal
    root = Path(__file__).resolve().parents[1]
    source = (root/'src/CrowdSim/humanoid/runtime/sealed/sdk6_terminal.py').read_text()
    original = terminal.relocate_sdk_path
    def relocated(value, *, root):
        return original(value,root=root).replace('/project/EXP/SangforAndSaut/IsaacLab-Protomotions',str(tmp_path/'SDK with spaces'))
    monkeypatch.setattr(terminal,'relocate_sdk_path',relocated)
    derived, proof = terminal.derive_relocated_checkout(source,root,tmp_path)
    assert proof['absolute_source_path_changes'] >= 14
    assert proof['exact_inverse'] and not proof['shutdown_operations_changed']
    compile(derived,'test_sealed_terminal','exec')


def test_scene_cfg_copy_retains_single_provider_owner():
    from copy import deepcopy
    from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
    owner = object.__new__(DynBenchSceneProvider)
    copied = deepcopy({'spawn':owner._spawn})
    assert copied['spawn'].__self__ is owner
    assert deepcopy(owner) is owner
