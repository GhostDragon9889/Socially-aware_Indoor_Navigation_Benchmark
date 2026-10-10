from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,numpy as np,pytest
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('floor_evidence_fixtures',root/'tests/test_dynbench_scene_provider.py')
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
evidence=f.evidence

def test_explicit_floor_feature_does_not_admit_blocked_or_exterior_support(evidence):
    scene=f.admitted_fixture(evidence)
    before=scene._map.copy()
    scene.bind_native_query(SimpleNamespace(raycast_all=lambda *args:None),physics_step_count=lambda:9)
    points=np.array([[-.15,-.05],[-.05,-.05],[.05,-.05],[1.,1.]])
    scene.configure_policy_height_feature('reference_floor')
    np.testing.assert_array_equal(scene.height_at(points),np.zeros(4))
    np.testing.assert_array_equal(scene._map,before)
    assert scene.describe()['policy_height_padding_claims_walkable_support'] is False
    with pytest.raises(ValueError,match='finite reference floor'):
        scene.ground_height_at([[1.,1.]])
    with pytest.raises(ValueError,match='Unknown AMP'):
        scene.configure_policy_height_feature('silent_mode')
    scene.configure_policy_height_feature('native_obstacle_cap')
    np.testing.assert_array_equal(scene.height_at(points),[0.,1.8,1.8,1.8])
