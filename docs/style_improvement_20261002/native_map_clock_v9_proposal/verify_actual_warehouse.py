"""Read completed Warehouse candidate through staged consumers; CPU only."""
from pathlib import Path
import importlib.util
import json
import sys

import CrowdSim

DOC = Path(__file__).resolve().parent
ROOT = Path(CrowdSim.__file__).resolve().parents[1]
PROPOSAL = DOC / 'proposed'


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    package = 'CrowdSim.humanoid.runtime.'
    verifier = source(package + 'native_map_v9', PROPOSAL / 'CrowdSim/humanoid/runtime/native_map_v9.py')
    provider_module = source(package + 'dynbench_scene', PROPOSAL / 'CrowdSim/humanoid/runtime/dynbench_scene.py')
    result = ROOT / 'output/scene_maps/native_warehouse_anonymous_root_v9_candidate_v1/Warehouse/result.json'
    data = json.loads(result.read_text())
    provider = provider_module.DynBenchSceneProvider.from_preflight(data['preflight']['path'], expected_sha256=data['preflight']['sha256'])
    provider.admit_native_map(result, expected_sha256=provider_module.file_record(result)['sha256'])
    checked = verifier.verify_v9(provider, data, admitted=True)
    report = {'status': 'CPU_recomputation_of_actual_Warehouse_candidate_passed', 'CPU_only': True, 'SDK_started_by_this_reader': False,
              'GPU_used_by_this_reader': False, 'result': provider_module.file_record(result), 'admission': checked,
              'candidate_only': True, 'four_scenes_native_verified': False}
    with (DOC / 'actual_Warehouse_candidate_admission.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'canonical_known5_cells': checked['canonical_known5_cells']}))


if __name__ == '__main__':
    main()
