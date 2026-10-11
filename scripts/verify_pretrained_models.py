#!/usr/bin/env python3
"""Inventory every supplied model file and validate actual checkpoint tensors.

Current ProtoMotions model families additionally undergo strict architecture
loading. Archived references retain their declared branch compatibility status.
Tensor validity and architecture loading do not establish policy quality.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def tensors(value, prefix=''):
    import torch
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, child in value.items():
            yield from tensors(child, prefix+'/'+str(name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensors(child, prefix+'/'+str(index))


def audit(directory):
    import torch
    from protomotions.agents.common.config import PretrainedModelConfig
    from protomotions.agents.common.pretrained import load_pretrained_model_module
    from protomotions.utils.pretrained_compat import archived_upstream_reference
    torch.set_num_threads(4)
    report = {'schema':'social_navigation.pretrained_validation.v1', 'python':sys.executable,
              'directory':str(directory), 'files':[], 'policy_quality_verified':False,
              'all_functionalities_verified':False, 'failures':[]}
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        before = path.stat()
        row = {'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
               'bytes':before.st_size, 'sha256':digest(path)}
        archived = archived_upstream_reference(path, ROOT)
        if archived:
            row['compatibility'] = archived['status']
            row['reason'] = archived['reason']
        if path.suffix in ('.pth','.ckpt'):
            try:
                checkpoint = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                count, elements = 0, 0
                sentinels = []
                for name, tensor in tensors(checkpoint):
                    count += 1; elements += tensor.numel()
                    flat = tensor.reshape(-1)
                    for start in range(0,flat.numel(),4*1024*1024):
                        if not torch.isfinite(flat[start:start+4*1024*1024]).all():
                            # Lightning initializes an unfilled min-mode top-k
                            # checkpoint slot to +inf. It is callback bookkeeping,
                            # not a learned parameter or optimizer tensor.
                            if (name.startswith('/callbacks/ModelCheckpoint') and name.endswith('/kth_value')
                                    and "'mode': 'min'" in name and tensor.shape == torch.Size([])
                                    and torch.isposinf(tensor)):
                                sentinels.append(name)
                            else:
                                raise ValueError('Non-finite checkpoint tensor '+name)
                if not count: raise ValueError('No checkpoint tensors')
                row.update(tensors=count,elements=elements,learned_tensors_finite=True,
                           lightning_positive_infinity_sentinels=sentinels,restricted_load_passed=True)
                del checkpoint
                relative = path.relative_to(directory)
                if not archived and relative.parts[0] in ('motion_tracker','gpc_prior','masked_mimic'):
                    model = load_pretrained_model_module(PretrainedModelConfig(
                        checkpoint_path=str(path),module_path='',freeze=True),
                        device=torch.device('cpu'),prefer_inference_config=True)
                    row.update(strict_model_loaded=True,model_class=type(model).__module__+'.'+type(model).__name__)
                    del model
                row['validation_status'] = 'passed'
            except Exception as error:
                row.update(validation_status='failed',error=type(error).__name__+': '+str(error))
                report['failures'].append({'path':row['path'],'error':row['error']})
            gc.collect()
        if path.stat().st_size != before.st_size or path.stat().st_mtime_ns != before.st_mtime_ns:
            raise RuntimeError('Model file changed during verification: '+str(path))
        report['files'].append(row)
    report['inventoried_files'] = len(report['files'])
    report['checkpoints'] = sum(row['path'].endswith(('.pth','.ckpt')) for row in report['files'])
    report['inventory_scope_percent'] = 100
    report['checkpoint_validation_passed'] = not report['failures']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'data/pretrained_models')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Use a new output file')
    report=audit(args.directory.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('inventoried_files','checkpoints','checkpoint_validation_passed','failures')}))
    return 0 if report['checkpoint_validation_passed'] else 1


if __name__=='__main__': raise SystemExit(main())
