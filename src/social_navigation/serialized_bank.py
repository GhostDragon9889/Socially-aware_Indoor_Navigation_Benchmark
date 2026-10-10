"""Reconstruct an immutable Torch container from verified tensor payloads.

The template contains ZIP structure and source metadata, never motion tensors.
Every payload and the complete resulting file must match the recorded SHA256.
This avoids dependence on output paths, pickle memo layout or ZIP writer details.
"""
import base64
import hashlib
import json
from pathlib import Path


def write_bank(template_path, tensors, target):
    document = json.loads(Path(template_path).read_text())
    if document.get('schema') != 'social_navigation.tensor_bank_template.v1':
        raise ValueError('Unsupported tensor bank template')
    target = Path(target)
    if target.exists():
        raise FileExistsError('Preserve existing bank: '+str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    with target.open('xb') as output:
        for piece in document['pieces']:
            if 'literal_base64' in piece:
                content = base64.b64decode(piece['literal_base64'], validate=True)
                output.write(content)
                checksum.update(content)
                continue
            value = tensors[piece['field']]
            if list(value.shape) != piece['shape'] or str(value.dtype) != piece['dtype']:
                raise ValueError('Fixed tensor schema changed: '+piece['field'])
            if value.device.type != 'cpu' or not value.is_contiguous():
                raise ValueError('Container writer requires contiguous CPU tensors')
            payload = memoryview(value.detach().numpy()).cast('B')
            if payload.nbytes != piece['bytes'] or hashlib.sha256(payload).hexdigest() != piece['sha256']:
                raise ValueError('Fixed tensor values changed: '+piece['field'])
            for start in range(0, payload.nbytes, 8 * 1024 * 1024):
                chunk = payload[start:start + 8 * 1024 * 1024]
                output.write(chunk)
                checksum.update(chunk)
    if target.stat().st_size != document['bytes'] or checksum.hexdigest() != document['sha256']:
        raise ValueError('Full reconstructed bank SHA differs; output retained')
    return {'path': str(target.resolve()), 'bytes': target.stat().st_size,
            'sha256': checksum.hexdigest(), 'all_tensor_payloads_verified': True}
