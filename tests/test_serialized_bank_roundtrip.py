"""Reconstruct a real torch.save container and reject tensor/metadata corruption."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile

import pytest
import torch

from social_navigation.serialized_bank import write_bank


@pytest.fixture
def torch_container(tmp_path):
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    stream = io.BytesIO(); torch.save({'pose': value}, stream)
    raw = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info, = [item for item in archive.infolist() if re.search(r'/data/0$', item.filename)]
    name_size, extra_size = struct.unpack_from('<HH', raw, info.header_offset + 26)
    offset = info.header_offset + 30 + name_size + extra_size
    payload = raw[offset:offset+info.file_size]
    assert payload == value.numpy().tobytes()
    document = {'schema': 'social_navigation.tensor_bank_template.v1', 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'pieces': [
                    {'literal_base64': base64.b64encode(raw[:offset]).decode()},
                    {'field': 'pose', 'shape': list(value.shape), 'dtype': str(value.dtype),
                     'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()},
                    {'literal_base64': base64.b64encode(raw[offset+len(payload):]).decode()}]}
    template = tmp_path / 'template.json'; template.write_text(json.dumps(document))
    return template, value, raw, document


def test_exact_torch_bank_is_loadable_by_sdk_torch(torch_container, tmp_path):
    template, value, raw, _ = torch_container
    target = tmp_path / 'new container 中文' / 'bank.pt'
    receipt = write_bank(template, {'pose': value}, target)
    assert target.read_bytes() == raw
    restored = torch.load(target, weights_only=True, map_location='cpu')
    torch.testing.assert_close(restored['pose'], value, rtol=0, atol=0)
    assert receipt['all_tensor_payloads_verified']
    with pytest.raises(FileExistsError, match='Preserve existing'):
        write_bank(template, {'pose': value}, target)
    assert target.read_bytes() == raw


@pytest.mark.parametrize('defect', ['shape', 'dtype', 'values', 'noncontiguous'])
def test_changed_tensor_is_rejected(torch_container, tmp_path, defect):
    template, value, _, _ = torch_container
    changed = {'shape': value.reshape(3, 2), 'dtype': value.double(),
               'values': value + 1, 'noncontiguous': value.transpose(0, 1).contiguous().transpose(0, 1)}[defect]
    with pytest.raises(ValueError, match='schema|values|contiguous'):
        write_bank(template, {'pose': changed}, tmp_path / 'rejected.pt')


@pytest.mark.parametrize('field', ['bytes', 'sha256'])
def test_full_container_identity_mismatch_retains_diagnostic_output(torch_container, tmp_path, field):
    template, value, raw, document = torch_container
    document[field] = len(raw)+1 if field == 'bytes' else '0'*64
    template.write_text(json.dumps(document))
    target = tmp_path / 'diagnostic.pt'
    with pytest.raises(ValueError, match='output retained'):
        write_bank(template, {'pose': value}, target)
    assert target.read_bytes() == raw


def test_unknown_template_is_rejected_before_output_creation(torch_container, tmp_path):
    template, value, _, document = torch_container
    document['schema'] = 'unknown'; template.write_text(json.dumps(document))
    target = tmp_path / 'must not create.pt'
    with pytest.raises(ValueError, match='Unsupported'):
        write_bank(template, {'pose': value}, target)
    assert not target.exists()
