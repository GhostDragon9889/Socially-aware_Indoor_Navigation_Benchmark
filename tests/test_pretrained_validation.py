"""Validate finite weights while retaining Lightning's documented top-k sentinel."""
import torch
import pytest
from scripts.verify_pretrained_models import audit


@pytest.mark.parametrize('sentinel, mode, passed', [(float('inf'),'min',True),
                                                  (float('nan'),'min',False),
                                                  (float('inf'),'max',False)])
def test_callback_sentinel_is_explicit_and_does_not_mask_nan(tmp_path,sentinel,mode,passed):
    payload={'state_dict':{'weight':torch.ones(2)},
             'callbacks':{f"ModelCheckpoint{{'mode': '{mode}'}}":{'kth_value':torch.tensor(sentinel)}}}
    torch.save(payload,tmp_path/'model.ckpt')
    result=audit(tmp_path)
    assert result['checkpoint_validation_passed'] is passed
    assert result['checkpoints']==1
    assert result['inventoried_files']==1


def test_infinity_in_learned_weight_is_rejected(tmp_path):
    torch.save({'model':{'weight':torch.tensor([float('inf')])}},tmp_path/'model.ckpt')
    result=audit(tmp_path)
    assert not result['checkpoint_validation_passed']
    assert 'Non-finite checkpoint tensor /model/weight' in result['failures'][0]['error']
