"""Real SDK Torch sensor-network gradients; these are not native camera tests."""
import pytest
import torch

from crowdsim_ppo import ppo_policy as standalone_policy
from CrowdSim.ppo import ppo_policy as native_policy

pytestmark = pytest.mark.parametrize('policy', ['standalone', 'native'], indirect=True)


@pytest.fixture
def policy(request):
    return {'standalone': standalone_policy, 'native': native_policy}[request.param]


@pytest.mark.parametrize('encoder_name,channels', [('DepthEncoder', 1),
                                                  ('MapEncoder', 1),
                                                  ('RGBEncoder', 3)])
@pytest.mark.parametrize('channel_last', [False, True])
def test_shared_convolution_parameters_retain_all_layers_and_gradients(policy, encoder_name, channels, channel_last):
    encoder = getattr(policy, encoder_name)
    model = encoder(channels=(4, 8), kernel_sizes=(3,), strides=(1,), paddings=(0,), feature_dim=5)
    image = torch.randn(2, channels, 32, 32)
    if channel_last:
        image = image.permute(0, 2, 3, 1) if channels == 3 else image[:, 0]
    output = model(image)
    assert output.shape == (2, 5) and torch.isfinite(output).all()
    convolutions = [layer for layer in model.cnn if isinstance(layer, torch.nn.Conv2d)]
    assert len(convolutions) == 2 and convolutions[-1].out_channels == 8
    assert all(layer.padding == (0, 0) for layer in convolutions)
    (output * torch.arange(5)).sum().backward()
    assert convolutions[0].weight.grad is not None
    assert torch.isfinite(convolutions[0].weight.grad).all()
    assert convolutions[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('encoder_name', ['DepthEncoder', 'MapEncoder', 'RGBEncoder'])
@pytest.mark.parametrize('field', ['kernel_sizes', 'strides', 'paddings'])
def test_inconsistent_layer_parameters_fail_before_silent_truncation(policy, encoder_name, field):
    encoder = getattr(policy, encoder_name)
    with pytest.raises(ValueError, match='length'):
        encoder(channels=(4, 8), **{field: (1, 1, 1)})


@pytest.mark.parametrize('value', [0, '0,1', [0, 1], (0, 0)])
def test_zero_padding_is_valid_without_admitting_zero_network_width(policy, value):
    assert min(policy.normalize_paddings(value)) == 0
    with pytest.raises(ValueError, match='positive'):
        policy.normalize_dims(0)


@pytest.mark.parametrize('value', [[], -1, (-0.5,), ''])
def test_invalid_padding_is_rejected(policy, value):
    with pytest.raises(ValueError, match='nonnegative|integers'):
        policy.normalize_paddings(value)


def test_zero_padding_config_runs_fused_depth_map_network_and_real_adam_update(policy):
    kwargs = policy.robot_network_kwargs({
        'hidden_dims': [8],
        'depth_encoder': {'channels': [4, 8], 'paddings': [0], 'feature_dim': 5},
        'map_encoder': {'channels': [4, 8, 8], 'paddings': [0], 'feature_dim': 6},
        'attention': {'enabled': True, 'dim': 8},
    }, depth_size=32, map_size=32, num_neighbors=0)
    assert len(kwargs['depth_encoder_kernel_sizes']) == 2
    assert len(kwargs['map_encoder_kernel_sizes']) == 3
    model = policy.RobotActorCritic(obs_dim=8, action_dim=2, **kwargs)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    obs = torch.randn(3, 8); depth = torch.rand(3, 32, 32); map_patch = torch.rand(3, 32, 32)
    actions, raw, log_prob, values = model.act(obs, depth=depth, map_patch=map_patch)
    assert actions.shape == raw.shape == (3, 2)
    assert ((actions[:, 0] >= 0) & (actions[:, 0] <= 1)).all()
    assert actions[:, 1].abs().max() <= 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = (values - torch.arange(3)).square().mean() - log_prob.mean()
    loss.backward(); optimizer.step()
    for prefix in ('vector_encoder.', 'depth_encoder.', 'map_encoder.', 'fusion.', 'critic.'):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
                   if name.startswith(prefix))
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_generic_conv_encoder_supports_zero_padding_and_shared_specs(policy):
    model = policy.make_conv_encoder((4, 8), (3,), (1,), (0,))
    output = model(torch.randn(2, 1, 16, 16))
    assert output.shape == (2, 8 * 12 * 12)


def test_direct_encoder_defaults_adapt_to_custom_channel_count(policy):
    model = policy.DepthEncoder(depth_size=32, channels=(4, 8, 8), feature_dim=5)
    assert model(torch.rand(2, 32, 32)).shape == (2, 5)
