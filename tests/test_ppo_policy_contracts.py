"""Exercise policy channels, buffer admission, and checkpoint compatibility."""

from copy import deepcopy
import importlib

import pytest
import torch


@pytest.fixture(params=["crowdsim_ppo.ppo_policy", "CrowdSim.ppo.ppo_policy"])
def module(request):
    return importlib.import_module(request.param)


def test_multimodal_attention_fills_absent_channels_exactly_as_zero(module):
    model = module.RobotActorCritic(
        obs_dim=5,
        action_dim=2,
        hidden_dims=(8,),
        num_neighbors=1,
        depth_enabled=True,
        depth_size=8,
        map_enabled=True,
        map_size=8,
        depth_encoder_channels=(2,),
        depth_encoder_kernel_sizes=(3,),
        depth_encoder_strides=(2,),
        depth_encoder_paddings=(1,),
        depth_feature_dim=8,
        map_encoder_channels=(2,),
        map_encoder_kernel_sizes=(3,),
        map_encoder_strides=(2,),
        map_encoder_paddings=(1,),
        map_feature_dim=8,
        attn_enabled=True,
        attn_dim=8,
    ).eval()
    obs = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10
    neighbors, mask = torch.ones(2, 1, 5), torch.ones(2, 1, dtype=torch.bool)
    with torch.no_grad():
        absent = model.encode(obs, neighbors=neighbors, neighbor_mask=mask)
        empty = model.encode(
            obs,
            depth=torch.empty(0, 8, 8),
            map_patch=torch.empty(0, 8, 8),
            neighbors=neighbors,
            neighbor_mask=mask,
        )
        zeros = model.encode(
            obs,
            depth=torch.zeros(2, 8, 8),
            map_patch=torch.zeros(2, 8, 8),
            neighbors=neighbors,
            neighbor_mask=mask,
        )
    assert absent.shape == (2, 24) and torch.isfinite(absent).all()
    torch.testing.assert_close(absent, empty)
    torch.testing.assert_close(absent, zeros)


def test_neighbor_shape_and_required_channel_validation(module):
    model = module.RobotActorCritic(
        5, 2, hidden_dims=(8,), num_neighbors=1, depth_enabled=False, map_enabled=False
    )
    obs = torch.zeros(2, 5)
    with pytest.raises(ValueError, match="required"):
        model.encode(obs)
    with pytest.raises(ValueError, match="Expected neighbors"):
        model.encode(
            obs,
            neighbors=torch.zeros(2, 2, 5),
            neighbor_mask=torch.ones(2, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="Expected neighbors"):
        module.flatten_neighbor_observation(
            obs, torch.zeros(2, 1, 5), torch.ones(2, 2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="same batch size"):
        module.flatten_neighbor_observation(
            obs, torch.zeros(3, 1, 5), torch.ones(3, 1, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        module.RobotActorCritic(5, 2, num_neighbors=-1)
    with pytest.raises(ValueError, match="robot action"):
        module.bounded_robot_action(torch.zeros(2, 3))


def test_config_helpers_and_convolution_execute_real_shapes(module):
    dimensions = module.normalize_dims("4, 8, ,")
    assert dimensions == (4, 8)
    network = module.make_mlp(5, dimensions, output_dim=2)
    assert network(torch.zeros(2, 5)).shape == (2, 2)
    with pytest.raises(ValueError, match="Network dimensions must be positive"):
        module.normalize_dims("4, 0")
    kwargs = module.robot_network_kwargs(
        {"map_encoder": None, "depth_encoder": [], "attention": "off"},
        hidden_dim_override=8,
        num_layers_override=2,
    )
    assert (
        kwargs["hidden_dims"] == (8, 8)
        and kwargs["depth_enabled"]
        and not kwargs["attn_enabled"]
    )
    assert module.optional_hidden_dims([4, 8], default=(8,)) == (4, 8)
    assert module.network_hidden_dims([4, 8], 7) == (4, 8)
    with pytest.raises(ValueError, match="length"):
        module.expand_or_validate([1, 2], 3, "kernel_sizes")
    with pytest.raises(ValueError, match="Network dimensions must be positive"):
        module.normalize_dims([])
    convolution = module.make_conv_encoder((2,), (3,), (2,), (1,))
    result = convolution(torch.ones(2, 1, 8, 8))
    assert result.shape == (2, 32) and torch.isfinite(result).all()


@pytest.mark.parametrize(
    "field,value",
    [("num_neighbors", -1), ("depth_size", -1), ("map_size", -1), ("neighbor_dim", 0)],
)
def test_invalid_buffer_channels_are_rejected(module, field, value):
    kwargs = {"num_neighbors": 1, field: value}
    with pytest.raises(ValueError, match="nonnegative|positive"):
        module.RobotRolloutBuffer(1, 1, 5, 2, torch.device("cpu"), **kwargs)


def test_optional_channels_are_present_in_shuffled_batches_and_missing_depth_warns_once(
    module, caplog
):
    buffer = module.RobotRolloutBuffer(
        2, 1, 5, 2, torch.device("cpu"), num_neighbors=1, depth_size=4, map_size=4
    )
    common = (
        torch.zeros(1, 5),
        torch.zeros(1, 2),
        torch.zeros(1),
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1),
    )
    for _ in range(2):
        buffer.add(
            *common,
            neighbors=torch.ones(1, 1, 5),
            neighbor_mask=torch.ones(1, 1, dtype=torch.bool),
            map_patch=torch.ones(1, 4, 4),
        )
    assert len([r for r in caplog.records if "depth=None" in r.message]) == 1
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(*common)
    buffer.compute_returns_and_advantages(torch.zeros(1), 0.99, 0.95)
    batches = list(buffer.batches(1))
    assert len(batches) == 2
    for batch in batches:
        assert batch["depth"].count_nonzero() == 0 and batch["map"].sum() == 16
        assert batch["neighbors"].sum() == 5 and batch["neighbor_mask"].all()
        torch.testing.assert_close(batch["returns"], torch.ones(1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("ppo_epochs", 0),
        ("minibatch_size", 0),
        ("lr", -1.0),
        ("lr", float("nan")),
        ("max_grad_norm", 0.0),
        ("max_grad_norm", float("inf")),
    ],
)
def test_invalid_optimizer_configuration_fails_before_model_allocation(
    module, field, value
):
    config = module.RobotPPOConfig(obs_dim=5, **{field: value})
    with pytest.raises(ValueError, match="positive"):
        module.RobotPPOTrainer(config, torch.device("cpu"))


@pytest.mark.parametrize("defect", ["schema", "parameterization", "vector_input"])
def test_checkpoint_architecture_failure_preserves_live_weights_and_optimizer(
    module, tmp_path, defect
):
    config = module.RobotPPOConfig(
        obs_dim=5,
        hidden_dims=(8,),
        num_neighbors=0,
        map_enabled=False,
        depth_enabled=False,
    )
    trainer = module.RobotPPOTrainer(config, torch.device("cpu"))
    checkpoint = tmp_path / "current.pt"
    trainer.save(checkpoint, 17, {"fixture": True})
    payload = torch.load(checkpoint, weights_only=False)
    before = deepcopy(trainer.model.state_dict())
    optimizer = deepcopy(trainer.optimizer.state_dict())
    if defect == "schema":
        payload["schema_version"] = 2
    else:
        payload["architecture"][
            "action_parameterization"
            if defect == "parameterization"
            else "vector_input_dim"
        ] = "incompatible"
    corrupt = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt)
    with pytest.raises(ValueError, match="predates|parameterization|vector_input_dim"):
        trainer.load(corrupt)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert trainer.optimizer.state_dict() == optimizer
