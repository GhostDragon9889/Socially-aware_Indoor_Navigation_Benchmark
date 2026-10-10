"""Stdlib-only contract for a public state prior with zero action extension.

Validation reads saved JSON provenance only; resume/inference never reopen the
public checkpoint or require trained action columns to remain zero.
"""
from pathlib import Path

PUBLIC_PACER_SHA256 = "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615"
STRATEGY = "public_pacer_state_prior_zero_action_extension"
PUBLIC_COUNTS = {"self_normalizer": 61931520001.0, "task_normalizer": 61931520001.0,
                 "amp_normalizer": 15482880001.0, "value_normalizer": 6881280001.0}
EXTENSION = {"schema": "navisaaclab.action_state_prior_initialization.v1",
             "history_columns": 1950, "action_columns": 69,
             "action_column_initialization": "zeros",
             "public_prior_training_data_overlap_with_current_amass": "unknown"}


def validate_action_state_prior_config(config):
    enabled = getattr(config, "warm_start_action_discriminator_from_state_prior", False)
    if type(enabled) is not bool:
        raise ValueError("warm_start_action_discriminator_from_state_prior must be an explicit boolean")
    if enabled and (config.warm_start_discriminator is not False
            or config.discriminator_input != "state_action"
            or config.amp_normalization != "per_history_coordinate"
            or (config.history_frames, config.amp_frame_dim, tuple(config.discriminator_layers)) != (10, 195, (1024, 512))
            or config.normalize_observations is not True or config.normalize_values is not True
            or config.normalization_clip != 5.):
        raise ValueError("Action state prior requires state_action10x195 exact layers/history RMS, normalized observations/values clip5 and warm_start_discriminator=False")
    return enabled


def validate_action_state_prior_initialization(config, initialization):
    """Require the coupled D/history-RMS provenance without mutating it."""
    enabled = validate_action_state_prior_config(config)
    if not enabled:
        if type(initialization) is dict and (
                "state_prior_action_extension" in initialization
                or any(type(initialization.get(name)) is str and initialization[name] == STRATEGY
                       for name in ("discriminator_initialization", "action_conditioned_discriminator_initialization"))):
            raise ValueError("Action state prior initialization provenance requires its explicit flag")
        return
    if type(initialization) is not dict:
        raise ValueError("Action state prior requires complete initialization provenance")
    expected = {"mode": "transfer_warm_start_public_pacer_actor_critic",
                "sha256": PUBLIC_PACER_SHA256, "actor_critic_task_transferred": True,
                "observation_value_moments_transferred": True,
                "action_conditioned_discriminator_initialization": STRATEGY,
                "discriminator_initialization": STRATEGY, "discriminator_input": "state_action",
                "amp_normalization": "per_history_coordinate",
                "published_state_only_discriminator_transferred": True,
                "published_amp_history_moments_transferred": True,
                "author_navisaaclab_weights": False}
    for name, value in expected.items():
        if type(initialization.get(name)) is not type(value) or initialization[name] != value:
            raise ValueError(f"Invalid action state prior initialization provenance: {name}")
    path = initialization.get("path")
    if type(path) is not str or not path.strip() or "://" in path or not Path(path).is_absolute():
        raise ValueError("Invalid action state prior initialization provenance: path")
    extension = initialization.get("state_prior_action_extension")
    if (type(extension) is not dict or set(extension) != set(EXTENSION)
            or any(type(extension[name]) is not type(value) or extension[name] != value
                   for name, value in EXTENSION.items())):
        raise ValueError("Invalid action state prior initialization provenance: extension")
    counts = initialization.get("normalizer_counts_before_cap")
    if (type(counts) is not dict or set(counts) != set(PUBLIC_COUNTS)
            or any(type(counts[name]) is not float or counts[name] != value
                   for name, value in PUBLIC_COUNTS.items())):
        raise ValueError("Invalid action state prior initialization provenance: public counts")
    cap = initialization.get("normalizer_count_cap")
    if ((cap is not None and type(cap) not in (int, float))
            or cap != config.warm_start_normalizer_count_cap
            or "normalizer_count_cap" not in initialization):
        raise ValueError("Invalid action state prior initialization provenance: count cap")
