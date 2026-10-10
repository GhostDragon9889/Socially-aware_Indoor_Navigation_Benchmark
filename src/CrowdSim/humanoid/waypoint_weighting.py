"""Explicit evaluation waypoint priorities; stdlib only, safe before Kit.

This redistribution is an engineering experiment, not an author setting. A
zero mass returns None so callers retain their original mean operation exactly.
"""

from __future__ import annotations

import math
from numbers import Real


NEAR_HORIZON_SECONDS = 1.0


def validate_near_horizon_mass(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Near-horizon waypoint mass must be a finite number in [0,1], not a boolean")
    return float(value)


def _waypoint_times(times):
    try:
        values = tuple(times)
    except TypeError as error:
        raise ValueError("Waypoint times must be a nonempty sequence of finite positive seconds") from error
    if not values or any(isinstance(value, bool) or not isinstance(value, Real)
                         or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Waypoint times must be a nonempty sequence of finite positive seconds")
    return tuple(float(value) for value in values)


def near_horizon_waypoint_weights(times, near_horizon_mass=0.0):
    """Return normalized priorities, or None for the untouched mean path.

    For eta>0, weights=(1-eta)/K + eta/m for each of the m times<=1s.
    Every waypoint occurrence counts, including repeated times; no sorting,
    clipping, or unsupported-near-window redistribution is performed.
    """
    mass = validate_near_horizon_mass(near_horizon_mass)
    if mass == 0:
        return None
    values = _waypoint_times(times)
    near = tuple(value <= NEAR_HORIZON_SECONDS for value in values)
    count = sum(near)
    if count == 0:
        raise ValueError("Positive near-horizon mass requires at least one waypoint at or before1s")
    uniform = (1-mass)/len(values)
    return tuple(uniform+mass/count*selected for selected in near)


def near_horizon_weighting_description(times, near_horizon_mass=0.0):
    """JSON-safe runtime metadata, separate from training configuration."""
    values = _waypoint_times(times)
    mass = validate_near_horizon_mass(near_horizon_mass)
    weighted = near_horizon_waypoint_weights(values, mass)
    weights = list(weighted) if weighted is not None else [1/len(values)]*len(values)
    return {"variant": "engineering_near_horizon_priority" if mass else "original_uniform_mean",
            "engineering_variant_enabled": mass > 0, "author_hyperparameters": False,
            "near_horizon_mass": mass, "near_horizon_seconds": NEAR_HORIZON_SECONDS,
            "waypoint_times_s": list(values), "normalized_waypoint_weights": weights,
            "weight_sum": math.fsum(weights), "total_goal_weight_changed": False,
            "near_waypoint_probability_mass": math.fsum(weight for weight, time in zip(weights, values, strict=True)
                                                        if time <= NEAR_HORIZON_SECONDS),
            "probability_interpretation": "normalized objective coefficients, not stochastic sample probabilities",
            "gradient_and_filter_use_same_weights": True,
            "default_original_mean_path": mass == 0,
            "formula": "(1-eta)/K + eta/m * indicator(time<=1s)"}
