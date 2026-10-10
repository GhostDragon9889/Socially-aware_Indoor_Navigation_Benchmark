"""Read-only scalar admission for recipe-induced FP32 normalization constants."""
import math
import struct

EMA_RECIPE = {"decay": .995, "every_optimizer_updates": 1, "start_optimizer_update": 2000,
              "timing": "before_first_micro_forward_of_optimizer_update",
              "warmup": "whole_state_dict_copy", "state": "whole_state_dict_including_buffers",
              "integer_buffers": "author_float_blend_copy_cast"}


def fp32(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite numeric normalization scalar required")
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("Representable FP32 normalization scalar required") from error
    if not math.isfinite(rounded):
        raise ValueError("Finite FP32 normalization scalar required")
    return rounded


def ordered_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return 0x80000000 - (bits & 0x7fffffff) if bits & 0x80000000 else 0x80000000 + bits


def expected_blend(value, events):
    raw, previous = fp32(value), fp32(value)
    alpha, complement = fp32(EMA_RECIPE["decay"]), fp32(1. - EMA_RECIPE["decay"])
    raw_term = fp32(complement * raw)
    for _ in range(events):
        updated = fp32(fp32(alpha * previous) + raw_term)
        if updated == previous:
            break  # A constant deterministic recurrence stays at this fixed point.
        previous = updated
    return previous


def validate_normalization(actual, config, optimizer_updates, *, weight_source, ema_recipe):
    """Caller supplies original FP32 tensors' lists; nothing is rewritten."""
    if (weight_source not in ("raw", "EMA") or type(optimizer_updates) is not int or optimizer_updates < 1
            or ema_recipe != EMA_RECIPE or set(actual) != {"state_mean", "state_std"}):
        raise ValueError("Exact weight source, update count and original EMA recipe required")
    events = max(0, optimizer_updates - EMA_RECIPE["start_optimizer_update"])
    buffers = {}
    for key in ("state_mean", "state_std"):
        if len(actual[key]) != 5 or len(config[key]) != 5:
            raise ValueError("Exactly five normalization coefficients required")
        observed = [fp32(value) for value in actual[key]]
        if list(actual[key]) != observed:
            raise ValueError("Caller normalization values must already be FP32")
        configured = [fp32(value) for value in config[key]]
        expected = [expected_blend(value, events) for value in configured]
        if key == "state_std" and any(value <= 0 for value in configured + observed):
            raise ValueError("Positive normalization standard deviations required")
        if any(value != raw and (weight_source != "EMA" or value != blend)
               for value, raw, blend in zip(observed, configured, expected)):
            raise ValueError("Normalization differs from exact config or recipe-supported EMA values")
        buffers[key] = {"actual": observed, "config_fp32": configured, "recipe_expected_fp32": expected,
                        "signed_delta_config": [value - raw for value, raw in zip(observed, configured)],
                        "signed_delta_expected": [value - blend for value, blend in zip(observed, expected)],
                        "ULP_distance_config": [abs(ordered_bits(value) - ordered_bits(raw)) for value, raw in zip(observed, configured)],
                        "ULP_distance_expected": [abs(ordered_bits(value) - ordered_bits(blend)) for value, blend in zip(observed, expected)]}
    return {"schema": "official_orca.normalization_constant_EMA_contract.v1", "weight_source": weight_source,
            "optimizer_updates": optimizer_updates, "EMA_blend_events": events, "recipe": EMA_RECIPE,
            "allowed_values_policy": "raw exact configFP32; EMA each element in {configFP32, checkpoint-count recipeFP32}",
            "buffers": buffers, "buffers_modified": False, "unrestricted_allclose_used": False}
