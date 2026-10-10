from copy import deepcopy
import json
from pathlib import Path

import pytest
import run_native15_CPU_metrics_v2 as runner


def values():
    catalog = json.loads(runner.CATALOG.read_text())
    raw = json.loads(Path(catalog["sources"]["actual_native15_raw_audit"]["path"]).read_text())
    return raw, catalog


def test_actual_raw15_canonical_revision_is_bound_without_array_or_model_io():
    raw, catalog = values()
    runner.validate_revision_report(raw, catalog)
    assert raw["compiled_v1_v2_routes_math_changed"] is False
    assert len(raw["adapter_proof"]["origin_shape_constant_changes"]) == 4


@pytest.mark.parametrize("change", ["missing_adapter", "wrong_adapter", "math_changed", "not_representation_only", "proof_value_changed", "drop_first_failure", "outer_registration_changed"])
def test_revision_gate_rejects_unbound_or_semantic_raw_adapters(change):
    raw, catalog = values()
    raw = deepcopy(raw)
    if change == "missing_adapter":
        raw.pop("representation_adapter")
    elif change == "wrong_adapter":
        raw["representation_adapter"]["sha256"] = "0" * 64
    elif change == "math_changed":
        raw["compiled_v1_v2_routes_math_changed"] = True
    elif change == "not_representation_only":
        raw["proof_tuple_list_difference_only"] = False
    elif change == "proof_value_changed":
        raw["adapter_proof"]["origin_shape_constant_changes"][0][2] = 14
    elif change == "drop_first_failure":
        raw.pop("first_actual_CPU_failure")
    else:
        raw["representation_revision_registration"]["source_count"] = 174
    with pytest.raises(ValueError):
        runner.validate_revision_report(raw, catalog)
