"""SDK-free admission tests; no unfinished capture is loaded."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native15_prepared_reader", DOC / "audit_native15.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def actual_entry():
    return json.loads((reader.RUN / "deployment_entry.json").read_text())


def test_shape_bridge_reverses_to_entire_original_AST_and_never_changes_base_formulas():
    _, proof = reader.readers()
    assert proof["restored_full_AST_equals_original"]
    assert proof["frozen_base_and_route_bytes_unchanged"]
    assert len(proof["four_shape_constant_changes"]) == 4
    assert {tuple(change[1:]) for change in proof["four_shape_constant_changes"]} == {(4, 15), (132, 143)}


def test_unrecognized_origin_gate_label_cannot_silently_skip_adaptation():
    source = (DOC / "frozen_v2.py").read_text()
    source = source.replace('"exact ray roles"', '"unexpected renamed ray roles"')
    with pytest.raises(ValueError, match="Exactly four"):
        reader.adapt_origin_checks(ast.parse(source))


def test_actual15_preflight_admitted_without_terminal_artifacts():
    args = reader.validate_entry(actual_entry())
    assert args["num_envs"] == 15 and args["evaluate_steps"] == 6000


@pytest.mark.parametrize("change", ["wrong_actor", "wrong_budget", "missing_measurement", "false_physical_claim"])
def test_registered_entry_rejects_changed_actor_budget_and_contract(change):
    entry = deepcopy(actual_entry())
    if change == "wrong_actor":
        entry["parsed_driver_arguments"]["num_envs"] = 4
    elif change == "wrong_budget":
        entry["physical_transitions_planned"] = 24000
    elif change == "missing_measurement":
        entry["parsed_driver_arguments"]["measure_ground_support"] = False
    else:
        entry["physical_execution_verified"] = True
    with pytest.raises(ValueError):
        reader.validate_entry(entry)
