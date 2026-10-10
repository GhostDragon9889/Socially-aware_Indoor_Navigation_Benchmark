"""Small source-only counterexamples; no future process, checkpoint deserialization or arrays."""
import copy
import json
from pathlib import Path
import sys

from physical_evidence import HERE, read, record, require, source_binding, zero_physical


def main():
    plan = read(HERE / "registration.json")
    checks = []
    valid = {"schema": "default_newDP.independent_physical_CPU_audit.v1", "status": "passed", "phase": "fixture_new",
             "actual_capture_independently_audited": True,
             "physical_gate_counts": {"controls": 6000, "actors": 4, "duration_s": 200.0, "falls": 0,
                                      "static_actor_controls": 0, "peer_pair_controls": 0}}
    zero_physical(valid, "fixture_new")
    checks.append("Synthetic complete zero-physical report passes only numerical gate, not actual source admission")
    for field, value in (("falls", 1), ("static_actor_controls", 1), ("peer_pair_controls", 1), ("controls", 5999)):
        wrong = copy.deepcopy(valid)
        wrong["physical_gate_counts"][field] = value
        try:
            zero_physical(wrong, "fixture_new")
        except ValueError:
            checks.append("Synthetic wrong " + field + " rejected")
        else:
            raise AssertionError("Wrong physical result accepted: " + field)
    wrong = copy.deepcopy(plan)
    wrong["inputs"]["default_controller"]["sha256"] = "0" * 64
    try:
        source_binding(wrong)
    except ValueError:
        checks.append("Wrong legacy controller SHA rejected before future capture access")
    else:
        raise AssertionError("Wrong source accepted")
    require(all(not Path(case["output"]).exists() and not Path(case["audit"]).exists() for case in plan["cases"].values()),
            "Preparation cannot be relabeled after actual runs")
    require(all(case["command_arrays_generated"] is False for case in plan["native_future_path_specifications"].values()),
            "Native future stages contain paths, not extra execution arrays")
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "Preparation proof imported model/SDK")
    proof = {"schema": "default_newDP.source_only_preparation_proof.v1", "status": "passed", "checks": checks,
             "registration": record(HERE / "registration.json"), "proof_reader": record(__file__),
             "actual_future_outputs_read": False, "models_loaded": False, "GPU_used": False, "SDK_started": False,
             "future_effect": False, "sealed34_outputs_read": False}
    with (HERE / "preparation_proof.json").open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "future_effect": False}))


if __name__ == "__main__":
    main()
