"""Independent cold verification of both actually completed SDK smoke stages."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import own_shared_social_episode_pipeline as owner  # noqa: E402

HERE = Path(__file__).resolve().parent
plan = owner.admit_plan(HERE / "owner_plan.json")
state = json.loads((HERE / "owner_state.json").read_text())
proof = []
for selected in plan["stages"][:2]:
    stage = selected["stage"]
    output = Path(selected["output_directory"])
    steps = [row for row in state["steps"] if row["name"] in (stage + "_training", stage + "_CPU_audit")]
    assert [row["name"] for row in steps] == [stage + "_training", stage + "_CPU_audit"]
    assert all(row["status"] == "passed" and owner.release.recovery.exact_integer(row["actual_exit_code"], 0)
               and row["exit_observation"] == "subprocess.Popen.wait" for row in steps)
    for row in steps:
        for key in ("log", "result"):
            assert owner.release.record(row[key]["path"]) == row[key]
    result = json.loads((output / "result.json").read_text())
    audit_path = HERE / (stage + "_CPU_audit.json")
    audit = json.loads(audit_path.read_text())
    _, binding, _ = owner.training.prepare_training(HERE / "serial_release.json", stage)
    owner.validate_stage(result, audit, selected, binding)
    actual = result["episode_reference_experiment"]["actual_reference_configuration"]
    registered_probe = json.loads((owner.training.original.HERE / "actual_entry_probe.json").read_text())
    profile_name = Path(selected["profile"]["path"]).name
    assert actual == registered_probe["cohorts"][profile_name]["actual_entry_report"][
        "episode_reference_experiment"]["actual_reference_configuration"]
    provenance = json.loads((output / "resolved_configuration.json").read_text())["provenance"]
    assert provenance["episode_after_recovery_execution"] == binding
    assert result["source_profile_before_after_equal"] is True
    assert result["expert_contract"]["sources"]["world"] == binding["consumer_World_admission"]["selected_World"]
    assert result["expert_contract"]["original_expert_AST_sha256"] == "dbf7c3d3419e4c9cb6a4be571a726a854f41c9f6f9fd633d703093256c5be787"
    for path in [audit_path, *[p for p in output.rglob("*") if p.is_file()]]:
        assert not path.is_symlink() and not path.stat().st_mode & 0o222
    proof.append({"stage": stage, "actual_training_exit_code": steps[0]["actual_exit_code"],
                  "actual_CPU_audit_exit_code": steps[1]["actual_exit_code"],
                  "completed_updates": result["completed_updates"], "physical_transitions": audit["physical_transitions"],
                  "finite_tensors_checked": audit["finite_tensors_checked"], "checkpoint": audit["checkpoint"],
                  "result": steps[0]["result"], "audit": steps[1]["result"],
                  "registered_reference_configuration_preserved": True,
                  "actual_consumer_World_and_clock_derivation_matched": True, "cleanup_succeeded": True})
assert "torch" not in sys.modules
report = {"status": "passed_independent_cold_smoke_verification", "observer": owner.release.record(__file__),
          "plan": owner.release.record(HERE / "owner_plan.json"), "sources_checked": len(plan["sources"]),
          "inputs_checked": len(plan["inputs"]), "smokes": proof, "Torch_imported": False,
          "SDK_started_by_verifier": False, "paper_effect_achieved": False, "pilot2000_completion_claimed": False}
owner.io.write_json(HERE / "smokes_external_verification.json", report, readonly=True)
print(json.dumps({key: value for key, value in report.items() if key not in ("smokes", "observer", "plan")}))
