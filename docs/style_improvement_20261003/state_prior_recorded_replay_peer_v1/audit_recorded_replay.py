"""CPU source admission plus unchanged frozen recorded-media audit."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE_PREP = ROOT / "docs/style_improvement_20261003/state_prior_native_reader_preparation_v1"
RUNS = ROOT / "docs/style_improvement_20261003/state_prior_native_actual_runs_v1"
OUTPUT = ROOT / "output/diffusion_amp/native_warehouse_state_prior_recorded_indoor_30s_v1"
RAW_AUDIT = ROOT / "docs/style_improvement_20261003/state_prior_native_actual_review_v1/native4_audit.json"
RAW_SHA = "5655f818094a360a50ad7cc3ecb3100965cbf13e5e248e85ccc70edff9f75862"
NATIVE_FREEZE_SHA = "154c4eb9debc41a83a487e4549c73009dcc3c4f15ab3c36e85e0990129a2ac50"
LAUNCHER_SHA = "138b85604a829c715f4fdf436c557b069db97a7b84d3bf33571ddb4d121186c0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_bridge():
    import hashlib
    freeze = NATIVE_PREP / "source_freeze_v1.json"
    require(hashlib.sha256(freeze.read_bytes()).hexdigest() == NATIVE_FREEZE_SHA, "Exact frozen native evidence helper required")
    manifest = json.loads(freeze.read_text())
    for row in manifest["files"] + [manifest["archive"]]:
        path = Path(row["path"])
        require(path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"] and path.stat().st_mode & 0o222 == 0, "Frozen native reader artifact changed")
    return load("recorded_state_prior_native_evidence", NATIVE_PREP / "native_evidence.py")


def media_reader(bridge):
    identity = bridge.read(HERE / "reader_identity.json")
    for key in ("original", "copy"):
        row = bridge.record(identity[key])
        require(row["bytes"] == identity["bytes"] and row["sha256"] == identity["sha256"] == "7a94eb9c5d26a96c81731fb1522a46823c67fc78bd00e7408eb6cbf1f3260914", "Old media reader bytes changed")
    return load("state_prior_frozen_media_reader", HERE / "frozen_media_audit.py")


def validate_preflight(report, evidence):
    require(report["status"] == "preflight_passed" and report["mode"] == "recorded_pose_visualization" and report["rendering_only"] is True and report["frames"] == 0, "Preflight must remain non-rendered")
    require(report["sources"]["result"] == evidence["source_capture"]["result.json"] and report["sources"]["states"] == evidence["source_capture"]["physical_states.npz"], "Real new native4 capture closure required")
    args = report["arguments"]
    require(args["output"] == str(OUTPUT) and args["start_step"] == 0 and args["steps"] == 900 and args["capture_every_steps"] == 2 and args["width"] == 1280 and args["height"] == 960 and args["camera_eye"] == [-6.75, -8.95, 3.2] and args["camera_target"] == [.35, 6.35, 1.] and args["preflight_only"] is True and args["native_scene_visuals"] is True, "Exact original indoor camera/900/stride2 recipe required")
    preflight = report["preflight"]
    require(preflight["frames"] == 6000 and preflight["actors"] == 4 and preflight["frequency_hz"] == 30 and preflight["video_fps"] == 15 and preflight["captured_step_indices"] == list(range(0, 900, 2)) and len(preflight["scene"]["source_layers"]) == 53, "Full physical source/450 frames/53 USD closure required")
    for name in ("closed_loop_policy_rollout", "physics_verification", "physical_humanoids_spawned", "physics_scene_created"):
        require(report[name] is False, "Replay cannot claim new physics/policy")
    for name in ("physics_steps", "policy_forward_calls"):
        require(type(report[name]) is int and report[name] == 0, "Replay zero count must be a strict integer")


def prepare_evidence():
    bridge = native_bridge()
    raw = bridge.read(RAW_AUDIT)
    require(bridge.record(RAW_AUDIT)["sha256"] == RAW_SHA and raw["native_200s_artifacts_independently_audited"] is True and raw["source_epoch"]["source_count"] == 183 and raw["counts"]["controls"] == 6000 and raw["counts"]["actors"] == 4, "Actual full new native4 raw audit required")
    registered = bridge.read(NATIVE_PREP / "native4_registration.json")
    admitted = bridge.admit_closed_native(4, registered)
    require(admitted == raw["actual_source_admission"], "Native source/model closure differs from real raw audit")
    for row in raw["artifacts"].values():
        require(bridge.record(row["path"]) == row, "Actual native source capture changed")
    preflight_path = RUNS / "recorded_replay_CPU_preflight_launch.json"
    launch = bridge.read(preflight_path)
    require(launch["schema"] == "state_prior.actual_recorded_replay_launch.v1" and launch["status"] == "closed" and launch["exit_code"] == 0 and launch["child_exit_observed"] is True and launch["prepare_only"] is True and launch["inputs_unchanged"] is True and launch["source_before"] == launch["source_after"] and launch["source_before"]["manifest"] == admitted["production_source"]["manifest"] and launch["source_before"]["archive"] == admitted["production_source"]["archive"] and launch["source_before"]["all183_exact"] is True and launch["source_before"]["complete_runtime_import_ledger"] is False and launch["launcher"]["sha256"] == LAUNCHER_SHA and bridge.record(launch["launcher"]["path"]) == launch["launcher"], "Actual closed exact replay CPU preflight required")
    evidence = {"native_source_admission": admitted, "native_raw_audit": bridge.record(RAW_AUDIT), "native_reader_freeze": bridge.record(NATIVE_PREP / "source_freeze_v1.json"), "source_capture": {name: raw["artifacts"][name] for name in ("result.json", "physical_states.npz")}, "closed_owner_preflight": bridge.record(preflight_path), "preflight_log": bridge.record(launch["log"]), "owner_launcher": launch["launcher"], "preflight_command": launch["command"], "declared_env": launch["declared_env"], "expected_preflight": bridge.read(launch["log"])}
    require(launch["inputs"] == evidence["source_capture"] and launch["visualization_only"] is True and launch["fresh_physics_or_policy_claimed"] is False and launch["complete_runtime_import_ledger_claimed"] is False, "Exact recorded-only source/claims required")
    validate_preflight(evidence["expected_preflight"], evidence)
    for row in evidence["expected_preflight"]["preflight"]["scene"]["source_layers"]:
        require(bridge.record(row["path"]) == row, "Actual53 original USD layer bytes changed")
    selected = {row["path"]: row for row in bridge.read(bridge.SOURCE)["files"]}
    for row in evidence["expected_preflight"]["source_code"].values():
        require(bridge.record(row["path"]) == row == selected[row["path"]], "Replay producer source must belong to exact V11")
    media_reader(bridge)
    return evidence


def registration():
    evidence = prepare_evidence()
    return {"schema": "public_state_prior.recorded_replay_CPU_registration.v1", "status": "prepared_actual_render_pending", "created_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False, "SDK_started": False, "actual_render_audited": False, "actual_video_success_verified": False, "output": str(OUTPUT), "evidence": evidence, "reader": native_bridge().record(__file__), "media_mathematics_or_encoding_checks_modified": False, "actual_visual_visibility_judged": False, "fresh_physics_or_objective_style_improvement_proven": False}


def validate_render_metadata(result, evidence):
    expected = evidence["expected_preflight"]
    args = {**expected["arguments"], "preflight_only": False}
    require(result["status"] in {"passed", "failed"} and result["arguments"] == args and result["sources"] == expected["sources"] and result["source_code"] == expected["source_code"] and result["preflight"] == expected["preflight"] and result["native_visual_scene"] == expected["native_visual_scene"], "Actual render/source/camera/source-code closure differs from frozen preflight")


def audit(registered):
    bridge = native_bridge()
    require(registered["schema"] == "public_state_prior.recorded_replay_CPU_registration.v1" and registered["actual_render_audited"] is False and registered["reader"] == bridge.record(__file__) and registered["output"] == str(OUTPUT), "Exact pending media reader registration required")
    evidence = prepare_evidence()
    require(registered["evidence"] == evidence, "Editable registration cannot rewrite physical/media expectations")
    launch_path = RUNS / "recorded_replay_actual_launch.json"
    launch = bridge.read(launch_path)
    require(launch["schema"] == "state_prior.actual_recorded_replay_launch.v1" and launch["status"] == "closed" and launch["child_exit_observed"] is True and type(launch["exit_code"]) is int and launch["prepare_only"] is False and launch["inputs_unchanged"] is True and launch["command"] == evidence["preflight_command"][:-1] and evidence["preflight_command"][-1] == "--preflight-only" and launch["declared_env"] == evidence["declared_env"] and launch["cwd"] == str(ROOT) and launch["launcher"] == evidence["owner_launcher"] and launch["inputs"] == evidence["source_capture"] and launch["visualization_only"] is True and launch["fresh_physics_or_policy_claimed"] is False and launch["complete_runtime_import_ledger_claimed"] is False, "Actual closed recorded-only launch required before media read")
    for when in ("source_before", "source_after"):
        require(launch[when]["manifest"] == evidence["native_source_admission"]["production_source"]["manifest"] and launch[when]["archive"] == evidence["native_source_admission"]["production_source"]["archive"] and launch[when]["all183_exact"] is True and launch[when]["complete_runtime_import_ledger"] is False, "Actual selected183 pre/post source binding differs")
    result_path = OUTPUT / "result.json"
    result = bridge.read(result_path)
    validate_render_metadata(result, evidence)
    require((result["status"] == "passed" and launch["exit_code"] == 0) or (result["status"] == "failed" and launch["exit_code"] != 0), "Actual renderer failure cannot become video success")
    report = media_reader(bridge).verify(OUTPUT)
    require(report["result"] == bridge.record(result_path) and evidence["source_capture"] == {"result.json": report["source_closure"]["result"], "physical_states.npz": report["source_closure"]["states"]} and report["source_physical_capture_outcomes"]["actual_control_steps"] == 6000, "Complete source poses changed during media audit")
    require(prepare_evidence() == evidence, "Native/preflight closure changed during media audit")
    report.update(schema="public_state_prior.recorded_replay_actual_CPU_audit.v1", reader=bridge.record(__file__), frozen_media_reader=bridge.record(HERE / "frozen_media_audit.py"), registration=registered, actual_closed_renderer_launch=bridge.record(launch_path), actual_owner_log=bridge.record(launch["log"]), media_source_binding=evidence, media_math_or_encoding_checks_modified=False, actual_visual_visibility_judged=False, fresh_physics_or_objective_style_improvement_proven=False, source_pose_bytes_unchanged=True)
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "CPU media reader imported model/SDK")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve earlier media evidence")
    try:
        require(args.register_only or args.registration is not None, "Actual stage requires registration")
        report = registration() if args.register_only else audit(json.loads(args.registration.read_text()))
    except Exception as error:
        report = {"schema": "public_state_prior.recorded_replay_actual_CPU_audit.v1", "status": "audit_rejected", "CPU_only_reader": True, "audit_error": f"{type(error).__name__}: {error}", "actual_video_success_verified": False, "no_failure_promoted_to_success": True}
        exit_code = 1
    else:
        exit_code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("status", "renderer_terminal_status", "actual_video_success_verified", "audit_error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
