"""Read-only QC of the two actual completed CPU tracking reports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METRICS = ROOT / "docs/style_improvement_20261003/state_prior_metrics_v1"
ACTUAL = ROOT / "docs/style_improvement_20261003/state_prior_metrics_actual_runs_v1"
REFERENCE_PEER = ROOT / "docs/state_prior_tracking_reference_peer_v1"
CATALOG = ROOT / "docs/style_improvement_20261003/state_prior_CPU_metrics_preparation_v1/commands_CPU_metrics_v1.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    receipt_path = ACTUAL / "tracking_CPU_completion_v1.json"
    require(record(receipt_path)["sha256"] == "c3a79e233286fab9b216b1b64f2cbe7c49d5571c8745be4d51692b9a07ed8f44",
            "Actual completion receipt changed")
    receipt = read(receipt_path)
    require(receipt["status"] == "passed" and receipt["CPU_only"] is True and
            receipt["GPU_used"] is False and receipt["sealed34_outputs_read"] is False,
            "Completed actual CPU receipt required")
    for row in receipt["artifact_records"]:
        require(record(row["path"]) == row, "Actual metric/ledger/launch artifact changed")
    require(record(CATALOG) == receipt["frozen_catalog"], "Actual frozen catalog changed")
    catalog = read(CATALOG)
    source_path = Path(receipt["source_manifest"]["path"])
    require(record(source_path) == receipt["source_manifest"], "Actual source manifest changed")
    manifest = read(source_path)
    declared = {row["path"]: row for row in manifest["files"]}
    code = Path(catalog["bundle"]) / "code"
    reference_peer_path = REFERENCE_PEER / "actual_reference_pairing.json"
    require(record(reference_peer_path)["sha256"] == "63284b01f538cbe45de39bc981d936b92df46fe3a6e8dee53064189cc77da955",
            "Independent actual source/reference peer changed")
    reference_peer = read(reference_peer_path)
    require(reference_peer["status"] == "passed" and
            reference_peer["actual_source_admission"] == record(REFERENCE_PEER / "actual_source_admission.json"),
            "Real terminal/new934 source admission was not independently checked")
    independently_admitted = read(REFERENCE_PEER / "actual_source_admission.json")
    result = read(receipt["actual_tracking_capture"]["path"])
    require(record(receipt["actual_tracking_capture"]["path"]) == receipt["actual_tracking_capture"],
            "Actual full22 result changed")
    require(record(receipt["actual_terminal_admission"]["path"]) == receipt["actual_terminal_admission"],
            "Actual terminal admission changed")
    require(independently_admitted["actual_terminal_admission"] == receipt["actual_terminal_admission"],
            "Metrics and independent peer use different terminal admission")
    style = read(METRICS / "tracking90/evaluation.json")
    decomposition = read(METRICS / "tracking_decomposition/decomposition.json")
    comparison = read(METRICS / "tracking_decomposition/comparison.json")
    run = style["runs"]["state_prior_tracking2000"]
    validation = run["capture_validation"]
    ids = [row["motion_id"] for row in result["sequences"]]
    failed = [row for row in result["sequences"] if row["tracking_success"] is False]
    failed_ids = [row["motion_id"] for row in failed]
    require(len(ids) == 22 and len(failed) == 5, "Actual22/5failed contract differs")
    require(style["status"] == decomposition["status"] == comparison["status"] == "passed" and
            style["device"] == "CPU" and style["kit_started"] is False,
            "Actual report did not complete on CPU")
    require(validation["clips"] == 22 and validation["metric_frames"] == 11982 and
            validation["initial_reset_frames_discarded"] == 22 and
            validation["valid_same_clip_style_intervals"] == 11960,
            "Full failed-inclusive style population differs")
    require([row["motion_id"] for row in validation["sources"]] == ids and
            [row["motion_id"] for row in decomposition["by_clip"]] == ids,
            "A clip was omitted from style/decomposition")
    generated = run["generated"]
    require(generated["same_episode_transition_count"] == 11960 and
            generated["excluded_reset_transitions"] == 21 and
            generated["valid_acceleration_triplets"] == 11938,
            "Original concatenated clip derivative selection changed")
    paired = {row["motion_id"]: row for row in validation["sources"]}
    per_clip = {row["motion_id"]: row for row in decomposition["by_clip"]}
    for row in result["sequences"]:
        clip = paired[row["motion_id"]]
        diagnostic = per_clip[row["motion_id"]]
        require(clip["path"] == row["capture"] and clip["sha256"] == row["capture_sha256"] and
                clip["metric_frames"] == diagnostic["metric_frames"] == row["observed_metric_steps"] and
                clip["source_path"] == diagnostic["source_path"] == row["source_path"] and
                diagnostic["tracking_success_primary"] is row["tracking_success"] and
                diagnostic["sequence_had_fall"] is row["sequence_had_fall"],
                "Actual per-clip source/frame/failure records differ")
    require(sum(row["metric_frames"] for row in validation["sources"]) == 11982 and
            decomposition["primary_aggregate_unchanged"] == result["aggregate"] == comparison["summary"]["aggregate"],
            "All22 primary aggregate was changed")
    require(decomposition["analytic_rigid_transform_check_passed"] is True,
            "Original same-frame decomposition analytic check failed")
    metric = run["inclusive_speed_conditioned"]
    require(metric["minimum_metric_samples_per_bin"] == 30 and
            metric["maximum_balanced_motion_samples_per_bin"] == 256,
            "Primary fixed sampling parameters changed")
    bins = metric["speed_bins"]
    require([row["lower_m_s"] for row in bins] == [0., .1, .5, 1., 1.5, 2., 3.] and
            sum(row["generated_count"] for row in bins) == 11960,
            "Original speed bins or inclusive sample population changed")
    values = [row["pose_velocity_gaussian_mmd2"]["value"] for row in bins
              if row["pose_velocity_gaussian_mmd2"] is not None]
    score = metric["equal_supported_bin_mean_pose_velocity_mmd2"]
    require(len(values) == metric["motion_supported_bins"] == 5 and
            math.isclose(sum(values) / len(values), score, abs_tol=1e-15, rel_tol=0) and
            score == comparison["summary"]["inclusive_mmd2"] == receipt["inclusive90_MMD2"],
            "Actual primary supported-bin mean differs from unchanged reports")
    require(all(row["pose_velocity_gaussian_mmd2"] is None for row in bins[-2:]),
            "Unsupported bins became numerical scores")
    actual_reference = record(METRICS / "tracking90/heldout_reference.json")
    require(all(actual_reference[key] == catalog["sources"]["style90_reference"][key]
                for key in ("bytes", "sha256")),
            "Frozen heldout reference changed")
    observed = {}
    for name in ("tracking90", "tracking_decomposition"):
        admission = read(METRICS / name / "actual_source_admission.json")
        require(admission["status"] == "passed" and admission["CPU_only"] is True and
                admission["GPU_used"] is False and admission["Kit_started"] is False and
                admission["actual_source_admission"] == independently_admitted and
                admission["runner"] == record(catalog["runner"]) and admission["catalog"] == record(CATALOG) and
                admission["fixed_metric_argv"] == catalog["metrics"][name]["metric_argv"],
                "Actual metric admission/source/command binding differs")
        for filename in ("GPU_import_ledger_as_observed.json", "GPU_import_events_as_observed.jsonl"):
            require((METRICS / name / filename).read_bytes() == (REFERENCE_PEER / filename).read_bytes(),
                    "Actual observed GPU source bytes differ from independently admitted capture")
        launch = receipt["actual_closed_CPU_launches"][name]
        require(launch["actual_started"] is True and launch["actual_closed"] is True and
                launch["exit_code"] == 0 and launch["argv"] == catalog["metrics"][name]["command"] and
                launch["environment"] == catalog["CPU_environment"], "Actual closed launch differs from frozen command")
        ledger_path = METRICS / (name + "_CPU_imports.json")
        ledger = read(ledger_path)
        events = [json.loads(line) for line in Path(ledger["event_ledger"]).read_text().splitlines()]
        require(ledger["status"] == "finalized" and ledger["phase"] == "passed" and
                ledger["source_manifest_sha256"] == receipt["source_manifest"]["sha256"] and
                ledger["all_observed_owned_modules_isolated"] is True and
                ledger["runtime_finalization_observed"] is True and
                len(events) == ledger["event_count"] and
                [event["sequence"] for event in events] == list(range(1, len(events) + 1)) and
                events[-1]["event"] == "final" and events[-1]["final_status"] == "passed",
                "Actual observed CPU import ledger is incomplete")
        executed = {event["module"]: event for event in events if event["event"] == "owned_execution"}
        require(len(executed) == len(ledger["owned_modules"]), "Actual CPU execution ledger disagrees")
        for module in ledger["owned_modules"]:
            path = Path(module["path"])
            require(path.resolve().is_relative_to(code) and record(path)["sha256"] == module["sha256"] and
                    record(path)["bytes"] == module["bytes"], "Executed CPU module escaped newCODE")
            selected = declared["code/" + str(path.relative_to(code))]
            require(module["sha256"] == selected["sha256"] and module["bytes"] == selected["bytes"] and
                    all(executed[module["module"]][key] == module[key] for key in
                        ("module", "path", "bytes", "sha256", "execution_observed")),
                    "Executed CPU module/event differs from frozen934 source")
        for stage in admission["staged_exact_external_tools"].values():
            require(record(stage["original"]["path"]) == stage["original"] and
                    record(stage["staged_execution"]["path"]) == stage["staged_execution"] and
                    stage["original"]["sha256"] == stage["staged_execution"]["sha256"],
                    "Executed exact staged original tool changed")
        observed[name] = {"ledger": record(ledger_path), "events": record(ledger["event_ledger"]),
                          "owned_executions": len(executed), "committed_events": len(events),
                          "runtime_finalization_actually_observed": True}
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
            "Report-only QC imported model/SDK")
    report = {"schema": "public_state_prior.actual_tracking_CPU_metrics_fresh_peer.v1", "status": "passed",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "GPU_used": False,
              "SDK_started": False, "sealed34_outputs_read": False, "metrics_recomputed": False,
              "reviewer": record(__file__), "actual_completion_receipt": record(receipt_path),
              "independently_admitted_reference_peer": record(reference_peer_path),
              "actual_tracking90_report": record(METRICS / "tracking90/evaluation.json"),
              "actual_decomposition_report": record(METRICS / "tracking_decomposition/decomposition.json"),
              "actual_comparison_report": record(METRICS / "tracking_decomposition/comparison.json"),
              "actual_terminal_report": receipt["actual_terminal_admission"],
              "actual_observed_CPU_ledgers": observed, "complete_failed_inclusive_clips": 22,
              "metric_frames": 11982, "same_clip_style_intervals": 11960,
              "original_initial_reset_frames_discarded": 22, "failed_motion_ids_retained": failed_ids,
              "failed_clip_metric_frames_retained": sum(row["observed_metric_steps"] for row in failed),
              "failed_clip_same_clip_style_intervals_retained": sum(row["observed_metric_steps"] - 1 for row in failed),
              "tracking_primary_aggregate_unchanged": True, "inclusive90_MMD2": score,
              "supported_speed_bins": 5, "unsupported_bins_remain_unknown": True,
              "metric_execution_pass_is_quality_pass": False,
              "scope": "Independently read actual closed CPU launches, exact report/source fingerprints, observed CPU/GPU ledger bytes and all22 per-clip source/frame/failure records. Supported-bin mean QC only; no metric evaluator/model rerun, navigation read or quality admission."}
    with (HERE / "actual_tracking_review.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
