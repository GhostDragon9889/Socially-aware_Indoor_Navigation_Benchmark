"""Compare saved native metrics without replacing their primary aggregates."""
import argparse
import json
import os
from pathlib import Path

from native_metric_sources import admit_run, record, require, validate_engine
from run_native_CPU_metrics import CATALOG, validate_catalog

CATALOG_SHA = "f3bed1b88466ad009fa44fa7a08ce2352115b328781753e14b5203fafd00c067"


def common_supported_bins(old, new):
    def supported(value):
        return {(b["lower_m_s"], b["upper_m_s"]): b for b in value["speed_bins"]
                if b["pose_velocity_gaussian_mmd2"] is not None}
    old_bins, new_bins = supported(old), supported(new)
    common = sorted(old_bins.keys() & new_bins.keys(), key=lambda x: x[0])
    rows = [{"lower_m_s": edge[0], "upper_m_s": edge[1],
             "old_MMD2": old_bins[edge]["pose_velocity_gaussian_mmd2"]["value"],
             "new_MMD2": new_bins[edge]["pose_velocity_gaussian_mmd2"]["value"],
             "old_generated_count": old_bins[edge]["generated_count"],
             "new_generated_count": new_bins[edge]["generated_count"]} for edge in common]
    return len(old_bins), len(new_bins), {"bins": rows,
        "old_equal_bin_mean_MMD2": sum(r["old_MMD2"] for r in rows) / len(rows) if rows else None,
        "new_equal_bin_mean_MMD2": sum(r["new_MMD2"] for r in rows) / len(rows) if rows else None}


def compare(actors, catalog):
    captures, scores = {}, {}
    for kind in ("old", "new"):
        scores[kind] = {}
        for dimension in (90, 45):
            key = f"{kind}{actors}_{dimension}"
            row = catalog["metrics"][key]
            directory = Path(row["output"])
            admission_path = directory / "actual_native_source_admission.json"
            admission = json.loads(admission_path.read_text())
            current = admit_run(catalog, row)
            require(admission.get("schema") == "public_state_prior.native_style_source_admission.v1"
                    and admission.get("status") == "passed" and admission.get("metric") == key
                    and admission.get("catalog") == record(CATALOG)
                    and admission.get("actual_native_source_admission") == current,
                    "Saved style metric is not bound to this exact native run")
            ledger_path = directory.parent / (key + "_CPU_imports.json")
            ledger = json.loads(ledger_path.read_text())
            require(ledger.get("status") == "passed" and ledger.get("runtime_finalization_observed") is True
                    and ledger.get("catalog") == record(CATALOG), "Actual finalized CPU source ledger required")
            report_path = directory / "evaluation.json"
            require(not any(p.stat().st_mode & 0o222 for p in (admission_path, ledger_path, report_path)),
                    "Compare only sealed completed style evidence")
            report = json.loads(report_path.read_text())
            item = report["runs"][row["run_name"]]
            require(item["sources"]["result"] == current["capture_artifacts"]["result.json"]
                    and item["sources"]["states"] == current["capture_artifacts"]["physical_states.npz"],
                    "Saved style/capture source identity differs")
            scores[kind][dimension] = {"item": item, "report": record(report_path),
                                       "admission": record(admission_path), "ledger": record(ledger_path)}
        require(scores[kind][90]["item"]["sources"]["result"] == scores[kind][45]["item"]["sources"]["result"],
                "90D and 45D must describe the same actual capture")
        raw = json.loads(Path(catalog["metrics"][f"{kind}{actors}_90"]["raw_audit"]).read_text())
        captures[kind] = {"counts": raw["counts"], "source_count": raw["source_epoch"]["source_count"],
                          "raw_status": raw["status"]}
    old = scores["old"][90]["item"]["inclusive_speed_conditioned"]
    new = scores["new"][90]["item"]["inclusive_speed_conditioned"]
    old_count, new_count, common = common_supported_bins(old, new)
    return {"schema": "public_state_prior.actual_native_same_scene_metric_comparison.v1", "status": "passed",
            "CPU_only": True, "GPU_used": False, "Kit_started": False, "actors": actors, "captures": captures,
            "primary90": {"old": old, "new": new, "primary_aggregates_replaced": False,
                          "old_supported_bins": old_count, "new_supported_bins": new_count},
            "secondary_common_speed_bins": common,
            "pose45": {kind: scores[kind][45]["item"]["poses"] for kind in ("old", "new")},
            "bound_metric_artifacts": {kind: {str(d): {k: v for k, v in score.items() if k != "item"}
                                                for d, score in group.items()} for kind, group in scores.items()},
            "catalog": record(CATALOG), "all_failed_contact_idle_samples_retained": True,
            "same_RNG_or_bitwise_planning_noise_claimed": False, "causal_single_factor_gain_claimed": False,
            "overall_style_gain_claimed": False, "default_promotion": False,
            "limits": "Same native scene and actor count; runtime epochs and network initialization differ. Numerical metrics do not establish causal improvement or automatic promotion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, choices=(4, 15), required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU comparison requires CUDA disabled")
    require(record(CATALOG)["sha256"] == CATALOG_SHA, "Frozen native commands changed")
    catalog = json.loads(CATALOG.read_text())
    validate_catalog(catalog)
    validate_engine(catalog)
    output = Path(catalog["comparisons"][str(args.actors)]["output"])
    require(not output.exists() and not output.is_symlink(), "Fresh native comparison output required")
    report = compare(args.actors, catalog)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "comparison.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)
    output.chmod(0o555)
    print(json.dumps({"status": "passed", "actors": args.actors, "output": str(output), "GPU_used": False}))


if __name__ == "__main__":
    main()
