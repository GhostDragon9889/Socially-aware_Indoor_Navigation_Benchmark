#!/usr/bin/env python3
"""Independent CPU measurement of saved navigation/ground-support arrays.

Counts every actor-control, including failures. These explicit denominators
are local measurements, not a reconstruction of unpublished paper rates.
Foot speeds use body origins; they are not contact-point slip measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.execution_receipt.read_text())
    require(receipt.get("schema") == "shared_social.actual_navigation200s_execution.v1"
            and receipt.get("actual_exit_code") == 0, "Require an actual completed navigation200s receipt")
    result_path = Path(receipt["result"]["path"])
    require(record(result_path) == receipt["result"], "Navigation terminal result changed")
    result = json.loads(result_path.read_text())
    require(result.get("status") == "passed" and result.get("cleanup_succeeded") is True
            and result.get("cleanup_errors") == [], "Require complete physical measurement and owned cleanup")
    import numpy as np

    run = result_path.parent
    paths = [run / "physical_states.npz", run / "ground_support.npz", run / "reference_plans.npz"]
    identities = [record(path) for path in paths]
    steps = result["actual_control_steps"]
    require(steps == 6000, "Require the complete 200s/30Hz physical capture")
    with np.load(paths[0], allow_pickle=False) as physical, np.load(paths[1], allow_pickle=False) as ground:
        for channel in (physical, ground):
            for name in channel.files:
                value = channel[name]
                if value.dtype.kind in "fc":
                    require(np.isfinite(value).all(), "Nonfinite physical capture: " + name)
        actors = physical["root_pos"].shape[1]
        require(actors == 4 and physical["root_pos"].shape == (steps, actors, 3), "Require the fixed four-actor scene")
        flags = {}
        for name in ("terminated", "truncated", "reset_mask", "static_contact", "route_vertex_reached"):
            value = physical[name]
            require(value.shape == (steps, actors) and value.dtype == np.bool_, "Invalid flag: " + name)
            flags[name] = value
        peer = physical["peer_contact"]
        require(peer.shape == (steps, actors, actors) and peer.dtype == np.bool_, "Invalid directed peer flags")
        require(not np.diagonal(peer, axis1=1, axis2=2).any(), "Self contacts entered peer measurements")
        threshold = result["physical_contact_measurement"]["threshold_n"]
        require(threshold == 1.0, "Contact threshold differs from the fixed baseline")
        require(np.array_equal(flags["static_contact"], physical["static_contact_max_force_n"] > threshold)
                and np.array_equal(peer, physical["peer_contact_max_force_n"] > threshold),
                "Contact flags differ from actual saved native force maxima")
        episode = physical["episode_ids"]
        require(episode.shape == (steps, actors) and episode.dtype.kind in "iu", "Invalid episode clock")
        delta = np.diff(episode, axis=0)
        require(np.isin(delta, (0, 1)).all() and flags["reset_mask"][0].all()
                and np.array_equal(delta == 1, flags["reset_mask"][1:]), "Reset mask/episode continuity mismatch")
        ended = flags["terminated"] | flags["truncated"]
        require(np.array_equal(ended[:-1], flags["reset_mask"][1:]), "Termination/reset phase mismatch")
        falls = int(flags["terminated"].sum())
        truncations = int((flags["truncated"] & ~flags["terminated"]).sum())
        arrivals = flags["route_vertex_reached"].sum(axis=0).tolist()
        require(falls == result["fall_resets"] == receipt["fall_resets"]
                and truncations == result["time_limit_resets"]
                and arrivals == result["route_vertices_reached"] == receipt["route_vertices_reached"],
                "Independent termination/route counts differ from terminal summaries")
        completed_lengths = []
        for actor in range(actors):
            start = 0
            for step in range(steps):
                if flags["reset_mask"][step, actor]:
                    start = step
                if ended[step, actor]:
                    completed_lengths.append(step - start + 1)
        require(sorted(completed_lengths) == sorted(result["completed_episode_lengths"]),
                "Saved episode lengths differ from actual captured phases")
        require(np.array_equal(ground["control_step_after"], np.arange(1, steps + 1))
                and (ground["substep_valid_count"] == 4).all() and ground["valid"].all()
                and np.array_equal(ground["physics_step_end"] - ground["physics_step_before"], np.full(steps, 4)),
                "Ground capture is missing control/substep measurements")
        for name in ("episode_ids", "reset_mask", "terminated", "truncated"):
            require(np.array_equal(ground[name], physical[name]), "Ground/physical phase mismatch: " + name)
        valid = ground["foot_contact_substeps"]
        speeds = ground["body_origin_tangential_speed_proxy_mps"][valid]
        # Independent actor-position displacement excludes every cross-reset
        # interval. All failed same-episode intervals remain included.
        same = delta == 0
        speed = np.linalg.norm(np.diff(physical["root_pos"][..., :2], axis=0), axis=-1)[same] * 30
        exposure = steps * actors
        unordered = np.logical_or(peer, peer.swapaxes(1, 2))[:, np.triu_indices(actors, 1)[0],
                                                                  np.triu_indices(actors, 1)[1]]
        report = {"schema": "shared_social.completed_navigation_CPU_array_audit.v1", "status": "passed",
            "reader": record(__file__), "execution_receipt": record(args.execution_receipt), "result": receipt["result"],
            "arrays": identities, "controls": steps, "actor_controls": exposure, "physics_substeps": steps * 4,
            "fall_termination_actor_controls": falls, "fall_terminations_per_actor_second": falls / (exposure / 30),
            "time_limit_terminations": truncations, "reset_actor_controls": int(flags["reset_mask"].sum()),
            "static_contact_actor_controls": int(flags["static_contact"].sum()),
            "peer_contact_actor_controls": int(peer.any(axis=-1).sum()),
            "unordered_peer_pair_controls": int(unordered.sum()), "native_normal_force_threshold_n": threshold,
            "route_vertices_reached": arrivals, "completed_episodes": len(completed_lengths),
            "completed_episode_length_mean_controls": float(np.mean(completed_lengths)) if completed_lengths else None,
            "same_episode_position_intervals": int(same.sum()), "same_episode_root_speed_mean_mps": float(speed.mean()),
            "no_support_substep_fraction_mean": float(ground["no_support_fraction"].mean()),
            "double_support_substep_fraction_mean": float(ground["double_support_fraction"].mean()),
            "supported_foot_origin_proxy_samples": int(speeds.size),
            "supported_foot_origin_speed_proxy_mean_mps": float(speeds.mean()) if speeds.size else None,
            "supported_foot_origin_speed_proxy_p95_mps": float(np.quantile(speeds, .95)) if speeds.size else None,
            "support_proxy_scope": "all measured supported body-origin substeps; not contact-point slip; failures included",
            "paper_rate_denominators_recovered": False, "quality_or_author_equivalence_established": False,
            "Torch_imported": "torch" in sys.modules, "SDK_executed": False}
    require([record(path) for path in paths] == identities and record(result_path) == receipt["result"],
            "Actual captures changed during CPU audit")
    require(report["Torch_imported"] is False, "CPU array audit must not import Torch")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "controls", "actor_controls",
        "fall_termination_actor_controls", "static_contact_actor_controls", "unordered_peer_pair_controls",
        "route_vertices_reached", "supported_foot_origin_speed_proxy_mean_mps")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
