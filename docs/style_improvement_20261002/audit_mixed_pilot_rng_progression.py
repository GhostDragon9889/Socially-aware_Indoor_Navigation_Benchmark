"""Verify actual three-stream RNG continuation using immutable pilot4/8 data.

Only CPU generators are advanced. Random observations/actions are not replayed
and there is no PhysX/cache replay claim. A raw draw-count equality is checked
against the actual serialized states, rather than assumed from matching seeds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from audit_amass_root_actual_checkpoint import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Fresh output and explicitly empty CUDA visibility required")
    torch.set_num_threads(4)
    full = json.loads(args.audit.read_text())
    if full["status"] != "passed" or len(full["checkpoints"]) != 2:
        parser.error("Complete successful two-checkpoint audit required first")
    checks = []

    def check(name, passed, evidence=None):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(name)

    bundles, results = [], []
    for row in full["checkpoints"]:
        check("sealed_identity:"+str(row["update_count"]), fingerprint(row["checkpoint"]["path"]) == row["checkpoint"]
              and fingerprint(row["result"]["path"]) == row["result"])
        bundles.append(torch.load(row["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True))
        results.append(json.loads(Path(row["result"]["path"]).read_text()))
    a, b = bundles
    ra, rb = results
    ca, cb = a["runner_state"]["curriculum_state"], b["runner_state"]["curriculum_state"]
    check("actual_immutable_step4_to8_declared_source", a["update_count"] == 4 and b["update_count"] == 8
          and Path(rb["arguments"]["resume"]).resolve() == Path(full["checkpoints"][0]["checkpoint"]["path"]))
    for bundle, result in zip(bundles, results, strict=True):
        arguments = result["arguments"]
        expected = {"replan_steps": arguments["replan_steps"], "episode_steps": arguments["episode_steps"],
                    "planner_batch_size": arguments["planner_batch_size"], "trajectory_source": "mixed_amass_procedural",
                    "planning_clock": "synchronized"}
        check("exact_runner_contract_no_unknown_extras:"+str(bundle["update_count"]),
              bundle["runner_state"]["runner_contract"] == expected)
    changed = {key: {"step4": ra["arguments"].get(key), "step8": rb["arguments"].get(key)}
               for key in set(ra["arguments"]) | set(rb["arguments"])
               if ra["arguments"].get(key) != rb["arguments"].get(key)}
    check("only_explicit_resume_argument_changed", set(changed) == {"resume"}, changed)
    real = cb["statistics"]["amass_reference_samples"]-ca["statistics"]["amass_reference_samples"]
    resets = cb["statistics"]["procedural_reference_samples"]-ca["statistics"]["procedural_reference_samples"]
    original = resets+(b["update_count"]-a["update_count"])*min(ra["arguments"]["num_envs"]*ra["arguments"]["rollout_steps"], 4096)
    plans = cb["statistics"]["procedural_plans"]-ca["statistics"]["procedural_plans"]
    check("same_uniform_expert_sampler_and_2848_source", a["config"]["expert_sampling"] is None
          and b["config"]["expert_sampling"] is None
          and len(ca["config"]["amass_config"]["motion_ids"]) == 2848)
    seeds = {"AMASS_private": ca["config"]["seed"]+1, "original_reset_expert": ra["arguments"]["seed"],
             "procedural_command": ca["config"]["seed"]}
    streams = [
        ("AMASS_private", ca["amass_state"]["generator_state"], cb["amass_state"]["generator_state"], real, "integer_and_uniform"),
        ("original_reset_expert", a["runner_state"]["corpus_rng"], b["runner_state"]["corpus_rng"], original, "integer_and_uniform"),
        ("procedural_command", ca["procedural_state"]["generator_state"], cb["procedural_state"]["generator_state"], plans*151, "uniform"),
    ]
    summaries = []
    for name, initial, final, count, kind in streams:
        generator = torch.Generator(device="cpu").set_state(initial)
        seeded = torch.Generator(device="cpu").manual_seed(seeds[name])
        for value in (generator, seeded):
            if kind == "integer_and_uniform":
                torch.randint(2848, (count,), generator=value)
            torch.rand(count, generator=value)
        check(name+":actual_saved4_draw_count_advances_exactly_to_saved8",
              torch.equal(generator.get_state(), final), {"draw_count": count, "kind": kind})
        check(name+":same_segment_from_fresh_seed_does_not_match_actual8",
              not torch.equal(seeded.get_state(), final))
        summaries.append({"stream": name, "count": count, "kind": kind, "exact_saved4_to8_state_equal": True,
                          "fresh_reseed_segment_matches_actual8": False})
    check("top_procedural_rng_alias_remains_exact", torch.equal(cb["generator_state"], cb["procedural_state"]["generator_state"]))
    changed_parameters = [key for key in a["model"] if not torch.equal(a["model"][key], b["model"][key])]
    check("actual_model_updated_not_only_counter_increment", bool(changed_parameters), {"changed_tensor_count": len(changed_parameters)})
    report = {"schema": "mixed_actual_pilot_three_RNG_continuation.v1", "status": "passed", "CPU_only": True,
              "GPU_used": False, "SDK_started": False, "heldout_outputs_read": False,
              "main_payload_audit": fingerprint(args.audit), "script": fingerprint(__file__),
              "checks": checks, "check_count": len(checks), "streams": summaries,
              "source_formula": {"AMASS_sampler": "one torch.randint(2848) and one float32 torch.rand per sample",
                                 "smooth_procedural": "one speed-goal plus three uniform draws for each of50 knots =151 per actor plan"},
              "limitations": ["Actual serialized RNG-state equality proves draw-count continuation under this frozen CPU Torch implementation.",
                              "It does not reconstruct the per-reset batching, sampled motion IDs or physical trajectory/actions.",
                              "No CUDA RNG or PhysX solver/contact caches were replayed."]}
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
