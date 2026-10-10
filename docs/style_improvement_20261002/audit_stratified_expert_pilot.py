"""CPU audit of actual GPU pilot bundles; no physical-equivalence claim."""

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
PILOT = ROOT/"output/diffusion_amp/amp_style_amass2848_stratified_experts_pilot_v1"


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equal_tree(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (tuple,list)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(equal_tree(a,b) for a,b in zip(actual,expected,strict=True))
    return actual == expected


def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location("checkpoint_watch_helpers",DOCS/"watch_stratified_expert_checkpoints.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    source_path = DOCS/"stratified_expert_training_source_v1.json"
    source, frozen, corpus_type, bridge_type, config_type = helpers.frozen_helpers(source_path)
    sys.path.insert(0,str(frozen))
    from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
    preflight = json.loads((DOCS/"expert_sampling_integration_preflight.json").read_text())
    config = preflight["TrainingConfig_difference_from_negative_backward_baseline"]["expert_sampling"]
    corpus = corpus_type(ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json","train",device="cpu",dof_velocity_frame="child_joint")
    bridge = bridge_type(corpus,config)
    bridge.caller_config = config
    results = {}
    for step in (4,8):
        path = PILOT/f"step_{step}.pt"
        bundle = torch.load(path,weights_only=True,map_location="cpu",mmap=True)
        validation = helpers.validate_bundle(bundle,bridge,config_type,torch)
        trainer, runner = PaperAMPTrainer.load(path,device="cpu",restore_rng=False)
        checks = {
            "configuration_exact":asdict(trainer.config) == bundle["config"],
            "model_exact":equal_tree(trainer.model.state_dict(),bundle["model"]),
            "optimizer_exact":equal_tree(trainer.optimizer.state_dict(),bundle["optimizer"]),
            "discriminator_optimizer_exact":equal_tree(trainer.discriminator_optimizer.state_dict(),bundle["discriminator_optimizer"]),
            "all_four_normalizers_exact":all(equal_tree(getattr(trainer,key).state_dict(),value) for key,value in bundle["normalizers"].items()),
            "full_policy_replay_exact":equal_tree(trainer.policy_replay.state_dict(),bundle["policy_replay"]),
            "runner_payload_exact":equal_tree(runner,bundle["runner_state"]),
            "deferred_all_global_rng_exact":equal_tree(trainer._pending_rng_state,bundle["rng"]),
        }
        assert all(checks.values())
        saved_private = runner["expert_sampling"]
        bridge.load_state_dict(saved_private)
        corpus.generator.set_state(runner["corpus_rng"].cpu())
        expected = bridge.sample_reference(16)
        bridge.load_state_dict(saved_private)
        corpus.generator.set_state(runner["corpus_rng"].cpu())
        actual = bridge.sample_reference(16)
        for left,right in zip(expected[:2],actual[:2],strict=True):
            assert equal_tree(vars(left),vars(right))
        assert all(torch.equal(a,b) for a,b in zip(expected[2:],actual[2:],strict=True))
        results[str(step)] = {"checkpoint":helpers.record(path), "validation":validation, "CPU_trainer_load_checks":checks,
            "private_and_shared_reference_RNG_deterministic_continuation":True,
            "saved_episode_id_sum":int(runner["episode_ids"].sum()),
            "saved_CUDA_rng_arrays":len(bundle["rng"]["torch_cuda"]),
            "actual_CUDA_rng_restored_in_this_audit":False}
        del trainer, bundle
    first = json.loads((PILOT/"result_after_4.json").read_text())
    resumed = json.loads((PILOT/"result.json").read_text())
    assert first["status"] == resumed["status"] == "passed"
    assert (first["update_count"],resumed["update_count"]) == (4,8)
    assert (first["physical_transition_count"],resumed["physical_transition_count"]) == (32768,65536)
    assert first["arguments"]["resume"] is None and resumed["arguments"]["resume"]
    assert first["training_config"] == resumed["training_config"]
    rows = [json.loads(row) for row in (PILOT/"metrics.jsonl").read_text().splitlines()]
    assert [row["update_count"] for row in rows] == list(range(1,9))
    assert [row["physical_transitions"] for row in rows] == [8192*step for step in range(1,9)]
    review = {"schema":"navisaaclab.stratified_expert_GPU_pilot_CPU_resume_review.v1","status":"passed",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),"CPU_only_audit":True,"Kit_started":False,"GPU_used_by_audit":False,
        "source_manifest":helpers.record(source_path),"prelaunch_source_archive":source["archive"],
        "imported_training_implementation":helpers.record(frozen/"CrowdSim/humanoid/training/trainer.py"),
        "checkpoints":results,"actual_GPU_run_results":{"first_segment":helpers.record(PILOT/"result_after_4.json"),"resumed_segment":helpers.record(PILOT/"result.json")},
        "actual_completed_updates":8,"actual_physical_transitions":65536,
        "segment_outcomes":{"first4":{"control_steps_this_invocation":first["actual_control_steps"],"fall_resets":first["fall_resets"],"time_limit_resets":first["time_limit_resets"]},
                            "resumed4":{"control_steps_this_invocation":resumed["actual_control_steps"],"fall_resets":resumed["fall_resets"],"time_limit_resets":resumed["time_limit_resets"]}},
        "sum_of_reported_segment_fall_resets":first["fall_resets"]+resumed["fall_resets"],
        "fall_counter_scope":"result_after_4 reports first segment5; final result reports resumed segment1, not cumulative8updates",
        "metrics_updates_and_transition_sequence_exact":True,
        "resumed_episode_id_increment":results["8"]["saved_episode_id_sum"]-results["4"]["saved_episode_id_sum"],
        "mechanism_conclusion":"Actual full GPU-generated bundles load on CPU with all model/optimizer/normalizer/replay/runner/RNG payloads exact; private sampler supports deterministic logical continuation",
        "limitations":["No step-for-step GPU rollout comparison was performed","Global CUDA RNG is preserved as pending payload; it was not restored on a device in this CPU audit",
                       "PhysX contact/solver caches are not serialized; logical state restoration is not bitwise physical-equivalence evidence",
                       "4+4 update pilot verifies learning/checkpoint integration, not convergence/style or paper reproduction"],
        "script":helpers.record(Path(__file__))}
    output = DOCS/"stratified_expert_pilot_resume_review.json"
    output.write_text(json.dumps(review,indent=2,allow_nan=False)+"\n")
    print("PASS actual4+4pilot full CPU payload restore;65,536transitions; segmentfalls5+1; no physical-equivalence claim")
    print(json.dumps({step:value["checkpoint"] for step,value in results.items()},indent=2))


if __name__ == "__main__":
    main()
