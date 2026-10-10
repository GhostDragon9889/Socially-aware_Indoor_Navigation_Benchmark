"""One source/archive and synthetic state1950 boundary; no checkpoint read."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "state_prior_stratified_training_audit_v1"
BUNDLE = ROOT / "output/isolated_training/mixed_state_terrain_state_prior_stratified_matched_v1"
PREP = HERE.parent / "state_only_matched_q_training_preparation_v1"


def record(path):
    path = Path(path)
    return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    edits = json.loads((HERE / "adapter_exact_replacements.json").read_text())
    main_text = (HERE / "audit_state_only_training.py").read_text()
    helper_text = (HERE / "state_prior_checks.py").read_text()
    inverse_main = main_text
    for before, after in reversed(edits["main_replacements"]):
        inverse_main = inverse_main.replace(after, before)
    inverse_helper = helper_text.replace(edits["helper_archive_derivation_after"], edits["helper_archive_derivation_before"])
    inverse_helper = inverse_helper.replace(edits["helper_state_prior_after"], edits["helper_state_prior_before"])
    for before, after in reversed(edits["helper_replacements"]):
        inverse_helper = inverse_helper.replace(after, before)
    assert inverse_main == (OLD / "audit_stratified_training.py").read_text()
    assert inverse_helper == (OLD / "state_prior_checks.py").read_text()
    assert ast.dump(ast.parse(inverse_main)) == ast.dump(ast.parse((OLD / "audit_stratified_training.py").read_text()))
    assert (HERE / "expert_sampling_checks.py").read_bytes() == (OLD / "expert_sampling_checks.py").read_bytes()
    sys.path[:0] = [str(HERE), str(BUNDLE / "code")]
    spec = importlib.util.spec_from_file_location("state_only_prepared_reader", HERE / "audit_state_only_training.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    checks = []

    def check(name, condition, **detail):
        checks.append({"name":name,"passed":bool(condition),**detail})
        if not condition:
            raise ValueError(name)

    source = json.loads((BUNDLE / "source_manifest.json").read_text())
    archive = auditor.verify_source_archive(BUNDLE, source, record, check)
    import torch
    from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES
    from CrowdSim.humanoid.application.configuration import load_amp_profile
    from state_prior_checks import PUBLIC_COUNTS, PUBLIC_SHA, validate_state_prior
    torch.set_num_threads(4)
    profile = load_amp_profile(BUNDLE / "code/configs/diffusion_amp/isolated_state_only_matched_q_absolute.yaml")
    config = deepcopy(profile["training"])
    config["action_mask"] = [int(name.rsplit("_",1)[0] not in {"L_Hand","R_Hand","L_Toe","R_Toe"}) for name in PACER_DOF_NAMES]
    initialization = {"mode":"transfer_warm_start_public_pacer_actor_critic","sha256":PUBLIC_SHA,
                      "actor_critic_task_transferred":True,"observation_value_moments_transferred":True,
                      "discriminator_initialization":"public_pacer_state_only","action_conditioned_discriminator_initialization":"not_applicable",
                      "discriminator_input":"state","amp_normalization":"per_history_coordinate",
                      "published_state_only_discriminator_transferred":True,"published_amp_history_moments_transferred":True,
                      "author_navisaaclab_weights":False,"normalizer_counts_before_cap":PUBLIC_COUNTS,
                      "normalizer_count_cap":200000.,"path":profile["initialization"]["checkpoint"]}
    shapes = {"discriminator.0.weight":(1024,1950),"discriminator.0.bias":(1024,),"discriminator.2.weight":(512,1024),
              "discriminator.2.bias":(512,),"discriminator.4.weight":(1,512),"discriminator.4.bias":(1,)}
    state = {"config":config,"update_count":4,"provenance":{"initialization":initialization,
              "amp_reward_timestamp":"pre-step h[t..]; state-only discriminator ignores applied a[t]",
              "expert_action_semantics":"not consumed by the explicit classic-AMP state-only discriminator"},
              "model":{key:torch.zeros(shape) for key,shape in shapes.items()},
              "normalizers":{"amp_normalizer":{"mean":torch.zeros(1950,dtype=torch.float64),
                    "variance":torch.ones(1950,dtype=torch.float64),"count":torch.tensor(200000.+4*12288,dtype=torch.float64)}}}
    validate_state_prior(state,check,"synthetic_metadata:")
    negatives = []
    for case in ("old_action_extension_flag","wrong_D2019_shape","action_extension_provenance","poststep_history"):
        mutant = deepcopy(state)
        if case == "old_action_extension_flag":
            mutant["config"]["warm_start_action_discriminator_from_state_prior"] = True
        elif case == "wrong_D2019_shape":
            mutant["model"]["discriminator.0.weight"] = torch.zeros(1024,2019)
        elif case == "action_extension_provenance":
            mutant["provenance"]["initialization"]["state_prior_action_extension"] = {}
        else:
            mutant["provenance"]["amp_reward_timestamp"] = "post-step h[t+1..] before resets"
        try:
            validate_state_prior(mutant,lambda name,condition: None if condition else (_ for _ in ()).throw(ValueError(name)))
        except ValueError as error:
            negatives.append({"case":case,"rejected":True,"error":str(error)})
        else:
            raise AssertionError(case)
    assert not torch.cuda.is_initialized() and not any(name.split(".")[0] in {"isaacsim","isaaclab","omni","pxr"} for name in sys.modules)
    report = {"schema":"state_only_matched_q.prepared_reader_CPU_boundary.v1","status":"passed","checks":checks,
              "source_archive":archive,"negative_cases":negatives,"main_inverse_whole_bytes_AST_exact":True,
              "helper_inverse_whole_bytes_AST_exact":True,"q_callback_bytes_exact":True,
              "original_full_model_2Adam_replay_RMS_RNG_terrain_q_restore_math_unchanged":True,
              "synthetic_tensor_metadata_only":True,"future_checkpoint_read":False,"actual_training_payload_admitted":False,
              "CPU_only":True,"GPU_used":False,"SDK_started":False,"models_read":False,"sealed34_outputs_read":False}
    with (HERE / "prepared_boundary_CPU.json").open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":"passed","source_and_boundary_checks":len(checks),"negative_cases":len(negatives),
                      "source_count":source["candidate_inputs"]["parent_code_count"]+1,"proof":record(HERE / "prepared_boundary_CPU.json")}))


if __name__ == "__main__":
    main()
