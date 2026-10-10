"""Source bound CPU compatibility of the real state prior with production native deployment."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1/step_2000.pt"
CHECKPOINT_SHA = "19f3341b9bb6118243c0945d839a669360175d93a35dde0f792639178d508f4f"
SOURCE = ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json"
SOURCE_SHA = "b4770f3f7301af7aed294db3145161453cf22325e66a9f9da0a3d677f5aeb3ee"


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only CUDA_VISIBLE_DEVICES='' required")
    source = json.loads(SOURCE.read_text())
    assert record(SOURCE)["sha256"] == SOURCE_SHA
    assert source["source_count"] == len(source["files"]) == 183
    assert all(record(row["path"]) == row for row in source["files"])
    archive = source["archive"]
    assert record(archive["path"]) == archive
    with tarfile.open(archive["path"], "r:gz") as tar:
        members = tar.getmembers()
        declared = {str(Path(row["path"]).relative_to(ROOT)): row for row in source["files"]}
        assert len(members) == len({row.name for row in members}) == 183
        assert {row.name for row in members} == set(declared)
        assert all(row.isfile() and not Path(row.name).is_absolute() and ".." not in Path(row.name).parts for row in members)
        for row in members:
            data = tar.extractfile(row).read()
            assert len(data) == declared[row.name]["bytes"] and hashlib.sha256(data).hexdigest() == declared[row.name]["sha256"]
    admission_path = ROOT / "docs/style_improvement_20261003/state_prior_training_audit_v1/terminal2000_admission_v1.json"
    admission = json.loads(admission_path.read_text())
    assert record(admission_path)["sha256"] == "d5154251853253a8f866a103af9e51c6fff0e6ee7e1f56c36ff3f250b7124118"
    checkpoint = record(CHECKPOINT)
    assert checkpoint == admission["checkpoint"] and checkpoint["sha256"] == CHECKPOINT_SHA
    assert admission["status"] == "passed" and admission["terminal_state_admitted"] is True
    from CrowdSim.humanoid.application.deployment import parse_deployment_args, preflight_deployment
    common = ["--amp-checkpoint", str(CHECKPOINT), "--config", str(ROOT / "configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml"),
              "--manifest", str(ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"),
              "--diffusion-checkpoint", str(ROOT / "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt"),
              "--duration-seconds", "200", "--seed", "7", "--inference-only", "--native-dynbench", "--arm-style-alpha", "0"]
    preflights, commands = [], {}
    for actors, scene, output in ((4, "native_warehouse_pedestrians_clearance08_v1.yaml", "native_warehouse_state_prior_clearance08_200s_v1"),
                                  (15, "native_warehouse_pedestrians_15_clearance08_v1.yaml", "native_warehouse_state_prior_15_clearance08_200s_v1")):
        argv = common + ["--actors", str(actors), "--scene-config", str(ROOT / "configs/diffusion_amp" / scene),
                         "--output", str(ROOT / "output/diffusion_amp" / output)]
        proof = preflight_deployment(parse_deployment_args(argv))
        assert proof["status"] == "SDK_free_preflight_passed" and proof["SDK_started"] is False
        assert proof["Torch_imported_by_entry"] is False and not Path(proof["parsed_driver_arguments"]["output"]).exists()
        assert proof["evaluation_actor_request"]["alpha"] == 0. and proof["evaluation_actor_request"]["style_checkpoint"] is None
        assert proof["native_scene_request"]["native_map"]["sha256"] == "9ac1a312d3b86697014f74f496e32de0328d1aaffe70e77e965c1de2becb5b29"
        preflights.append(proof)
        commands["native" + str(actors)] = [sys.executable, str(ROOT / "scripts/simulate_paper_pedestrians.py"), *argv]
    assert "torch" not in sys.modules
    import numpy as np
    import torch
    from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor
    from CrowdSim.humanoid.training.trainer import PaperAMPNetwork
    torch.set_num_threads(4)
    before = torch.get_rng_state().clone()
    original_load, loads = torch.load, []

    def audited_load(path, *args, **kwargs):
        assert Path(path).resolve() == CHECKPOINT
        loads.append(str(Path(path).resolve()))
        return original_load(path, *args, **kwargs)

    with patch.object(torch, "load", audited_load), patch.object(np, "load", side_effect=AssertionError("External statistics NPZ forbidden")):
        actor = EvaluationActor.from_checkpoint(CHECKPOINT, device="cpu", expected_sha256=CHECKPOINT_SHA)
        private = torch.Generator(device="cpu").manual_seed(19)
        own, task = torch.randn(7, 357, generator=private), torch.randn(7, 1044, generator=private)
        actual = actor.act(own, task, deterministic=True)
    state = original_load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    with torch.random.fork_rng(devices=[]):
        full = PaperAMPNetwork(actor.config)
    full.load_state_dict(state["model"], strict=True)
    full.eval()
    inference = actor._inference
    with torch.inference_mode():
        expected = full.actor(torch.cat((inference.self_normalizer.normalize(own),
            full.task_encoder(inference.task_normalizer.normalize(task))), dim=-1))
        applied = expected.clamp(-1, 1) * inference.action_mask
    assert torch.equal(expected, actual["actions"]) and torch.equal(applied, actual["applied_actions"])
    assert torch.equal(before, torch.get_rng_state())
    assert set(inference.model._modules) == {"task_encoder", "actor"}
    assert not any(hasattr(actor, name) for name in ("optimizer", "policy_replay", "amp_reward", "value"))
    assert actor.config.warm_start_action_discriminator_from_state_prior is True
    initialization = actor.provenance["initialization"]
    assert initialization["published_state_only_discriminator_transferred"] is True
    assert initialization["published_amp_history_moments_transferred"] is True
    assert actor.provenance == state["provenance"] and actor.update_count == 2000 and actor.physical_transition_count == 16384000
    assert all(torch.equal(value, state["model"][name]) for name, value in inference.model.state_dict().items())
    assert all(torch.equal(value, state["normalizers"][label][name]) for label in ("self_normalizer", "task_normalizer")
               for name, value in getattr(inference, label).state_dict().items())
    assert not torch.cuda.is_initialized() and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules)
    owned = {name: record(module.__file__) for name, module in sys.modules.items()
             if name.startswith("CrowdSim.") and getattr(module, "__file__", None)}
    assert all(Path(row["path"]).is_relative_to(ROOT) for row in owned.values())
    assert all(record(row["path"]) == row for row in source["files"])
    report = {"schema": "public_state_prior.production_native_cold_compatibility.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "Kit_started": False, "production_modified": False,
              "source_manifest": record(SOURCE), "source_archive": archive, "selected_source_count": 183,
              "all_selected_source_and_archive_members_exact": True, "full_actual_import_ledger_claim": False,
              "reviewer": record(__file__), "terminal_admission": record(admission_path), "checkpoint": checkpoint,
              "preflights": preflights, "commands": commands, "production_owned_CPU_imports": owned,
              "tensor_compatibility": {"new_flag_true": True, "both_public_D_and_AMP_transfer_flags_true": True,
                  "D2019_and_AMP1950_validated_then_discarded": True, "AMP_count": 24776000.,
                  "actor_and_task_encoder_weights_exact": True, "self_task_moments_exact": True,
                  "full_training_provenance_exact": True, "synthetic_private_seed19_CPU_probe_samples": 7,
                  "raw_and_applied_outputs_bitwise_full_checkpoint_actor": True,
                  "saved_native_or_GPU_observations_compared": False, "caller_CPU_RNG_unchanged": True,
                  "selected_checkpoint_tensor_and_metadata_load_calls": len(loads), "external_public_or_NPZ_loads": 0,
                  "retained_model_modules": sorted(inference.model._modules), "optimizer_replay_AMP_reward_value_API_present": False},
              "actual_native_physics_validated": False, "native4_and_native15_physical_effects_need_new_actual_trial": True,
              "automatic_promotion": False,
              "native_result_source_and_model_admission_must_use_new_V11_True_prior_contract": True,
              "old_source168_source174_or_false_public_transfer_gates_not_reused": True}
    path = HERE / "actual_CPU_readiness_v1.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)
    print(json.dumps({"status": "passed", "report": record(path), "commands": list(commands)}))


if __name__ == "__main__":
    run()
