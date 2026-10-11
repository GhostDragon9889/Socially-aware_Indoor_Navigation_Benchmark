"""Pre-SDK configuration/resource failures and the existing CLI contract.

Small signature fixtures exercise the deliberately shallow file preflight;
they are not replacement model weights or proof of model compatibility.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from CrowdSim.humanoid.application.configuration import (
    ConfigurationError,
    load_amp_profile,
    parse_amp_args,
)
from CrowdSim.humanoid.application.resources import (
    ResourceError,
    inspect_resource,
    preflight_amp_resources,
    resolve_local_path,
)
from CrowdSim.humanoid.application.settings import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
SIGNATURE_FIXTURE = b"PK\x03\x04signature-check-only-not-a-model"


@pytest.fixture
def profile_data():
    return {"mode": "paper_method_reconstruction", "training": {},
            "initialization": {}, "author_experiment_reproduction": False}


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.fixture
def local_files(tmp_path, monkeypatch, profile_data):
    source = tmp_path / "resources"
    source.mkdir()
    rig = source / "assets/mjcf/smpl_humanoid.xml"
    rig.parent.mkdir(parents=True)
    rig.write_text('<mujoco model="minimal-file-preflight"><worldbody/></mujoco>')
    monkeypatch.setenv("PROTOMOTIONS_ASSET_ROOT", str(rig.parents[1]))
    packages = {}
    for split in ("train", "validation", "test"):
        path = source / f"{split}.pt"
        path.write_bytes(SIGNATURE_FIXTURE)
        packages[split] = {"path": path.name, "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "clip_count": 1, "clips": [{"motion_id": 0,
                               "source_path": f"{split}/walking.npz", "diffusion_eligible": True}]}
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"schema": "navisaaclab.amass_reconstruction.v1", "splits": packages}))
    profile_path = write_yaml(source / "profile.yaml", profile_data)
    diffusion = source / "diffusion.ckpt"
    diffusion.write_bytes(SIGNATURE_FIXTURE)
    cwd = tmp_path / "another-working-directory"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    args = SimpleNamespace(config=str(profile_path), manifest=str(manifest),
                           diffusion_checkpoint=str(diffusion), from_scratch=True,
                           resume=None, num_envs=2, scene_config=None, capture=False, seed=7)
    return args, profile_data, packages


def test_cli_preserves_paths_and_legacy_defaults_and_explicit_new_flags(tmp_path):
    diffusion = tmp_path / "diffusion.ckpt"
    diffusion.write_bytes(SIGNATURE_FIXTURE)
    args = parse_amp_args(["--diffusion-checkpoint", str(diffusion), "--output", str(tmp_path / "run")])
    assert args.config == "configs/diffusion_amp/method_reconstruction.yaml"
    assert args.manifest == "data/diffusion_amp/amass_manifest.json"
    assert (args.num_envs, args.updates, args.rollout_steps, args.replan_steps) == (64, 1000, 32, 15)
    assert args.route_guidance == "endpoint"
    assert args.diffusion_guidance_mode == "legacy_posterior"
    assert args.guidance_weight is None
    assert args.agent_collision_weight == args.scene_collision_weight == 0
    assert not args.pairwise_contacts and args.contact_threshold_n == 1
    assert args.spawn_margin_m == 6


@pytest.mark.parametrize("flags", [
    ["--guidance-weight", "nan"],
    ["--guidance-weight", "-1"],
    ["--contact-threshold-n", "0"],
    ["--pairwise-contacts"],
    ["--pairwise-contacts", "--evaluate-steps", "5", "--resume", "resume.pt",
     "--power-source", "legacy_pd_proxy_misaligned"],
    ["--capture", "--evaluate-steps", "5", "--resume", "resume.pt", "--physics-only"],
    ["--route-guidance", "timed_path", "--evaluate-steps", "5", "--resume", "resume.pt"],
    ["--agent-collision-weight", "1", "--evaluate-steps", "5", "--resume", "resume.pt"],
    ["--scene-collision-weight", "1", "--diffusion-guidance-mode", "trace_clean"],
])
def test_incompatible_physics_and_guidance_requests_fail_before_sdk(tmp_path, flags):
    diffusion = tmp_path / "diffusion.ckpt"
    diffusion.write_bytes(SIGNATURE_FIXTURE)
    with pytest.raises(SystemExit) as failure:
        parse_amp_args(["--diffusion-checkpoint", str(diffusion), "--output", str(tmp_path / "run"), *flags])
    assert failure.value.code == 2


def test_cli_allows_declared_contact_and_joint_clean_evaluation(tmp_path):
    diffusion = tmp_path / "diffusion.ckpt"
    diffusion.write_bytes(SIGNATURE_FIXTURE)
    args = parse_amp_args(["--diffusion-checkpoint", str(diffusion), "--output", str(tmp_path / "run"),
        "--evaluate-steps", "6000", "--resume", "resume.pt", "--num-envs", "4",
        "--scene-config", "declared-scene.yaml", "--route-guidance", "timed_path",
        "--diffusion-guidance-mode", "trace_clean_variance", "--guidance-weight", "10",
        "--pairwise-contacts", "--agent-collision-weight", "1", "--scene-collision-weight", "2"])
    assert args.guidance_weight == 10 and args.pairwise_contacts
    assert args.scene_collision_weight == 2 and args.agent_collision_weight == 1


@pytest.mark.parametrize("value", [None, [], "scalar", 4])
def test_nonmapping_yaml_is_an_actionable_configuration_error(tmp_path, value):
    path = write_yaml(tmp_path / "profile.yaml", value)
    with pytest.raises(ConfigurationError, match="mapping"):
        load_amp_profile(path)


@pytest.mark.parametrize("changes,match", [
    ({"mode": "strict_author_experiment"}, "unpublished"),
    ({"training": {"learnig_rate": 2e-5}}, "Unknown"),
    ({"training": {"history_frames": 11}}, "history10"),
    ({"training": {"self_dim": 3}}, "self357"),
    ({"training": {"freeze_toes": False}}, "frozen"),
    ({"training": {"control_dt": .02}}, "30Hz"),
    ({"training": {"normalize_observations": "false"}}, "booleans"),
    ({"reward": {"task_weight": 1.}}, "fixed"),
    ({"initialization": []}, "mapping"),
])
def test_invalid_or_unimplemented_profile_contracts_fail_before_sdk(tmp_path, profile_data, changes, match):
    profile_data.update(changes)
    path = write_yaml(tmp_path / "profile.yaml", profile_data)
    with pytest.raises(ConfigurationError, match=match):
        load_amp_profile(path)


def test_actual_method_and_public_pacer_ablation_profiles_pass_the_same_pure_validator():
    method = load_amp_profile(ROOT / "configs/diffusion_amp/method_reconstruction.yaml")
    baseline = load_amp_profile(ROOT / "configs/diffusion_amp/amp_pacer_adaptation.yaml")
    assert method["author_experiment_reproduction"] is False
    assert baseline["training"]["discriminator_input"] == "state"
    assert baseline["training"]["warm_start_discriminator"] is True


@pytest.mark.parametrize("key", [field.name for field in fields(TrainingConfig) if type(field.default) is float])
@pytest.mark.parametrize("boolean", [False, True])
def test_yaml_boolean_cannot_be_interpreted_as_any_numeric_training_scalar(tmp_path, profile_data, key, boolean):
    profile_data["training"][key] = boolean
    path = write_yaml(tmp_path / "profile.yaml", profile_data)
    with pytest.raises(ConfigurationError, match=f"{key}.*boolean"):
        load_amp_profile(path)


def test_full_preflight_is_manifest_relative_and_does_not_overstate_signature_fixtures(local_files):
    args, profile, _ = local_files
    report = preflight_amp_resources(args, profile)
    assert report.require("amass_train") == Path(args.manifest).parent / "train.pt"
    assert report.split_clip_counts == {"train": 1, "validation": 1, "test": 1}
    evidence = report.as_dict()
    assert evidence["sdk_started"] is evidence["models_deserialized"] is False
    assert evidence["author_experiment_reproduced"] is False
    assert report.require("smpl_physics_rig").name == "smpl_humanoid.xml"
    assert all(not record.sha256_recomputed for record in report.records)
    assert next(record.header_kind for record in report.records if record.name == "amass_train") == "torch_zip_signature_only"
    json.dumps(evidence)
    with pytest.raises(ResourceError, match="not selected"):
        report.require("warm_start_checkpoint")


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_cli_seed_override_cannot_defer_failure_until_after_sdk(local_files, seed):
    args, profile, _ = local_files
    args.seed = seed
    with pytest.raises(ResourceError, match="seed"):
        preflight_amp_resources(args, profile)


@pytest.mark.parametrize("payload,match", [
    (b"", "empty"),
    (b"version https://git-lfs.github.com/spec/v1\noid sha256:123", "LFS"),
    (b"html server error", "signature"),
])
def test_corrupt_missing_downloads_are_not_accepted_as_checkpoint_presence(tmp_path, payload, match):
    path = tmp_path / "weight.pt"
    path.write_bytes(payload)
    with pytest.raises(ResourceError, match=match):
        inspect_resource("checkpoint", path, kind="torch")


def test_selected_warm_start_profile_path_and_checksum_are_used_instead_of_default(local_files):
    args, profile, _ = local_files
    checkpoint = Path(args.manifest).parent / "explicit-pacer.pt"
    checkpoint.write_bytes(SIGNATURE_FIXTURE)
    profile["initialization"] = {"checkpoint": str(checkpoint),
                                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    args.from_scratch = False
    report = preflight_amp_resources(args, profile)
    assert report.require("warm_start_checkpoint") == checkpoint
    assert next(record.sha256_recomputed for record in report.records if record.name == "warm_start_checkpoint")
    checkpoint.write_bytes(SIGNATURE_FIXTURE + b"changed")
    with pytest.raises(ResourceError, match="checksum"):
        preflight_amp_resources(args, profile)


@pytest.mark.parametrize("change,match", [
    ({"clip_count": True}, "count"),
    ({"bytes": 1}, "byte size"),
    ({"sha256": "invalid"}, "SHA256"),
    ({"clips": [{"motion_id": True, "source_path": "train/walk"}]}, "clip ID"),
])
def test_manifest_inventory_contract_errors_fail_before_motion_mmap(local_files, change, match):
    args, profile, packages = local_files
    packages["train"].update(change)
    Path(args.manifest).write_text(json.dumps({"schema": "navisaaclab.amass_reconstruction.v1", "splits": packages}))
    with pytest.raises(ResourceError, match=match):
        preflight_amp_resources(args, profile)


def test_heldout_source_identity_leak_is_rejected_before_sdk(local_files):
    args, profile, packages = local_files
    packages["validation"]["clips"][0]["source_path"] = packages["train"]["clips"][0]["source_path"]
    Path(args.manifest).write_text(json.dumps({"schema": "navisaaclab.amass_reconstruction.v1", "splits": packages}))
    with pytest.raises(ResourceError, match="leakage"):
        preflight_amp_resources(args, profile)


def test_from_scratch_cannot_claim_transferred_public_discriminator(local_files):
    args, profile, _ = local_files
    profile["training"]["warm_start_discriminator"] = True
    with pytest.raises(ResourceError, match="From-scratch"):
        preflight_amp_resources(args, profile)


@pytest.mark.parametrize("change,match", [
    ({"origins_xy": [[0, 0]]}, "one explicit"),
    ({"origins_xy": [[0, 0], [7, 0]]}, "outside"),
    ({"cuboids": [{"center": [0, 0, 1], "size": [1, -1, 2]}]}, "sizes"),
    ({"arrival_radius_m": float("nan")}, "arrival_radius"),
])
def test_bad_shared_scene_metadata_is_rejected_before_spawn(local_files, change, match):
    args, profile, _ = local_files
    scene = {"schema": "navisaaclab.reconstruction_scene.v1", "bounds_xy": [-2, -2, 6, 6],
             "origins_xy": [[0, 0], [1, 1]], "routes_xy": [[[2, 2], [4, 4]], [[2, 4], [4, 2]]]}
    scene.update(change)
    args.scene_config = str(write_yaml(Path(args.manifest).parent / "scene.yaml", scene))
    with pytest.raises(ResourceError, match=match):
        preflight_amp_resources(args, profile)


def test_capture_uses_existing_model_override_and_optional_files_are_only_inventory(local_files, monkeypatch):
    args, profile, _ = local_files
    visual_dir = Path(args.manifest).parent / "separately-managed-smpl"
    (visual_dir / "smpl").mkdir(parents=True)
    model = visual_dir / "smpl/SMPL_NEUTRAL.pkl"
    model.write_bytes(b"numeric-model-presence-only")
    monkeypatch.setenv("CROWDSIM_SMPL_MODEL_DIR", str(visual_dir))
    args.capture = True
    assert preflight_amp_resources(args, profile).require("smpl_visual_model") == model
    assert not preflight_amp_resources(args, profile).optional_capture_textures_present
    (visual_dir / "smpl_uv.npz").write_bytes(b"presence-only")
    (visual_dir / "texture.png").write_bytes(b"presence-only")
    assert preflight_amp_resources(args, profile).optional_capture_textures_present


def test_explicit_local_paths_are_not_replaced_with_other_assets(tmp_path):
    with pytest.raises(ResourceError, match="local resources"):
        resolve_local_path("https://example.org/model.pt")
    with pytest.raises(ResourceError, match="missing"):
        inspect_resource("checkpoint", tmp_path / "wrong.pt")
    assert resolve_local_path("relative.pt", relative_to=tmp_path) == tmp_path / "relative.pt"


def test_application_and_legacy_entry_imports_are_torch_and_kit_free_in_fresh_python(tmp_path):
    source = """
import json, runpy, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + '/src')
from CrowdSim.humanoid.application.configuration import load_amp_profile
from CrowdSim.humanoid.application.resources import preflight_amp_resources
load_amp_profile(sys.argv[1] + '/configs/diffusion_amp/method_reconstruction.yaml')
runpy.run_path(sys.argv[1] + '/scripts/train_paper_amp.py', run_name='import_probe')
for prefix in ('torch', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'warp'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
print(json.dumps({'torch_or_sdk_imported': False}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", source, str(ROOT)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout)["torch_or_sdk_imported"] is False


def test_preflight_module_runs_the_same_complete_file_gate_without_torch(local_files):
    args, _, _ = local_files
    source = """
import json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + '/src')
from CrowdSim.humanoid.application.preflight import main
status = main(sys.argv[2:])
for prefix in ('torch', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'warp'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
raise SystemExit(status)
"""
    result = subprocess.run([sys.executable, "-I", "-c", source, str(ROOT),
        "--config", args.config, "--manifest", args.manifest,
        "--diffusion-checkpoint", args.diffusion_checkpoint,
        "--output", str(Path(args.manifest).parent / "unused-run"), "--from-scratch"],
        capture_output=True, text=True, timeout=20, check=True)
    report = json.loads(result.stdout)["resource_preflight"]
    assert report["status"] == "local_file_preflight_passed"
    assert report["split_clip_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["models_deserialized"] is report["sdk_started"] is False


def test_profile_loader_does_not_mutate_yaml_content(tmp_path, profile_data):
    expected = copy.deepcopy(profile_data)
    assert load_amp_profile(write_yaml(tmp_path / "profile.yaml", profile_data)) == expected
