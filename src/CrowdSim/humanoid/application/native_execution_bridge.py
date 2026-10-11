"""Bind registered native commands to externally observed tool exits.

No child is launched here. A tools API observer supplies its actual terminal
response. Training/owner completion, resources and result identities are cold
admitted before a successful SDK receipt can enter the raw/style readers.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex

from CrowdSim.humanoid.analysis import native_navigation_seeded_audit as raw
from CrowdSim.humanoid.analysis import native_navigation_seeded_style as style
from CrowdSim.humanoid.analysis import native_seed_protocol as protocol
from scripts.compare_shared_social_evidence import bound

ROOT, record, require = raw.ROOT, raw.original.record, raw.original.require
PYTHON = ROOT / ".venv/bin/python"
MATRIX = ROOT / "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/registered_S3_native_matrix.json"
MATRIX_SHA = "e8a72e6d0971459eac68ac499a2f05b6253cda7804502afea6b5dc591730d50d"
CURRENT_MATRIX_SHA = "29423976c0e6d40e940ca2ba057950aeefa8c4fbc927046324791ce132af44c2"
S3 = ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1"
TRAINING_RECEIPT = S3 / "budget12500_execution.json"
CHECKPOINT = ROOT / "output/diffusion_amp/shared_social_budget12500_v1/step_012500.pt"
OWNER_SHA = "e93799be375dcd52dae3c897b9e26fad448cb1e3136611e725ba7ee4c4744f9e"
DIFFUSION = ROOT / "output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt"
DIFFUSION_SHA = "4581f29de20c91865ad30645564f4c93a0edd47a8d88f9e496359b9c52d270ff"
OBSERVATION_SCHEMA = "shared_social.external_native_tool_observation.v1"


def readonly_json(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_nlink == 1 and not path.stat().st_mode & 0o222,
            "Require an ordinary readonly external observation/registration")
    identity = record(path)
    value = json.loads(path.read_text())
    require(record(path) == identity, "Observation changed while reading")
    return value, identity


def current_matrix():
    """Admit the whole current matrix before selecting any individual case."""
    from . import native_matrix_migration
    historical, original_identity = readonly_json(MATRIX)
    require(original_identity["sha256"] == MATRIX_SHA, "Exact native156 pre-registration changed")
    matrix, identity = readonly_json(native_matrix_migration.MATRIX)
    require(identity["sha256"] == CURRENT_MATRIX_SHA
            and matrix["registration_migration"]["historical_matrix"] == original_identity
            and matrix["registration_migration"]["adapter"] == record(native_matrix_migration.__file__)
            and matrix["protocol_source"] == record(protocol.__file__)
            and matrix["sources"] == raw.entry.source_bindings(),
            "Current native156 registration or sources changed")
    return matrix, identity


def prepare_case(case_id):
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene

    matrix, identity = current_matrix()
    selected = [row for row in matrix["cases"] if row["case_id"] == case_id]
    require(len(selected) == 1, "Select exactly one pre-registered native case")
    case = selected[0]
    require(case["status"] == "pending_actual_training_and_native_SDK" and len(case["arms"]) == 2,
            "Retain native resource rejection; no replacement or resampling")
    raw._protocol(case["actors"], case["control_steps"])
    raw.entry.validate_seed(case["seed"])
    request = preflight_native_scene(case["scene_configuration"]["path"], actors=case["actors"])
    require(request["configuration_file"] == case["scene_configuration"]
        and case["same_AMP_checkpoint_for_both_arms"] == str(CHECKPOINT), "Case resource or terminal candidate differs")
    parent = Path(case["arms"][0]["output"]).parent
    require(parent.is_relative_to(ROOT / "output/diffusion_amp") and not parent.is_symlink(), "Native output outside owned namespace")
    common = ["--actors", str(case["actors"]), "--seed", str(case["seed"]),
        "--scene-config", request["configuration_file"]["path"], "--scene-sha256", request["configuration_file"]["sha256"]]
    stages = []
    for controls in (30, 6000):
        for arm in case["arms"]:
            planner = arm["planner"]
            output = Path(arm["output"]) if controls == 6000 else parent / "smoke30" / planner
            sdk = [str(PYTHON), str(raw.ENTRY), *common, "--planner", planner,
                "--execution-receipt", str(TRAINING_RECEIPT), "--diffusion-checkpoint", str(DIFFUSION),
                "--diffusion-sha256", DIFFUSION_SHA, "--control-steps", str(controls), "--output", str(output)]
            sdk_receipt = parent / "observations" / f"{planner}_C{controls}_SDK_exit.json"
            audit = parent / "observations" / f"{planner}_C{controls}_raw.json"
            audit_receipt = parent / "observations" / f"{planner}_C{controls}_raw_CPU_exit.json"
            raw_command = [str(PYTHON), str(style.RAW_CLI), *common, "--planner", planner,
                "--control-steps", str(controls), "--execution-receipt", str(sdk_receipt), "--output", str(audit)]
            stages.append({"planner": planner, "controls": controls, "output": str(output), "SDK_command": sdk,
                "SDK_receipt": str(sdk_receipt), "raw_command": raw_command, "raw_report": str(audit),
                "raw_CPU_receipt": str(audit_receipt), "CPU_or_SDK_exit_observed": False})
    comparisons = []
    for controls in (30, 6000):
        outputs = {row["planner"]: Path(row["output"]) for row in stages if row["controls"] == controls}
        comparisons.append({"controls": controls, "command": [str(PYTHON),
            str(ROOT / "scripts/compare_shared_social_native_initial_conditions.py"),
            "--left", str(outputs["diffusion"] / "paired_initial_condition.json"),
            "--right", str(outputs["sfm"] / "paired_initial_condition.json"),
            "--output", str(parent / "observations" / f"initial_pair_C{controls}.json")]})
    full = {row["planner"]: row for row in stages if row["controls"] == 6000}
    style_command = [str(PYTHON), str(ROOT / "scripts/evaluate_shared_social_native_seeded_style.py"),
        *common, "--diffusion-audit-execution", full["diffusion"]["raw_CPU_receipt"],
        "--sfm-audit-execution", full["sfm"]["raw_CPU_receipt"],
        "--output", str(parent / "observations" / "inclusive_style_200s.json")]
    return {"schema": "shared_social.registered_native_case_tool_plan.v1", "case": case, "matrix": identity,
        "request": request, "stages": stages, "source": record(__file__), "raw_sources": raw.source_bindings(),
        "style_sources": style.source_bindings(), "initial_comparisons": comparisons, "style_command": style_command,
        "both_live_initial_pairings_required": True, "readonly_raw_array_seal_required_before_style": True,
        "GPU_or_SDK_started": False, "actual_execution_admitted": False}


def tool_command(argv, *, cuda):
    return shlex.join(["env", "CUDA_VISIBLE_DEVICES=" + cuda, "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
                      "OPENBLAS_NUM_THREADS=4", "PYTHONDONTWRITEBYTECODE=1", *argv])


def validate_tool_observation(packet, *, command, cuda):
    require(isinstance(packet, dict) and isinstance(packet.get("response"), dict),
            "Actual terminal tools API observation must be a mapping with a response")
    response = packet.get("response", {})
    require(packet.get("schema") == OBSERVATION_SCHEMA and packet.get("cmd") == tool_command(command, cuda=cuda)
        and packet.get("workdir") == str(ROOT) and packet.get("actual_terminal_response_observed") is True
        and type(response.get("exit_code")) is int and isinstance(response.get("chunk_id"), str)
        and bool(response["chunk_id"]), "Require this command's actual terminal tools API response; no inferred or invented exit")
    return {"actual_exit_code": response["exit_code"], "terminal_tool_chunk": response["chunk_id"]}


def admit_S3_owner_completion(certificate_path):
    from scripts.evaluate_shared_social_tracking import admit_training

    certificate, certificate_identity = readonly_json(certificate_path)
    owner_source = record(S3 / "terminal_tool_owner.js")
    require(certificate.get("schema") == "shared_social.external_S3_owner_completion.v1"
        and certificate.get("completed_exec_cell_id") == "581" and certificate.get("completion_observed_by_root") is True
        and certificate.get("owner_source") == owner_source and owner_source["sha256"] == OWNER_SHA,
        "Require externally observed S3 owner completion; training receipt alone does not release its later GPU stages")
    owner = bound(certificate["terminal_owner_state"])
    require(owner["stage"] == "completed_measurements_quality_unclassified" and owner["training_exit_observed"] is True
        and type(owner["training_terminal"]["exit_code"]) is int and owner["training_terminal"]["exit_code"] == 0,
        "S3 training and its serial post-evaluation owner are not complete")
    checkpoint, sha, training = admit_training(TRAINING_RECEIPT)
    result = bound(training["result"])
    require(checkpoint == CHECKPOINT and result["completed_updates"] == 12500
        and result["actual_training_transitions"] == 102400000 and training["value_normalization"] == "popart",
        "Require the complete preselected S3 terminal candidate, never an intermediate or S1 checkpoint")
    require(record(DIFFUSION)["sha256"] == DIFFUSION_SHA, "Fixed formal Diffusion40k changed")
    return {"external_owner_completion": certificate_identity, "training": training, "checkpoint_sha256": sha}


def _stage(plan, planner, controls):
    require(planner in ("diffusion", "sfm") and type(controls) is int and controls in (30, 6000),
            "Explicit registered planner/control stage required")
    selected = [row for row in plan["stages"] if row["planner"] == planner and row["controls"] == controls]
    require(len(selected) == 1, "Native tool stage is ambiguous")
    return selected[0]


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    path.chmod(0o444)
    return record(path)


def navigation_summary(result, actors):
    """Publish driver counts for the independent raw-array cross-check."""
    falls, arrivals = result.get("fall_resets"), result.get("route_vertices_reached")
    require(type(actors) is int and 3 <= actors <= 15
            and type(falls) is int and falls >= 0
            and type(arrivals) is list and len(arrivals) == actors
            and all(type(value) is int and value >= 0 for value in arrivals),
            "Require complete integer navigation termination and route summaries")
    return {"fall_resets": falls, "route_vertices_reached": list(arrivals)}


def publish_sdk_exit(case_id, *, planner, controls, observation_path, owner_completion):
    admitted = admit_S3_owner_completion(owner_completion)
    plan = prepare_case(case_id)
    stage = _stage(plan, planner, controls)
    packet, observed_identity = readonly_json(observation_path)
    observed = validate_tool_observation(packet, command=stage["SDK_command"], cuda="0")
    result_path = Path(stage["output"]) / "result.json"
    receipt = {"schema": raw.EXECUTION_SCHEMA, **observed, "owner": record(__file__),
        "tool_observation": observed_identity, "actors": plan["case"]["actors"], "control_steps": controls,
        "planner": planner, "seed": plan["case"]["seed"], "scene_configuration": plan["request"]["configuration_file"],
        "status": "actual_failed_SDK_exit" if observed["actual_exit_code"] else "completed_measurement_quality_unclassified",
        "paper_effect_achieved": False, "default_model_promoted": False}
    if observed["actual_exit_code"] != 0:
        return _write_new(stage["SDK_receipt"], receipt), False
    result = json.loads(result_path.read_text())
    require(result["shared_navigation_migration"]["training_execution_admission"] == admitted["training"],
            "Actual native SDK used another terminal AMP candidate")
    receipt.update(navigation_summary(result, plan["case"]["actors"]))
    receipt.update(result=record(result_path), raw_arrays=[record(Path(stage["output"]) / name) for name in
        ("physical_states.npz", "ground_support.npz", "reference_plans.npz")])
    identity = _write_new(stage["SDK_receipt"], receipt)
    raw.admit_execution(identity["path"], actors=plan["case"]["actors"], controls=controls,
        planner=planner, seed=plan["case"]["seed"], scene_config=plan["request"]["configuration_file"]["path"],
        scene_sha256=plan["request"]["configuration_file"]["sha256"])
    return identity, True


def publish_raw_CPU_exit(case_id, *, planner, controls, observation_path, owner_completion):
    admit_S3_owner_completion(owner_completion)
    plan = prepare_case(case_id)
    stage = _stage(plan, planner, controls)
    packet, observed_identity = readonly_json(observation_path)
    observed = validate_tool_observation(packet, command=stage["raw_command"], cuda="")
    receipt = {"schema": style.OWNER_SCHEMA, **observed, "owner": record(__file__), "tool_observation": observed_identity,
        "actors": plan["case"]["actors"], "control_steps": controls, "planner": planner, "seed": plan["case"]["seed"],
        "scene_configuration": plan["request"]["configuration_file"], "paper_effect_achieved": False}
    if observed["actual_exit_code"] != 0:
        return _write_new(stage["raw_CPU_receipt"], receipt), False
    audit_identity = record(stage["raw_report"])
    audit = json.loads(Path(stage["raw_report"]).read_text())
    selected = raw.admit_execution(stage["SDK_receipt"], actors=plan["case"]["actors"], controls=controls,
        planner=planner, seed=plan["case"]["seed"], scene_config=plan["request"]["configuration_file"]["path"],
        scene_sha256=plan["request"]["configuration_file"]["sha256"])
    kernels = raw.prepare_kernels(actors=plan["case"]["actors"], controls=controls, seed=plan["case"]["seed"])
    exposures = {"actors": plan["case"]["actors"], "actual_controls": controls,
        "actor_controls": controls * plan["case"]["actors"], "physics_substeps": controls * 4,
        "native_actor_substep_samples": controls * 4 * plan["case"]["actors"],
        "unordered_pair_controls_exposure": controls * plan["case"]["actors"] * (plan["case"]["actors"] - 1) // 2}
    require(audit["schema"] == raw.AUDIT_SCHEMA and audit["status"] == "passed"
        and all(type(audit.get(key)) is int and audit[key] == value for key, value in exposures.items())
        and type(audit["seed"]) is int and audit["seed"] == plan["case"]["seed"] and audit["planner"] == planner
        and audit["execution_receipt"] == record(stage["SDK_receipt"])
        and audit["reader"] == record(style.RAW_CLI) and audit["kernel_sources"] == kernels.sources
        and audit["kernel_derivation"] == kernels.derivation and audit["execution_admission"] == selected.admission
        and audit["native_scene_request"] == selected.request and audit["result"] == selected.receipt["result"]
        and audit["Torch_imported"] is False and audit["SDK_executed_by_reader"] is False
        and audit["metric_formulae_changed"] is False, "Actual raw CPU report disagrees with registered SDK/source/stage")
    receipt["audit"] = audit_identity
    if controls == 6000:
        style.validate_audit_contract(receipt, audit, kernels, planner=planner, seed=plan["case"]["seed"])
    require(record(stage["raw_report"]) == audit_identity and record(observation_path) == observed_identity,
            "Raw CPU report or actual tool observation changed during publication")
    return _write_new(stage["raw_CPU_receipt"], receipt), True
