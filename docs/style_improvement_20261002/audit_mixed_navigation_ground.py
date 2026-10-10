"""CPU admission for registered mixed-AMP four-person navigation captures.

Reuses every frozen force/COM/contact check. Only the exact138 prelaunch
source admission changes. Does not import archived code, Torch, Kit or models.
Preparations certify no executed rollout; complete captures get independent
clock/plan/route/reset checks and retain every failed/fallen frame.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent.parent
SOURCE = DOCS / "mixed_curriculum_navigation_source_v1.json"
SOURCE_SHA = "285e4126f5683281437e312cc02c369fec58beaa8a901b649334340f4dbc637c"
ARCHIVE_SHA = "a06eae3d815e45fa2a67f6c40fda00d12354ad40717b3ff4ab46fe8f2bf48742"
TRAINING_SHA = "8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e"
ADMISSION = DOCS / "mixed_curriculum_terminal500_1000_2000_independent_audit_v1.json"
ADMISSION_SHA = "1894c4b252fc83c84328ff156c5e13547a574d22c560d0954187e4d7e2381e79"
HELPER = DOCS / "audit_ground_support_capture.py"
HELPER_SHA = "f851e7d5f90d48d830c564fdc2b2e446dfe35daa36ec636aba6f946ec7752c3f"
BASELINE_AUDIT = DOCS / "ground_support_200s_actual_verification.json"
BASELINE_AUDIT_SHA = "fb32c5fa56dae76478919e4d8c91748f6bd78a57a7b8d1bb9fc5741919410307"
CONFIGURATION = "CrowdSim/humanoid/application/configuration.py"
RUNNER = "scripts/train_paper_amp.py"
CONTROL_AST_SHA = "1adbfe51b09a2311b3e450925cface7b4f81a1fbb122dd945838d16a9272a74f"
STEPS = (500, 1000, 2000)
CHECKPOINT_SHA = {
    500: "42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b",
    1000: "76708bfa8cc2a55d621f8c3b39a864b66a3b2522eb15d2ca7f451583db165d7d",
    2000: "28aed874464379f0b0628719290d36af0a8b0c53691c0dd51f56102c5f428f72",
}
NEW_ENTRY_MODULES = ("CrowdSim.humanoid.runtime.amp_inference", "CrowdSim.humanoid.runtime.dynbench_scene",
                     "CrowdSim.humanoid.application.deployment")


def resolve(path):
    path = Path(path)
    return (ROOT/path if not path.is_absolute() else path).resolve()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_equal(left, right):
    """Python equality alone would accept YAML bool as an integer contract."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def argument_changes(candidate, baseline):
    """Only registered controller/profile/output differ; absent old eta is 0."""
    old = dict(baseline)
    if "near_horizon_waypoint_mass" not in old:
        old["near_horizon_waypoint_mass"] = 0.0
    changes = {k: {"baseline": old.get(k), "candidate": candidate.get(k)}
               for k in old.keys() | candidate.keys()
               if k not in old or k not in candidate or not strict_equal(old[k], candidate[k])}
    if set(changes) != {"config", "resume", "output"}:
        raise ValueError(f"Unregistered baseline navigation argument difference: {sorted(changes)}")
    return changes


def select_checkpoint(admission, step):
    if (type(step) is not int or step not in STEPS or admission.get("status") != "passed"
            or admission.get("CPU_only") is not True or admission.get("GPU_used") is not False
            or admission.get("SDK_started") is not False or admission.get("heldout_outputs_read") is not False
            or admission.get("check_count") != 220 or len(admission.get("checks", [])) != 220
            or not all(c.get("passed") is True for c in admission["checks"])):
        raise ValueError("Exact terminal220 CPU admission is required")
    rows = [c for c in admission["checkpoints"] if c["update_count"] == step]
    if len(rows) != 1:
        raise ValueError("Selected checkpoint admission is missing or duplicated")
    row = rows[0]
    if (row["checkpoint"]["sha256"] != CHECKPOINT_SHA[step] or row["checkpoint"]["bytes"] != 652179357
            or row["physical_transitions"] != step*8192 or row["control_step"] != step*32
            or row["num_envs"] != 256 or row["group_statistics"]["amass_actor_transitions"] != step*4096
            or row["group_statistics"]["procedural_actor_transitions"] != step*4096):
        raise ValueError("Selected checkpoint counter/hash/assignment differs from admitted milestone")
    return row


def ground_cli_nodes(text):
    selected = []
    for function in ast.parse(text).body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in function.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if any(isinstance(arg, ast.Constant) and arg.value in {
                        "--measure-ground-support", "--ground-support-threshold-n"} for arg in node.value.args):
                    selected.append((function.name, ast.dump(node, include_attributes=False)))
            if isinstance(node, ast.If) and any(isinstance(child, ast.Attribute) and child.attr in {
                    "measure_ground_support", "ground_support_threshold_n"} for child in ast.walk(node.test)):
                selected.append((function.name, ast.dump(node, include_attributes=False)))
    return selected


def control_loop(text):
    loops = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "step"]
    if len(loops) != 1:
        raise ValueError("One full physical control loop is required")
    return ast.dump(loops[0], include_attributes=False)


def validate_member_paths(records, members):
    declared = {item["path"]: item for item in records}
    if (len(declared) != len(records) or any(PurePosixPath(name).is_absolute() or
            ".." in PurePosixPath(name).parts or "\\" in name for name in declared)
            or len(members) != len(declared) or {m.name for m in members} != set(declared)
            or not all(m.isfile() and m.size <= 16*1024*1024 for m in members)):
        raise ValueError("Exact unique regular source archive paths required; no traversal/links/extras")
    return declared


def load_helper():
    if sha(HELPER.read_bytes()) != HELPER_SHA:
        raise ValueError("Frozen measurement auditor changed")
    spec = importlib.util.spec_from_file_location("mixed_navigation_frozen_measurement_auditor", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(module, audit, path):
    audit.require(resolve(path) == SOURCE.resolve() and sha(SOURCE.read_bytes()) == SOURCE_SHA,
                  "exact known138 navigation prelaunch source manifest")
    document = json.loads(SOURCE.read_text())
    archive_path = resolve(document["archive"]["path"])
    archive = module.fingerprint(archive_path)
    audit.require(archive["sha256"] == ARCHIVE_SHA == document["archive"]["sha256"]
                  and archive["bytes"] == document["archive"]["bytes"], "138 archive full SHA/size")
    with tarfile.open(archive_path) as stream:
        members = stream.getmembers()
        declared = validate_member_paths(document["files"], members)
        audit.require(len(declared) == 138, "registered138 exact safe member count")
        source = {}
        for member in members:
            raw = stream.extractfile(member).read()
            expected = declared[member.name]
            audit.require(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"],
                          "exact138 source member:"+member.name)
            source[member.name] = raw
    training_path = resolve(document["previous_source_manifest"])
    audit.require(sha(training_path.read_bytes()) == TRAINING_SHA, "parent135 training source identity")
    training = {r["path"]: r for r in json.loads(training_path.read_text())["files"]}
    changed = {name for name in training if declared[name]["sha256"] != training[name]["sha256"]
               or declared[name]["bytes"] != training[name]["bytes"]}
    registrations = {f"docs/style_improvement_20261002/mixed_curriculum_step{s}_ground_support_200s_preparation_v1.json" for s in STEPS}
    audit.require(len(training) == 135 and set(training) <= set(declared)
                  and changed == {"docs/style_improvement_20261002/audit_mixed_actual_checkpoint.py"}
                  and set(declared)-set(training) == registrations and set(document["registrations"]) == registrations,
                  "all134 GPU/business parent paths exact; only actual CPU auditor+three preregistrations differ")
    old_configuration = DOCS/"ground_support_capture_source_freeze"/CONFIGURATION
    audit.require(len(ground_cli_nodes(old_configuration.read_text())) == 6
                  and ground_cli_nodes(source[CONFIGURATION].decode()) == ground_cli_nodes(old_configuration.read_text()),
                  "unchanged exact ground CLI options and four gates")
    old_runner = DOCS/"ground_support_capture_source_freeze"/RUNNER
    loop = control_loop(source[RUNNER].decode())
    audit.require(sha(loop.encode()) == CONTROL_AST_SHA and loop == control_loop(old_runner.read_text()),
                  "complete physical For AST equals original frozen measured-control loop")
    imports = []
    for name, raw in source.items():
        if name.endswith(".py") and name.startswith(("CrowdSim/", "scripts/", "protomotions/")):
            tree = ast.parse(raw)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if any(new in node.value for new in NEW_ENTRY_MODULES):
                        raise ValueError("New unregistered shim/provider literal occurs in captured source")
    audit.require(not any(new == name or name.startswith(new+".") for new in NEW_ENTRY_MODULES for name in imports)
                  and not any(new.replace(".", "/")+".py" in source for new in NEW_ENTRY_MODULES),
                  "archived original entry has no new AMP inference/deployment/DynBench provider imports")
    return {"manifest": module.fingerprint(SOURCE), "archive": archive, "member_count": 138,
            "member_records": document["files"], "current_tree_unchanged_claim": False,
            "parent135_GPU_runtime_sources_unchanged": True, "new_entry_modules_integrated": False}, source


def install_source_adapter(module, source):
    """Amend one expected configuration identity; reverse projection is exact."""
    tree = ast.parse(inspect.getsource(module.verify_sources))
    original = copy.deepcopy(tree)
    function = tree.body[0]
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "frozen" for t in node.targets))
    function.body.insert(index+1, ast.parse(f"frozen[{CONFIGURATION!r}] = NAV_CONFIGURATION_IDENTITY").body[0])
    restored = copy.deepcopy(tree)
    del restored.body[0].body[index+1]
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError("Source adapter changes additional frozen measurement code")
    module.NAV_CONFIGURATION_IDENTITY = {"bytes": len(source[CONFIGURATION]), "sha256": sha(source[CONFIGURATION])}
    exec(compile(ast.fix_missing_locations(tree), str(HELPER), "exec"), module.__dict__)
    module.verify_archive = lambda audit, path: verify_source(module, audit, path)


def verify_preparation(module, audit, path, source, step, candidate, admission_path):
    expected = DOCS/f"mixed_curriculum_step{step}_ground_support_200s_preparation_v1.json"
    audit.require(resolve(path) == expected.resolve() and expected.read_bytes() == source[str(expected.relative_to(ROOT))],
                  "exact candidate registration bytes are included in source138")
    preparation = json.loads(expected.read_text())
    checkpoint = preparation["checkpoint"]
    audit.require(resolve(admission_path) == ADMISSION.resolve() and sha(ADMISSION.read_bytes()) == ADMISSION_SHA,
                  "exact completed220-check terminal admission identity")
    admission = json.loads(ADMISSION.read_text())
    row = select_checkpoint(admission, step)
    audit.require(admission["source_manifest"]["sha256"] == TRAINING_SHA,
                  "selected checkpoint terminal admission binds frozen135 source")
    audit.require(strict_equal(checkpoint["sha256"], row["checkpoint"]["sha256"])
                  and checkpoint["update_count"] == step and checkpoint["physical_transition_count"] == step*8192
                  and resolve(checkpoint["path"]) == resolve(row["checkpoint"]["path"]),
                  "candidate registered exact admitted milestone; evaluation updates arg is not model budget")
    actual_checkpoint = module.fingerprint(resolve(checkpoint["path"]))
    audit.require(actual_checkpoint["sha256"] == checkpoint["sha256"] and actual_checkpoint["bytes"] == checkpoint["bytes"],
                  "selected controller complete file SHA/bytes recomputed")
    parsed = preparation["parsed_arguments"]
    audit.require(resolve(parsed["output"]) == resolve(candidate) and parsed["evaluate_steps"] == 6000
                  and parsed["num_envs"] == 4 and parsed["seed"] == 7 and parsed["physics_only"] is True
                  and parsed["capture"] is False and parsed["deterministic_evaluation"] is True
                  and parsed["ground_support_threshold_n"] == parsed["contact_threshold_n"] == 1.0,
                  "registered complete four-person200s physical-only1N evaluation")
    baseline_path = resolve(preparation["baseline_result"])
    audit.require(sha(baseline_path.read_bytes()) == preparation["baseline_sha256"], "registered historical baseline exact result")
    baseline = json.loads(baseline_path.read_text())
    changes = argument_changes(parsed, baseline["arguments"])
    audit.require(True, "baseline navigation arguments identical except profile/controller/output; old absent eta=0", differences=changes)
    audit.require(preparation["status"] == "SDK-free preflight passed" and preparation["actual_GPU_rollout_performed"] is False
                  and preparation["quality_claim"] is False, "registration is not an actual-rollout or quality claim")
    return preparation, row, {"registration": module.fingerprint(expected), "checkpoint": actual_checkpoint,
                             "terminal_admission": module.fingerprint(ADMISSION), "baseline_result": module.fingerprint(baseline_path),
                             "baseline_argument_changes": changes}, baseline


def verify_navigation_arrays(audit, arrays, plan, result, *, frames=6000, actors=4):
    """Independent actual-clock replay of archived plan selection; no best alignment."""
    n = len(plan["control_step"])
    shapes = {"control_step": (n,), "actor_ids": (n,), "selected_sample": (n,), "raw_xy": (n, 51, 2),
              "reference_xy": (n, 151, 2), "reference_velocity_xy": (n, 151, 2),
              "initial_overspeed": (n,), "speed_violation": (n, 151), "deviation_m": (n, 151)}
    audit.require(set(plan) == set(shapes) and n > 0, "exact nonempty archived raw/reference plan field set")
    for name, shape in shapes.items():
        value = plan[name]
        kind = "iu" if name in ("control_step", "actor_ids", "selected_sample") else ("b" if name in ("initial_overspeed", "speed_violation") else "f")
        audit.require(value.shape == shape and value.dtype.kind in kind and (kind != "f" or np.isfinite(value).all()),
                      "actual plan shape/dtype/finite:"+name)
    steps, ids = plan["control_step"], plan["actor_ids"]
    audit.require(np.all((steps >= 0) & (steps < frames)) and np.all(np.diff(steps) >= 0)
                  and np.all((ids >= 0) & (ids < actors)) and (plan["selected_sample"] == 0).all(),
                  "plan controls/actors monotonic bounded; one sample selected")
    audit.require(len(np.unique(np.stack((steps, ids), axis=-1), axis=0)) == n,
                  "no duplicate actor/control plan publication")
    terminal, reset, reached = arrays["terminated"], arrays["reset_mask"], arrays["route_vertex_reached"]
    audit.require(reached.shape == (frames, actors) and reached.dtype == np.bool_ and not (reached & terminal).any(),
                  "actual route arrival excludes all fallen frames")
    counts = reached.sum(0).astype(int).tolist()
    audit.require(counts == result["route_vertices_reached"], "all physical route arrival events reconcile runtime actor totals")
    clock = result["planning_clock"]
    expected_steps = np.flatnonzero((np.arange(frames) % 15 == 0) | reset.any(-1)
                                   | np.r_[False, reached[:-1].any(-1)])
    events, event_count = np.unique(steps, return_counts=True)
    audit.equal(events, expected_steps, "all and only scheduled/reset/vertex forced scene replans")
    audit.require((event_count == actors).all() and clock["mode"] == "synchronized" and clock["control_step"] == frames
                  and clock["maximum_interval_steps"] == 15 and clock["forward_calls"] == len(events)
                  and clock["mean_agents_per_forward"] == float(actors)
                  and result["diffusion_plans"] == result["trajectory_plans"] == n,
                  "joint-scene replan count matches raw plans and complete control clock")
    expected = np.zeros((frames, actors, 2), np.float32)
    expected_raw = np.zeros_like(expected)
    expected_vel = np.zeros_like(expected)
    endpoint = np.zeros_like(expected)
    raw_endpoint = np.zeros_like(expected)
    age_summary = []
    for actor in range(actors):
        rows = np.flatnonzero(ids == actor)
        audit.require(len(rows) == len(events) and steps[rows[0]] == 0, f"actor{actor} real initial plan exists")
        for j, row in enumerate(rows):
            start = int(steps[row])
            stop = int(steps[rows[j+1]]) if j+1 < len(rows) else frames
            ages = np.arange(1, stop-start+1)
            audit.require(ages[-1] <= 15, f"actor{actor}plan@{start} bounded age")
            expected[start:stop, actor] = plan["reference_xy"][row, ages]
            expected_vel[start:stop, actor] = plan["reference_velocity_xy"][row, ages]
            phase = ages.astype(np.float32)/np.float32(30)*np.float32(10)
            lower = np.floor(phase).astype(int)
            fraction = phase-lower
            expected_raw[start:stop, actor] = (plan["raw_xy"][row, lower]*(1-fraction[:, None])
                                               + plan["raw_xy"][row, lower+1]*fraction[:, None])
            endpoint[start:stop, actor] = plan["reference_xy"][row, -1]
            raw_endpoint[start:stop, actor] = plan["raw_xy"][row, -1]
        age_summary.append({"actor": actor, "plan_records": len(rows)})
    audit.close(arrays["target_xy"], expected, "every physical target follows exact most-recent151 plan post-step phase", atol=2e-5, rtol=0)
    audit.close(arrays["raw_target_xy"], expected_raw, "every raw target follows most-recent51 10Hz interpolation", atol=2e-5, rtol=0)
    audit.close(arrays["reference_target_velocity_xy"], expected_vel, "every reference velocity matches saved151 phase", atol=2e-6, rtol=0)
    audit.close(arrays["reference_deviation_m"], np.linalg.norm(expected-expected_raw, axis=-1), "saved executed/raw deviation reconstructs", atol=2e-5, rtol=0)
    audit.close(arrays["plan_endpoint_goal_error"], np.linalg.norm(endpoint-arrays["global_goal_xy"], axis=-1), "reference endpoint to conditioning goal residual reconstructs", atol=2e-5, rtol=0)
    audit.close(arrays["raw_plan_endpoint_goal_error"], np.linalg.norm(raw_endpoint-arrays["global_goal_xy"], axis=-1), "raw endpoint to conditioning goal residual reconstructs", atol=2e-5, rtol=0)
    failures, boundaries = [], []
    for frame, actor in np.argwhere(terminal | arrays["truncated"]):
        failures.append({"frame": int(frame), "actor": int(actor), "time_s": (int(frame)+1)/30,
                         "episode_id": int(arrays["episode_ids"][frame, actor]),
                         "fall": bool(terminal[frame, actor]), "time_limit": bool(arrays["truncated"][frame, actor]),
                         "root_xyz": arrays["root_pos"][frame, actor].tolist(),
                         "static_contact": bool(arrays["static_contact"][frame, actor]),
                         "peer_contact_actors": np.flatnonzero(arrays["peer_contact"][frame, actor]).tolist()})
    for frame, actor in np.argwhere(reset):
        boundaries.append({"frame": int(frame), "actor": int(actor), "first_poststep_time_s": (int(frame)+1)/30,
                           "initial": int(frame) == 0, "episode_id": int(arrays["episode_ids"][frame, actor])})
    routes = np.asarray(result["evaluation_scene"]["routes_xy"])
    audit.require(routes.shape == (actors, 4, 2), "same declared four-vertex actor routes")
    indices = arrays["route_indices"]
    audit.require(indices.shape == (frames, actors) and indices.dtype.kind in "iu" and ((indices >= 0) & (indices < 4)).all(),
                  "route indices are bounded genuine integers")
    previous = np.zeros(actors, dtype=np.int64)
    per_episode = []
    for actor in range(actors):
        for episode in np.unique(arrays["episode_ids"][:, actor]):
            mask = arrays["episode_ids"][:, actor] == episode
            vertices = int(reached[mask, actor].sum())
            per_episode.append({"actor": actor, "episode_id": int(episode), "vertices": vertices, "full_four_vertex_cycles": vertices//4})
    expected_vertices = np.empty_like(arrays["route_vertex_xy"])
    expected_indices = np.empty_like(indices)
    for frame in range(frames):
        previous[reset[frame]] = 0
        expected_vertices[frame] = routes[np.arange(actors), previous]
        previous = (previous+reached[frame].astype(np.int64)) % 4
        expected_indices[frame] = previous
    audit.close(arrays["route_vertex_xy"], expected_vertices,
                "every physical route vertex before advancement", atol=0, rtol=0)
    audit.equal(indices, expected_indices, "every poststep/reset-safe route index")
    return {"plan_records": n, "joint_forward_calls": len(events), "maximum_age_controls": 15,
            "all_failed_and_fallen_frames_retained": True, "failure_events": failures, "reset_events": boundaries,
            "per_actor_plan_counts": age_summary, "per_episode_routes": per_episode,
            "full_cycles": sum(r["full_four_vertex_cycles"] for r in per_episode)}


def capture_markdown(module, report, arguments):
    """Keep measurement narrative; direct reproduction through the actual wrapper."""
    import shlex

    text = module.markdown(report)
    prefix = text[:text.index("## 可复现 CPU 命令")]
    command = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "PYTHONPATH=.",
               sys.executable, str(Path(__file__).resolve()), "--candidate-dir", str(resolve(arguments.candidate_dir)),
               "--candidate-step", str(arguments.candidate_step), "--registered-preparation", str(resolve(arguments.registered_preparation)),
               "--source-manifest", str(resolve(arguments.source_manifest)), "--checkpoint-admission", str(resolve(arguments.checkpoint_admission)),
               "--output", "/tmp/mixed_navigation_ground_reaudit.json"]
    return prefix+"## 可复现 CPU 命令\n\n```bash\n"+shlex.join(command)+"\n```\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-step", type=int, choices=STEPS, required=True)
    parser.add_argument("--registered-preparation", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-admission", type=Path, default=ADMISSION)
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Explicit empty CUDA visibility and fresh outputs required")
    module = load_helper()
    audit = module.Audit()
    try:
        source_identity, source = verify_source(module, audit, args.source_manifest)
        prep, row, identity, baseline = verify_preparation(module, audit, args.registered_preparation, source,
            args.candidate_step, args.candidate_dir, args.checkpoint_admission)
        report = {"status": "passed", "preparation_only": args.preparation_only, "source_preparation": source_identity,
                  "candidate_admission": identity, "candidate_update": args.candidate_step,
                  "actual_native_capture_validated": False, "quality_acceptance": None}
        if not args.preparation_only:
            install_source_adapter(module, source)
            report = module.audit_capture(resolve(args.candidate_dir), audit, source_archive_manifest=resolve(args.source_manifest))
            result = json.loads((resolve(args.candidate_dir)/"result.json").read_text())
            audit.require(strict_equal(result["arguments"], prep["parsed_arguments"]), "actual runtime complete argv equals preregistered candidate")
            audit.require(result["update_count"] == args.candidate_step and result["physical_transition_count"] == args.candidate_step*8192
                          and strict_equal(result["training_config"], row["configuration"]), "runtime actor budget/config match exact admitted checkpoint")
            audit.require(result["actual_control_steps"] == 6000 and result["author_experiment_reproduced"] is False,
                          "complete 6000 frames; author-equivalence remains false")
            audit.require(strict_equal(result["evaluation_scene"], baseline["evaluation_scene"])
                          and strict_equal(result["physics"], baseline["physics"]), "baseline scene and live physics profile exact")
            plan_path = resolve(result["reference_adaptation"]["plan_capture"])
            audit.require(plan_path == resolve(args.candidate_dir)/"reference_plans.npz", "plan capture belongs to exact candidate directory")
            with np.load(plan_path, allow_pickle=False) as archive:
                plans = {k: archive[k] for k in archive.files}
            with np.load(resolve(args.candidate_dir)/"physical_states.npz", allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            with np.load(resolve(args.candidate_dir)/"physical_roots.npz", allow_pickle=False) as archive:
                audit.require(archive.files == ["root_pos"], "roots archive exact one-field schema")
                audit.equal(archive["root_pos"], arrays["root_pos"], "separate physical roots bitwise full-frame capture")
            temporal = verify_navigation_arrays(audit, arrays, plans, result)
            for name, record in report["artifacts"].items():
                audit.require(module.fingerprint(record["path"]) == record,
                              "actual capture input remains immutable through complete CPU audit:"+name)
            audit.require(sha(BASELINE_AUDIT.read_bytes()) == BASELINE_AUDIT_SHA, "historical baseline native audit immutable identity")
            baseline_audit = json.loads(BASELINE_AUDIT.read_text())
            audit.require(baseline_audit["status"] == "passed" and all(c["passed"] is True for c in baseline_audit["checks"])
                          and baseline_audit["artifacts"]["result"]["sha256"] == prep["baseline_sha256"],
                          "baseline already fully native-audited; exact registered result")
            report.update(status="passed", preparation_only=False, source_preparation=source_identity,
                candidate_admission=identity, candidate_update=args.candidate_step, quality_acceptance=None,
                temporal_navigation=temporal, plan_capture=module.fingerprint(plan_path),
                roots_capture=module.fingerprint(resolve(args.candidate_dir)/"physical_roots.npz"),
                baseline_comparison={"source": module.fingerprint(BASELINE_AUDIT),
                    "arguments": identity["baseline_argument_changes"], "counts": baseline_audit["counts"],
                    "support_statistics": baseline_audit["support_statistics"], "COM_statistics": baseline_audit["COM_statistics"],
                    "contacts": baseline_audit["original_obstacle_peer_channels"],
                    "attribution": "Controller training data/curriculum/budget differ; shared seed is not identical physical rollout. No single-factor or author metric claim."})
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                  "input": str(resolve(args.candidate_dir)), "preparation_only": args.preparation_only,
                  "actual_native_capture_validated": False, "quality_acceptance": None}
    report.update(CPU_only=True, GPU_used_by_auditor=False, SDK_started_by_auditor=False,
        heldout34_test_outputs_read=False, all_frozen_measurement_checks_unchanged=True,
        checks=audit.checks, check_count=len(audit.checks), audit_script=module.fingerprint(__file__),
        unchanged_measurement_auditor=module.fingerprint(HELPER), command=sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if report["status"] == "passed":
        text = ("# 混合控制器导航独立 CPU 审查\n\n"+f"结果：{report['status']}；检查{len(audit.checks)}项。\n\n")
        if args.preparation_only:
            text += "仅准入预启动138源码、注册参数与实际sealed checkpoint，不读取或声称已有原生导航捕获。\n"
        else:
            text += capture_markdown(module, report, args)
            text += "\n完整6000控制帧／24000actor帧／96000actor子步全部保留；实际raw/reference计划、重规划、路线、每次跌倒及reset事件见JSON。状态passed是采集与证据完整性，不是质量验收。\n"
        with args.output.with_suffix(".md").open("x") as stream:
            stream.write(text)
    print(json.dumps({"status": report["status"], "checks": len(audit.checks), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
