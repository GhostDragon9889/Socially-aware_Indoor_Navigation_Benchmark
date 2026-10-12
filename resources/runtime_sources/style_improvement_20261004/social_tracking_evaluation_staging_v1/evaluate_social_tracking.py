#!/usr/bin/env python3
"""Source-derived full22 tracking entry for a strict Social AMP actor snapshot.

Prepared source only. Actual NN/SDK/checkpoint/reference execution is scheduled
separately. The original N1 physics, reference timing and metrics are retained.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import stat
import sys
from types import ModuleType

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
STAGES = ROOT / "resources/runtime_sources/style_improvement_20261004"
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
PINS = {
    "original": (ROOT / "scripts/benchmark_paper_motion_tracking.py", 28945,
                 "a89f05f94869695c9f908f8e9681f130a12b7e1f76085b6a5a8b91598efc616e"),
    "metrics": (ROOT / "CrowdSim/humanoid/motion_tracking.py", 8986,
                "da482dcd565d42ef9f6165606428ad510f077bdd68c3fab09c73a840149eac56"),
    "world": (ROOT / "CrowdSim/humanoid/amp_world.py", 30450,
              "1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83"),
    "observations": (ROOT / "CrowdSim/humanoid/amp/observations.py", 10586,
                     "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba"),
    "simulator": (ROOT / "protomotions/simulator/isaaclab/simulator.py", 55888,
                  "207ff0cbfb65fd48092e73e36c97b2e1b8e2ab55cae79bc2e7edf562528e2ad5"),
    "corpus": (ROOT / "CrowdSim/humanoid/amass_corpus.py", 12126,
               "2d4e4f7549aa38f5fd6763de4d3924a9cdd969746682c649ab773f033d747cf7"),
    "loader": (STAGES / "social_checkpoint_inference_staging_v1/checkpoint_inference.py", 16281,
               '875d9fd21af190cd0405b90d214aad71d1dffcb4678661ce7e187e61c91dc954'),
    "producer": (STAGES / "social_evaluation_observation_staging_v1/observation_producer.py", 17810,
                 'e1c0eba40a895c4bcfe06c3fb1823ad961f29ca4e29cd922b34df021d715d4f0'),
}
ENTRY_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
POSE_FIELDS = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
               "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel")
DIAGNOSTIC_FIELDS = ("actor_ids", "scene_ids", "age", "episode_ids", "actual_pelvis_quat_xyzw",
                     "actual_ground_height_world_z", "actual_pelvis_height_above_ground_m", "actual_up_z",
                     "used_task_heading", "physical_fall", "task_heading_defined", "singular_heading_fallback")

# Each literal occurs once; the complete inverse must recover all 28,945 bytes.
REPLACEMENTS = (
    ("repository root", 'ROOT = Path(__file__).resolve().parents[1]\n', 'ROOT = Path(_social_root)\n'),
    ("fixed heldout manifest", 'default=ROOT/"data/diffusion_amp/amass_manifest.json")',
     'default=ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")'),
    ("full22 arguments", '    return args\n\n\ndef preflight',
     '    _social_validate_args(args)\n    return args\n\n\ndef preflight'),
    ("actual social training provenance", '    dataset = provenance.get("dataset") if isinstance(provenance, dict) else None\n',
     '    dataset = provenance.get("AMASS") if isinstance(provenance, dict) else None\n'),
    ("strict actor loader", '''def load_evaluation_actor(path, *, device):
    """Load the fixed model and moments; optimizer/replay stay unused on CPU."""
    import torch
    from CrowdSim.humanoid.training import PaperAMPTrainer

    trainer, _ = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
    if (trainer.config.self_dim, trainer.config.task_dim, trainer.config.history_frames) != (357, 1044, 10):
        raise ValueError("Tracking requires the raw PACER 357/1044 observation and 10-history contract")
    if not math.isclose(trainer.config.control_dt, 1/30, abs_tol=1e-9) or not math.isclose(trainer.config.target_horizon_s, 5., abs_tol=1e-9):
        raise ValueError("Tracking requires 30Hz control and a 5-second XY command horizon")
    if not trainer.config.freeze_toes:
        raise ValueError("This AMPWorld physical decoder freezes toe actions; checkpoint freeze_toes must agree")
    if not all(torch.isfinite(value).all() for value in trainer.model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model weights")
    trainer.device = torch.device(device)
    trainer.model.to(trainer.device).eval().requires_grad_(False)
    trainer.action_mask = trainer.action_mask.to(trainer.device)
    for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
        getattr(trainer, name).to(trainer.device).eval()
    return trainer''', '''def load_evaluation_actor(path, *, device, expected_sha256):
    """Strict Social snapshot; no legacy trainer/optimizer/replay construction."""
    return _social_load_actor(path, device=device, expected_sha256=expected_sha256)'''),
    ("producer parameter", 'def run_reference_clip(world, app, corpus, actor, record, output, threshold_m):',
     'def run_reference_clip(world, app, corpus, actor, record, output, threshold_m, producer):'),
    ("notify actual reset", '    actual, translation = reset_reference_clip(world, corpus, record)\n',
     '    actual, translation = reset_reference_clip(world, corpus, record)\n    producer.begin_episode()\n'),
    ("capture observation parameter", 'def append_frame(state, frame, command, query_phase, held, actions, applied, *, measured, fall, timeout):',
     'def append_frame(state, frame, command, query_phase, held, actions, applied, observation, *, measured, fall, timeout):'),
    ("raw observation capture", '        for name, value in values.items():\n',
     '        _social_capture_values(values, observation)\n        for name, value in values.items():\n'),
    ("initial observation", '    append_frame(actual, first, command, first, held, zero, zero, measured=False, fall=False, timeout=False)\n',
     '    observation = _social_observe(producer, command, actual)\n'
     '    append_frame(actual, first, command, first, held, zero, zero, observation, measured=False, fall=False, timeout=False)\n'),
    ("raw social actor inputs", '            observations = world.observations(command, actual)\n'
     '            prediction = actor.act(*observations, deterministic=True)\n',
     '            observation = _social_observe(producer, command, actual)\n'
     '            prediction = actor.act(observation.raw_self, observation.raw_task, deterministic=True)\n'),
    ("preaction packet beside postphysics pose", '                         prediction["actions"], prediction["applied_actions"],\n',
     '                         prediction["actions"], prediction["applied_actions"], observation,\n'),
    ("source and full22 admission", '    resources = preflight(args)\n',
     '    resources = preflight(args)\n    _social_resources(resources)\n'),
    ("new report identity", '    app = world = None\n',
     '    _social_initial_report(report)\n    app = world = None\n'),
    ("immutable checkpoint identity", '        actor = load_evaluation_actor(args.checkpoint, device="cuda:0")\n',
     '        actor = load_evaluation_actor(args.checkpoint, device="cuda:0", expected_sha256=args.checkpoint_sha256)\n'),
    ("create producer after original world", '        torch.manual_seed(args.seed)\n        report.update(physics=world.physics_manifest()',
     '        producer = _social_create_producer(world, actor)\n'
     '        _social_actor_report(report, actor, producer)\n'
     '        torch.manual_seed(args.seed)\n        report.update(physics=world.physics_manifest()'),
    ("actual new loader source in report", 'ROOT/"CrowdSim/humanoid/amp_world.py", ROOT/"CrowdSim/humanoid/training/trainer.py",',
     'ROOT/"CrowdSim/humanoid/amp_world.py", Path(_social_loader_path), Path(_social_producer_path),'),
    ("producer clip call", 'run_reference_clip(world, app, corpus, actor, record, captures, args.failure_threshold_m)',
     'run_reference_clip(world, app, corpus, actor, record, captures, args.failure_threshold_m, producer)'),
    ("complete inclusive evaluation and sources", '        report.update(status="passed", aggregate=aggregate_sequences(report["sequences"]), elapsed_s=time.monotonic()-started)\n',
     '        _social_complete(report)\n'
     '        report.update(status="passed", aggregate=aggregate_sequences(report["sequences"]), elapsed_s=time.monotonic()-started)\n'),
)


def _sources():
    records, data = {}, {}
    for name, (path, size, sha) in PINS.items():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Single-link ordinary source required: " + str(path))
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if len(content) != size or actual != sha:
            raise ValueError("Reviewed source identity changed: " + name)
        records[name] = {"path": str(path), "bytes": len(content), "sha256": actual}
        data[name] = content
    entry = Path(__file__)
    content = entry.read_bytes()
    if hashlib.sha256(content).hexdigest() != ENTRY_SHA:
        raise ValueError("Entry source changed after import")
    records["entry"] = {"path": str(entry), "bytes": len(content), "sha256": ENTRY_SHA}
    return records, data


def derive(original):
    """Restricted byte deltas; no evaluation execution or module imports."""
    derived = original
    for name, before, after in REPLACEMENTS:
        if derived.count(before) != 1:
            raise ValueError("Missing/ambiguous replacement: " + name)
        derived = derived.replace(before, after, 1)
    # One explicit try surrounds the existing initial observation, control loop
    # and normal metric/save/return statements. Its normal AST is not rewritten.
    run = next(n for n in ast.parse(derived).body if isinstance(n, ast.FunctionDef) and n.name == "run_reference_clip")
    starts = [n for n in run.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "first" for t in n.targets)]
    if len(starts) != 1:
        raise ValueError("Unique initial-frame control/finalization boundary required")
    lines = derived.splitlines(keepends=True)
    tail = "".join(lines[starts[0].lineno - 1:run.end_lineno])
    wrapped = '    try:\n' + "".join("    " + line if line.strip() else line for line in tail.splitlines(keepends=True))
    wrapped += ('    except BaseException as error:\n'
                '        try:\n'
                '            _social_partial_capture(capture, record, output, error, locals().get("observation"))\n'
                '        except BaseException:\n'
                '            pass  # Archival failure cannot replace the original control failure.\n'
                '        raise\n')
    if derived.count(tail) != 1:
        raise ValueError("Unique control/finalization source tail required")
    before_wrap = derived
    derived = derived.replace(tail, wrapped, 1)
    wrapped_run = next(n for n in ast.parse(derived).body if isinstance(n, ast.FunctionDef) and n.name == run.name)
    guarded = wrapped_run.body[-1]
    normal_tail = run.body[run.body.index(starts[0]):]
    if not isinstance(guarded, ast.Try) or ast.dump(ast.Module(body=normal_tail, type_ignores=[]), include_attributes=False) != (
            ast.dump(ast.Module(body=guarded.body, type_ignores=[]), include_attributes=False)):
        raise ValueError("Normal control/finalization AST changed inside partial guard")
    if derived.count(wrapped) != 1 or derived.replace(wrapped, tail, 1) != before_wrap:
        raise ValueError("Partial guard exact inverse differs")
    inverse = derived.replace(wrapped, tail, 1)
    for name, before, after in reversed(REPLACEMENTS):
        if inverse.count(after) != 1:
            raise ValueError("Missing/ambiguous inverse: " + name)
        inverse = inverse.replace(after, before, 1)
    if inverse != original:
        raise ValueError("Whole-source inverse differs")
    old = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in ast.parse(derived).body if isinstance(n, ast.FunctionDef)}
    if old.keys() != new.keys():
        raise ValueError("Original function inventory changed")
    changed = {"parse_args", "verify_actor_training_split", "load_evaluation_actor", "run_reference_clip", "main"}
    unchanged = {}
    for name in old.keys() - changed:
        segment = ast.get_source_segment(original, old[name])
        if segment != ast.get_source_segment(derived, new[name]) or (
                ast.dump(old[name], include_attributes=False) != ast.dump(new[name], include_attributes=False)):
            raise ValueError("Original untouched function changed: " + name)
        unchanged[name] = hashlib.sha256(segment.encode()).hexdigest()
    old_nested = {n.name: n for n in old["run_reference_clip"].body if isinstance(n, ast.FunctionDef)}
    new_nested = {n.name: n for n in new["run_reference_clip"].body if isinstance(n, ast.FunctionDef)}
    if ast.get_source_segment(original, old_nested["trajectory"]) != ast.get_source_segment(derived, new_nested["trajectory"]):
        raise ValueError("True reference trajectory/endhold changed")
    if any(isinstance(n, ast.ImportFrom) and n.module == "CrowdSim.humanoid.training"
           for n in ast.walk(new["load_evaluation_actor"])):
        raise ValueError("Legacy trainer remains in derived loader")
    compile(derived, str(__file__) + ":full22_derived", "exec")
    return derived, {"replacement_count": len(REPLACEMENTS) + 1,
                     "replacement_names": [x[0] for x in REPLACEMENTS] + ["partial failure guard around intact control/finalize AST"],
                     "whole_byte_reverse_exact": True, "unchanged_function_source_sha256": unchanged,
                     "trajectory_nested_source_exact": True,
                     "normal_control_and_finalization_AST_exact_within_guard": True,
                     "derived_source_sha256": hashlib.sha256(derived.encode()).hexdigest()}


def _validate_args(args):
    if args.checkpoint_sha256 is None:
        raise ValueError("An explicit immutable Social checkpoint SHA256 is mandatory")
    if (args.manifest != MANIFEST or args.split != "validation" or args.motion_ids is not None
            or args.max_clips is not None or args.max_steps is not None):
        raise ValueError("This entry measures the entire fixed v7 validation22; no selections/caps")
    if args.minimum_evaluation_seconds != 1. or args.failure_threshold_m != .5:
        raise ValueError("Retain original full22 minimum duration1s and tracking threshold.5m")


def _load_pinned(role):
    records, data = _sources()
    name = "_social_full22_" + role
    if name in sys.modules:
        raise ValueError("Refuse a previously imported entry-owned module: " + name)
    module = ModuleType(name)
    module.__file__ = records[role]["path"]
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(data[role], module.__file__, "exec"), module.__dict__)
    return module


def _load_actor(path, *, device, expected_sha256):
    actor = _load_pinned("loader").FrozenSocialCheckpointInference.from_checkpoint(
        path, expected_sha256=expected_sha256, device=device)
    config = actor.config
    if ((config.self_dim, config.task_dim, config.history_frames) != (357, 12308, 10)
            or not math.isclose(config.control_dt, 1 / 30, abs_tol=1e-9)
            or not math.isclose(config.target_horizon_s, 5., abs_tol=1e-9) or config.freeze_toes is not True):
        raise ValueError("Social actor differs from original30Hz/history10/frozen-toe physical decoder")
    return actor


def _create_producer(world, actor):
    import torch

    if world.num_envs != 1 or world.shared_world or world.scene_provider is not None or world.validation_scene is not None:
        raise ValueError("Full22 uses the original isolated N1 unobstructed flat-floor world")
    ids = torch.zeros(1, dtype=torch.int64, device=world.device)
    producer = _load_pinned("producer").RawSocialEvaluationObservations.from_world(
        world, ids, ids.clone(), grid_center="root", grid_heading="root", height_reference="static_ground",
        include_ego=False, terrain_sampler=world.height_at,
        terrain_sampler_semantics="original AMPWorld.height_at absolute static flat-ground feature; not measured body surface")
    producer.require_policy_contract(actor.describe())
    return producer


def _observe(producer, command, actual):
    import torch

    packet = producer.observe(command)
    for name in POSE_FIELDS:
        old, current = getattr(actual, name), getattr(packet.state, name)
        if old.dtype != current.dtype or old.device != current.device or not torch.equal(old, current):
            raise RuntimeError("Current social snapshot differs from original same-phase physical state: " + name)
    return packet


def _capture_values(values, packet):
    import numpy as np

    for name in ("raw_self", "raw_task", "owners"):
        values["observation_" + name] = getattr(packet, name).detach().cpu().numpy().copy()
    diagnostics = packet.diagnostics
    values["observation_physics_step"] = np.asarray(diagnostics["physics_step"])
    for name in DIAGNOSTIC_FIELDS:
        values["observation_" + name] = diagnostics[name].detach().cpu().numpy().copy()


def _save_partial(capture, record, output, error, latest_observation):
    """Archive independent channels honestly, without partial metric/success math."""
    import numpy as np

    path = output / f"motion_{record['motion_id']:06d}_partial_failure.npz"
    metadata = {"purpose": "unexpected failure archival; not a complete sequence or metric result",
                "motion_id": record["motion_id"], "source_path": record["source_path"],
                "tracking_window": deepcopy(record["tracking_window"]),
                "original_error_type": type(error).__name__, "original_error": str(error),
                "partial_capture": str(path), "channels": {}, "status": "write_failed",
                "postphysics_and_preaction_channels_keep_independent_lengths": True,
                "latest_owned_observation": "independent last returned packet; may already occur in completed rows; not certified as failed-action phase",
                "original_exception_rethrown": True, "physics_retry_or_frames_added": False}
    try:
        archive = dict(capture)
        if latest_observation is not None:
            try:
                latest = {}
                _capture_values(latest, latest_observation)
                archive.update({"latest_owned_" + name: [value] for name, value in latest.items()})
            except BaseException as archive_error:
                metadata["latest_observation_archive_error"] = f"{type(archive_error).__name__}: {archive_error}"
        arrays = {}
        for name, rows in archive.items():
            channel = {"rows": len(rows), "row_shapes": sorted({str(value.shape) for value in rows})}
            metadata["channels"][name] = channel
            try:
                arrays[name] = np.stack(rows)
                channel["stored_shape"] = list(arrays[name].shape)
            except (ValueError, TypeError) as archive_error:
                channel["not_stored"] = f"{type(archive_error).__name__}: {archive_error}"
        if path.exists():
            raise FileExistsError("Do not overwrite a partial failure capture")
        np.savez_compressed(path, **arrays)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata.update(status="partial_saved", bytes=path.stat().st_size, sha256=digest)
    except BaseException as archive_error:
        metadata["archive_error"] = f"{type(archive_error).__name__}: {archive_error}"
    return metadata


def main(argv=None):
    # Bind all sources before deriving, parsing reference metadata or SDK/NN imports.
    binding, sources = _sources()
    derived, derivation = derive(sources["original"].decode("utf-8"))
    partial_captures = []

    def resource_contract(resources):
        rows = resources["selected_clips"]
        if (resources["manifest_sha256"] != MANIFEST_SHA or len(rows) != 22
                or any(not r["tracking_window"]["eligible"] or not r["tracking_window"]["full_clip_requested"] for r in rows)
                or sum(r["tracking_window"]["available_steps"] for r in rows) != 11982):
            raise ValueError("Entire original validation22/11982 postphysics-frame population required")
        resources["social_entry_sources"] = deepcopy(binding)
        resources["social_source_derivation"] = deepcopy(derivation)

    def initial_report(report):
        report.update(schema="local.neutral_smpl.social_full22_tracking.v1", entry_sources=deepcopy(binding),
                      source_derivation=deepcopy(derivation), original_benchmark_executed_bitexact=False,
                      unexpected_failure_partial_captures=partial_captures)
        report["protocol"].update(actor_observations="raw self357 + trajectory20/social64x64x3 HWC =12308; RMS inside frozen actor",
            peers_added_to_single_actor_evaluation=0, reference_selection="all22 fixed v7 validation;11982 postphysics controls",
            observation_capture_phase="preaction current COMMON snapshot; postphysics pose fields retain original same-reference phase",
            initial_observation="source-frame9 reset state; action_validFalse and metric_frameFalse",
            singular_task_heading="only physical fall with defined same-episode prior heading; actual physical quaternion retained",
            original_self_heading_math_changed=False, geometry_proxy_is_collision_truth=False)

    def actor_report(report, actor, producer):
        report["frozen_social_actor"] = actor.describe()
        report["social_observation_producer"] = producer.describe()
        report["protocol"]["action_projection"] = "deterministic Gaussian mean; clamp[-1,1] then canonical57-active/69 mask; original action_to_pd"

    def partial_capture(capture, record, output, error, latest_observation):
        partial_captures.append(_save_partial(capture, record, output, error, latest_observation))

    def complete(report):
        aggregate = report["sequences"]
        if (len(aggregate) != 22 or sum(r["observed_metric_steps"] for r in aggregate) != 11982
                or sum(bool(r["in_full_sequence_success_denominator"]) for r in aggregate) != 22):
            raise RuntimeError("Incomplete full22 measurement; failures remain in report, no success promotion")
        if _sources()[0] != binding:
            raise RuntimeError("Reviewed source changed during full22 measurement")

    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "_social_full22_derived",
                 "_social_root": str(ROOT), "_social_loader_path": str(PINS["loader"][0]),
                 "_social_producer_path": str(PINS["producer"][0]), "_social_validate_args": _validate_args,
                 "_social_load_actor": _load_actor, "_social_create_producer": _create_producer,
                 "_social_observe": _observe, "_social_capture_values": _capture_values,
                 "_social_partial_capture": partial_capture,
                 "_social_resources": resource_contract, "_social_initial_report": initial_report,
                 "_social_actor_report": actor_report, "_social_complete": complete}
    exec(compile(derived, str(__file__) + ":full22_derived", "exec"), namespace)
    return namespace["main"](argv)


if __name__ == "__main__":
    raise SystemExit(main())
