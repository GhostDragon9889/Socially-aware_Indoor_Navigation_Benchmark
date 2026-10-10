"""Recover optimizer training at a saved boundary with an explicit World restart.

Standard-library import only. This does not restore a PhysX episode, curriculum
RNG, contact caches or histories. The original immutable World and PPO/AMP
mathematics are reused, and the recovery is recorded separately from S3.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .social_evaluation_lifecycle import ROOT, record

SCHEMA = "shared_social.optimizer_recovery_registration.v1"
BASE_UPDATES = 12000
CHECKPOINT = ROOT / "output/diffusion_amp/shared_social_budget12500_v1/step_012000.pt"
CHECKPOINT_SHA = "62872fc8376a848c46e02e9540467864a61835c930b9b67e9c695c0ff774a8b7"
ORIGINAL_PROFILE = ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/training.json"
ORIGINAL_METRICS = CHECKPOINT.parent / "metrics.jsonl"
CANONICAL_SHA = "a1247911b78cc9adce2ee67da42e2afefb43c785fcdbd8afe02e97b1b572cacc"


def validate_profile(profile, original, stage):
    """Allow only the registered terminal target and checkpoint/archive cadence."""
    if stage not in ("smoke", "training"):
        raise ValueError("Unknown recovery stage")
    if profile.get("schema") != original["schema"] or profile.get("training") != original["training"]:
        raise ValueError("Recovery cannot change training mathematics")
    expected = dict(original["runtime"])
    expected.update(updates=12001 if stage == "smoke" else 12500,
                    checkpoint_every_updates=25, archive_every_updates=125)
    if profile.get("runtime") != expected or profile.get("purpose") != original["purpose"]:
        raise ValueError("Recovery runtime differs from its registered 256-actor boundary")
    return expected


def read_retained_metrics(path=ORIGINAL_METRICS):
    """Retain only updates represented by the checkpoint; preserve the old log."""
    data = Path(path).read_bytes()
    lines = data.splitlines(keepends=True)
    retained, maximum = [], 0
    for index, line in enumerate(lines, 1):
        row = json.loads(line)
        if row["update_count"] != index:
            raise ValueError("Original metric sequence is not contiguous")
        maximum = index
        if index <= BASE_UPDATES:
            retained.append(line)
    if len(retained) != BASE_UPDATES:
        raise ValueError("Checkpoint training prefix is missing")
    prefix = b"".join(retained)
    last = json.loads(retained[-1])
    expected = BASE_UPDATES * 256 * 32
    if last["physical_transitions"] != expected:
        raise ValueError("Original metric prefix disagrees with checkpoint exposure")
    return {"source": record(path), "retained_updates": BASE_UPDATES,
            "retained_physical_transitions": expected, "prefix_bytes": len(prefix),
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "source_logged_max_update": maximum,
            "uncheckpointed_updates_not_inherited": maximum - BASE_UPDATES}, prefix


def derive_run(source):
    """Derive only recovery admission and segment counters; invert all edits."""
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_run")
    original = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    edits = (
        ('        trainer = trainer_api.SocialAMPTrainer.prepare(config, device=args.device, provenance=provenance)\n',
         '        trainer = _recovery_restore(trainer_api, config, corpus, world, report, args.device, torch)\n'),
        ('        with metrics_path.open("x") as log:\n            for update in range(1, runtime["updates"] + 1):\n',
         '        trainer.restore_rng()  # After all World/expert/collector setup, before the first action.\n'
         '        report["recovery"]["global_training_rng_restored_after_setup"] = True\n'
         '        trainer.provenance["training_recovery"]["global_training_rng_restored_after_setup"] = True\n'
         '        with metrics_path.open("x") as log:\n            for update in range(_recovery_base + 1, runtime["updates"] + 1):\n'),
        ('or world.control_step != update * runtime["rollout_controls"]',
         'or world.control_step != (update - _recovery_base) * runtime["rollout_controls"]'),
        ('cumulative_expert_windows_drawn=expert.windows,',
         'cumulative_expert_windows_drawn=_recovery_base * expert_count + expert.windows,'),
        ('                json.dump(metrics, log, allow_nan=False)\n',
         '                metrics.update(recovery_segment_update=update - _recovery_base,\n'
         '                    retained_training_controls=update * runtime["rollout_controls"],\n'
         '                    physical_world_restarted=True, physical_counters_scope="recovery_segment_only")\n'
         '                json.dump(metrics, log, allow_nan=False)\n'),
    )
    derived = original
    for before, after in edits:
        if derived.count(before) != 1:
            raise ValueError("Ambiguous recovery derivation")
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(edits):
        if inverse.count(after) != 1:
            raise ValueError("Ambiguous recovery derivation inverse")
        inverse = inverse.replace(after, before, 1)
    if inverse != original:
        raise ValueError("Recovery did not preserve the complete original physical runner")
    compile(derived, "shared_social_recovery_run", "exec")
    return derived, {"original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "replacement_count": len(edits), "exact_inverse": True,
                     "World_reward_PD_reset_collector_expert_cleanup_math_changed": False}


def admit_checkpoint(state, api, profile):
    """Strict original Pop-Art metadata admission, plus this boundary's counts."""
    config = api.SocialAMPTrainer._admit_metadata(state)
    runtime = profile["runtime"]
    if (state["update_count"] != BASE_UPDATES
            or state["physical_transition_count"] != BASE_UPDATES * runtime["actors"] * runtime["rollout_controls"]
            or state["runner_state"] is not None or state["physical_runner_state_included"] is not False
            or config.actor_count != runtime["actors"] or config.seed != runtime["seed"]):
        raise ValueError("Require the exact optimizer-only S3 update12000 boundary")
    if state["provenance"]["profile"] != record(ORIGINAL_PROFILE):
        raise ValueError("Recovery checkpoint is not from the registered original S3 profile")
    if state["provenance"]["AMASS"]["clips"] != 2848 or state["provenance"]["AMASS"]["split"] != "train":
        raise ValueError("Recovery must retain the real fixed2848 training data")
    return config


def resolve_requested_config(config, api):
    """Expand the original constructor's canonical mask before comparison."""
    resolved = config.detached_copy()
    if resolved.action_mask is None:
        resolved.action_mask = api._settings.PHYSICAL_ACTION_MASK
    resolved.validate()
    return resolved


def prepare_entry(registration_path, stage):
    from scripts import train_shared_social as canonical

    if record(canonical.__file__)["sha256"] != CANONICAL_SHA:
        raise ValueError("Original canonical training entry changed")
    registration_path = Path(registration_path).resolve()
    registration_identity = record(registration_path)
    registration = json.loads(registration_path.read_text())
    if registration.get("schema") != SCHEMA or registration_path.stat().st_mode & 0o222:
        raise ValueError("Require a readonly optimizer recovery registration")
    entry = record(ROOT / "scripts/recover_shared_social_training.py")
    helper = record(__file__)
    if registration["sources"] != {"entry": entry, "helper": helper}:
        raise ValueError("Registered recovery implementation changed")
    checkpoint = record(CHECKPOINT)
    if checkpoint["sha256"] != CHECKPOINT_SHA or registration["checkpoint"] != checkpoint:
        raise ValueError("Registered original checkpoint changed")
    selected = registration["stages"][stage]
    if record(selected["configuration"]["path"]) != selected["configuration"]:
        raise ValueError("Registered recovery profile changed")
    original = json.loads(ORIGINAL_PROFILE.read_text())
    profile = json.loads(Path(selected["configuration"]["path"]).read_text())
    validate_profile(profile, original, stage)
    metrics, prefix = read_retained_metrics()
    if registration["retained_metrics"] != metrics:
        raise ValueError("Original training metrics changed after recovery registration")
    module, canonical_binding = canonical.prepare_entry("popart")
    parent_sources = module._sources
    parent_rows = parent_sources()
    if registration["original_entry_sources"] != parent_rows:
        raise ValueError("Original S3 implementation sources changed")
    derived, proof = derive_run(canonical.PARENT.read_text())
    binding = {"registration": registration_identity, "entry": entry, "helper": helper,
               "checkpoint": checkpoint, "retained_metrics": metrics,
               "canonical_binding": canonical_binding, "derivation": proof, "stage": stage,
               "configuration": selected["configuration"], "output": selected["output"]}

    def sources():
        if (record(__file__) != helper or record(entry["path"]) != entry
                or record(registration_path) != registration_identity):
            raise ValueError("Recovery sources or registration changed during execution")
        rows = parent_sources()
        rows["canonical_training_entry"] = rows["entry"]
        rows.update(entry=entry, training_recovery_helper=helper)
        return rows

    def restore(api, config, corpus, world, report, device, torch):
        if record(CHECKPOINT) != checkpoint:
            raise ValueError("Original checkpoint changed before restore")
        state = torch.load(CHECKPOINT, weights_only=True, map_location="cpu", mmap=True)
        admitted = admit_checkpoint(state, api, profile)
        if admitted.to_document() != resolve_requested_config(config, api).to_document():
            raise ValueError("Recovery actual training config differs from checkpoint")
        if state["provenance"]["entry_sources"] != parent_rows:
            raise ValueError("Checkpoint provenance differs from admitted original S3 sources")
        if state["provenance"]["AMASS"] != corpus.provenance():
            raise ValueError("Recovery actual AMASS differs from checkpoint")
        if state["provenance"]["physical_scene"]["group_actor_rows"] != world.rows.cpu().tolist():
            raise ValueError("Recovery actor grouping differs from checkpoint")
        del state
        trainer, runner_state = api.SocialAMPTrainer.load(CHECKPOINT, device=device, restore_rng=False)
        if runner_state is not None:
            raise ValueError("This recovery does not support opaque physical runner restoration")
        recovery = {"schema": "shared_social.optimizer_recovery_segment.v1", "binding": binding,
                    "initial_update_count": BASE_UPDATES,
                    "initial_physical_transition_count": BASE_UPDATES * 256 * 32,
                    "model_both_Adam_four_RMS_replay_restored": True,
                    "physical_world_curriculum_history_restored": False,
                    "physical_world_restarted": True,
                    "curriculum_and_corpus_private_RNG_reinitialized_seed": 7,
                    "global_training_rng_restored_after_setup": False,
                    "continuous_original_S3_execution_claimed": False}
        trainer.provenance["training_recovery"] = recovery.copy()
        report["recovery"] = recovery
        report["resume_implemented"] = False
        report["optimizer_recovery_implemented"] = True
        report["initialization"] = "S3 update12000 optimizer recovery; physical World/curriculum/history restarted"
        report["canonical_entry"] = canonical_binding
        with (Path(binding["output"]) / "retained_metrics_prefix.jsonl").open("xb") as stream:
            stream.write(prefix)
        return trainer

    module._sources = sources
    module.__dict__.update(_recovery_restore=restore, _recovery_base=BASE_UPDATES)
    exec(compile(derived, str(canonical.PARENT) + ":optimizer_recovery", "exec"), module.__dict__)
    return module, binding, profile
