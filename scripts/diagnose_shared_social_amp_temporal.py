#!/usr/bin/env python3
"""Diagnose completed S3-R's temporal/action sensitivity on its final archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import amp_temporal_diagnostic as diagnostic  # noqa: E402
from CrowdSim.humanoid.application import recovery_evaluation_admission as admission  # noqa: E402
from scripts import resume_shared_social_recovery_evaluation as io  # noqa: E402

HERE = ROOT / "docs/style_improvement_20261008/shared_social_S3R_temporal_discriminator_CPU_v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    admission.require(not output.exists(), "Preserve previous diagnostics; use a fresh output")
    documents, identities, _ = admission.completed_documents()
    archive_identity = documents["audit"]["rollout"]
    checkpoint_identity = documents["audit"]["checkpoint"]
    inputs = [*identities.values(), archive_identity, checkpoint_identity]
    sources = [admission.record(path) for path in (Path(__file__), Path(diagnostic.__file__),
        ROOT / "CrowdSim/humanoid/training/trainer.py",
        ROOT / "docs/style_improvement_20261004/social_trainer_staging_v1/social_trainer.py")]
    admission.require(sources[2]["sha256"] == "707f355acbd8b5d33aa40c3299a2540610ea695ae4bb9705ede3498e27224f9d",
                      "Original AMP numerical code changed")
    for row in inputs:
        admission.require(admission.readonly_identity(row["path"]) == row, "Completed diagnostic input changed")
    output.mkdir(parents=True)
    import numpy as np
    import torch

    torch.set_num_threads(2)
    # CPU mapping with restricted tensor/primitive loading follows pinned docs:
    # https://docs.pytorch.org/docs/2.11/generated/torch.load.html
    state = torch.load(checkpoint_identity["path"], map_location="cpu", weights_only=True, mmap=True)
    archive = torch.load(archive_identity["path"], map_location="cpu", weights_only=True, mmap=True)
    admission.require(state["update_count"] == 12500 and state["physical_transition_count"] == 102400000,
                      "Select the actual preselected completed S3-R terminal model")
    proxy = diagnostic.load_discriminator(state)
    rollout, expert = archive["collection"]["rollout"], archive["expert"]
    populations = {"physical_policy": (rollout["amp_history"].reshape(-1, 10, 195),
                                        rollout["applied_actions"].reshape(-1, 69)),
                   "actual_AMASS_expert": (expert["history"], expert["actions"])}
    admission.require(len(populations["physical_policy"][0]) == 8192 and len(expert["history"]) == 4096
        and bool(torch.equal(rollout["history_episode_ids"], rollout["episode_ids"].unsqueeze(-1).expand(-1, -1, 10)))
        and bool(((expert["motion_ids"] >= 0) & (expert["motion_ids"] < 2848)).all()),
        "Keep the full actual physical/expert archive without filtering failures")
    summaries, arrays = {}, {}
    for role, (history, actions) in populations.items():
        summaries[role], scores = diagnostic.measure(proxy, history, actions)
        for name, values in scores.items():
            for kind, value in values.items():
                arrays[role + "__" + name + "__" + kind] = value.numpy()
    np.savez_compressed(output / "scores.npz", **arrays)
    for row in inputs + sources:
        admission.require(admission.record(row["path"]) == row, "Diagnostic input/source changed during inference")
    admission.require(torch.cuda.is_initialized() is False
                      and not any(name.startswith(("isaacsim", "isaaclab", "omni.")) for name in sys.modules),
                      "Temporal diagnostic must stay CPU-only without an SDK")
    report = {"schema": "shared_social.frozen_S3R_temporal_discriminator_diagnostic.v1", "status": "passed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "inputs": inputs,
        "scores": admission.record(output / "scores.npz"), "conditions": list(diagnostic.CONDITIONS),
        "summaries": summaries, "history_order": "newest-first; current index0 remains fixed for past permutations",
        "normalization": "saved per-frame shared195 RMS; original full flattened2019 discriminator",
        "scoring_phase": "post-update12500 checkpoint on that update's captured pre-update rollout/expert inputs",
        "original_logged_AMP_reward_trace_reproduced": False,
        "actual_expert_unique_motion_ids": int(expert["motion_ids"].unique().numel()),
        "samples_are_training_inputs_not_heldout_quality_evaluation": True,
        "causal_training_failure_claimed": False, "model_RMS_or_original_metric_changed": False,
        "Torch_version": torch.__version__, "CUDA_initialized": False, "SDK_started": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    (output / "scores.npz").chmod(0o444)
    io.write_json(output / "result.json", report, readonly=True)
    print(io.json.dumps({"status": "passed", "result": admission.record(output / "result.json"),
                        "summaries": summaries, "CUDA_initialized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
