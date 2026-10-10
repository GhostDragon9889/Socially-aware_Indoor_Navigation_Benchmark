"""Read-only post-training audit; no simulator, GPU, optimization, or test split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def artifact(path):
    path = ROOT / path
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def summarize(path):
    rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]
    result = {"artifact": artifact(path), "updates": len(rows), "windows": {}}
    keys = ("policy_loss", "value_loss", "motion_symmetry_loss", "amp_rewards_mean",
            "trajectory_rewards_mean", "energy_rewards_mean", "direction_rewards_mean",
            "disc_policy_probability", "disc_expert_probability", "approximate_kl",
            "policy_target_kl_stopped", "forward_velocity_mean", "policy_grad_norm")
    for label, window in (("first100", rows[:100]), ("last100", rows[-100:])):
        means = {key: sum(row[key] for row in window) / len(window) for key in keys}
        task = .5 * sum(means[key] for key in ("trajectory_rewards_mean", "energy_rewards_mean", "direction_rewards_mean"))
        means.update({"current_outer2_AMP_mean_contribution": 2 * means["amp_rewards_mean"],
                      "counterfactual_outer0_5_same_raw_AMP_mean_contribution": .5 * means["amp_rewards_mean"],
                      "task_weight0_5_mean_contribution": task,
                      "current_mean_AMP_to_task_contribution_ratio": 2 * means["amp_rewards_mean"] / task,
                      "counterfactual_mean_AMP_to_task_contribution_ratio": .5 * means["amp_rewards_mean"] / task,
                      "weighted_symmetry_mean_loss": 50 * means["motion_symmetry_loss"]})
        result["windows"][label] = means
    return result


def main():
    torch.set_num_threads(4)
    relative = "output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_snapshots_v1/step_2000.pt"
    checkpoint = torch.load(ROOT / relative, map_location="cpu", weights_only=True, mmap=True)
    model = nn.Sequential(nn.Linear(2019, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    model.load_state_dict({key.removeprefix("discriminator."): value
                           for key, value in checkpoint["model"].items() if key.startswith("discriminator.")})
    model.eval()
    replay = checkpoint["policy_replay"]
    # Fixed deterministic subset from already saved *training* replay only.
    ids = torch.linspace(0, replay["size"] - 1, 1024).long()
    history = replay["history"][ids].flatten(1)
    actions = replay["actions"][ids]
    moments = checkpoint["normalizers"]["amp_normalizer"]
    normalized = ((history - moments["mean"].float()) / torch.sqrt(moments["variance"].float() + 1e-5)).clamp(-5, 5)
    mask = torch.tensor(checkpoint["config"]["action_mask"], dtype=torch.float32)
    features = torch.cat((normalized, actions), -1).detach().requires_grad_(True)
    logits = model(features).squeeze(-1)
    gradient = torch.autograd.grad(F.softplus(logits).sum(), features)[0]
    with torch.no_grad():
        zero_logits = model(torch.cat((normalized, torch.zeros_like(actions)), -1)).squeeze(-1)
        shuffled_logits = model(torch.cat((normalized, actions.roll(1, 0)), -1)).squeeze(-1)
        weight = model[0].weight
        changes = logits.detach() - zero_logits
        shuffled_changes = logits.detach() - shuffled_logits
        sensitivity = {"samples": len(ids), "sampling": "1024 linearly spaced indices from checkpoint's saved train policy replay", 
                       "first_layer_history_weight_Frobenius": float(weight[:, :1950].norm()),
                       "first_layer_active_action_weight_Frobenius": float(weight[:, 1950:][:, mask.bool()].norm()),
                       "frozen_action_columns_exact_zero": bool((weight[:, 1950:][:, ~mask.bool()] == 0).all()),
                       "AMP_normalizer_count": float(moments["count"]),
                       "zero_action_logit_difference_mean_absolute": float(changes.abs().mean()),
                       "zero_action_logit_difference_max_absolute": float(changes.abs().max()),
                       "shuffled_action_logit_difference_mean_absolute": float(shuffled_changes.abs().mean()),
                       "softplus_history_gradient_L2_mean": float(gradient[:, :1950].norm(dim=-1).mean()),
                       "softplus_active_action_gradient_L2_mean": float(gradient[:, 1950:][:, mask.bool()].norm(dim=-1).mean()),
                       "normalized_history_clip_fraction": float((normalized.abs() == 5).float().mean()),
                       "causal_limit": "Saved replay intervention is a fixed-state discriminator diagnostic; not a physics rerun or causal quality gain."}
    sources = ["CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/amp_world.py",
               "CrowdSim/humanoid/amass_corpus.py", "CrowdSim/humanoid/training/symmetry.py",
               "scripts/train_paper_amp.py", "third_party/pacer/pacer/learning/amp_continuous.py",
               "configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml"]
    report = {"schema": "root.independent_training_style_root_cause_CPU.v1", "GPU_used": False,
              "optimizer_steps": 0, "sealed34_outputs_read": False,
              "checkpoint": artifact(relative), "configuration": checkpoint["config"],
              "source_artifacts": [artifact(path) for path in sources],
              "state_prior_metrics": summarize("output/diffusion_amp/amp_style_mixed_state_action_terrain_state_prior_isolated_v1/sealed_terminal/metrics.jsonl"),
              "random_action_D_protected_metrics": summarize("output/diffusion_amp/amp_style_mixed_state_action_terrain_protected_isolated_v1/metrics.jsonl"),
              "saved_training_replay_action_sensitivity": sensitivity,
              "verified_findings": {
                  "state_action_pairing": "Pre-step newest-first history s_t..s_t-9 with actual a_t. Expert same t history with normalized q_t+1 PD target proxy. No temporal offset found.",
                  "state_only_pairing": "Explicit PACER-compatible post-step history; distinct declared objective, not silently mistaken state-action pairing.",
                  "symmetry50": "Official pinned PACER _sym_loss contains literal 50. Current effective-active normalization averages57 rather than public69 coordinates, giving69/57=1.2105263 greater weight for equal active residuals; declared engineering change.",
                  "effective_AMP_weight": "F.softplus(logit)*amp_reward_scale2 then amp_reward_weight2 gives4. Public PACER scale2*disc_reward_w0.5 gives1.",
                  "history_RMS": "1950 per-history coordinates. update reshapes complete10x195 histories; same transferred public history moments. Additional69 action inputs separately normalized PD targets, no implicit RMS mix.",
                  "expert_action_limit": "AMASS contains poses rather than measured commands; q_t+1/scale is a kinematic PD target proxy, not inverse dynamics or true optimal tracking torque command.",
                  "upper_limb_observability": "Policy receives XY future and current pose, no target pose/semantic latent. Unconditional AMP can reward realistic arm motions but cannot require the unique upper-body AMASS reference trajectory for an otherwise identical XY command.",
                  "critical_correctness_bug": None},
              "next_single_variable": {"amp_reward_weight": {"from": 2.0, "to": .5},
                                      "effective_softplus_coefficient": {"from": 4.0, "to": 1.0},
                                      "metadata_reward_amp_weight_must_match": True,
                                      "matches_public_PACER_weight": True,
                                      "NavIsaacLab_exact_hyperparameters_claim": False,
                                      "requires_fresh_training_and_same_full_validation": True,
                                      "guaranteed_style_or_route_improvement": False},
              "limitations": ["Training reward contribution means do not identify gradient attribution or predict retraining quality.",
                              "No sealed test tuning, no original paper8-scene protocol, no author's exact2848 list.",
                              "No production/default/historical snapshot changes; no GPU inference or simulator launch."]}
    destination = OUT / "training_root_cause_CPU.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(destination), "replay": sensitivity, "last100": report["state_prior_metrics"]["windows"]["last100"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
