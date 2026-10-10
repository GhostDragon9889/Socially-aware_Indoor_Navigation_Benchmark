"""Stage declared source deltas only; no model/data/SDK execution."""
import ast
import json
from pathlib import Path

from convert_official_raw import record

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

PAPER = [
    ("from ..waypoint_weighting import", "from CrowdSim.humanoid.waypoint_weighting import"),
    ("from .base_models import", "from CrowdSim.humanoid.diffusion.base_models import"),
    ("from .temporal import", "from CrowdSim.humanoid.diffusion.temporal import"),
    ("from .trace_helpers import", "from CrowdSim.humanoid.diffusion.trace_helpers import"),
    ("        _, grid = self.map_encoder(raster)\n", "        map_available = context.get(\"map_condition_mask\")\n        if map_available is None or bool(map_available.all()):\n            _, grid = self.map_encoder(raster)\n        else:\n            height, width = (int(value) for value in self.map_encoder.feat_map_out_dim(224, 224))\n            grid = raster.new_zeros((batch, self.config.map_grid_feature_dim, height, width))\n            if bool(map_available.any()):\n                _, available_grid = self.map_encoder(raster[map_available])\n                grid = grid.index_copy(0, map_available.nonzero().flatten(), available_grid)\n"),
    ("                        neighbor_mask=None):\n", "                        neighbor_mask=None, map_condition_mask=None):\n"),
    ("        return {\"history\": world_to_agent_state(history, current), \"history_mask\": mask,\n", "        map_available = torch.ones(batch, device=self.device, dtype=torch.bool) if map_condition_mask is None else self.tensor(map_condition_mask, boolean=True)\n        if map_available.shape != (batch,):\n            raise ValueError(\"map_condition_mask[B] required\")\n        return {\"history\": world_to_agent_state(history, current), \"history_mask\": mask,\n"),
    ("                \"raster\": image}\n", "                \"raster\": image, \"map_condition_mask\": map_available}\n"),
]

TRAINER = [
    ("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path(__file__).resolve().parents[3]"),
    ('PHASE = "new_orca_map_neighbor_conditional_finetune"', 'PHASE = "new_official_raw_interact_resampled_neighbor_only_finetune"'),
    ('    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner, PaperDiffusionTrainBatch', '    from paper_interact import PaperDiffusionPlanner, PaperDiffusionTrainBatch'),
    ('    from CrowdSim.humanoid.orca_corpus import ORCACorpus, sha256', '    from interaction_corpus import InteractionORCACorpus as ORCACorpus, sha256'),
    ('default=ROOT / "data/diffusion_amp/orca_social_pilot_v1/manifest.json"', 'default=ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json"'),
    ('output/diffusion_amp/social_diffusion_training"', 'output/diffusion_amp/social_diffusion_interact_staged_v1"'),
    ('"data": "New pinned TRACE/RVO2 ORCA experts with actual obstacle maps and synchronized neighbors"', '"data": "Official Interact raw trajectories independently resampled to100 true t0..9.9 points; no maps"'),
    ('"Velocity heading proxy of0.4m discs, not SMPL poses or AMP reference motions"', '"Current interpolated velocity heading proxy with explicit stationary0; actual variable disc radii, not SMPL/AMP poses"'),
    ('"31ego/history frames, real neighbors within15m and their valid masks;2x224x224 local occupancy"', '"31ego/history frames,1..19 real neighbors within15m; unknown .5 raster, explicit false map mask and zero map-grid features"'),
    ('"Scene/episode/layout disjoint validation/test; fixed heldout noise and masks; open-loop trajectory QA is not a simulator rollout"', '"Scene/episode-disjoint registered800/100/100 split; true empty layouts repeat legally; open-loop QA is not a simulator rollout"'),
    ('history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"])', 'history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], map_condition_mask=sample["map_condition_mask"])'),
    ('history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], seed=args.seed + 400 + start)', 'history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], map_condition_mask=sample["map_condition_mask"], seed=args.seed + 400 + start)'),
    ('if not all(math.isfinite(value) and value > 0 for value in norms.values()):', 'map_available = bool(sample["map_condition_mask"].any())\n            if not (math.isfinite(norms["neighbor_encoder"]) and norms["neighbor_encoder"] > 0\n                    and math.isfinite(norms["map_encoder"]) and (norms["map_encoder"] > 0 if map_available else norms["map_encoder"] == 0)): '),
    ('"batch_occupied_pixels": int((sample["raster"][:, 1] == 1).sum()),', '"batch_occupied_pixels": int((sample["raster"][:, 1] == 1).sum()),\n                                      "batch_available_map_windows": int(sample["map_condition_mask"].sum()),'),
]

NEW_METRICS = '''def offline_plan_metrics(corpus, sample, predicted_xy):
    """Original ADE/FDE formulas; real radii and no-map metric N/A semantics."""
    predicted_xy = np.asarray(predicted_xy)
    if predicted_xy.shape != (len(sample["states"]), 50, 2) or not np.isfinite(predicted_xy).all():
        raise ValueError("Finite predicted future XY[B,50,2] required")
    target = np.asarray(sample["future"])[..., :50, :2]
    error = np.linalg.norm(predicted_xy - target, axis=-1)
    pair_gaps = []
    for row, index in enumerate(np.asarray(sample["window_indices"])):
        episode_index, agent, frame = corpus.windows[index]
        entry = corpus.episodes[episode_index]
        others = np.delete(entry["states"][:, frame + 1:frame + 51, :2], agent, axis=0)
        radii = np.asarray(entry["agent_radii_m"])
        gaps = np.linalg.norm(predicted_xy[row][None] - others, axis=-1)
        pair_gaps.append((gaps - (radii[agent] + np.delete(radii, agent))[:, None]).min(axis=0))
    pair_gaps = np.asarray(pair_gaps)
    collision = pair_gaps < -1e-4
    return {"windows": len(error), "ADE_m": float(error.mean()), "FDE_m": float(error[:, -1].mean()),
            "min_neighbor_surface_gap_m": float(pair_gaps.min()), "min_obstacle_surface_gap_m": None,
            "neighbor_collision_point_fraction": float(collision.mean()), "obstacle_collision_point_fraction": None,
            "neighbor_collision_trajectory_fraction": float(collision.any(axis=1).mean()),
            "obstacle_collision_trajectory_fraction": None, "qa_tolerance_m": 1e-4,
            "scope": "Open-loop ego predictions against recorded neighbors at aligned10Hz with actual agent radii; no obstacle map available, not a closed-loop rollout"}


'''


def stage(parent, target, operations):
    before = parent.read_text()
    text = before
    for old, new in operations:
        if text.count(old) != 1:
            raise ValueError("Expected one exact source fragment: " + old[:80])
        text = text.replace(old, new)
    ast.parse(text)
    target.write_text(text)
    inverse = text
    for old, new in reversed(operations):
        inverse = inverse.replace(new, old)
    if inverse != before or ast.dump(ast.parse(inverse)) != ast.dump(ast.parse(before)):
        raise ValueError("Staged delta does not reverse to complete source")
    return {"parent": record(parent), "staged": record(target), "operations": operations,
            "inverse_entire_bytes_AST_exact": True}


def main():
    paper = stage(ROOT / "CrowdSim/humanoid/diffusion/paper.py", HERE / "paper_interact.py", PAPER)
    parent = ROOT / "scripts/train_orca_diffusion.py"
    text = parent.read_text()
    old_metrics = text[text.index("def offline_plan_metrics("):text.index("def parse_args(")]
    operations = TRAINER + [(old_metrics, NEW_METRICS),
        ('for key in reports[0] if key not in ("scope", "qa_tolerance_m", "windows", "min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m")',
         'for key in reports[0] if reports[0][key] is not None and key not in ("scope", "qa_tolerance_m", "windows", "min_neighbor_surface_gap_m", "min_obstacle_surface_gap_m")'),
        ('min_obstacle_surface_gap_m=min(row["min_obstacle_surface_gap_m"] for row in reports))',
         'min_obstacle_surface_gap_m=None, obstacle_collision_point_fraction=None, obstacle_collision_trajectory_fraction=None)')]
    trainer = stage(parent, HERE / "train_interact_diffusion.py", operations)
    report = {"schema": "local_orca_interact.staged_source_derivation.v1", "status": "source_only_staged",
              "paper_delta": paper, "trainer_delta": trainer,
              "DDPM_network_loss_optimizer_guidance_unchanged_except_explicit_map_availability": True,
              "full_conversion_executed": False, "training_executed": False, "GPU_used": False,
              "SDK_started": False, "models_or_weights_read": False}
    with (HERE / "source_derivation_v2.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "paper": paper["staged"], "trainer": trainer["staged"]}))


if __name__ == "__main__":
    main()
