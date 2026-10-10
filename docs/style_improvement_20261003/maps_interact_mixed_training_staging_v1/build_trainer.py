"""Build a declared metadata/data adapter from the frozen Interact trainer."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "orca_interact_usable_staging_v1/train_interact_diffusion.py"
source = BASE.read_text()
text = source
changes = []


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact occurrence: {old[:80]!r}")
    text = text.replace(old, new)
    changes.append({"old": old, "new": new})


replace('PHASE = "new_official_raw_interact_resampled_neighbor_only_finetune"',
        'PHASE = "new_maps_interact_union_uniform_window_finetune"')
replace('def offline_plan_metrics(corpus, sample, predicted_xy):',
        'def _interaction_offline_plan_metrics(corpus, sample, predicted_xy):')
replace('def parse_args(argv=None):', '''def offline_plan_metrics(corpus, sample, predicted_xy):
    """Dispatch to each unchanged domain metric, preserving no-map N/A."""
    if corpus.domain == "maps":
        from scripts.train_orca_diffusion import offline_plan_metrics as maps_metrics
        return maps_metrics(corpus.corpus, sample, predicted_xy)
    return _interaction_offline_plan_metrics(corpus.corpus, sample, predicted_xy)


def encoding_map_mask(sample, mode):
    """Keep data presence false; separately select the encoder input policy."""
    mask = sample["map_condition_mask"]
    if mode == "skip_encoder":
        return mask
    if mode != "trace_unknown_raster":
        raise ValueError("Unknown missing-map encoding")
    return mask.new_ones(mask.shape) if hasattr(mask, "new_ones") else np.ones_like(mask, dtype=bool)


def make_training_batch(planner, sample, mode):
    from paper_interact import PaperDiffusionTrainBatch
    return PaperDiffusionTrainBatch.from_world(planner, sample["states"], sample["history"],
        sample["neighbors"], sample["raster"], sample["future"],
        history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"],
        map_condition_mask=encoding_map_mask(sample, mode))


def validate_resume_metadata(metadata, corpora, mode):
    requested = {corpus.split: corpus.resume_identity() for corpus in corpora}
    if (metadata.get("phase") != PHASE or metadata.get("corpus_identity") != requested
            or metadata.get("missing_map_encoding") != mode):
        raise ValueError("Resume phase, union corpora/sampling identity or missing-map encoding differs")


def parse_args(argv=None):''')
replace('    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json")',
        '''    parser.add_argument("--maps-manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_maps_expanded_1000_v1/manifest.json")
    parser.add_argument("--interact-manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json")
    parser.add_argument("--missing-map-encoding", choices=("trace_unknown_raster", "skip_encoder"), default="trace_unknown_raster")''')
replace('    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/social_diffusion_interact_staged_v1")',
        '    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/social_diffusion_maps_interact_staged_v1")')
replace('    from paper_interact import PaperDiffusionPlanner, PaperDiffusionTrainBatch\n    from interaction_corpus import InteractionORCACorpus as ORCACorpus, sha256',
        '''    from mixed_corpus import MixedORCACorpus, INTERACT_STAGE, sha256
    sys.path.insert(0, str(INTERACT_STAGE))
    from paper_interact import PaperDiffusionPlanner''')
replace('    train = ORCACorpus(args.manifest, "train", device=args.device, seed=args.seed)\n    validation = ORCACorpus(args.manifest, "validation", device=args.device, seed=args.seed + 1)\n    test = ORCACorpus(args.manifest, "test", device=args.device, seed=args.seed + 2)',
        '''    train = MixedORCACorpus(args.maps_manifest, args.interact_manifest, "train", device=args.device, seed=args.seed)
    validation = MixedORCACorpus(args.maps_manifest, args.interact_manifest, "validation", device=args.device, seed=args.seed + 1)
    test = MixedORCACorpus(args.maps_manifest, args.interact_manifest, "test", device=args.device, seed=args.seed + 2)''')
replace('    assumptions = {"phase": PHASE, "data": "Official Interact raw trajectories independently resampled to100 true t0..9.9 points; no maps",\n                   "author_dataset": False, "author_checkpoint": False,\n                   "body_reference": "Current interpolated velocity heading proxy with explicit stationary0; actual variable disc radii, not SMPL/AMP poses",\n                   "conditions": "31ego/history frames,1..19 real neighbors within15m; unknown .5 raster, explicit false map mask and zero map-grid features",\n                   "initialization": "Independent AMASS model and its normalization; new social optimizer/RNG, no author or TRACE4D weights",\n                   "validation": "Scene/episode-disjoint registered800/100/100 split; true empty layouts repeat legally; open-loop QA is not a simulator rollout"}',
        '''    assumptions = {"phase": PHASE,
                   "data": "Existing Maps reconstruction plus official raw Interact resampled to100 true t0..9.9 points; preserved splits",
                   "author_dataset": False, "author_checkpoint": False,
                   "body_reference": "Velocity-derived disc heading proxy, not SMPL/AMP poses; each domain retains its original metric radii",
                   "conditions": "31 inclusive ego/history frames,19 neighbor slots with real masks; true no-map .5 raster and false data presence",
                   "missing_map_encoding": args.missing_map_encoding,
                   "missing_map_encoding_definition": "trace_unknown_raster encodes .5 rasters with encoder availability True; skip_encoder passes actual presence and is a local ablation",
                   "sampling": "Uniform over union valid windows with replacement; no imposed50:50 domain ratio, not author epoch-shuffle identity",
                   "initialization": "Independent AMASS model and normalization; new optimizer/RNG, no author or TRACE4D weights",
                   "validation": "Per-domain original open-loop metrics; Interact obstacle metrics remain N/A, not zero; no physical rollout claim"}''')
replace('        if metadata.get("phase") != PHASE or metadata.get("train", {}).get("manifest_sha256") != train.provenance()["manifest_sha256"]:\n            raise ValueError("Resume phase or source manifest differs")',
        '        validate_resume_metadata(metadata, (train, validation, test), args.missing_map_encoding)')
replace('        return PaperDiffusionTrainBatch.from_world(planner, sample["states"], sample["history"],\n            sample["neighbors"], sample["raster"], sample["future"],\n            history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], map_condition_mask=sample["map_condition_mask"])',
        '        return make_training_batch(planner, sample, args.missing_map_encoding)')
replace('    def evaluate(corpus):\n        previous',
        '''    def evaluate(corpus):
        if hasattr(corpus, "domains"):
            return {name: evaluate(domain) for name, domain in corpus.domains.items()}
        previous''')
replace('    def sample_evaluation(corpus):\n        previous',
        '''    def sample_evaluation(corpus):
        if hasattr(corpus, "domains"):
            return {name: sample_evaluation(domain) for name, domain in corpus.domains.items()}
        previous''')
replace('                    history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], map_condition_mask=sample["map_condition_mask"], seed=args.seed + 400 + start)',
        '                    history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"], map_condition_mask=encoding_map_mask(sample, args.missing_map_encoding), seed=args.seed + 400 + start)')
replace('                      min_obstacle_surface_gap_m=None, obstacle_collision_point_fraction=None, obstacle_collision_trajectory_fraction=None)',
        '''                      min_obstacle_surface_gap_m=(min(row["min_obstacle_surface_gap_m"] for row in reports) if reports[0]["min_obstacle_surface_gap_m"] is not None else None))
        for key in ("obstacle_collision_point_fraction", "obstacle_collision_trajectory_fraction"):
            if reports[0][key] is None:
                result[key] = None''')
replace('        metadata = {"phase": PHASE, "initialization": initialization, "assumptions": assumptions,',
        '''        metadata = {"phase": PHASE, "initialization": initialization, "assumptions": assumptions,
                    "missing_map_encoding": args.missing_map_encoding,
                    "corpus_identity": {corpus.split: corpus.resume_identity() for corpus in (train, validation, test)},''')
replace('        # The first update explicitly exercises both conditions for audit evidence.\n        values = planner.train_step(from_sample(sample), optimizer, drop_conditioning=first_gradient_evidence is not None)\n        if first_gradient_evidence is None:\n            norms = {name: math.sqrt(sum(float(parameter.grad.detach().square().sum()) for parameter in module.parameters() if parameter.grad is not None))\n                     for name, module in (("map_encoder", planner.model.map_encoder), ("neighbor_encoder", planner.model.neighbor_encoder))}\n            map_available = bool(sample["map_condition_mask"].any())\n            if not (math.isfinite(norms["neighbor_encoder"]) and norms["neighbor_encoder"] > 0\n                    and math.isfinite(norms["map_encoder"]) and (norms["map_encoder"] > 0 if map_available else norms["map_encoder"] == 0)): \n                raise RuntimeError(f"Actual conditional encoder gradients are absent: {norms}")\n            first_gradient_evidence = {"step": planner.trained_steps, "grad_norms_after_clip": norms,\n                                      "batch_real_neighbor_slots": int(sample["neighbor_mask"][:, :, -1].sum()),\n                                      "batch_occupied_pixels": int((sample["raster"][:, 1] == 1).sum()),\n                                      "batch_available_map_windows": int(sample["map_condition_mask"].sum()),\n                                      "conditioning_dropout": False}',
        '''        # Until each encoder has evidence, do not drop its available condition.
        evidence_complete = (first_gradient_evidence is not None
                             and first_gradient_evidence["map_encoder"] is not None
                             and first_gradient_evidence["neighbor_encoder"] is not None)
        values = planner.train_step(from_sample(sample), optimizer, drop_conditioning=evidence_complete)
        if first_gradient_evidence is None:
            first_gradient_evidence = {"map_encoder": None, "neighbor_encoder": None, "first_all_missing_map": None}
        if not evidence_complete:
            for name, module in (("map_encoder", planner.model.map_encoder), ("neighbor_encoder", planner.model.neighbor_encoder)):
                gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
                norm = math.sqrt(sum(float(gradient.detach().square().sum()) for gradient in gradients))
                encoded_map = bool(encoding_map_mask(sample, args.missing_map_encoding).any())
                required = name != "map_encoder" or encoded_map
                if required and not (math.isfinite(norm) and norm > 0):
                    raise RuntimeError(f"Actual available conditional gradients are absent: {name}={norm}")
                if not required and gradients:
                    raise RuntimeError("Absent map encoder unexpectedly has gradients")
                if required and first_gradient_evidence[name] is None:
                    first_gradient_evidence[name] = {
                        "step": planner.trained_steps, "grad_norm_after_clip": norm,
                        "data_map_windows": int(sample["map_condition_mask"].sum()),
                        "encoded_map_windows": int(encoding_map_mask(sample, args.missing_map_encoding).sum()),
                        "batch_real_neighbor_slots": int(sample["neighbor_mask"][:, :, -1].sum()),
                        "batch_occupied_pixels": int((sample["raster"][:, 1] == 1).sum()),
                        "conditioning_dropout": False,
                        "unknown_raster_encoding_is_obstacle_evidence": False}
                if not required and first_gradient_evidence["first_all_missing_map"] is None:
                    first_gradient_evidence["first_all_missing_map"] = {"step": planner.trained_steps, "all_map_gradients_none": True}
        values["batch_domain_counts"] = {name: sample["domain_names"].count(name) for name in ("maps", "interact")}''')
destination = HERE / "train_mixed_diffusion.py"
destination.write_text(text)
reverse = text
for change in reversed(changes):
    if reverse.count(change["new"]) != 1:
        raise ValueError("Reverse ambiguity")
    reverse = reverse.replace(change["new"], change["old"])
if reverse != source or ast.dump(ast.parse(reverse)) != ast.dump(ast.parse(source)):
    raise ValueError("Inverse byte/AST identity failed")
record = {"status": "declared_adapter_source_only", "base": str(BASE),
          "base_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
          "derived": str(destination), "derived_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
          "inverse_entire_bytes_exact": True, "inverse_entire_AST_exact": True,
          "replacements": changes, "NN_DDPM_optimizer_schedule_math_changed": False,
          "actual_model_smoke_completed": False}
(HERE / "trainer_derivation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "replacements"}))

