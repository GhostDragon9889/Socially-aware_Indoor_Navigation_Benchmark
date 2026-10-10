#!/usr/bin/env python3
"""CPU motion-style diagnosis against frozen held-out contiguous AMASS windows."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def identity(value):
    return str(value).replace("\\", "/")


def reference_selection(document, *, split="validation", windows=256, window_frames=61, seed=7):
    if split not in {"validation", "test"}:
        raise ValueError("Style reference must be held-out validation or test, never train")
    if type(windows) is not int or windows < 1 or type(window_frames) is not int or window_frames < 3:
        raise ValueError("Reference window counts must be positive and windows have at least3 frames")
    eligible = [clip for clip in document["splits"][split]["clips"] if clip["frames"] >= window_frames]
    if not eligible:
        raise ValueError("No held-out clip supports the complete contiguous window")
    generator = np.random.default_rng(seed)
    selected = []
    for _ in range(windows):
        clip = eligible[int(generator.integers(len(eligible)))]
        start = int(generator.integers(clip["frames"]-window_frames+1))
        if abs(clip["dt"]-1/30) > 1e-5:
            raise ValueError("Style windows require30Hz references")
        selected.append({"motion_id": clip["motion_id"], "source_path": clip["source_path"],
                         "start_frame": start, "stop_frame_exclusive": start+window_frames})
    return {"split": split, "seed": seed, "windows": selected, "window_frames": window_frames,
        "window_duration_s": (window_frames-1)/30, "eligible_clips": len(eligible),
        "excluded_short_clips": len(document["splits"][split]["clips"])-len(eligible),
        "sampling": "uniform clip then uniform valid contiguous window, with replacement; overlaps are not independent observations"}


def assert_heldout_disjoint(selection, training_document):
    selected = {identity(window["source_path"]) for window in selection["windows"]}
    training = {identity(clip["source_path"]) for clip in training_document["splits"]["train"]["clips"]}
    overlap = selected & training
    if overlap:
        raise ValueError(f"Style reference leaks original source files into training: {sorted(overlap)[:3]}")
    return {"selected_original_sources": len(selected), "declared_training_original_sources": len(training),
            "original_source_intersection": 0, "subject_session_independence_proven": False}


def load_reference(manifest, selection):
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    corpus = AMASSCorpus(manifest, selection["split"], device="cpu", seed=selection["seed"])
    metadata = {clip["motion_id"]: clip for clip in corpus.entry["clips"]}
    indices = []
    for window in selection["windows"]:
        clip = metadata.get(window["motion_id"])
        if clip is None or identity(clip["source_path"]) != identity(window["source_path"]):
            raise ValueError("Frozen held-out motion identity differs from the manifest")
        start, stop = window["start_frame"], window["stop_frame_exclusive"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= clip["frames"] or stop-start != selection["window_frames"]:
            raise ValueError("Frozen reference window would cross a clip or violate its duration")
        indices.append(int(corpus.data["length_starts"][window["motion_id"]])+torch.arange(start, stop))
    absolute = torch.stack(indices)
    body = corpus.data["gts"][absolute].numpy().transpose(1, 0, 2, 3).copy()
    rotation = corpus.data["grs"][absolute, 0].numpy().transpose(1, 0, 2).copy()
    digests = {"body_pos": hashlib.sha256(body.tobytes()).hexdigest(), "root_rot": hashlib.sha256(rotation.tobytes()).hexdigest()}
    return body, rotation, digests, corpus.provenance()


def subset(samples, mask):
    return {**samples, "speed_m_s": samples["speed_m_s"][mask],
            "metrics": {name: value[mask] for name, value in samples["metrics"].items()},
            "metric_valid": {name: value[mask] for name, value in samples["metric_valid"].items()},
            "local_pose": samples["local_pose"][mask], "local_pose_velocity": samples["local_pose_velocity"][mask]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=physical capture directory; repeat for comparisons")
    parser.add_argument("--reference-manifest", type=Path, default=ROOT/"data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--reference-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--reference-artifact", type=Path, help="Reuse the exact saved held-out selection, source and spec")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--window-frames", type=int, default=61)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--minimum-bin-samples", type=int, default=30)
    parser.add_argument("--maximum-motion-samples", type=int, default=256)
    parser.add_argument("--verify-package-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be fresh; historical evaluation artifacts are never overwritten")
    if args.cpu_threads < 1:
        parser.error("CPU threads must be positive")
    import torch
    from CrowdSim.humanoid.style_evaluation import BODY_NAMES, StyleSpec, compare_speed_conditioned, sequence_style_samples
    from CrowdSim.humanoid.training.manifests import validate_training_manifest
    from scripts.evaluate_paper_pedestrians import read_capture
    torch.set_num_threads(args.cpu_threads)
    spec = StyleSpec()
    artifact = None
    if args.reference_artifact:
        artifact = json.loads(args.reference_artifact.read_text())
        if artifact.get("schema") != "navisaaclab.heldout_style_reference.v1" or artifact["metric_spec"] != json.loads(json.dumps(asdict(spec))):
            raise ValueError("Frozen reference schema or metric specification differs")
        manifest = Path(artifact["manifest"]["path"])
        if record(manifest) != artifact["manifest"]:
            raise ValueError("Frozen reference manifest changed")
        selection = artifact["selection"]
    else:
        manifest = args.reference_manifest.resolve()
        selection = reference_selection(json.loads(manifest.read_text()), split=args.reference_split,
            windows=args.windows, window_frames=args.window_frames, seed=args.seed)
    gate = validate_training_manifest(manifest, verify_package_hashes=args.verify_package_hashes)
    if selection["split"] not in {"validation", "test"}:
        raise ValueError("Frozen style reference must be held-out")
    body, rotation, sample_hashes, provenance = load_reference(manifest, selection)
    if artifact and sample_hashes != artifact["actual_selected_array_sha256"]:
        raise ValueError("Actual frozen held-out samples changed")
    artifact = {"schema": "navisaaclab.heldout_style_reference.v1", "manifest": record(manifest), "selection": selection,
        "metric_spec": json.loads(json.dumps(asdict(spec))), "actual_selected_array_sha256": sample_hashes,
        "source": provenance, "package_hash_recomputed": args.verify_package_hashes,
        "model_selection_role": "validation for iterative choices; test scores historically exposed, not a pristine blind test"}
    t, b = body.shape[:2]
    episodes = np.ones((t, b), dtype=np.int64)
    resets = np.zeros((t, b), dtype=bool)
    resets[0] = True
    report = {"schema": "navisaaclab.speed_conditioned_style.v1", "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": "CPU", "kit_started": False,
        "author_parameters_reproduced": False, "author_experiment_reproduced": False, "convergence_claim": False,
        "reference": artifact, "manifest_gate": gate, "runs": {}, "sources": {
            "implementation": record(Path(__file__)), "module": record(ROOT/"CrowdSim/humanoid/style_evaluation.py")}}
    for run in args.run:
        name, separator, path = run.partition("=")
        if not separator or not name or name in report["runs"]:
            raise ValueError("Run must provide a unique NAME=directory")
        directory = Path(path).resolve()
        states_path, result_path = directory/"physical_states.npz", directory/"result.json"
        arrays, runner, capture = read_capture(states_path, result_path)
        declared = Path(runner["state_capture"]["path"])
        if (ROOT/declared if not declared.is_absolute() else declared).resolve() != states_path:
            raise ValueError("Runner result identifies a different physical-state capture")
        training_manifest = Path(runner["arguments"]["manifest"])
        training_manifest = ROOT/training_manifest if not training_manifest.is_absolute() else training_manifest
        training_document = json.loads(training_manifest.read_text())
        disjoint = assert_heldout_disjoint(selection, training_document)
        training_gate = validate_training_manifest(training_manifest, verify_package_hashes=False)
        masses = runner.get("physics", {}).get("live_body_masses_kg")
        mass_names = runner.get("physics", {}).get("body_names")
        if masses is not None:
            if len(masses) != 24 or set(mass_names or ()) != set(BODY_NAMES):
                raise ValueError("Live mass metadata must use the exact declared common SMPL body mapping")
            generated_masses = [masses[mass_names.index(body_name)] for body_name in capture["body_names"]]
            reference_masses = [masses[mass_names.index(body_name)] for body_name in BODY_NAMES]
        else:
            generated_masses = reference_masses = None
        generated = sequence_style_samples(arrays["body_pos"], arrays["root_rot"], body_names=capture["body_names"],
            episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"], spec=spec, body_masses_kg=generated_masses)
        reference = sequence_style_samples(body, rotation, episode_ids=episodes, reset_mask=resets, spec=spec, body_masses_kg=reference_masses)
        compare_kwargs = {"spec": spec, "minimum_bin_samples": args.minimum_bin_samples, "maximum_motion_samples": args.maximum_motion_samples}
        inclusive = compare_speed_conditioned(generated, reference, **compare_kwargs)
        unique_reference_clips = sorted({window["motion_id"] for window in selection["windows"]})
        group_a, group_b = set(unique_reference_clips[::2]), set(unique_reference_clips[1::2])
        window_a = np.array([window["motion_id"] in group_a for window in selection["windows"]])
        sample_a = window_a[reference["actor_indices"]]
        calibration = compare_speed_conditioned(subset(reference, sample_a), subset(reference, ~sample_a), **compare_kwargs) if group_a and group_b else None
        gm = (generated["metrics"]["root_up_z"] >= .8)&(generated["metrics"]["pelvis_height_m"] >= .7)
        rm = (reference["metrics"]["root_up_z"] >= .8)&(reference["metrics"]["pelvis_height_m"] >= .7)
        upright = compare_speed_conditioned(subset(generated, gm), subset(reference, rm), **compare_kwargs) if gm.any() and rm.any() else None
        model_files = {}
        for label, key in (("controller", "resume"), ("diffusion", "diffusion_checkpoint")):
            value = runner["arguments"].get(key)
            if value:
                model_path = Path(value)
                model_files[label] = record(ROOT/model_path if not model_path.is_absolute() else model_path)
        report["runs"][name] = {"sources": {"states": record(states_path), "result": record(result_path),
            "declared_training_manifest": record(training_manifest), "models_resolved_from_evaluation_arguments": model_files},
            "capture_validation": capture, "reference_not_in_declared_training": disjoint, "training_manifest_gate": training_gate,
            "rig_mass_metadata": {"body_names": mass_names, "masses_kg": masses, "reference_proxy_uses_same_virtual_rig_masses": masses is not None},
            "generated": generated["summary"], "reference": reference["summary"], "inclusive_speed_conditioned": inclusive,
            "heldout_between_clip_group_calibration": {"group_a_motion_ids": sorted(group_a), "group_b_motion_ids": sorted(group_b),
                "motion_id_intersection": 0, "comparison": calibration,
                "qualification": "Natural between-clip reference heterogeneity, not a confidence interval or automatic pass threshold"},
            "upright_subset": {"root_up_z_min": .8, "pelvis_height_m_min": .7, "generated_fraction_retained": float(gm.mean()),
                "reference_fraction_retained": float(rm.mean()), "comparison": upright},
            "source_navigation_outcomes": {key: runner.get(key) for key in ("fall_resets", "route_vertices_reached", "minimum_height_m")},
            "actual_foot_ground_contact": None, "true_com_world": None,
            "physical_static_peer_contacts": "Preserved in separate existing pedestrian_metrics; not inferred from toe support proxy"}
    report["limitations"] = ["Toe/sole geometry, actual foot-ground contact and body COM positions were not captured; toe metrics and mass-weighted joint origins are proxies",
        "Physical-speed conditioning does not match turn rate, terrain or social/task conditions; deployment changes are not single-factor controller ablations",
        "Window overlaps and adjacent frames are dependent; no IID confidence interval or author-equivalent threshold is claimed",
        "Original source-file disjointness is verified against declared manifests, not subject/session independence or undisclosed public PACER pretraining clips",
        "Test split has already appeared in historical reports; use validation for iterative selection, and disclose repeated exposure"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"heldout_reference.json").write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n")
    (args.output/"evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "reference_split": selection["split"], "runs": list(report["runs"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
