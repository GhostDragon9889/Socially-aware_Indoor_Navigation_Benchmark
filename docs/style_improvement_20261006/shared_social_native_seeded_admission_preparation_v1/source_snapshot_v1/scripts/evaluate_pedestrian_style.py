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
    from CrowdSim.humanoid.style_evaluation import assert_parent_split_disjoint, parent_source_identity
    if split not in {"validation", "test"}:
        raise ValueError("Style reference must be held-out validation or test, never train")
    if type(windows) is not int or windows < 1 or type(window_frames) is not int or window_frames < 3:
        raise ValueError("Reference window counts must be positive and windows have at least3 frames")
    assert_parent_split_disjoint(document)
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
        window = {"motion_id": clip["motion_id"], "source_path": clip["source_path"],
                  "start_frame": start, "stop_frame_exclusive": start+window_frames}
        if "parent_source_path" in clip or "#" in clip["source_path"]:
            window["parent_source_path"] = parent_source_identity(clip)
            for key in ("parent_control_frame_start", "parent_control_frame_stop", "original_motion_id",
                        "parent_source_frame_start", "parent_source_frame_stop"):
                if key in clip:
                    window[key] = clip[key]
        selected.append(window)
    return {"split": split, "seed": seed, "windows": selected, "window_frames": window_frames,
        "window_duration_s": (window_frames-1)/30, "eligible_clips": len(eligible),
        "excluded_short_clips": len(document["splits"][split]["clips"])-len(eligible),
        "sampling": "uniform clip then uniform valid contiguous window, with replacement; overlaps are not independent observations"}


def assert_heldout_disjoint(selection, training_document, reference_document=None):
    from CrowdSim.humanoid.style_evaluation import assert_parent_split_disjoint, parent_source_identity
    if not isinstance(selection, dict) or selection.get("split") not in {"validation", "test"}:
        raise ValueError("Style reference must declare a held-out split")
    if not isinstance(selection.get("windows"), list) or not selection["windows"]:
        raise ValueError("Style reference requires explicit nonempty held-out windows")
    selected = {parent_source_identity(window) for window in selection["windows"]}
    training = assert_parent_split_disjoint(training_document)["train"]
    if reference_document is not None:
        reference_parents = assert_parent_split_disjoint(reference_document)
        for split in ("validation", "test"):
            overlap = training & reference_parents.get(split, set())
            if overlap:
                raise ValueError(f"Style reference leaks held-out {split} parent source files into training: {sorted(overlap)[:3]}")
    overlap = selected & training
    if overlap:
        raise ValueError(f"Style reference leaks original parent source files into training: {sorted(overlap)[:3]}")
    return {"selected_original_sources": len(selected), "declared_training_original_sources": len(training),
            "original_source_intersection": 0, "subject_session_independence_proven": False}


def load_reference(manifest, selection):
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.style_evaluation import parent_source_identity
    corpus = AMASSCorpus(manifest, selection["split"], device="cpu", seed=selection["seed"])
    metadata = {clip["motion_id"]: clip for clip in corpus.entry["clips"]}
    indices = []
    for window in selection["windows"]:
        clip = metadata.get(window["motion_id"])
        if clip is None or identity(clip["source_path"]) != identity(window["source_path"]):
            raise ValueError("Frozen held-out motion identity differs from the manifest")
        if parent_source_identity(clip) != parent_source_identity(window):
            raise ValueError("Frozen held-out parent source identity differs from the manifest")
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


def _result_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Recorded resource paths must be nonempty strings")
    path = Path(value)
    return (ROOT/path if not path.is_absolute() else path).resolve()


def tracking_training_manifest(runner):
    """Bind the actual controller training manifest, not the oracle test manifest."""
    dataset = runner.get("controller_provenance", {}).get("dataset")
    if not isinstance(dataset, dict) or dataset.get("split") != "train":
        raise ValueError("Tracking controller must declare its actual training dataset")
    path = _result_path(dataset.get("manifest"))
    document = json.loads(path.read_text())
    entry = document["splits"]["train"]
    if (record(path)["sha256"] != dataset.get("manifest_sha256") or
            entry["sha256"] != dataset.get("source_sha256") or
            len(entry["clips"]) != dataset.get("clips")):
        raise ValueError("Tracking controller training dataset identity changed")
    return path, document


def read_tracking_capture(directory):
    """Concatenate declared post-step clips; no glob, initial reset, or cross-clip derivative."""
    from CrowdSim.humanoid.style_evaluation import BODY_NAMES, parent_source_identity
    directory = Path(directory).resolve()
    runner = json.loads((directory/"result.json").read_text())
    if runner.get("status") != "passed":
        raise ValueError("Tracking capture execution must have passed")
    sequences = runner.get("sequences")
    selected = runner.get("preflight", {}).get("selected_clips")
    if not isinstance(sequences, list) or not sequences or not isinstance(selected, list) or len(sequences) != len(selected):
        raise ValueError("Tracking result must account for every explicitly selected capture")
    protocol = runner.get("protocol", {})
    expected_phase = "post-physics at 30Hz vs same reference frame; no pelvis/yaw/frame best-match alignment"
    if (protocol.get("metric_phase") != expected_phase or
            protocol.get("initial_frame_excluded_from_errors") is not True or
            protocol.get("fall_reset_within_sequence") is not False or
            protocol.get("reference_3d_poses_enter_actor_or_actions") is not False or
            protocol.get("commands") != "ten true reference root XY positions at +.5..5s; clip-tail hold is command-only"):
        raise ValueError("Unknown tracking metric phase or oracle trajectory task contract")
    if runner.get("physics", {}).get("control_hz") != 30 or tuple(runner.get("physics", {}).get("body_names", ())) != BODY_NAMES:
        raise ValueError("Tracking captures require canonical24-body SMPL at30Hz")
    manifest_path = _result_path(runner.get("arguments", {}).get("manifest"))
    document = json.loads(manifest_path.read_text())
    split = runner.get("arguments", {}).get("split")
    if split not in {"validation", "test"} or record(manifest_path)["sha256"] != runner["preflight"].get("manifest_sha256"):
        raise ValueError("Tracking held-out manifest identity or split changed")
    metadata = {clip["motion_id"]: clip for clip in document["splits"][split]["clips"]}
    ids = [row.get("motion_id") for row in sequences]
    if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Tracking source motion IDs must be unique nonnegative integers")
    if [row.get("motion_id") for row in selected] != ids:
        raise ValueError("Tracking sequence/source selection IDs disagree")
    velocity = runner.get("reference_velocity_contract")
    if not isinstance(velocity, dict):
        raise ValueError("Tracking capture requires explicit reference velocity provenance")
    output_frame = velocity.get("reference_dof_velocity_frame")
    if (velocity.get("expert_dof_velocity_mode") not in {"legacy_stored", "child_joint"} or
            output_frame not in {"parent_joint", "child_joint"} or
            (velocity.get("expert_dof_velocity_mode") == "child_joint" and output_frame != "child_joint") or
            velocity.get("initial_reset_dof_velocity_frame") != output_frame or
            velocity.get("amp_reference_history_dof_velocity_frame") != output_frame or
            velocity.get("physical_native_dof_velocity_frame") != "child_joint" or
            velocity.get("poses_position_rotation_metric_changed") is not False):
        raise ValueError("Tracking reference/native velocity contracts disagree")
    tracking_training_manifest(runner)
    parts, records = [], []
    initial_frames = fall_frames = total = 0
    for ordinal, (sequence, selection) in enumerate(zip(sequences, selected)):
        clip = metadata.get(sequence["motion_id"])
        if clip is None or any(parent_source_identity(value) != parent_source_identity(clip) for value in (sequence, selection)):
            raise ValueError("Tracking source ID/path differs from the held-out manifest")
        if sequence["source_path"] != clip["source_path"] or selection["source_path"] != clip["source_path"]:
            raise ValueError("Tracking capture must preserve the exact source segment identity")
        if selection["frames"] != clip["frames"] or abs(clip["dt"]-1/30) > 1e-5:
            raise ValueError("Tracking source clip count or clock differs")
        path = _result_path(sequence.get("capture"))
        if path.parent != (directory/"captures").resolve() or path.name != f"motion_{sequence['motion_id']:06d}.npz":
            raise ValueError("Tracking capture path escapes its declared result directory or source ID")
        digest = record(path)
        if digest["sha256"] != sequence.get("capture_sha256"):
            raise ValueError("Tracking capture SHA differs from its result")
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in (
                "rigid_body_pos", "root_rot", "reference_frame", "reference_time_s", "metric_frame",
                "native_reset_frame", "episode_id", "fall_mask")}
        frames = len(values["reference_frame"])
        for field, shape in (("rigid_body_pos", (frames, 1, 24, 3)), ("root_rot", (frames, 1, 4))):
            if values[field].shape != shape or values[field].dtype.kind != "f" or not np.isfinite(values[field]).all():
                raise ValueError(f"Tracking {field} must be finite and have the declared shape")
        if not np.allclose(np.linalg.norm(values["root_rot"], axis=-1), 1., atol=1e-3, rtol=0):
            raise ValueError("Tracking root quaternion must be unit XYZW")
        metric, reset, fall = (values[key] for key in ("metric_frame", "native_reset_frame", "fall_mask"))
        if any(value.shape != (frames,) or value.dtype != np.bool_ for value in (metric, reset, fall)):
            raise ValueError("Tracking metric/reset/fall masks require explicit boolean frame vectors")
        if frames < 4 or metric[0] or not metric[1:].all() or not np.array_equal(reset, ~metric):
            raise ValueError("Tracking must have one discarded initial reset followed by true post-step frames")
        window = sequence["tracking_window"]
        count = frames-1
        if (type(sequence["observed_metric_steps"]) is not int or sequence["observed_metric_steps"] != count or
                type(sequence["expected_control_steps"]) is not int or sequence["expected_control_steps"] != count or
                window != selection["tracking_window"] or window["requested_steps"] != count or
                sequence.get("complete_clip") is not True):
            raise ValueError("Tracking frame count/window differs from its complete-sequence result")
        ref_frame, ref_time = values["reference_frame"], values["reference_time_s"]
        if (ref_frame.dtype.kind not in "iu" or ref_frame.shape != (frames,) or
                not np.array_equal(ref_frame, np.arange(window["initial_frame"], window["initial_frame"]+frames)) or
                ref_frame[-1] != clip["frames"]-1 or window["last_frame"] != ref_frame[-1]):
            raise ValueError("Tracking reference phase must be consecutive and end at the actual clip boundary")
        if (ref_time.shape != (frames,) or ref_time.dtype.kind != "f" or not np.isfinite(ref_time).all() or
                not np.allclose(ref_time, ref_frame/30, atol=1e-9, rtol=0)):
            raise ValueError("Tracking reference timestamps must stay on the true30Hz grid")
        episodes = values["episode_id"]
        if episodes.shape != (frames, 1) or episodes.dtype.kind not in "iu" or np.unique(episodes).size != 1:
            raise ValueError("Tracking captured clip must remain in one physical episode")
        selected_falls = fall[metric]
        if (sequence.get("fall_capture_frames") != np.flatnonzero(fall & metric).tolist() or
                sequence.get("sequence_had_fall") is not bool(selected_falls.any())):
            raise ValueError("Tracking fall evidence disagrees with its declared result")
        new_reset = np.zeros((count, 1), dtype=bool)
        new_reset[0] = True
        parts.append({"body_pos": values["rigid_body_pos"][metric], "root_rot": values["root_rot"][metric],
                      "episode_ids": np.full((count, 1), ordinal, dtype=np.int64), "reset_mask": new_reset,
                      "fall_mask": selected_falls[:, None]})
        records.append({**digest, "motion_id": sequence["motion_id"], "source_path": clip["source_path"],
            "metric_frames": count, "initial_reset_frames_discarded": 1,
            "source_reference_frame_start": int(ref_frame[1]), "source_reference_frame_stop_inclusive": int(ref_frame[-1]),
            "source_physical_episode_id": int(episodes[0, 0]), "concatenated_episode_id": ordinal,
            "fall_metric_frames_kept": int(selected_falls.sum())})
        total += count
        initial_frames += 1
        fall_frames += int(selected_falls.sum())
    if runner.get("aggregate", {}).get("observed_metric_steps") != total or runner["aggregate"].get("measured_sequences") != len(sequences):
        raise ValueError("Tracking aggregate frame/sequence count differs from declared captures")
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    capture = {"input_kind": "full_tracking_per_clip_npz", "body_names": list(BODY_NAMES),
        "metric_frames": total, "clips": len(sequences), "initial_reset_frames_discarded": initial_frames,
        "valid_same_clip_style_intervals": total-len(sequences), "fall_metric_frames_kept": fall_frames,
        "sources": records, "heldout_manifest": record(manifest_path), "heldout_split": split,
        "task": "truefutureXYoracle fulltracking; not a closed-loopdiffusionnavigation scenario",
        "reference_velocity_contract": velocity, "new_clip_boundary_before_frame_mask": True,
        "no_time_alignment_or_fall_filter": True}
    return arrays, runner, capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=physical capture or declared fulltracking directory; repeat for comparisons")
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
    from CrowdSim.humanoid.style_evaluation import BODY_NAMES, StyleSpec, assert_parent_split_disjoint, compare_speed_conditioned, sequence_style_samples
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
    reference_document = json.loads(manifest.read_text())
    assert_parent_split_disjoint(reference_document)
    gate = validate_training_manifest(manifest, verify_package_hashes=args.verify_package_hashes)
    if selection["split"] not in {"validation", "test"}:
        raise ValueError("Frozen style reference must be held-out")
    body, rotation, sample_hashes, provenance = load_reference(manifest, selection)
    if artifact and sample_hashes != artifact["actual_selected_array_sha256"]:
        raise ValueError("Actual frozen held-out samples changed")
    if artifact is None:
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
        "reference": artifact, "reference_load_provenance": provenance, "manifest_gate": gate, "runs": {}, "sources": {
            "implementation": record(Path(__file__)), "module": record(ROOT/"CrowdSim/humanoid/style_evaluation.py")}}
    for run in args.run:
        name, separator, path = run.partition("=")
        if not separator or not name or name in report["runs"]:
            raise ValueError("Run must provide a unique NAME=directory")
        directory = Path(path).resolve()
        states_path, result_path = directory/"physical_states.npz", directory/"result.json"
        tracking_input = not states_path.exists()
        if tracking_input:
            arrays, runner, capture = read_tracking_capture(directory)
            training_manifest, training_document = tracking_training_manifest(runner)
            capture_sources = {"tracking_captures": capture["sources"], "result": record(result_path)}
        else:
            arrays, runner, capture = read_capture(states_path, result_path)
            declared = Path(runner["state_capture"]["path"])
            if (ROOT/declared if not declared.is_absolute() else declared).resolve() != states_path:
                raise ValueError("Runner result identifies a different physical-state capture")
            training_manifest = Path(runner["arguments"]["manifest"])
            training_manifest = ROOT/training_manifest if not training_manifest.is_absolute() else training_manifest
            training_document = json.loads(training_manifest.read_text())
            capture_sources = {"states": record(states_path), "result": record(result_path)}
        disjoint = assert_heldout_disjoint(selection, training_document, reference_document)
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
            if tracking_input and label == "controller":
                key = "checkpoint"
            value = runner["arguments"].get(key)
            if value:
                model_path = Path(value)
                model_files[label] = record(ROOT/model_path if not model_path.is_absolute() else model_path)
                if tracking_input and model_files[label]["sha256"] != runner["arguments"].get("checkpoint_sha256"):
                    raise ValueError("Tracking controller checkpoint changed after its evaluation")
        report["runs"][name] = {"sources": {**capture_sources,
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
        if tracking_input:
            if generated["summary"]["same_episode_transition_count"] != capture["valid_same_clip_style_intervals"]:
                raise ValueError("Tracking style derivative intervals crossed a source boundary")
            report["runs"][name]["task_contract"] = {"kind": capture["task"],
                "benchmark_primary_aggregate_unchanged": runner["aggregate"],
                "reference_velocity_contract": runner["reference_velocity_contract"],
                "comparable_as_same_navigation_scene": False,
                "falls_kept_in_inclusive_style_metrics": True}
            report["runs"][name]["physical_static_peer_contacts"] = None
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
