#!/usr/bin/env python3
"""CPU verification of an existing recorded-pose render; never launches Kit."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.replay_paper_pedestrians import bind_source_path, file_record, stamp_recorded_frame, validate_arrays  # noqa: E402


def checked_file(declared):
    actual = file_record(declared["path"])
    for field in ("sha256", "bytes"):
        if field in declared and actual[field] != declared[field]:
            raise ValueError(f"Changed {field}: {declared['path']}")
    return actual


def verify(directory):
    if not __debug__:
        raise RuntimeError("This verification uses admission assertions; run Python without -O")
    directory = Path(directory).resolve()
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "passed" and result["mode"] == "recorded_pose_visualization"
    assert result["rendering_only"] is True
    for flag in ("closed_loop_policy_rollout", "physics_verification", "physical_humanoids_spawned", "physics_scene_created"):
        assert result[flag] is False
    assert result["physics_steps"] == result["policy_forward_calls"] == result["simulation_time_advanced_s"] == 0
    contract = result["capture_contract"]
    for flag in ("orchestrator_enabled", "capture_on_play", "annotator_reference_time_gate", "timeline_auto_update", "play_simulations"):
        assert contract[flag] is False
    assert contract["timeline_playing"] is True
    closure = {name: checked_file(record) for name, record in result["sources"].items()}
    closure["render_result"] = file_record(result_path)
    source = json.loads(Path(closure["result"]["path"]).read_text())
    bind_source_path(closure["states"]["path"], source)
    with np.load(closure["states"]["path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arguments = result["arguments"]
    admission = validate_arrays(arrays, source, start_step=arguments["start_step"], steps=arguments["steps"],
                                capture_every_steps=arguments["capture_every_steps"])
    assert admission == result["preflight"]
    indices = admission["captured_step_indices"]
    frames = result["frame_records"]
    assert len(frames) == result["frames"] == len(indices)
    before, after = result["timeline_time_before_s"], result["timeline_time_after_s"]
    assert before == after == 0
    for number, (index, frame) in enumerate(zip(indices, frames, strict=True)):
        assert frame["frame"] == number and frame["source_step_index"] == index
        assert frame["source_time_s"] == (index + 1) / 30
        assert frame["source_reset_mask"] == arrays["reset_mask"][index].tolist()
        assert frame["source_episode_ids"] == arrays["episode_ids"][index].tolist()
        assert frame["timeline_before_s"] == frame["timeline_after_s"] == before
        assert (48 if number == 0 else 3) <= frame["app_updates"] <= (120 if number == 0 else 24)
        assert 3 <= frame["rgb_mean"] <= 245 and frame["rgb_max"] > 0
    expected_pngs = [directory / "frames" / f"frame_{number:05d}.png" for number in range(len(frames))]
    assert sorted((directory / "frames").glob("*.png")) == expected_pngs
    closure["video"] = checked_file(result["video_file"])
    probe_cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames:format=duration,size",
        "-of", "json", closure["video"]["path"]]
    probe = json.loads(subprocess.run(probe_cmd, check=True, capture_output=True, text=True).stdout)
    stream = probe["streams"][0]
    assert int(stream["nb_frames"]) == int(stream["nb_read_frames"]) == len(frames)
    assert stream["width"] == arguments["width"] and stream["height"] == arguments["height"]
    assert float(Fraction(stream["avg_frame_rate"])) == admission["video_fps"]
    assert float(stream["duration"]) == len(frames) / admission["video_fps"] == result["video_duration_at_declared_fps_s"]
    assert int(probe["format"]["size"]) == closure["video"]["bytes"]
    samples, pictures = [], []
    for number in (0, len(frames) // 2, len(frames) - 1):
        path = expected_pngs[number]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        assert rgb.shape == (arguments["height"], arguments["width"], 3)
        # Re-stamping uses the frozen public banner function and makes the
        # existing expected text/background identical without touching pose pixels.
        assert np.array_equal(stamp_recorded_frame(rgb, frames[number]["source_time_s"]), rgb)
        pose_pixels = rgb[60:]
        assert 3 < pose_pixels.mean() < 245 and pose_pixels.std() > 0 and pose_pixels.min() != pose_pixels.max()
        pictures.append(rgb)
        index = indices[number]
        samples.append({"frame": number, "source_step_index": index, "source_time_s": frames[number]["source_time_s"],
            "png": file_record(path), "pixel_mean": float(rgb.mean()), "pixel_std": float(rgb.std()),
            "pixel_min": int(rgb.min()), "pixel_max": int(rgb.max()), "below_banner_mean": float(pose_pixels.mean()),
            "below_banner_std": float(pose_pixels.std()), "banner_matches_source_time": True,
            "source_root_positions_m": arrays["root_pos"][index].tolist()})
    image_changes = []
    for earlier, later, left, right in zip(samples[:-1], samples[1:], pictures[:-1], pictures[1:], strict=True):
        difference = np.abs(left[60:].astype(np.int16) - right[60:].astype(np.int16))
        assert np.any(difference)
        image_changes.append({"from_frame": earlier["frame"], "to_frame": later["frame"],
            "changed_pixels_below_banner": int(np.count_nonzero(np.any(difference > 0, axis=-1))),
            "mean_absolute_rgb_change_below_banner": float(difference.mean())})
    closure["source_code"] = {name: checked_file(record) for name, record in result["source_code"].items()}
    closure["neutral_model"] = checked_file(result["licensed_neutral_model"])
    closure["uv"] = checked_file(result["visual_resources"]["uv"])
    closure["textures"] = [checked_file(record) for record in result["visual_resources"]["textures"]]
    assert len(result["visual_resources"]["actor_texture_paths"]) == admission["actors"]
    assert set(result["visual_resources"]["actor_texture_paths"]) <= {item["path"] for item in closure["textures"]}
    # Resolve the evaluation arguments, rather than assuming inherited training
    # provenance identifies the planner/controller used for this source capture.
    model_records = {}
    for name, key in (("diffusion", "diffusion_checkpoint"), ("amp_controller", "resume")):
        path = Path(source["arguments"][key])
        model_records[name] = file_record(ROOT / path if not path.is_absolute() else path)
    return {"schema": 1, "status": "passed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_device": "CPU", "kit_started_by_verifier": False, "rendering_only": True,
        "closed_loop_policy_rollout": False, "physics_verification": False,
        "source_closure": closure, "source_models_resolved_from_evaluation_arguments": model_records,
        "source_capture": {"status": source["status"], "actors": admission["actors"], "control_frames": admission["frames"],
            "frequency_hz": 30, "fall_resets": source["fall_resets"], "minimum_height_m": source["minimum_height_m"],
            "route_vertices_reached": source["route_vertices_reached"], "author_experiment_reproduced": source["author_experiment_reproduced"],
            "convergence_claim": source["convergence_claim"]},
        "render": {"frames": len(frames), "video_fps": admission["video_fps"], "width": arguments["width"],
            "height": arguments["height"], "video_duration_s": float(stream["duration"]),
            "first_source_index": indices[0], "last_source_index": indices[-1], "source_stride": arguments["capture_every_steps"],
            "first_source_time_s": frames[0]["source_time_s"], "last_source_time_s": frames[-1]["source_time_s"],
            "all_indices_times_episode_reset_flags_match_source": True, "all_timeline_records_zero": all(f["timeline_after_s"] == 0 for f in frames),
            "simulation_time_advanced_s": 0, "physics_steps": 0, "policy_forward_calls": 0,
            "app_update_histogram": dict(Counter(f["app_updates"] for f in frames)),
            "raw_rgb_mean_min_max": [min(f["rgb_mean"] for f in frames), max(f["rgb_mean"] for f in frames)],
            "mesh_mode": result["mesh_mode"], "elapsed_wall_s": result["elapsed_wall_s"]},
        "ffprobe_command": probe_cmd, "ffprobe": probe, "sample_png_verification": samples,
        "image_changes_below_banner": image_changes, "runtime_manifest": result["runtime"],
        "capture_contract": result["capture_contract"],
        "limitations": ["RGBA/projection/occlusion are rendered anew; no original mesh-vertex or framebuffer bit-equivalence claim",
            "Source physics metrics remain the source rollout evidence; replay adds no fresh policy/contact/fall verification",
            "Author scene/checkpoint/2848-clip protocol is not reproduced by this shared fixture"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.render_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "frames": result["render"]["frames"], "duration_s": result["render"]["video_duration_s"],
                      "video_sha256": result["source_closure"]["video"]["sha256"]}))


if __name__ == "__main__":
    main()
