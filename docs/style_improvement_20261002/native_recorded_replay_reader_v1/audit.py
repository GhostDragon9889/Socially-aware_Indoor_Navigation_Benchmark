"""Read an actual native recorded-pose renderer terminal result on CPU.

No Kit, Torch, optimizer, policy, or physics is imported or executed. Both
failed and successful executions are recorded without deleting earlier evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(declared):
    check(isinstance(declared, dict) and {"path", "bytes", "sha256"} <= declared.keys(), "Complete file declaration required")
    actual = record(declared["path"])
    check(actual == declared, "Changed declared source: " + declared["path"])
    return actual


def verify_png(path, frame, arguments, producer):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    check(rgb.shape == (arguments["height"], arguments["width"], 3), "PNG dimensions changed")
    check(np.array_equal(rgb, producer.stamp_recorded_frame(rgb, frame["source_time_s"])), "Permanent PNG source-time banner missing/changed")
    below = rgb[60:]
    check(below.size and below.std() > 0 and 3 < below.mean() < 245, "Blank/severely exposed pose pixels")
    return rgb


def check_frame_metadata(frame, number, index, arrays):
    check(frame["frame"] == number and frame["source_step_index"] == index and frame["source_time_s"] == (index+1)/30,
          "Frame/source post-control timestamp changed")
    check(frame["source_reset_mask"] == arrays["reset_mask"][index].tolist() and
          frame["source_episode_ids"] == arrays["episode_ids"][index].tolist(), "Frame/source episode/reset mapping changed")
    check(frame["timeline_before_s"] == frame["timeline_after_s"] == 0, "Replay timeline advanced")
    check((48 if number == 0 else 3) <= frame["app_updates"] <= (120 if number == 0 else 24), "Capture exceeded bounded update contract")


def verify(directory):
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only CUDA_VISIBLE_DEVICES required")
    directory = Path(directory).resolve()
    result_path = directory/"result.json"
    result = json.loads(result_path.read_text())
    check(result.get("status") in {"passed", "failed"}, "Only actual terminal passed/failed result admitted")
    check(result.get("mode") == "recorded_pose_visualization" and result.get("rendering_only") is True, "Not recorded visual replay")
    for key in ("closed_loop_policy_rollout", "physics_verification", "physical_humanoids_spawned", "physics_scene_created"):
        check(result.get(key) is False, "Unexpected physics/policy flag: " + key)
    for key in ("physics_steps", "policy_forward_calls"):
        check(type(result.get(key)) is int and result[key] == 0, "Unexpected execution count: " + key)
    closure = {name: checked_file(value) for name, value in result["sources"].items()}
    source_result = json.loads(Path(closure["result"]["path"]).read_text())
    producer_path = ROOT/"scripts/replay_paper_pedestrians.py"
    spec = importlib.util.spec_from_file_location("recorded_replay_producer_CPU_helpers", producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    producer.bind_source_path(closure["states"]["path"], source_result)
    with np.load(closure["states"]["path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arguments = result["arguments"]
    check(arguments.get("native_scene_visuals") is True, "Native visual scene opt-in absent")
    admission = producer.validate_arrays(arrays, source_result, start_step=arguments["start_step"], steps=arguments["steps"],
        capture_every_steps=arguments["capture_every_steps"], native_scene_visuals=True)
    admission["scene"]["recorded_source_result"] = closure["result"]
    admission["scene"]["recorded_states"] = closure["states"]
    check(json.loads(json.dumps(admission)) == result["preflight"], "Actual source admission differs")
    scene = result["native_visual_scene"]
    check(scene == result["preflight"]["scene"], "Native prepared scene differs from recorded preflight")
    layers = [checked_file(value) for value in scene["source_layers"]]
    check(len(layers) == 53, "Registered Warehouse source must retain53 exact source layers")
    check(checked_file(scene["source_USD"]) == layers[0], "Source USD/closure root differs")
    code = {}
    for path, declared in result["source_code"].items():
        actual = record(declared["path"]) if Path(declared["path"]).is_file() else None
        code[path] = {"declared_by_renderer": declared, "current_file": actual, "current_bytes_match": actual == declared}
    report = {"schema": "native_recorded_replay.actual_CPU_audit.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed_execution_recorded" if result["status"] == "failed" else "passed",
        "CPU_only_reader": True, "Kit_started_by_reader": False, "GPU_used_by_reader": False,
        "renderer_terminal_status": result["status"], "renderer_error": result.get("error"), "renderer_traceback": result.get("traceback"),
        "result": record(result_path), "source_closure": closure, "source_USD_layers_unchanged": layers,
        "source_code_current_byte_checks_not_runtime_import_proof": code,
        "source_physical_capture_outcomes": {k: source_result.get(k) for k in ("actual_control_steps", "fall_resets", "time_limit_resets", "route_vertices_reached", "minimum_height_m")},
        "zero_policy_physics_as_recorded": {"policy_forward_calls": 0, "physics_steps": 0, "new_physics_quality_validation": False},
        "rendered_frames_as_recorded": result.get("frames"), "PNG_files_observed": len(list((directory/"frames").glob("*.png"))),
        "actual_video_success_verified": False, "limits": ["Recorded poses are visualization; source physical outcomes are retained, not newly simulated.",
            "No original framebuffer/mesh-vertex equivalence, contact or new controller acceptance claim.",
            "Current code hashes do not prove all actual imports; historical mismatches are recorded, not rewritten.",
            "PNG banner is checked exactly; lossy MP4 banner uses disclosed pixel tolerances, not exact OCR of source-time digits."]}
    if result["status"] == "failed":
        check(isinstance(result.get("error"), str) and result["error"], "Failed execution lacks actual error")
        report["no_failure_promoted_to_success"] = True
        return report
    check(arguments["start_step"] == 0 and arguments["steps"] == 900 and arguments["capture_every_steps"] == 2,
          "Registered replay interval must be first900 at stride2")
    indices = admission["captured_step_indices"]
    frames = result["frame_records"]
    check(len(frames) == result["frames"] == len(indices) == 450 and admission["video_fps"] == 15., "Expected450 frames15FPS")
    check(result["timeline_time_before_s"] == result["timeline_time_after_s"] == result["simulation_time_advanced_s"] == 0,
          "Unexpected timeline/simulation advance")
    capture = result["capture_contract"]
    for key in ("orchestrator_enabled", "capture_on_play", "annotator_reference_time_gate", "timeline_auto_update", "play_simulations"):
        check(capture[key] is False, "Unexpected capture gate: " + key)
    check(capture["timeline_playing"] is True, "Actual paused clock capture requires timeline playback gate")
    composition = result["native_visual_stage"]
    check(composition["source_root_layer_occurrences"] == 1 and composition["source_session_occurrences"] == 0 and
          composition["source_is_weak_root_sublayer_not_session_sublayer"] is True, "Source layer strength/occurrence mismatch")
    check(composition["visual_floor_has_collision"] is False and composition["source_USD_saved"] is False and
          composition["physics_steps"] == composition["policy_forward_calls"] == 0, "Recorded visual physics/resource contract mismatch")
    expected = [directory/"frames"/f"frame_{number:05d}.png" for number in range(450)]
    check(sorted((directory/"frames").glob("*.png")) == expected, "450 exact ordered PNGs required")
    banners = []
    pngs = []
    samples = []
    for number, (frame, index, path) in enumerate(zip(frames, indices, expected, strict=True)):
        check_frame_metadata(frame, number, index, arrays)
        rgb = verify_png(path, frame, arguments, producer)
        pngs.append(record(path))
        banners.append(rgb[:60].copy())
        if number in (0, 225, 449):
            samples.append({"frame": number, "source_step_index": index, "source_time_s": frame["source_time_s"],
                "PNG": pngs[-1], "below_banner_mean": float(rgb[60:].mean()), "below_banner_std": float(rgb[60:].std()),
                "source_root_positions_m": arrays["root_pos"][index].tolist()})
    video = checked_file(result["video_file"])
    video_path = Path(video["path"])
    check(video_path.parent == directory, "Video is outside actual run directory")
    probe_command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,duration,nb_frames,nb_read_frames:format=duration,size", "-of", "json", str(video_path)]
    probe = json.loads(subprocess.run(probe_command, check=True, capture_output=True, text=True, timeout=45).stdout)
    stream = probe["streams"][0]
    check(int(stream["nb_frames"]) == int(stream["nb_read_frames"]) == 450 and
          float(Fraction(stream["avg_frame_rate"])) == 15. and float(stream["duration"]) == 30. and
          float(probe["format"]["duration"]) == 30. and int(probe["format"]["size"]) == video["bytes"], "Actual encoded frame/fps/duration changed")
    check(stream["width"] == arguments["width"] and stream["height"] == arguments["height"] and
          result["video_duration_at_declared_fps_s"] == 30., "Actual encoded dimensions/declaration changed")
    decode_command = ["ffmpeg", "-v", "error", "-threads", "4", "-i", str(video_path), "-vf", "crop=iw:60:0:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    decoded = subprocess.run(decode_command, check=True, capture_output=True, timeout=45).stdout
    width = arguments["width"]
    check(len(decoded) == 450*60*width*3, "All450 encoded banner crops must decode")
    decoded = np.frombuffer(decoded, dtype=np.uint8).reshape(450, 60, width, 3)
    banner_errors, white_retention = [], []
    for png, encoded in zip(banners, decoded, strict=True):
        error = float(np.abs(png.astype(np.int16)-encoded.astype(np.int16)).mean())
        white = (png.min(-1) > 240)
        check(white.sum() > 100, "PNG lacks substantial permanent text glyphs")
        retained = float((encoded.min(-1)[white] > 130).mean())
        check(error <= 15 and retained >= .9, "Encoded frame lacks corresponding visible permanent banner")
        banner_errors.append(error)
        white_retention.append(retained)
    report.update(actual_video_success_verified=True, render={"frames": 450, "fps": 15, "duration_s": 30.,
        "PNG_source_time_banners_exact": 450, "all_encoded_banner_crops_decoded": 450,
        "lossy_banner_mean_abs_RGB_error_max": max(banner_errors), "minimum_bright_glyph_retention": min(white_retention),
        "lossy_banner_gate": "meanabs RGB<=15; >=90% PNG white glyph pixels retain all RGB>130; no OCR/exact lossy timestamp digit claim",
        "first_source_time_s": frames[0]["source_time_s"], "last_source_time_s": frames[-1]["source_time_s"],
        "native_visual_stage_as_recorded": composition}, video=video, ffprobe=probe, ffprobe_command=probe_command,
        decode_command=decode_command, all_PNG_records=pngs, sampled_PNGs=samples)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Refuse overwrite of existing audit")
    try:
        report = verify(args.run)
    except Exception as error:
        report = {"schema": "native_recorded_replay.actual_CPU_audit.v1", "status": "audit_rejected", "CPU_only_reader": True,
            "run": str(args.run.resolve()), "audit_error": f"{type(error).__name__}: {error}", "no_failed_execution_promoted": True}
        exit_code = 1
    else:
        exit_code = 0
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report.get(k) for k in ("status", "renderer_terminal_status", "rendered_frames_as_recorded", "actual_video_success_verified", "audit_error")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
