"""Independent historical source/clock audit; reuse prior complete media checks."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v6/native_pedestrian_source_v6.json"
MEDIA = ROOT / "docs/style_improvement_20261002/native_recorded_replay_reader_v1/actual_replay30_v2.json"
READER = ROOT / "docs/style_improvement_20261002/native_recorded_replay_reader_v1/source_freeze.json"
CHECKS = []


def require(value, message):
    if not value:
        raise ValueError(message)
    CHECKS.append(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(declared):
    actual = record(declared["path"])
    require(actual == declared, "actual file SHA/size " + actual["path"])
    return actual


def verify(run):
    result = json.loads((run/"result.json").read_text())
    media = json.loads(MEDIA.read_text())
    require(media["result"] == record(run/"result.json") and media["actual_video_success_verified"] is True,
            "completed prior all450 PNG/encoded-banner audit bound to actual result bytes")
    reader = json.loads(READER.read_text())
    audit_row = next(v for v in reader["files"] if v["path"].endswith("/audit.py"))
    require(all(record(ROOT/audit_row["path"])[k] == audit_row[k] for k in ("bytes", "sha256")), "prior media reader frozen identity")
    require(result["mode"] == "recorded_pose_visualization" and result["status"] == "passed" and result["rendering_only"] is True,
            "actual recorded visualization execution, not fresh physics rollout")
    require(all(result[k] is False for k in ["closed_loop_policy_rollout", "physics_verification", "physical_humanoids_spawned", "physics_scene_created"]), "no physics/controller quality promotion")
    require(result["physics_steps"] == result["policy_forward_calls"] == result["simulation_time_advanced_s"] == 0,
            "recorded zero physics/policy/advancing time")
    closure = {k: checked(v) for k,v in result["sources"].items()}
    require(closure == media["source_closure"], "prior media/source closure identical")
    source = json.loads(Path(closure["result"]["path"]).read_text())
    with np.load(closure["states"]["path"], allow_pickle=False) as a:
        reset, episodes = a["reset_mask"], a["episode_ids"]
    indices = np.arange(0, 900, 2)
    require(len(result["frame_records"]) == result["frames"] == len(indices) == 450
            and result["preflight"]["captured_step_indices"] == indices.tolist(), "all450 exact source indices0..898 stride2")
    for number,(frame,index) in enumerate(zip(result["frame_records"], indices, strict=True)):
        require(frame["frame"] == number and frame["source_step_index"] == int(index)
                and frame["source_time_s"] == (index+1)/30
                and frame["source_reset_mask"] == reset[index].tolist()
                and frame["source_episode_ids"] == episodes[index].tolist(), "source timestamp/reset/episode frame " + str(number))
        require(frame["timeline_before_s"] == frame["timeline_after_s"] == 0, "actual replay frozen timeline frame " + str(number))
    scene = result["native_visual_stage"]
    require(scene["source_root_layer_occurrences"] == 1 and scene["source_session_occurrences"] == 0
            and scene["source_is_weak_root_sublayer_not_session_sublayer"] is True
            and scene["source_USD_saved"] is False and scene["visual_floor_has_collision"] is False, "unique weak root-source composition, no source saves or physical floor")
    layers = [checked(v) for v in result["native_visual_scene"]["source_layers"]]
    require(len(layers) == 53 and layers == media["source_USD_layers_unchanged"], "53 original USD layer bytes retained")
    v6 = json.loads(V6.read_text())
    files = {v["path"]:v for v in v6["files"]}
    require(len(files) == v6["source_count"] == 171, "actual historical V6 source171 scope")
    archive = ROOT/v6["archive"]["path"]
    require(all(record(archive)[k] == v6["archive"][k] for k in ("bytes", "sha256")), "historical V6 archive identity")
    imports = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == 171 and {m.name for m in members} == set(files)
                and all(m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members), "V6 bounded171 regular source members")
        for member in members:
            raw = tar.extractfile(member).read()
            require(len(raw) == files[member.name]["bytes"] and hashlib.sha256(raw).hexdigest() == files[member.name]["sha256"], "historical member SHA " + member.name)
        for name,declared in result["source_code"].items():
            require(all(declared[k] == files[name][k] for k in ("bytes", "sha256")), "actual five renderer declared modules match historical V6 " + name)
            imports[name] = {"actual_renderer_declared":declared,"historical_V6_archive_member":files[name],
                             "current_file_informational":record(declared["path"]),"not_full_import_mapping":True}
    video = checked(result["video_file"])
    require(video == media["video"], "actual MP4 bound to existing full encoded-banner audit")
    command = ["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
               "stream=width,height,avg_frame_rate,duration,nb_frames,nb_read_frames:format=size,duration","-of","json",video["path"]]
    probe = json.loads(subprocess.run(command,check=True,capture_output=True,text=True,timeout=45).stdout)
    stream = probe["streams"][0]
    require(int(stream["nb_read_frames"]) == int(stream["nb_frames"]) == 450
            and float(Fraction(stream["avg_frame_rate"])) == 15 and float(stream["duration"]) == 30
            and stream["width"] == 1280 and stream["height"] == 960, "actual independent encoded framecount/duration/FPS/dimensions")
    sample = next(v for v in media["sampled_PNGs"] if v["frame"] == 225)
    checked(sample["PNG"])
    return {"schema":"native_recorded_replay.independent_historical_epoch_and_clock.v1",
            "status":"source_clock_media_contracts_passed_visual_delivery_rejected_roof_occlusion",
            "CPU_only":True,"SDK_or_GPU_started":False,"production_or_actual_artifacts_modified":False,
            "checks_passed":len(CHECKS),"checks":CHECKS,"actual_result":record(run/"result.json"),
            "historical_V6_manifest":record(V6),"historical_V6_archive":record(archive),"declared_render_modules":imports,
            "media_all450_audit_reused_not_redecoded":record(MEDIA),"media_reader_freeze":record(READER),
            "video":video,"independent_ffprobe":probe,"ffprobe_command":command,
            "rendered_frames":450,"fps":15,"duration_s":30,"timeline_s_before_after":[0,0],
            "physics_steps":0,"policy_forward_calls":0,
            "source_physical_outcomes_retained_not_newly_simulated":{k:source[k] for k in ["actual_control_steps","fall_resets","time_limit_resets","route_vertices_reached"]},
            "manual_visual_observation":{"frame":225,"source_step_index":450,"source_time_s":15.033333333333333,
                "PNG":sample["PNG"],"observed":"The exterior overhead view is dominated by the Warehouse roof; source pedestrian poses are occluded.",
                "pedestrian_visible_style_delivery_accepted":False,"all450frames_manually_inspected":False},
            "limits":["Integrity of a recorded replay does not prove new physical stability, policy inference, natural style or visible delivery.",
                      "Historical whole V6 source and five declared renderer modules are checked; no full runtime import ledger was saved.",
                      "An indoor fresh output must receive separate artifact and visibility checks; this exterior failure remains retained."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),"new independent report only")
    report=verify(args.run.resolve())
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":report["status"],"checks_passed":report["checks_passed"],"rendered_frames":450}))


if __name__=="__main__":
    main()
