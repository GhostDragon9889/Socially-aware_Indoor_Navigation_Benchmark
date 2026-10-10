"""CPU-only audit of saved native velocities and a real curated AMASS bank."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def stats(value):
    value = np.asarray(value, dtype=np.float64)
    return {"rms_rad_s": float(np.sqrt(np.mean(value*value))),
            "mean_abs_rad_s": float(np.abs(value).mean()), "max_abs_rad_s": float(np.abs(value).max())}


def inverse_rotate(quaternion, velocity):
    xyz, w = quaternion[..., :3], quaternion[..., 3:4]
    return velocity*(2*w*w-1)-2*w*np.cross(xyz, velocity)+2*xyz*np.sum(xyz*velocity, axis=-1, keepdims=True)


def main():
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from protomotions.components.pose_lib import compute_angular_velocity
    from protomotions.utils.rotations import quaternion_to_matrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--usd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite historical velocity evidence")
    torch.set_num_threads(4)
    result = json.loads(args.result.read_text())
    state = np.load(args.capture, allow_pickle=False)
    names = result["physics"]["body_names"]
    dofs = result["physics"]["dof_names"]
    parents = {}

    def descend(body, parent=None):
        name = body.attrib["name"]
        parents[name] = parent
        for child in body.findall("body"):
            descend(child, name)

    mjcf = ROOT/"protomotions/data/assets/mjcf/smpl_humanoid.xml"
    for root in ET.parse(mjcf).getroot().findall("worldbody/body"):
        descend(root)
    if names[0] != "Pelvis" or dofs != [f"{name}_{axis}" for name in names[1:] for axis in "xyz"]:
        raise ValueError("Capture is not the declared common SMPL24/69 ordering")
    parent_indices = [names.index(parents[name]) for name in names[1:]]
    q = state["rigid_body_rot"].astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    angular = state["rigid_body_ang_vel"].astype(np.float64)
    native = state["dof_vel"].astype(np.float64)
    delta_world = angular[..., 1:, :]-angular[..., parent_indices, :]
    child = inverse_rotate(q[..., 1:, :], delta_world).reshape(native.shape)
    parent = inverse_rotate(q[..., parent_indices, :], delta_world).reshape(native.shape)
    selected = [index for index, name in enumerate(dofs)
        if name.rsplit("_", 1)[0] not in {"L_Toe", "R_Toe", "L_Hand", "R_Hand"}]
    mask = state["metric_frame"] & ~state["native_reset_frame"]
    live = {"frames": int(mask.sum()), "selected_dofs": len(selected),
        "sameinstant_native_minus_child": stats((native-child)[mask][..., selected]),
        "sameinstant_native_minus_parent": stats((native-parent)[mask][..., selected]),
        "native_velocity": stats(native[mask][..., selected]),
        "method": "Same captured instant, world angular child-parent difference rotated into each body frame; no time differencing"}
    usd = None
    if args.usd:
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(args.usd))
        joints = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                joints.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                    "localRot0_wxyz": str(joint.GetLocalRot0Attr().Get()),
                    "localRot1_wxyz": str(joint.GetLocalRot1Attr().Get())})
        identity = all(joint["localRot0_wxyz"] == "(1, 0, 0, 0)" and
                       joint["localRot1_wxyz"] == "(1, 0, 0, 0)" for joint in joints)
        if len(joints) != 23 or not identity:
            raise ValueError("Probe frame formula requires the actual23identity joint frames")
        usd = {"root_layer": file_record(args.usd), "active_joints": joints,
               "all_local_joint_frames_identity": identity, "standalone_usd_only_kit_not_started": True}
    legacy = AMASSCorpus(args.manifest)
    converted = AMASSCorpus(args.manifest, dof_velocity_frame="child_joint")
    bank_rows = []
    for clip in legacy.entry["clips"][:32]:
        motion_id = clip["motion_id"]
        count = int(legacy.data["motion_num_frames"][motion_id])
        start = int(legacy.data["length_starts"][motion_id])
        indices = torch.arange(start, start+count)
        local = legacy.data["lrs"][indices, 1:].clone()
        parent_delta = compute_angular_velocity(quaternion_to_matrix(local, w_last=True), 30, 1).reshape(count, 69)
        rotations = Rotation.from_quat(local.numpy().reshape(-1, 4))
        child_delta = rotations.inv().apply(parent_delta.numpy().reshape(-1, 3)).reshape(count, 69)
        stored = legacy.state_at(indices).dof_vel.numpy()
        actual = converted.state_at(indices).dof_vel.numpy()
        bank_rows.append({"motion_id": motion_id, "source_path": clip["source_path"], "frames": count,
            "stored_minus_parent_delta": stats((stored-parent_delta.numpy())[:, selected]),
            "stored_minus_child_delta": stats((stored-child_delta)[:, selected]),
            "adapter_minus_independent_child_delta": stats((actual-child_delta)[:, selected])})
    initial = Rotation.from_rotvec([.4, .6, -.2])
    velocity = np.array([.3, -.7, 1.1])
    future = initial*Rotation.from_rotvec(velocity/30)
    child_analytic = (initial.inv()*future).as_rotvec()*30
    parent_analytic = (future*initial.inv()).as_rotvec()*30
    report = {"schema": "navisaaclab.dof_velocity_coordinate_audit.v1", "status": "passed",
        "CPU_only": True, "kit_started_by_audit": False, "torch_cuda_initialized": torch.cuda.is_initialized(),
        "live_probe": live, "usd": usd, "bank_sample_clips": bank_rows,
        "bank_velocity_provenance": {"legacy": legacy.provenance(), "child_joint": converted.provenance()},
        "bank_full_sha_recomputed_by_this_audit": False,
        "analytic_example": {"initial_rotvec": [.4, .6, -.2], "native_child_velocity": velocity.tolist(),
            "child_delta": child_analytic.tolist(), "parent_delta": parent_analytic.tolist(),
            "expmap_component_derivative": ((future.as_rotvec()-initial.as_rotvec())*30).tolist()},
        "sources": {"capture": file_record(args.capture), "result": file_record(args.result),
            "mjcf": file_record(mjcf), "manifest": file_record(args.manifest),
            "implementation": file_record(ROOT/"CrowdSim/humanoid/amass_corpus.py"),
            "audit": file_record(__file__)}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "native": live,
        "bank_clips": len(bank_rows), "cuda_initialized": report["torch_cuda_initialized"]}))


if __name__ == "__main__":
    main()
