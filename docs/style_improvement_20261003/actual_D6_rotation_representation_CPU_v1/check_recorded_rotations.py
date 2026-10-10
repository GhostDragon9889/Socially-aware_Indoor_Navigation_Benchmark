"""Compare recorded common DOFs with actual parent-relative body rotations.

Pure NumPy/SciPy on already closed validation captures. This does not reset,
run, or import a simulator, model, or controller. Alternative Euler conventions
are diagnostics; no convention is selected from held-out test outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULT = ROOT / "output/diffusion_amp/state_only_matched_q_tracking2000_v1/result.json"
RIG = ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"
EXPECTED = {
    RESULT: "d23d5f4b8a436e4f6cf7cfc08e7003937107fac7949beaa8330132077c4f88db",
    RIG: "58dc997c830b3ded2d2b4528797ed2768f9528b2f25aa008e541fe3a9083e7cc",
}


def record(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected an independent regular source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def summary(errors):
    values = np.concatenate(errors)
    return {"observations": int(values.size), "mean_rad": float(values.mean()),
            "max_rad": float(values.max()),
            "p50_rad": float(np.quantile(values, .5)),
            "p95_rad": float(np.quantile(values, .95))}


def main():
    sources = {}
    for path, expected in EXPECTED.items():
        sources[str(path)] = record(path)
        if sources[str(path)]["sha256"] != expected:
            raise ValueError(f"Previously sealed source changed: {path}")
    result = json.loads(RESULT.read_text())
    if result["status"] != "passed" or result["preflight"]["split"] != "validation":
        raise ValueError("Closed validation capture required")
    body_names, parents, joint_names = [], [], []

    def visit(body, parent):
        index = len(body_names)
        body_names.append(body.attrib["name"])
        parents.append(parent)
        if body.attrib.get("quat", "1 0 0 0").split() != ["1", "0", "0", "0"]:
            raise ValueError("This diagnostic requires identity body rest rotations")
        if "euler" in body.attrib or "axisangle" in body.attrib:
            raise ValueError("Unsupported body rest rotation")
        axes = body.findall("joint")
        if parent >= 0:
            for node, axis in zip(axes, "xyz", strict=True):
                expected_axis = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[axis]
                if (node.attrib["name"] != f"{body_names[-1]}_{axis}"
                        or node.attrib["type"] != "hinge" or node.attrib["axis"] != expected_axis):
                    raise ValueError("Expected common SMPL semantic XYZ joint order")
                joint_names.append(node.attrib["name"])
        for child in body.findall("body"):
            visit(child, index)

    visit(ET.parse(RIG).getroot().find("worldbody/body"), -1)
    if len(body_names) != 24 or len(joint_names) != 69:
        raise ValueError("Expected original24-body/69-DOF rig")
    if (result["physics"]["body_names"] != body_names
            or result["physics"]["dof_names"] != joint_names):
        raise ValueError("Recorded common names/order disagree with the source rig")
    alternatives = ("exp_map", "intrinsic_XYZ", "extrinsic_xyz")
    all_errors = {key: [] for key in alternatives}
    reference_errors = []
    per_clip = []
    for sequence in result["sequences"]:
        path = Path(sequence["capture"])
        entry = record(path)
        if path.stat().st_mode & 0o222 or entry["sha256"] != sequence["capture_sha256"]:
            raise ValueError("Capture must retain its actual closed/readonly identity")
        with np.load(path, allow_pickle=False) as archive:
            angles = archive["dof_pos"].astype(np.float64)
            quat = archive["rigid_body_rot"].astype(np.float64)
            ref_angles = archive["reference_dof_pos"].astype(np.float64)
            ref_quat = archive["reference_rigid_body_rot"].astype(np.float64)
        frames = len(angles)
        if angles.shape != (frames, 1, 69) or quat.shape != (frames, 1, 24, 4):
            raise ValueError("Unexpected recorded common DOF/body shape")
        if ref_angles.shape != angles.shape or ref_quat.shape != quat.shape:
            raise ValueError("Reference/common shape mismatch")
        angles, quat = angles[:, 0], quat[:, 0]
        ref_angles, ref_quat = ref_angles[:, 0], ref_quat[:, 0]
        for value in (angles, quat, ref_angles, ref_quat):
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite source")
        # SciPy takes XYZW and normalizes quaternions. Absolute error is SO(3)
        # geodesic magnitude, so q/-q and 2pi rotvec aliases are equivalent.
        rotations = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(frames, 24, 3, 3)
        local = rotations[:, np.asarray(parents[1:])].swapaxes(-1, -2) @ rotations[:, 1:]
        actual = Rotation.from_matrix(local.reshape(-1, 3, 3))
        dofs = angles.reshape(-1, 3)
        predictions = {"exp_map": Rotation.from_rotvec(dofs),
                       "intrinsic_XYZ": Rotation.from_euler("XYZ", dofs),
                       "extrinsic_xyz": Rotation.from_euler("xyz", dofs)}
        clip = {"motion_id": sequence["motion_id"], "frames_including_initial": frames,
                "tracking_success": sequence["tracking_success"], "source_capture": entry,
                "alternatives": {}}
        for key, predicted in predictions.items():
            error = (actual.inv() * predicted).magnitude().reshape(frames, 23)
            all_errors[key].append(error.ravel())
            clip["alternatives"][key] = summary([error.ravel()])
        ref_rot = Rotation.from_quat(ref_quat.reshape(-1, 4)).as_matrix().reshape(frames, 24, 3, 3)
        ref_local = ref_rot[:, np.asarray(parents[1:])].swapaxes(-1, -2) @ ref_rot[:, 1:]
        ref_error = (Rotation.from_matrix(ref_local.reshape(-1, 3, 3)).inv()
                     * Rotation.from_rotvec(ref_angles.reshape(-1, 3))).magnitude()
        reference_errors.append(ref_error)
        clip["reference_exp_map"] = summary([ref_error])
        per_clip.append(clip)
    if len(per_clip) != 22:
        raise ValueError("All22 validation captures required")
    report = {"schema": "actual_D6.recorded_common_rotation_diagnostic.v1",
              "status": "completed", "source_records": sources,
              "body_names": body_names, "body_parent_indices": parents,
              "joint_names": joint_names, "actual_alternatives": {
                  key: summary(value) for key, value in all_errors.items()},
              "reference_exp_map": summary(reference_errors), "clips": per_clip,
              "all_failed_clips_and_initial_frames_retained": True,
              "scope": "actual common DOFs vs same-frame body rotations from already recorded IsaacLab validation; no new simulator query",
              "rest_frame_assumption": "identity body rotations declared in source MJCF",
              "CPU_only": True, "model_deserialized": False, "SDK_started": False,
              "sealed34_outputs_read": False, "rotation_convention_changed": False,
              "runtime_or_quality_equivalence_claimed": False}
    if any(name == "torch" or name.startswith(("isaacsim", "omni", "isaaclab")) for name in sys.modules):
        raise RuntimeError("Diagnostic imported a heavy runtime")
    with (HERE / "actual_rotation_comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "clips": len(per_clip),
                      "actual_alternatives": report["actual_alternatives"],
                      "reference_exp_map": report["reference_exp_map"]}, allow_nan=False))


if __name__ == "__main__":
    main()
