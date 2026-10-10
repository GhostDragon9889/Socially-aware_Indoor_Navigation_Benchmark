"""Memory mapped AMASS references for new pedestrian training.

The manifest is a reproducible reconstruction, not the unpublished 2,848 clip
list. File-name action labels and explicit upright/velocity filters are used;
we do not silently classify unlabeled numeric CMU sequences as locomotion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import torch

from .application.resources import resolve_local_path


SCHEMA = "navisaaclab.amass_reconstruction.v1"
ACTION_PATTERN = r"walk|run|jog|turn|sidestep|side.?step"
EXCLUDE_PATTERN = r"jump|crawl|climb|stair|beam|handrail|cartwheel|somersault"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_manifest(sources, output):
    """Select labeled upright locomotion, retaining existing source splits."""
    manifest = {"schema": SCHEMA, "author_clip_list": False, "seed": 7,
                "selection": {"action_regex": ACTION_PATTERN,
                              "exclude_regex": EXCLUDE_PATTERN,
                              "minimum_frames": 32, "control_hz": 30,
                              "upright_fraction_minimum": .95,
                              "root_height_range_m": [.7, 1.2],
                              "root_up_z_minimum": .8,
                              "planar_speed_median_range_m_s": [.05, 4.],
                              "diffusion_window": "31 past/current plus 52 future at 10Hz",
                              "expert_actions": "next-frame mocap joint position mapped to normalized PD targets; no measured expert torques"},
                "splits": {}}
    for split, source in sources.items():
        path = Path(source).resolve()
        data = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        records = []
        for index, filename in enumerate(data["motion_files"]):
            name = Path(filename).name
            if not re.search(ACTION_PATTERN, name, re.I) or re.search(EXCLUDE_PATTERN, name, re.I):
                continue
            count = int(data["motion_num_frames"][index])
            dt = float(data["motion_dt"][index])
            if count < 32 or abs(dt - 1/30) > 1e-5:
                continue
            start = int(data["length_starts"][index])
            sample = torch.arange(start, start + count, max(1, count // 128))
            position = data["gts"][sample, 0]
            rotation = data["grs"][sample, 0]
            velocity = data["gvs"][sample, 0]
            if not all(torch.isfinite(value).all() for value in (position, rotation, velocity)):
                continue
            up_z = 1 - 2 * rotation[:, :2].square().sum(-1)
            upright = ((position[:, 2] >= .7) & (position[:, 2] <= 1.2) & (up_z >= .8)).float().mean()
            speed = velocity[:, :2].norm(dim=-1).median()
            if upright < .95 or not .05 <= speed <= 4:
                continue
            records.append({"motion_id": index, "source_path": filename,
                            "frames": count, "dt": dt, "upright_fraction": float(upright),
                            "median_speed_m_s": float(speed),
                            "diffusion_eligible": count >= 247})
        manifest["splits"][split] = {"path": str(path), "bytes": path.stat().st_size,
                                     "sha256": file_sha256(path), "clips": records,
                                     "clip_count": len(records),
                                     "diffusion_clip_count": sum(r["diffusion_eligible"] for r in records)}
        del data
    if not manifest["splits"].get("train", {}).get("clip_count"):
        raise ValueError("No labeled locomotion training clips passed the declared filters")
    names = [{r["source_path"] for r in entry["clips"]} for entry in manifest["splits"].values()]
    if any(left & right for i, left in enumerate(names) for right in names[i+1:]):
        raise ValueError("Training and held-out manifests share motion files")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class AMASSCorpus:
    """Gather small reference batches on CPU; never load the 9.5GB bank on GPU."""

    def __init__(self, manifest, split="train", *, device="cpu", seed=7,
                 dof_velocity_frame="legacy_stored"):
        if not isinstance(dof_velocity_frame, str) or dof_velocity_frame not in {"legacy_stored", "child_joint"}:
            raise ValueError("dof_velocity_frame must be legacy_stored or child_joint")
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unknown AMASS reconstruction manifest")
        self.split = split
        self.entry = self.manifest["splits"][split]
        path = resolve_local_path(self.entry["path"], relative_to=self.manifest_path.parent)
        if path.stat().st_size != self.entry["bytes"]:
            raise ValueError("AMASS source size changed since manifest creation")
        self.data = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        declarations = {}
        for label, source in (("entry", self.entry), ("bank", self.data)):
            if "dof_velocity_frame" in source:
                value = source["dof_velocity_frame"]
                if not isinstance(value, str) or value not in {"parent_joint", "child_joint"}:
                    raise ValueError(f"Unknown {label} stored DOF velocity frame")
                declarations[label] = value
        if len(set(declarations.values())) > 1:
            raise ValueError("AMASS entry/bank stored DOF velocity frames disagree")
        self.dof_velocity_frame = dof_velocity_frame
        self.stored_dof_velocity_frame = next(iter(declarations.values()), "parent_joint")
        self.output_dof_velocity_frame = (
            self.stored_dof_velocity_frame if dof_velocity_frame == "legacy_stored" else "child_joint")
        self._velocity_declarations = declarations
        self._convert_dof_velocity = dof_velocity_frame == "child_joint" and self.stored_dof_velocity_frame == "parent_joint"
        if self._convert_dof_velocity:
            local = self.data.get("lrs")
            if not isinstance(local, torch.Tensor) or local.shape != (len(self.data["dvs"]), 24, 4):
                raise ValueError("Parent-to-child DOF velocity conversion requires actual local XYZW rotations lrs[N,24,4]")
        self.ids = torch.tensor([clip["motion_id"] for clip in self.entry["clips"]])
        self.diffusion_ids = torch.tensor([clip["motion_id"] for clip in self.entry["clips"] if clip["diffusion_eligible"]], dtype=torch.long)
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(seed)

    def _indices(self, batch, *, past=9, future=1, diffusion=False):
        eligible = self.diffusion_ids if diffusion else self.ids
        if not len(eligible):
            raise ValueError(f"No eligible {self.split} clips for this window")
        ids = eligible[torch.randint(len(eligible), (batch,), generator=self.generator)]
        count = self.data["motion_num_frames"][ids]
        room = count - past - future
        if torch.any(room <= 0):
            raise ValueError("Reference window would cross a clip boundary")
        frame = past + (torch.rand(batch, generator=self.generator) * room).long()
        return self.data["length_starts"][ids] + frame, ids, frame

    def state_at(self, indices):
        fields = {name: self.data[key][indices].to(self.device) for name, key in (
            ("rigid_body_pos", "gts"), ("rigid_body_rot", "grs"),
            ("rigid_body_vel", "gvs"), ("rigid_body_ang_vel", "gavs"),
            ("dof_pos", "dps"), ("dof_vel", "dvs"))}
        if self._convert_dof_velocity:
            velocity = fields["dof_vel"]
            if velocity.shape[-1:] != (69,) or not velocity.is_floating_point() or not torch.isfinite(velocity).all():
                raise ValueError("Parent-to-child DOF velocity conversion requires finite floating-point69D velocities")
            local = self.data["lrs"][indices][..., 1:, :].to(device=self.device, dtype=velocity.dtype)
            norm = local.norm(dim=-1, keepdim=True)
            if (not torch.isfinite(local).all() or
                    not torch.allclose(norm, torch.ones_like(norm), atol=1e-3, rtol=0)):
                raise ValueError("Parent-to-child DOF velocity conversion requires unit finite local XYZW rotations")
            local = local/norm
            vector = velocity.reshape(*velocity.shape[:-1], 23, 3)
            xyz, w = local[..., :3], local[..., 3:4]
            # q_t^-1 omega_parent equals Log(q_t^-1 q_next)/dt for
            # the stored Log(q_next q_t^-1)/dt. No time differencing,
            # exp-map derivative, or world-frame substitution is introduced.
            fields["dof_vel"] = (vector*(2*w.square()-1) - 2*w*torch.cross(xyz, vector, dim=-1)
                + 2*xyz*(xyz*vector).sum(-1, keepdim=True)).reshape_as(velocity)
        fields.update(root_pos=fields["rigid_body_pos"][..., 0, :],
                      root_rot=fields["rigid_body_rot"][..., 0, :],
                      root_vel=fields["rigid_body_vel"][..., 0, :],
                      root_ang_vel=fields["rigid_body_ang_vel"][..., 0, :])
        return SimpleNamespace(**fields)

    def sample_reference(self, batch, history_frames=10, *, future_frames=1):
        if type(future_frames) is not int or future_frames < 1:
            raise ValueError("Reference future_frames must be a positive integer")
        indices, ids, frame = self._indices(batch, past=history_frames-1, future=future_frames)
        history_indices = indices[:, None] - torch.arange(history_frames)
        return self.state_at(indices), self.state_at(history_indices.reshape(-1)), self.data["dps"][indices+1].to(self.device), ids, frame

    def planar_at(self, indices):
        pos = self.data["gts"][indices, 0, :2].to(self.device)
        vel = self.data["gvs"][indices, 0, :2].to(self.device)
        q = self.data["grs"][indices, 0].to(self.device)
        yaw = torch.atan2(2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1]), 1-2*(q[..., 1].square()+q[..., 2].square()))
        return torch.cat((pos, vel, yaw[..., None]), -1)

    def diffusion_batch(self, batch):
        indices, ids, frame = self._indices(batch, past=90, future=156, diffusion=True)
        history = self.planar_at(indices[:, None] + torch.arange(-90, 1, 3))
        future = self.planar_at(indices[:, None] + torch.arange(3, 157, 3))
        return {"states": history[:, -1], "history": history, "future": future,
                "motion_ids": ids, "frame_indices": frame}

    def provenance(self):
        return {"manifest": str(self.manifest_path), "manifest_sha256": file_sha256(self.manifest_path),
                "split": self.split, "clips": len(self.ids), "diffusion_clips": len(self.diffusion_ids),
                "source_sha256": self.entry["sha256"], "author_clip_list": False,
                "dof_velocity": {
                    "requested_mode": self.dof_velocity_frame,
                    "stored_frame": self.stored_dof_velocity_frame,
                    "output_frame": self.output_dof_velocity_frame,
                    "storage_declarations": self._velocity_declarations,
                    "stored_frame_assumed": not bool(self._velocity_declarations),
                    "undeclared_storage_assumption": (
                        "parent_joint, matching current ProtoMotions AMASS converter; not a guarantee for arbitrary external banks"
                        if not self._velocity_declarations else None),
                    "conversion": "current-frame local XYZW inverse rotation; no derivative" if self._convert_dof_velocity else "none",
                }}
