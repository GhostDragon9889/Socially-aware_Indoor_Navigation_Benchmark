# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording and rendering logic extracted from the Simulator base class.

This module contains the RecordingMixin class which provides video recording,
frame capture, marker management, and motion/object serialization functionality.
The mixin is designed to be used with the Simulator class, accessing simulator
state and methods via self.
"""

from collections import deque
from dataclasses import replace
from datetime import datetime
import logging
import math
import os
from typing import Dict, Optional

import torch

from protomotions.utils import rotations
from protomotions.simulator.base_simulator.config import MarkerState
from protomotions.simulator.base_simulator.utils import build_motion_data
from protomotions.utils.checkpoint import atomic_torch_save

log = logging.getLogger(__name__)


class RecordingMixin:
    """Mixin providing recording and rendering capabilities for simulators.

    This mixin expects the following attributes/methods to be provided by the
    host class (Simulator):
        - self.headless, self.config, self.scene_lib, self.num_envs
        - self._num_dof, self._proj_config
        - self._original_marker_configs
        - self.get_robot_state(), self.get_object_root_state()
        - self._get_projectile_positions_rotations()
        - self._write_viewport_to_file(file_name)
        - self._update_simulator_markers(markers_state)
    """

    # -------------------------
    # Initialization
    # -------------------------

    def _recording_fps(self) -> float:
        """Frame cadence [Hz]: render records one frame per control step."""
        dt = getattr(self, "dt", None)
        if dt is None:
            sim = getattr(self.config, "sim", None)
            if sim is not None:
                dt = float(sim.decimation) / float(sim.fps)
        # Lightweight legacy recording hosts may not expose a control clock.
        if dt is None:
            return 30.0
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("Recording requires a positive, finite control timestep")
        return 1.0 / float(dt)

    def _init_recording_state(self) -> None:
        """Initialize all recording-related attributes."""
        self._camera_target: Dict[str, int] = {"env": 0, "element": 0}
        self._show_markers: bool = True

        self._user_is_recording, self._user_recording_state_change = False, False
        self._user_recording_video_queue_size = 100000
        self._delete_user_viewer_recordings = False
        os.makedirs("output/renderings", exist_ok=True)
        self._user_recording_video_path = os.path.join(
            "output/renderings", f"{self.config.experiment_name}-%s"
        )

        # Last markers state for recording (set each step)
        self._last_markers_state: Optional[Dict[str, MarkerState]] = None

    # -------------------------
    # Recording state control
    # -------------------------

    def _toggle_video_record(self):
        self._user_is_recording = not self._user_is_recording
        self._user_recording_state_change = True

    def _cancel_video_record(self):
        self._user_is_recording = False
        self._user_recording_state_change = False
        self._delete_user_viewer_recordings = True

    # -------------------------
    # Camera target
    # -------------------------

    def _toggle_camera_target(self) -> None:
        """
        Toggle the camera target between different environments and objects.

        The target cycles through all objects in the scene, with 0 referring to the environment.
        """
        if self.scene_lib.num_objects_per_scene > 0:
            self._camera_target["element"] = (self._camera_target["element"] + 1) % (
                self.scene_lib.num_objects_per_scene + 1
            )
            print("Updated camera target to element", self._camera_target["element"])

        if self._camera_target["element"] == 0:
            self._camera_target["env"] = (
                self._camera_target["env"] + 1
            ) % self.num_envs
            print("Updated camera target to env", self._camera_target["env"])

    # -------------------------
    # Marker management
    # -------------------------

    def _toggle_markers(self):
        self._show_markers = not self._show_markers
        print(f"Markers are now {'visible' if self._show_markers else 'hidden'}")

    def _update_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Update visualization markers for the simulator.

        Converts marker orientations if necessary and delegates to the simulator-specific update.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """

        if not markers_state or len(markers_state) == 0:
            return

        # Keep canonical targets intact for controllers and recording. Native
        # conversion and viewer visibility must not mutate the caller's state.
        native_markers = {}
        for name, marker in markers_state.items():
            orientation = marker.orientation
            if orientation is not None:
                orientation = (orientation.clone() if self.config.w_last
                               else rotations.xyzw_to_wxyz(orientation))
            translation = marker.translation
            if translation is not None:
                translation = (translation.clone() if self._show_markers
                               else torch.full_like(translation, -1000000))
            native_markers[name] = replace(marker, translation=translation, orientation=orientation)
        self._update_simulator_markers(native_markers)

    def _build_markers_save_data(self) -> dict:
        """Build markers data dictionary for saving to .markers.pt file."""
        markers_data = {"fps": self._recording_fps(), "markers": {}}
        for name, frame_list in self._recorded_markers.items():
            translations = torch.stack([f[0] for f in frame_list], dim=0)
            orientations = torch.stack([f[1] for f in frame_list], dim=0)
            # Get marker config metadata from the original (pre-simulator)
            # configs, since simulator-specific init may wrap/replace them
            marker_config = self._original_marker_configs.get(name)
            marker_type = "sphere"
            color = (1.0, 0.0, 0.0)
            sizes = []
            if marker_config is not None:
                marker_type = marker_config.type
                color = marker_config.color
                sizes = [m.size for m in marker_config.markers]

            markers_data["markers"][name] = {
                "type": marker_type,
                "color": color,
                "sizes": sizes,
                "translation": translations,
                "orientation": orientations,
            }
        return markers_data

    # -------------------------
    # Object serialization
    # -------------------------

    def _build_terrain_save_data(self) -> Optional[dict]:
        """Build terrain data dictionary for saving to .terrain.pt file.

        Returns None if terrain is flat or not available.
        """
        terrain = getattr(self, "terrain", None)
        if terrain is None:
            return None

        # Skip saving for flat terrains — Blender's default ground plane suffices
        if terrain.is_flat():
            return None

        return {
            "height_field_raw": terrain.height_field_raw,
            "horizontal_scale": terrain.horizontal_scale,
            "vertical_scale": terrain.vertical_scale,
        }

    def _build_objects_save_data(self) -> dict:
        """Build objects data dictionary for saving to .objects.pt file."""
        objects_list = []

        # Scene objects
        if self._recorded_objects:
            from protomotions.components.scene_lib import (
                BoxSceneObject,
                SphereSceneObject,
                CylinderSceneObject,
                MeshSceneObject,
            )

            translations = torch.stack([f[0] for f in self._recorded_objects], dim=0)
            rotations = torch.stack([f[1] for f in self._recorded_objects], dim=0)

            eid = self._recording_env_id
            scene_idx = self.scene_lib._scene_to_original_scene_id[eid].item()
            scene = self.scene_lib._original_scenes[scene_idx]

            for obj_idx, obj in enumerate(scene.objects):
                obj_info = {
                    "name": f"object_{obj_idx}",
                    "translation": translations[:, obj_idx, :],
                    "rotation": rotations[:, obj_idx, :],
                }
                if isinstance(obj, BoxSceneObject):
                    obj_info["shape"] = "box"
                    obj_info["size"] = [obj.width, obj.depth, obj.height]
                elif isinstance(obj, SphereSceneObject):
                    obj_info["shape"] = "sphere"
                    obj_info["size"] = [obj.radius]
                elif isinstance(obj, CylinderSceneObject):
                    obj_info["shape"] = "cylinder"
                    obj_info["size"] = [obj.radius, obj.height]
                elif isinstance(obj, MeshSceneObject):
                    obj_info["shape"] = "mesh"
                    obj_info["size"] = []
                    obj_info["mesh_path"] = obj.object_path
                    obj_info["scale"] = list(obj.scale)
                else:
                    obj_info["shape"] = "box"
                    dims = obj.object_dims
                    if dims is not None:
                        obj_info["size"] = [
                            dims[1] - dims[0],
                            dims[3] - dims[2],
                            dims[5] - dims[4],
                        ]
                    else:
                        obj_info["size"] = [0.1, 0.1, 0.1]

                objects_list.append(obj_info)

        # Projectiles
        if self._recorded_projectiles:
            proj_pos = torch.stack(
                [f[0] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 3]
            proj_rot = torch.stack(
                [f[1] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 4]

            half_sizes = self._proj_config.get_sizes()
            hide_z = self._proj_config.hide_z

            for p in range(proj_pos.shape[1]):
                # Only include projectiles that were visible at some point
                if (proj_pos[:, p, 2] > hide_z + 0.5).any():
                    hs = half_sizes[p]
                    full_size = hs * 2
                    objects_list.append(
                        {
                            "name": f"projectile_{p}",
                            "shape": "box",
                            "size": [full_size, full_size, full_size],
                            "translation": proj_pos[:, p, :],
                            "rotation": proj_rot[:, p, :],
                        }
                    )

        return {"fps": self._recording_fps(), "objects": objects_list}

    # -------------------------
    # Main render loop
    # -------------------------

    def _clear_recording_buffers(self):
        self._recorded_motion = None
        self._recorded_markers = None
        self._recorded_objects = None
        self._recorded_projectiles = None

    def _cleanup_user_recording(self):
        folder = getattr(self, "_curr_user_recording_name", None)
        if folder is not None and os.path.isdir(folder):
            for image in os.listdir(folder):
                if image.endswith(".png"):
                    os.remove(os.path.join(folder, image))
            if not os.listdir(folder):
                os.rmdir(folder)
        self._delete_user_viewer_recordings = False
        self._clear_recording_buffers()

    def _finalize_user_recording(self):
        folder = getattr(self, "_curr_user_recording_name", None)
        motion = getattr(self, "_recorded_motion", None)
        if folder is None or not motion or not motion.get("gts"):
            self._delete_user_viewer_recordings = True
            return

        # Preserve physical motion even if the encoder or viewport capture fails.
        motion_data = build_motion_data(motion, fps=self._recording_fps(), num_dof=self._num_dof)
        atomic_torch_save(motion_data, f"{folder}.motion")
        print(f"Motion saved to {folder}.motion")
        images = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".png")]
        images.sort(key=lambda path: (
            not os.path.splitext(os.path.basename(path))[0].isdigit(),
            int(os.path.splitext(os.path.basename(path))[0]) if os.path.splitext(os.path.basename(path))[0].isdigit() else 0,
            path,
        ))
        if not images:
            raise RuntimeError(f"No viewport PNG frames were captured; motion preserved at {folder}.motion")
        try:
            from moviepy import ImageSequenceClip
        except ImportError:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(images, fps=self._recording_fps())
        try:
            clip.write_videofile(
                f"{folder}.mp4", codec="libx264", audio=False, threads=2,
                preset="veryfast", ffmpeg_params=["-profile:v", "main", "-level", "4.0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "23",
                    "-x264-params", "keyint=60:min-keyint=30"],
            )
        finally:
            clip.close()
        print(f"Video saved to {folder}.mp4")
        try:
            if self._recorded_markers:
                atomic_torch_save(self._build_markers_save_data(), f"{folder}.markers.pt")
            if self._recorded_objects or self._recorded_projectiles:
                atomic_torch_save(self._build_objects_save_data(), f"{folder}.objects.pt")
            terrain = self._build_terrain_save_data()
            if terrain is not None:
                atomic_torch_save(terrain, f"{folder}.terrain.pt")
        except Exception as exc:
            # Retain source frames and buffers for a recoverable export failure.
            print(f"Warning: failed to save markers/objects/terrain: {exc}")
            return
        self._delete_user_viewer_recordings = True
        self._clear_recording_buffers()

    def render(self):
        """
        Render the current simulation state and handle video recording if enabled.

        This method manages:
        1. Video recording state transitions and initialization
        2. Frame capture and saving during recording
        3. Video compilation when recording ends
        4. Cleanup of temporary image files
        """
        if not self.headless:
            # Finish a cancelled session before a pending start can replace its
            # folder/buffers. Cancel + start may both arrive between two renders.
            if self._delete_user_viewer_recordings:
                self._cleanup_user_recording()
            # Handle recording state transitions
            if self._user_recording_state_change:
                if self._user_is_recording:
                    # A stop and a new start can arrive between two renders.
                    # Publish the old session before replacing its buffers.
                    if getattr(self, "_recorded_motion", None):
                        self._finalize_user_recording()
                        if not self._delete_user_viewer_recordings:
                            raise RuntimeError("Recording export is incomplete; previous frames and buffers retained")
                        self._cleanup_user_recording()
                    # Initialize new recording
                    self._user_recording_video_queue = deque(
                        maxlen=self._user_recording_video_queue_size
                    )
                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                    self._curr_user_recording_name = (
                        self._user_recording_video_path % curr_date_time
                    )
                    self._user_recording_frame = 0

                    self._recorded_motion = {
                        "gts": [],  # rigid_body_pos (global translations)
                        "grs": [],  # rigid_body_rot (global rotations)
                        "gvs": [],  # rigid_body_vel (global velocities)
                        "gavs": [],  # rigid_body_ang_vel (global angular velocities)
                        "dps": [],  # dof_pos
                        "dvs": [],  # dof_vel
                        "contacts": [],  # rigid_body_contacts
                    }
                    self._recorded_markers = {}
                    self._recorded_objects = []
                    self._recorded_projectiles = []
                    self._recording_env_id = self._camera_target["env"]

                    if not os.path.exists(self._curr_user_recording_name):
                        os.makedirs(self._curr_user_recording_name)
                    print(
                        f"Started recording to folder {self._curr_user_recording_name}"
                    )
                else:
                    self._finalize_user_recording()

                self._user_recording_state_change = False

            # Capture frame if recording
            if self._user_is_recording:
                file_name = (
                    self._curr_user_recording_name
                    + "/%04d.png" % self._user_recording_frame
                )
                self._write_viewport_to_file(file_name)
                self._user_recording_frame += 1

                eid = self._recording_env_id

                # Record motion (single env only)
                robot_state = self.get_robot_state()
                self._recorded_motion["gts"].append(
                    robot_state.rigid_body_pos[eid].cpu().clone()
                )
                self._recorded_motion["grs"].append(
                    robot_state.rigid_body_rot[eid].cpu().clone()
                )
                if robot_state.rigid_body_vel is not None:
                    self._recorded_motion["gvs"].append(
                        robot_state.rigid_body_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_ang_vel is not None:
                    self._recorded_motion["gavs"].append(
                        robot_state.rigid_body_ang_vel[eid].cpu().clone()
                    )
                if robot_state.dof_pos is not None:
                    self._recorded_motion["dps"].append(
                        robot_state.dof_pos[eid].cpu().clone()
                    )
                if robot_state.dof_vel is not None:
                    self._recorded_motion["dvs"].append(
                        robot_state.dof_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_contacts is not None:
                    self._recorded_motion["contacts"].append(
                        robot_state.rigid_body_contacts[eid].cpu().clone()
                    )

                # Record markers (single env only, skip terrain markers)
                if self._last_markers_state:
                    for name, ms in self._last_markers_state.items():
                        if name == "terrain_markers":
                            continue
                        if name not in self._recorded_markers:
                            self._recorded_markers[name] = []
                        self._recorded_markers[name].append(
                            (
                                ms.translation[eid].cpu().clone(),
                                ms.orientation[eid].cpu().clone(),
                            )
                        )

                # Record objects (single env only)
                if (
                    self.scene_lib is not None
                    and self.scene_lib.num_objects_per_scene > 0
                ):
                    obj_state = self.get_object_root_state()
                    self._recorded_objects.append(
                        (
                            obj_state.root_pos[eid].cpu().clone(),
                            obj_state.root_rot[eid].cpu().clone(),
                        )
                    )

                # Record projectiles (single env only)
                if (
                    self._proj_config is not None
                    and self._proj_config.num_projectiles > 0
                ):
                    pos, rot = self._get_projectile_positions_rotations()
                    self._recorded_projectiles.append(
                        (
                            pos[eid].cpu().clone(),
                            rot[eid].cpu().clone(),
                        )
                    )

            # Clean up temporary files if needed
            if self._delete_user_viewer_recordings:
                self._cleanup_user_recording()
