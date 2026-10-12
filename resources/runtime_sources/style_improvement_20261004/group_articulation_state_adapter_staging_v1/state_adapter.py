"""Lazy four-Articulation link snapshots and named COMMON position/reset writes."""
from __future__ import annotations

import ast
import __future__
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping

import torch

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
LAB = Path("lab:")
PINS = {
    "bridge": (ROOT / "resources/runtime_sources/style_improvement_20261004/group_articulation_tensor_bridge_staging_v1/tensor_bridge.py", 6035, '2f5dc721f739d26ace02cc0831864523e605587f7e960f0fd32b82326ec8c9ec'),
    "state": (ROOT / "protomotions/simulator/base_simulator/simulator_state.py", 34638, '31cf14fab319151602f43fcf6ad2751b71088cda3a3f4feefadfa61262102731'),
    "rotations": (ROOT / "protomotions/utils/rotations.py", 28136, 'ef1475d33d6f941ccdea4a612477483db4fad8236ba7786fbe8bf91a65c1cef8'),
    "resolver": (ROOT / "protomotions/simulator/isaaclab/utils/actuator_groups.py", 5693, '1de8f8483a407128d3bddf5df713919dbf62dad5513c55746ca9b5a2f752af3a'),
    "SDK_writes": (LAB / "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py", 224858, "2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87"),
    "SDK_data": (LAB / "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py", 95901, "6579e5b3efcd13d113cda141945cd055a4d0b86a069f0a80fc2c73c4cfe4163b"),
}
_DEPS = None
BODY_FIELDS = {"rigid_body_pos": ("body_link_pos_w", (24, 3)), "rigid_body_rot": ("body_link_quat_w", (24, 4)),
               "rigid_body_vel": ("body_link_lin_vel_w", (24, 3)), "rigid_body_ang_vel": ("body_link_ang_vel_w", (24, 3)),
               "dof_pos": ("joint_pos", (69,)), "dof_vel": ("joint_vel", (69,))}
ROOT_FIELDS = {"root_pos": ("root_link_pos_w", (3,)), "root_rot": ("root_link_quat_w", (4,)),
               "root_vel": ("root_link_lin_vel_w", (3,)), "root_ang_vel": ("root_link_ang_vel_w", (3,))}
WRITERS = ("write_root_link_pose_to_sim_index", "write_root_link_velocity_to_sim_index",
           "write_joint_state_to_sim_index", "set_joint_position_target_index")


def _dependencies():
    global _DEPS
    if _DEPS is None:
        sources = {}
        for key, (path, size, sha) in PINS.items():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (key == "bridge" and info.st_mode & 0o222):
                raise ValueError("Invalid source file: " + key)
            source = path.read_bytes()
            if len(source) != size or hashlib.sha256(source).hexdigest() != sha:
                raise ValueError("Pinned source identity mismatch: " + key)
            sources[key] = source
        bridge = ModuleType("_frozen_actor_row_bridge")
        exec(compile(sources["bridge"], str(PINS["bridge"][0]), "exec"), bridge.__dict__)
        rotations = importlib.import_module("protomotions.utils.rotations")
        states = importlib.import_module("protomotions.simulator.base_simulator.simulator_state")
        for key, module in (("state", states), ("rotations", rotations)):
            if Path(module.__file__).resolve() != PINS[key][0].resolve():
                raise ValueError("Imported dependency origin mismatch: " + key)
        # Keep the existing resolver class/function AST unchanged; its other imports load MuJoCo.
        tree = ast.parse(sources["resolver"], filename=str(PINS["resolver"][0]))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ("IsaacLabJointNameMap", "build_isaaclab_joint_name_map")]
        if len(nodes) != 2:
            raise ValueError("Expected original resolver class and builder")
        resolver = ModuleType("_original_isaaclab_joint_name_resolver")
        resolver.__dict__.update(dataclass=dataclass, Mapping=Mapping)
        sys.modules[resolver.__name__] = resolver  # dataclass annotation lookup only.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PINS["resolver"][0]), "exec",
                     flags=__future__.annotations.compiler_flag), resolver.__dict__)
        _DEPS = bridge, states, resolver.build_isaaclab_joint_name_map
    return _DEPS


def _names(value, count, name):
    names = tuple(value)
    if len(names) != count or any(not isinstance(item, str) or not item for item in names) or len(set(names)) != count:
        raise ValueError(name + " requires complete unique names")
    return names


@dataclass(frozen=True)
class CommonSnapshot:
    robot_state: object  # COMMON RobotState: mapped bodies/DOFs, root properties are Pelvis views.
    root_state: object  # COMMON RootOnlyState: independent native root-link snapshot.


class FourArticulationAdapter:
    def __init__(self, assets, group_actor_rows, kinematic_info, *, device):
        bridge, states, resolver = _dependencies()
        self.device = bridge._device(device)
        self.rows = bridge._partition(group_actor_rows, self.device).clone()
        self.groups, self.count = len(self.rows), self.rows.numel()
        if not isinstance(assets, (list, tuple)) or len(assets) != 4:
            raise ValueError("Four slot-ordered actual assets or minimal mocks required")
        self.assets = tuple(assets)
        bodies = _names(kinematic_info.body_names, 24, "SMPL common bodies")
        dofs = _names(kinematic_info.dof_names, 69, "SMPL common DOFs")
        if kinematic_info.num_bodies != 24 or bodies[0] != "Pelvis":
            raise ValueError("SMPL common root must be Pelvis and body count24")
        counts = [len(kinematic_info.hinge_axes_map.get(index, ())) for index in range(24)]
        if counts != [0]+[3]*23 or not set(kinematic_info.hinge_axes_map).issubset(range(24)):
            raise ValueError("SMPL kinematic hinge triples must cover exactly69 DOFs")
        joint_names = resolver(kinematic_info)
        self.mappings = []
        for asset in self.assets:
            if asset.num_instances != self.groups or bridge._device(asset.device) != self.device:
                raise ValueError("Asset group count/device mismatch")
            native_bodies = _names(asset.body_names, 24, "Native body names")
            native_dofs = _names(asset.joint_names, 69, "Native joint names")
            try:
                semantic = tuple(joint_names.backend_to_semantic[name] for name in native_dofs)
            except KeyError as error:
                raise ValueError("Unexpected native joint name: " + str(error)) from error
            if set(native_bodies) != set(bodies) or set(semantic) != set(dofs):
                raise ValueError("Body/semantic DOF names must exactly match the supplied kinematics")
            if any(not callable(getattr(asset, name, None)) for name in WRITERS):
                raise ValueError("Asset is missing a required explicit index writer")
            def index(values):
                return torch.tensor(values, dtype=torch.int64, device=self.device)
            self.mappings.append(states.DataConversionMapping(
                body_convert_to_common=index([native_bodies.index(name) for name in bodies]),
                body_convert_to_sim=index([bodies.index(name) for name in native_bodies]),
                dof_convert_to_common=index([semantic.index(name) for name in dofs]),
                dof_convert_to_sim=index([dofs.index(name) for name in semantic]), sim_w_last=True))
        self.mappings = tuple(self.mappings)

    def _tensor(self, value, shape, name):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != self.device or value.dtype != torch.float32:
            raise ValueError(name + " requires float32 on-device Tensor" + str(shape))
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + " must be finite")
        if "rot" in name or "quat" in name:
            if not torch.allclose(value.norm(dim=-1), torch.ones(shape[:-1], dtype=value.dtype, device=self.device), atol=1e-3, rtol=0):
                raise ValueError(name + " requires unit XYZW quaternions")
        return value

    def _snapshot_field(self, data, property_name, tail):
        proxy = getattr(data, property_name)  # Fresh property access triggers current-timestamp getter.
        view = proxy if isinstance(proxy, torch.Tensor) else proxy.torch
        if not isinstance(view, torch.Tensor):
            raise ValueError("Snapshot property must expose a Torch tensor")
        value = view.clone()  # Own the zero-copy view before any subsequent refresh/step.
        return self._tensor(value, (self.groups, *tail), property_name)

    def snapshot(self):
        """Read current link snapshots; no update/step. Caller owns coherent physics timestamp."""
        bridge, states, _ = _dependencies()
        body_states, root_states = [], []
        for asset, mapping in zip(self.assets, self.mappings):
            data = asset.data
            body = {key: self._snapshot_field(data, prop, tail) for key, (prop, tail) in BODY_FIELDS.items()}
            root = {key: self._snapshot_field(data, prop, tail) for key, (prop, tail) in ROOT_FIELDS.items()}
            body_states.append(states.RobotState(state_conversion=states.StateConversion.SIMULATOR, **body).convert_to_common(mapping))
            root_states.append(states.RootOnlyState(state_conversion=states.StateConversion.SIMULATOR, **root).convert_to_common(mapping))
        def gather(states_, key):
            return bridge.assets_to_canonical(tuple(getattr(state, key) for state in states_), self.rows, device=self.device)
        robot = states.RobotState(state_conversion=states.StateConversion.COMMON, **{key: gather(body_states, key) for key in BODY_FIELDS})
        root = states.RootOnlyState(state_conversion=states.StateConversion.COMMON, **{key: gather(root_states, key) for key in ROOT_FIELDS})
        return CommonSnapshot(robot, root)

    def set_common_position_targets(self, common_pd_targets):
        """Buffer COMMON PD position targets; not raw policy actions; no flush or step."""
        bridge, _, _ = _dependencies()
        value = self._tensor(common_pd_targets, (self.count, 69), "COMMON position targets")
        slots = bridge.actions_to_assets(value, self.rows, device=self.device)
        prepared = tuple(slot.index_select(1, mapping.dof_convert_to_sim).detach() for slot, mapping in zip(slots, self.mappings))
        groups = torch.arange(self.groups, dtype=torch.int64, device=self.device)
        for asset, target in zip(self.assets, prepared):
            asset.set_joint_position_target_index(target=target, joint_ids=None, env_ids=groups.clone(), full_data=False)

    def write_common_reset(self, common_reset_state, canonical_rows):
        """Prepare all four partial resets, then write. SDK failures may leave partial writes."""
        bridge, states, _ = _dependencies()
        if not isinstance(common_reset_state, states.ResetState) or common_reset_state.state_conversion != states.StateConversion.COMMON:
            raise ValueError("COMMON ResetState required")
        group_ids = bridge.reset_group_env_ids(canonical_rows, self.rows, device=self.device)
        fields = {**{key: tail for key, (_, tail) in ROOT_FIELDS.items()}, "dof_pos": (69,), "dof_vel": (69,)}
        values = {key: self._tensor(getattr(common_reset_state, key), (self.count, *tail), key) for key, tail in fields.items()}
        if not len(group_ids[0]):
            return group_ids
        slots = {key: bridge.canonical_to_assets(value, self.rows, device=self.device) for key, value in values.items()}
        prepared = []
        for slot, mapping in enumerate(self.mappings):
            reset = states.ResetState(state_conversion=states.StateConversion.COMMON,
                                      **{key: value[slot].index_select(0, group_ids[slot]).detach() for key, value in slots.items()}).convert_to_sim(mapping)
            prepared.append((torch.cat((reset.root_pos, reset.root_rot), -1), torch.cat((reset.root_vel, reset.root_ang_vel), -1), reset.dof_pos, reset.dof_vel))
        for asset, ids, (pose, velocity, position, joint_velocity) in zip(self.assets, group_ids, prepared):
            asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids.clone(), full_data=False)
            asset.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=ids.clone(), full_data=False)
            asset.write_joint_state_to_sim_index(position=position, velocity=joint_velocity, joint_ids=None, env_ids=ids.clone(), full_data=False)
            asset.set_joint_position_target_index(target=position.clone(), joint_ids=None, env_ids=ids.clone(), full_data=False)
        return group_ids
