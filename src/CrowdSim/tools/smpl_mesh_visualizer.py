"""Visual-only SMPL mesh overlay for ProtoMotions IsaacLab humanoids."""

from __future__ import annotations

import inspect
import colorsys
import os
import pickle
import pickletools
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from protomotions.utils.rotations import (
    exp_map_to_quat,
    matrix_to_quaternion,
    quat_to_exp_map,
    quaternion_to_matrix,
)


SMPL_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

MJCF_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


def _parse_optional_int_env(name: str) -> int | None:
    """Read an optional non-negative int from an environment variable."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        v = int(raw)
        return v if v >= 0 else None
    except ValueError:
        return None


@dataclass
class HumanMeshConfig:
    model_dir: str
    num_envs: int
    device: str
    prim_root: str = "/World/CrowdSim/HumanMesh"
    color: tuple[float, float, float] = (0.12, 0.20, 0.28)
    opacity: float = 1.0
    hide_humanoid: bool = False
    texture_path: str | None = None
    texture_paths: tuple[str, ...] = ()
    appearance_seed: int = 42
    shape_std: float = 0.55
    shape_clip: float = 1.25
    # Number of active (navigating) humanoids.  Only these are hidden when
    # hide_humanoid is true; frozen filler slots (env_id >= num_active_humanoids)
    # stay visible so the depth camera can still see them as obstacles.
    num_active_humanoids: int | None = None
    vary_solid_colors: bool = True


class ProtoMotionsHumanMeshAdapter:
    """Adapter between ProtoMotions IsaacLab state and the SMPL mesh overlay."""

    def __init__(self, simulator, visualizer: "SMPLMeshVisualizer") -> None:
        self.simulator = simulator
        self.visualizer = visualizer
        self.pose_mapper = SMPLRobotPoseMapper()

    @classmethod
    def from_simulator(cls, simulator) -> "ProtoMotionsHumanMeshAdapter":
        from CrowdSim.scene_setup import resolve_repo_path
        model_dir = os.environ.get(
            "CROWDSIM_SMPL_MODEL_DIR", str(resolve_repo_path("data/smpl"))
        )
        texture_path = os.environ.get("CROWDSIM_SMPL_TEXTURE_PATH")
        if not texture_path:
            for name in ("smpl_body_texture.png", "smpl_uv_20200910.png"):
                candidate = Path(model_dir) / name
                if candidate.exists():
                    texture_path = str(candidate)
                    break
        texture_dir = Path(
            os.environ.get("CROWDSIM_SMPL_TEXTURE_DIR", model_dir)
        ).expanduser()
        texture_paths: list[str] = []
        if texture_path:
            texture_paths.append(str(Path(texture_path).expanduser()))
        if texture_dir.is_dir():
            supported_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            texture_paths.extend(
                str(path)
                for path in sorted(texture_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in supported_suffixes
            )
        texture_paths = list(dict.fromkeys(texture_paths))
        cfg = HumanMeshConfig(
            model_dir=model_dir,
            num_envs=int(getattr(simulator, "num_envs", 1)),
            device=str(getattr(simulator, "device", "cuda:0")),
            hide_humanoid=os.environ.get("CROWDSIM_HIDE_HUMANOID", "0") == "1",
            texture_path=texture_path,
            texture_paths=tuple(texture_paths),
            appearance_seed=int(os.environ.get("CROWDSIM_HUMAN_APPEARANCE_SEED", "42")),
            shape_std=float(os.environ.get("CROWDSIM_SMPL_SHAPE_STD", "0.55")),
            shape_clip=float(os.environ.get("CROWDSIM_SMPL_SHAPE_CLIP", "1.25")),
            num_active_humanoids=_parse_optional_int_env("CROWDSIM_NUM_ACTIVE_HUMANOIDS"),
        )
        return cls(simulator=simulator, visualizer=SMPLMeshVisualizer.from_config(cfg))

    def create(self) -> None:
        # Keep the physical skeleton visible until real, finite mesh geometry
        # has been written; creating zero-valued USD points alone is insufficient.
        try:
            self.visualizer.create()
            self.update()
            if self.visualizer.cfg.hide_humanoid:
                self._set_humanoid_visibility(visible=False)
        except Exception:
            self.disable()
            raise

    def update(self) -> None:
        from CrowdSim.control.drive import convert_quaternion_order

        try:
            robot = self.simulator._robot
            # ProxyArray.shape is the Warp shape and omits vec/quat components.
            # Read its explicit, zero-copy Torch view before shape validation.
            def torch_view(value):
                return value if isinstance(value, torch.Tensor) else getattr(value, "torch", value)

            joint_names = getattr(robot, "joint_names", None)
            if joint_names is None:
                joint_names = robot.data.joint_names
            body_pose = self.pose_mapper(torch_view(robot.data.joint_pos), list(joint_names))
            native_order = getattr(self.simulator, "native_quaternion_order", "xyzw")
            root_quat_wxyz = convert_quaternion_order(
                torch_view(robot.data.root_quat_w), native_order, "wxyz"
            )
            self.visualizer.update(
                body_pose=body_pose,
                root_pos=torch_view(robot.data.root_pos_w),
                root_quat_wxyz=root_quat_wxyz,
            )
        except Exception:
            self.disable()
            raise

    def disable(self) -> None:
        """Restore the physical skeleton if a mesh update cannot be completed."""
        self._set_humanoid_visibility(visible=True)
        if self.visualizer.stage is not None:
            from pxr import UsdGeom

            prim = self.visualizer.stage.GetPrimAtPath(self.visualizer.cfg.prim_root)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()

    def _set_humanoid_visibility(self, *, visible: bool) -> None:
        from pxr import UsdGeom

        stage = self.visualizer.stage
        if stage is None:
            return

        # Only hide ACTIVE humanoid slots.  Frozen filler slots (env_id >=
        # num_active_humanoids) must stay visible so the active robot's depth
        # camera can still perceive them as obstacles — otherwise the policy
        # has no observation signal for a physical body it can collide with.
        n_active = self.visualizer.cfg.num_active_humanoids
        if n_active is None:
            n_active = self.visualizer.cfg.num_envs
        for env_id in range(min(n_active, self.visualizer.cfg.num_envs)):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
            if prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeVisible() if visible else imageable.MakeInvisible()


class SMPLRobotPoseMapper:
    """Convert ProtoMotions SMPL robot exp-map DOFs into SMPL body_pose."""

    def __init__(self) -> None:
        self.smpl_to_robot = smpl_to_robot_matrix()
        self.body_names = [name for name in SMPL_JOINT_NAMES if name != "Pelvis"]

    def __call__(self, joint_pos: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be [N, D], got {tuple(joint_pos.shape)}")
        expected_dofs = 3 * len(self.body_names)
        if joint_pos.shape[1] != expected_dofs or len(joint_names) != expected_dofs:
            raise ValueError(
                f"Expected exactly {expected_dofs} named SMPL DOFs, got "
                f"{joint_pos.shape[1]} values and {len(joint_names)} names"
            )
        if not all(isinstance(name, str) for name in joint_names):
            raise ValueError("SMPL joint names must be strings")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("SMPL joint names must be unique")
        # PhysX orders articulation DOFs by its hierarchy, rather than MJCF's
        # depth-first order. Guessing the order cross-wires legs and arms.
        return self._map_by_name(joint_pos, joint_names)

    def _map_by_name(
        self, joint_pos: torch.Tensor, joint_names: Sequence[str]
    ) -> torch.Tensor:
        rotvecs: list[torch.Tensor] = []
        for body_name in self.body_names:
            indices = self._body_triplet_indices(body_name, joint_names)
            robot_expmap = joint_pos[:, indices]
            rotvecs.append(robot_expmap_to_smpl_rotvec(robot_expmap, self.smpl_to_robot))
        return torch.cat(rotvecs, dim=-1).contiguous()

    @staticmethod
    def _body_triplet_indices(
        body_name: str, joint_names: Sequence[str]
    ) -> list[int]:
        # Matches build_isaaclab_joint_name_map: three-axis D6 DOFs are named
        # after the first semantic hinge, followed by :0/:1/:2 for x/y/z.
        semantic = tuple(f"{body_name}_{axis}" for axis in "xyz")
        d6_backend = tuple(f"{body_name}_x:{axis}" for axis in range(3))
        actual = {name for name in joint_names if name.startswith(f"{body_name}_")}
        for supported in (semantic, d6_backend):
            if actual == set(supported):
                indices = {name: index for index, name in enumerate(joint_names)}
                return [indices[name] for name in supported]
        raise ValueError(
            f"Cannot map SMPL body {body_name}: expected {semantic} or "
            f"IsaacLab D6 {d6_backend}, got {sorted(actual)}. "
            "Joint order is never inferred."
        )


class SMPLMeshVisualizer:
    """Create and update batched UsdGeom.Mesh prims from SMPL vertices."""

    def __init__(
        self,
        cfg: HumanMeshConfig,
        smpl_model,
        faces: np.ndarray,
        uvs: np.ndarray | None = None,
        uv_faces: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.smpl_model = smpl_model.to(self.device).eval()
        self.faces = np.asarray(faces, dtype=np.int64)
        try:
            self.uvs = np.asarray(uvs, dtype=np.float32) if uvs is not None else None
            uv_indices = np.asarray(uv_faces) if uv_faces is not None else None
            self.uv_faces = (
                uv_indices.astype(np.int64)
                if uv_indices is not None and np.issubdtype(uv_indices.dtype, np.integer)
                else None
            )
        except (TypeError, ValueError):
            self.uvs = None
            self.uv_faces = None
        self.texture_paths = tuple(
            path
            for path in (
                cfg.texture_paths
                or ((cfg.texture_path,) if cfg.texture_path else ())
            )
            if Path(path).is_file()
        )
        self.has_texture = self._valid_uv_data() and bool(self.texture_paths)
        if not self.has_texture:
            self.uvs = None
            self.uv_faces = None
        rng = np.random.default_rng(int(cfg.appearance_seed))
        base_color = np.asarray(cfg.color, dtype=np.float64)
        if base_color.shape != (3,) or not np.isfinite(base_color).all() or np.any((base_color < 0) | (base_color > 1)):
            raise ValueError("SMPL solid color must contain three finite values in [0,1]")
        hue, saturation, brightness = colorsys.rgb_to_hsv(*base_color)
        self.person_colors = tuple(
            colorsys.hsv_to_rgb((hue + env_id * 0.61803398875) % 1.0, saturation, brightness)
            if cfg.vary_solid_colors else tuple(base_color)
            for env_id in range(cfg.num_envs)
        )
        if self.texture_paths:
            # Shuffle without replacement within each cycle, then reshuffle if
            # there are more people than textures. This is deterministic for a
            # given appearance_seed while avoiding sorted/modulo assignment.
            assigned: list[str] = []
            while len(assigned) < cfg.num_envs:
                order = rng.permutation(len(self.texture_paths))
                assigned.extend(self.texture_paths[int(index)] for index in order)
            self.person_texture_paths = tuple(assigned[:cfg.num_envs])
        else:
            self.person_texture_paths = ()
        shape_scale = np.asarray(
            [1.0, 0.85, 0.85, 0.7, 0.7, 0.55, 0.55, 0.45, 0.45, 0.4],
            dtype=np.float32,
        )
        betas = rng.normal(
            0.0, float(cfg.shape_std), size=(cfg.num_envs, 10)
        ).astype(np.float32)
        betas *= shape_scale[None]
        self.betas = torch.as_tensor(
            np.clip(betas, -float(cfg.shape_clip), float(cfg.shape_clip)),
            device=self.device,
        )
        self.shape_floor_offsets = self._compute_shape_floor_offsets()
        self.stage = None
        self._points_attrs: list = []
        self._created = False

    def _valid_uv_data(self) -> bool:
        if self.uvs is None or self.uv_faces is None:
            return False
        return bool(
            self.uvs.ndim == 2
            and self.uvs.shape[1] == 2
            and np.isfinite(self.uvs).all()
            and self.uv_faces.shape == self.faces.shape
            and self.uv_faces.size > 0
            and self.uv_faces.min() >= 0
            and self.uv_faces.max() < len(self.uvs)
        )

    @torch.no_grad()
    def _compute_shape_floor_offsets(self) -> torch.Tensor:
        """Match each shaped mesh's rest-pose floor to neutral SMPL."""
        dtype = self.betas.dtype
        zeros_orient = torch.zeros(
            (self.cfg.num_envs, 3), device=self.device, dtype=dtype
        )
        zeros_pose = torch.zeros(
            (self.cfg.num_envs, 3 * (len(SMPL_JOINT_NAMES) - 1)),
            device=self.device,
            dtype=dtype,
        )
        common = {
            "global_orient": zeros_orient,
            "body_pose": zeros_pose,
            "transl": zeros_orient,
            "return_verts": True,
        }
        shaped = self.smpl_model(betas=self.betas, **common)
        neutral = self.smpl_model(
            betas=torch.zeros_like(self.betas), **common
        )
        transform = smpl_to_robot_matrix(self.device, dtype).T

        def local_floor(output) -> torch.Tensor:
            centered = output.vertices - output.joints[:, 0:1, :]
            robot_vertices = torch.matmul(centered, transform)
            return robot_vertices[..., 2].amin(dim=1)

        return (local_floor(neutral) - local_floor(shaped)).contiguous()

    @classmethod
    def from_config(cls, cfg: HumanMeshConfig) -> "SMPLMeshVisualizer":
        install_smpl_pickle_compat()
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("Install smplx and place SMPL_*.pkl files in data/smpl.") from exc

        from smplx.utils import Struct

        # smplx's default pickle.load is unrestricted. Decode only the numeric
        # SMPL model types below, validate arrays, and pass a ready data struct.
        model_path = resolve_smpl_model_path(cfg.model_dir)
        model_data = load_smpl_model_data(model_path)
        smpl_model = smplx.SMPL(
            model_path=model_path,
            data_struct=Struct(**model_data),
            gender="neutral",
            num_betas=10,
            use_pca=False,
            batch_size=cfg.num_envs,
        )

        uvs: np.ndarray | None = None
        uv_faces: np.ndarray | None = None
        uv_npz = Path(cfg.model_dir) / "smpl_uv.npz"
        if uv_npz.exists() and (cfg.texture_path or cfg.texture_paths):
            try:
                data = np.load(str(uv_npz), allow_pickle=False)
                uvs = data["uvs"]
                uv_faces = data["uv_faces"]
            except Exception as exc:
                print(
                    f"[SMPLMesh] WARNING: failed to load UV data from {uv_npz}: {exc}\n"
                    "  Using a solid-color mesh; provide matching uvs/uv_faces "
                    "and body textures to enable textured appearance."
                )

        return cls(
            cfg,
            smpl_model,
            np.asarray(smpl_model.faces, dtype=np.int64),
            uvs=uvs,
            uv_faces=uv_faces,
        )

    def _create_person_material(self, env_id: int, Gf, Sdf, UsdShade):
        material_root = f"{self.cfg.prim_root}/Materials/person_{env_id}"
        material = UsdShade.Material.Define(self.stage, material_root)
        shader = UsdShade.Shader.Define(self.stage, f"{material_root}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            float(self.cfg.opacity)
        )
        if self.has_texture:
            tex_path = Path(self.person_texture_paths[env_id]).resolve()
            st_reader = UsdShade.Shader.Define(
                self.stage, f"{material_root}/StReader"
            )
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            tex_shader = UsdShade.Shader.Define(
                self.stage, f"{material_root}/TexShader"
            )
            tex_shader.CreateIdAttr("UsdUVTexture")
            tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_path))
            tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )
            tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
            st_input = material.CreateInput(
                "frame:stPrimvarName", Sdf.ValueTypeNames.Token
            )
            st_input.Set("st")
            st_reader.CreateInput(
                "varname", Sdf.ValueTypeNames.Token
            ).ConnectToSource(st_input)
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*self.person_colors[env_id])
            )

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        return material

    def create(self) -> None:
        if self._created:
            return

        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("Could not get active USD stage.")

        UsdGeom.Xform.Define(self.stage, self.cfg.prim_root).MakeVisible()

        # ── Mesh geometry ─────────────────────────────────────────
        num_vertices = int(getattr(self.smpl_model, "v_template").shape[0])
        zero_points = np.zeros((num_vertices, 3), dtype=np.float32)
        face_vertex_counts = [3] * int(self.faces.shape[0])
        face_vertex_indices = self.faces.reshape(-1).astype(np.int64).tolist()

        for env_id in range(self.cfg.num_envs):
            material = self._create_person_material(env_id, Gf, Sdf, UsdShade)
            mesh_path = f"{self.cfg.prim_root}/env_{env_id}/Body"
            UsdGeom.Xform.Define(self.stage, f"{self.cfg.prim_root}/env_{env_id}")
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
            mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*self.person_colors[env_id])]))
            mesh.CreateDisplayOpacityAttr([float(self.cfg.opacity)])
            self._points_attrs.append(mesh.CreatePointsAttr(to_vt_vec3f(zero_points)))

            # ── UV primvar (faceVarying — preserves seam discontinuities) ──
            #
            # 修复说明（Bug 2 + Bug 3）：
            #
            # Bug 2 — 错误的 V 轴翻转：
            #   原代码: 1.0 - vtx_uv[i, 1]
            #   USD UsdUVTexture 规范: st=(0,0) = 纹理左下角 (OpenGL 约定)，
            #   USD 在读取 PNG 时会内部翻转图像，使其符合 OpenGL 约定，
            #   因此应用方无需再手动翻转 V 轴。
            #   SMPL 官方 UV 数据同样使用 OpenGL 约定 (V=0 在底部)。
            #   双重翻转导致人脸区域 (V≈0.65-1.0) 被映射到纹理底部 (脚/手)。
            #
            # Bug 3 — vertex 插值在 UV 缝合线 (seam) 处的平均误差：
            #   SMPL 模型有多处 UV seam，seam 处同一几何顶点在不同面中
            #   对应完全不同的 UV 坐标（属于两个不相邻的 UV 岛）。
            #   简单平均后的 UV 落在两个岛中间，恰好可能位于人脸纹理区域，
            #   导致「人脸贴到肚子上」。
            #   修复：改用 faceVarying 插值，每个面的每个顶点角使用精确 UV，
            #   不做任何平均，完全保留 seam 处的不连续性。
            if self.has_texture and self.uvs is not None and self.uv_faces is not None:
                # faceVarying: 每个面×3个角有独立 UV，展开为 (N_faces*3,) 列表
                # uv_faces[f, c] → 在 uvs 中的 UV 顶点索引
                # uvs[uv_idx]    → (u, v)，SMPL OBJ/官方格式使用 OpenGL 约定
                #                  (V=0=底部)，与 USD 一致，无需翻转
                fv_uvs = self.uvs[self.uv_faces.reshape(-1)]  # (N_faces*3, 2)
                st_array = Vt.Vec2fArray([
                    Gf.Vec2f(float(fv_uvs[i, 0]), float(fv_uvs[i, 1]))
                    for i in range(fv_uvs.shape[0])
                ])
                primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
                st_primvar = primvars_api.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.faceVarying,  # ← 关键修复：faceVarying 而非 vertex
                )
                st_primvar.Set(st_array)

            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        self._created = True
        print(
            f"[SMPLMesh] Created {self.cfg.num_envs} meshes "
            f"(mode={'textured' if self.has_texture else 'solid-color'}, "
            f"texture_pool={len(self.texture_paths)}, random_assignment=true, "
            f"shape_std={self.cfg.shape_std:.2f}, "
            f"seed={self.cfg.appearance_seed}, "
            f"floor_offset=[{self.shape_floor_offsets.min().item():+.3f}, "
            f"{self.shape_floor_offsets.max().item():+.3f}]m)"
        )

    @torch.no_grad()
    def update(
        self,
        body_pose: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
    ) -> None:
        if not self._created:
            self.create()

        vertices = self.compute_world_vertices(body_pose, root_pos, root_quat_wxyz)
        vertices = vertices.detach().cpu().numpy().astype(np.float32)

        for env_id, points_attr in enumerate(self._points_attrs):
            points_attr.Set(to_vt_vec3f(vertices[env_id]))

    @torch.no_grad()
    def compute_world_vertices(
        self,
        body_pose: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        """Skin SMPL on CPU or GPU; the root quaternion is explicitly wxyz."""
        expected = ((self.cfg.num_envs, 69), (self.cfg.num_envs, 3), (self.cfg.num_envs, 4))
        for value, shape in zip((body_pose, root_pos, root_quat_wxyz), expected):
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"SMPL pose input must be finite with shape {shape}, got {tuple(value.shape)}")
        if bool((root_quat_wxyz.norm(dim=-1) < 1e-8).any()):
            raise ValueError("SMPL root quaternion must have nonzero norm")
        dtype = self.smpl_model.v_template.dtype
        body_pose = body_pose.to(device=self.device, dtype=dtype)
        root_pos = root_pos.to(device=self.device, dtype=dtype)
        root_quat_wxyz = root_quat_wxyz.to(device=self.device, dtype=dtype)
        zeros = torch.zeros((self.cfg.num_envs, 3), device=self.device, dtype=body_pose.dtype)
        betas = self.betas.to(dtype=body_pose.dtype)

        out = self.smpl_model(
            betas=betas,
            global_orient=zeros,
            body_pose=body_pose,
            transl=zeros,
            return_verts=True,
        )
        vertices = out.vertices - out.joints[:, 0:1, :]
        vertices = torch.matmul(vertices, smpl_to_robot_matrix(self.device, vertices.dtype).T)
        vertices[..., 2] += self.shape_floor_offsets.to(vertices.dtype)[:, None]
        vertices = quat_apply_wxyz(root_quat_wxyz[:, None, :], vertices) + root_pos[:, None, :]
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("SMPL skinning produced non-finite vertices")
        return vertices


def smpl_to_robot_matrix(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed SMPL local frame -> ProtoMotions SMPL robot local frame transform."""
    rx90 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    yaw90 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return yaw90 @ rx90


def robot_expmap_to_smpl_rotvec(
    robot_expmap: torch.Tensor,
    smpl_to_robot: torch.Tensor,
) -> torch.Tensor:
    robot_quat = exp_map_to_quat(robot_expmap, w_last=False)
    robot_rot = quaternion_to_matrix(robot_quat, w_last=False)
    transform = smpl_to_robot.to(device=robot_expmap.device, dtype=robot_expmap.dtype)
    smpl_rot = transform.T @ robot_rot @ transform
    smpl_quat = matrix_to_quaternion(smpl_rot, w_last=False)
    smpl_quat = smpl_quat / torch.clamp(smpl_quat.norm(dim=-1, keepdim=True), min=1e-8)
    return quat_to_exp_map(smpl_quat, w_last=False)


def quat_apply_wxyz(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.clamp(quat_wxyz.norm(dim=-1, keepdim=True), min=1e-8)
    q_vec = quat[..., 1:4]
    q_w = quat[..., 0:1]
    t = 2.0 * torch.cross(q_vec.expand_as(vec), vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec.expand_as(vec), t, dim=-1)


def resolve_smpl_model_path(model_dir: str) -> str:
    path = Path(model_dir).expanduser()
    if path.is_file():
        return str(path)
    smpl_sub = path / "smpl" / "SMPL_NEUTRAL.pkl"
    if smpl_sub.exists():
        return str(smpl_sub)
    flat_file = path / "SMPL_NEUTRAL.pkl"
    if flat_file.exists():
        return str(flat_file)
    raise FileNotFoundError(
        f"SMPL neutral model missing at {path}. Provide SMPL_NEUTRAL.pkl in "
        "this directory (or its smpl/ subdirectory), or pass an explicit model file."
    )


def _smpl_pickle_globals() -> dict[tuple[str, str], object]:
    """Narrow compatibility list for SMPL's numeric Python 2/3 model files."""
    import copyreg
    from scipy.sparse import csc_matrix
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:
        from numpy.core.multiarray import _reconstruct

    allowed = {
        ("__builtin__", "object"): object,
        ("__builtin__", "set"): set,
        ("builtins", "object"): object,
        ("builtins", "set"): set,
        ("copy_reg", "_reconstructor"): copyreg._reconstructor,
        ("copyreg", "_reconstructor"): copyreg._reconstructor,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
        ("scipy.sparse.csc", "csc_matrix"): csc_matrix,
        ("scipy.sparse._csc", "csc_matrix"): csc_matrix,
    }
    try:
        from chumpy.ch import Ch
    except ImportError:
        pass
    else:
        allowed[("chumpy.ch", "Ch")] = Ch
    return allowed


def inspect_smpl_pickle(path: str | Path) -> dict:
    """Inspect opcodes before loading; reject global types outside SMPL data."""
    install_smpl_pickle_compat()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SMPL model file not found: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"SMPL model is a Git LFS pointer, not parameter data: {path}")
    allowed = _smpl_pickle_globals()
    globals_seen: set[str] = set()
    with path.open("rb") as stream:
        for op, arg, _ in pickletools.genops(stream):
            if op.name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
                raise ValueError(f"Unsupported SMPL pickle opcode {op.name}")
            if op.name == "GLOBAL":
                module, name = arg.split(" ", 1)
                if (module, name) not in allowed:
                    raise ValueError(f"Unsupported SMPL pickle global {module}.{name}")
                globals_seen.add(f"{module}.{name}")
    return {"path": str(path), "bytes": path.stat().st_size, "globals": sorted(globals_seen)}


def load_smpl_model_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load validated local SMPL arrays through a restricted unpickler.

    This supports the audited legacy neutral model without allowing arbitrary
    pickle imports or passing its Chumpy objects into the runtime skinning code.
    """
    inspect_smpl_pickle(path)
    allowed = _smpl_pickle_globals()

    class SMPLDataUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return allowed[(module, name)]
            except KeyError as exc:
                raise pickle.UnpicklingError(f"Unsupported SMPL global {module}.{name}") from exc

        def persistent_load(self, pid):
            raise pickle.UnpicklingError("SMPL persistent pickle objects are unsupported")

    with Path(path).open("rb") as stream:
        model = SMPLDataUnpickler(stream, encoding="latin1").load()
    if not isinstance(model, dict):
        raise ValueError("SMPL parameter file must contain a dictionary")
    keys = ("v_template", "f", "weights", "posedirs", "shapedirs", "J_regressor", "kintree_table")
    missing = set(keys) - set(model)
    if missing:
        raise ValueError(f"SMPL model is missing required arrays: {sorted(missing)}")
    data = {}
    for key in keys:
        value = model[key]
        if hasattr(value, "toarray"):
            value = value.toarray()
        elif type(value).__module__ == "chumpy.ch":
            value = value.r
        data[key] = np.asarray(value)
        if not np.issubdtype(data[key].dtype, np.number) or not np.isfinite(data[key]).all():
            raise ValueError(f"SMPL array {key} must contain finite numeric values")
    n = len(data["v_template"])
    shapes = {"v_template": (n, 3), "weights": (n, 24), "J_regressor": (24, n), "kintree_table": (2, 24), "posedirs": (n, 3, 207)}
    for key, expected in shapes.items():
        if data[key].shape != expected:
            raise ValueError(f"SMPL array {key} must have shape {expected}, got {data[key].shape}")
    if data["shapedirs"].ndim != 3 or data["shapedirs"].shape[:2] != (n, 3) or data["shapedirs"].shape[2] < 10:
        raise ValueError("SMPL shapedirs must be [vertices, 3, at least 10 betas]")
    faces = data["f"]
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer) or faces.size == 0 or faces.min() < 0 or faces.max() >= n:
        raise ValueError("SMPL faces must be valid triangle vertex indices")
    tree = data["kintree_table"]
    if not np.array_equal(tree[1], np.arange(24)) or not (tree[0, 1:] < np.arange(1, 24)).all():
        raise ValueError("SMPL kinematic tree must use the standard 24-joint order")
    if not np.allclose(data["weights"].sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("SMPL skinning weights must sum to one for every vertex")
    return data


def install_smpl_pickle_compat() -> None:
    """Patch Python 3.11 / NumPy aliases used by older SMPL pickle files."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    for name, value in {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def to_vt_vec3f(points: np.ndarray):
    from pxr import Gf, Vt

    points = np.asarray(points, dtype=np.float32)
    try:
        return Vt.Vec3fArray.FromNumpy(points)
    except Exception:
        return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
