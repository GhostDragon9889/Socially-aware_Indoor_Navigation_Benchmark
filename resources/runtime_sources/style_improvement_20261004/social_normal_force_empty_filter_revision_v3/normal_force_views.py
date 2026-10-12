"""Lazy per-actor24 all-partner net-normal views for the frozen grouped World.

Import is stdlib only. Calls require the caller's initialized App/sim/scene;
no SDK/Torch import, SimulationView/App construction, step or USD authoring.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import stat
import sys
from types import ModuleType
import weakref

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
STAGES = ROOT / "resources/runtime_sources/style_improvement_20261004"
LAB = Path("lab:")
API = LAB / ".venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py"
PINS = {
    "world": (STAGES / "shared_group_world_owned_lifetime_revision_v2/shared_group_world.py", 39960, 'd7400a48364b64971f4e2e942d87ebd9e108b5a8ff018e8a8f404b817bd02038', True),
    "sampler": (STAGES / "social_net_normal_sampler_staging_v1/net_normal_sampler.py", 7461, '52ea1901985c6152429d00c7dc6261c3371a672009acc4b578a57e5cc0385517', True),
    "names": (ROOT / "CrowdSim/humanoid/amp/observations.py", 10586, '000f3980a8f4d3c1fac929c57029dd05a9ce19b1d8ce82423dad273c9315e599', False),
    "ownership": (ROOT / "CrowdSim/humanoid/physics_views.py", 1845, '1b13d982864a3a2dae6eaebccd15f7ecfdda666c1e1a8dbce4996786a59a0c21', False),
    "tensor_API": (API, 382927, "b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903", False),
    "articulation": (LAB / "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py", 224858, "2cbd53ccbebb7a7b038bed2b612b65e41d6674a4791beec68d84336d7bd00a87", False),
    "scene": (LAB / "source/isaaclab/isaaclab/scene/interactive_scene.py", 39939, "5d509530070ffb8f4616f769a6608299a386800d7e2582833181ba28dfee32d1", False),
    "sim": (LAB / "source/isaaclab/isaaclab/sim/simulation_context.py", 42425, "8c5d73af45f82b0d5c36b05a1c293094319c5d17a75f75a2fe679c382334ea89", False),
    "manager": (LAB / "source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py", 34974, "d83ef1b3a375dbbdbd538f7e5f552064ee57528ae7b35b9ba58719fd8a4cfda5", False),
    "stage": (LAB / "source/isaaclab/isaaclab/sim/utils/stage.py", 20480, "7443a822773b7d8152757854c75cd8104f34d3fe68bb0c9da1d6e0bb56ba6f89", False),
}
DT = 1 / 120
_DEPS = None


def _dependencies():
    global _DEPS
    if _DEPS is None:
        sources = {}
        for key, (path, size, sha, readonly) in PINS.items():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
                raise ValueError("Invalid reviewed source: " + key)
            data = path.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("Reviewed whole-byte source changed: " + key)
            sources[key] = data
        tree = ast.parse(sources["names"])
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "PACER_BODY_NAMES" for target in node.targets)]
        if len(nodes) != 1:
            raise ValueError("Expected one canonical SMPL body tuple")
        names = ast.literal_eval(nodes[0].value)
        if type(names) is not tuple or len(names) != 24 or len(set(names)) != 24:
            raise ValueError("Invalid canonical SMPL body tuple")
        helper = ModuleType("_reviewed_normal_view_ownership")
        exec(compile(sources["ownership"], str(PINS["ownership"][0]), "exec"), helper.__dict__)
        _DEPS = names, helper.resolve_native_contact_factory
    return _DEPS


def _current_stage():
    # SimulationContext has already imported this public module. No SDK import
    # or private manager state is used; reject a missing/wrong-source module.
    module = sys.modules.get("isaaclab.sim.utils.stage")
    if (module is None or Path(getattr(module, "__file__", "")).resolve() != PINS["stage"][0]
            or not callable(getattr(module, "get_current_stage", None))):
        raise RuntimeError("The caller's initialized reviewed Lab stage module is required")
    return module.get_current_stage()


def _path(value):
    return (type(value) is str and value.startswith("/") and not value.endswith("/")
            and not any(char in value for char in "*?[]") and all(part not in ("", ".", "..") for part in value[1:].split("/")))


class _Binding:
    def __init__(self, sim, scene, assets, resolve):
        complete, asset = False, None
        roots = []
        try:
            self._released, self._bundle = False, None
            self.sim, self.scene, self.assets, self.resolve = sim, scene, assets, resolve
            self.view, self.ownership = resolve(sim)
            self.stage = scene.stage
            for asset in assets:
                roots.append(asset.root_view)
            self.roots = tuple(roots)
            self.baseline = sim.get_physics_step_count()
            self.check()
            complete = True
        finally:
            if not complete:
                _drop_binding_aliases(self)
            roots.clear()
            self = sim = scene = assets = resolve = asset = None

    def check(self):
        active = _ownership = asset = root = None
        try:
            if self._released:
                raise RuntimeError("Normal-force bundle has been released")
            active, _ownership = self.resolve(self.sim)
            counter = self.sim.get_physics_step_count()
            dt = self.sim.get_physics_dt()
            if (active is not self.view or not self.view.is_valid
                    or self.scene.stage is not self.stage or self.sim.stage is not self.stage
                    or _current_stage() is not self.stage or self.sim.is_playing() is not True):
                raise RuntimeError("Native view/stage ownership changed or stopped; automatic recovery is not implemented")
            for asset, root in zip(self.assets, self.roots, strict=True):
                if asset.root_view is not root:
                    raise RuntimeError("Native view/stage ownership changed or stopped; automatic recovery is not implemented")
            if (type(self.baseline) is not int or self.baseline < 0 or type(counter) is not int or counter < self.baseline
                    or type(dt) not in (int, float) or not math.isfinite(dt) or not math.isclose(dt, DT, rel_tol=0, abs_tol=1e-12)):
                raise RuntimeError("Native baseline/current clock and actual physics dt1/120 must remain valid")
        finally:
            self = active = _ownership = asset = root = None


class _NormalView:
    """Minimal public sampler interface with owned-lifetime checks, no payload copy."""
    def __init__(self, native, binding):
        try:
            self.native, self.binding = native, binding
        finally:
            self = native = binding = None

    @property
    def sensor_count(self):
        try:
            if self.native is None or self.binding is None:
                raise RuntimeError("Normal-force bundle has been released")
            return self.native.sensor_count
        finally:
            self = None

    @property
    def sensor_paths(self):
        try:
            if self.native is None or self.binding is None:
                raise RuntimeError("Normal-force bundle has been released")
            return self.native.sensor_paths
        finally:
            self = None

    def get_net_contact_forces(self, dt):
        payload = None
        try:
            if self.native is None or self.binding is None:
                raise RuntimeError("Normal-force bundle has been released")
            self.binding.check()
            if type(dt) not in (int, float) or not math.isfinite(dt) or not math.isclose(dt, DT, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Net-force dt must be actual physics1/120, not control1/30")
            if not self.native.check():
                raise RuntimeError("Native contact objects became invalid; rebuild belongs to the caller")
            payload = self.native.get_net_contact_forces(dt=dt)
            self.binding.check()
            return payload  # The frozen sampler immediately detaches/clones native Warp data.
        finally:
            self = payload = None


def _drop_binding_aliases(binding):
    """Drop only this module's ownership fields; never call native teardown."""
    try:
        binding._released, binding._bundle = True, None
        binding.sim = binding.scene = binding.resolve = binding.view = binding.stage = binding.ownership = None
        binding.assets = binding.roots = ()
    finally:
        binding = None


def create_normal_force_bundle(sim, scene, assets, group_actor_rows):
    """Return (N per-actor24 views, layouts, body_names_by_view) after sim.reset.

    Rows use the caller's existing int64[G,4] on its sim device; no IDs are
    invented. All paths/names/schema/admission checks precede view creation.
    Creation/getter exceptions propagate. Failed creation drops only this
    module's unreturned aliases; SDK side effects and foreign owners are not
    rolled back. The successful bundle must later be explicitly released.
    """
    binding = native = root = asset = prim = resolve = bundle = None
    views = []
    returned = False
    try:
        canonical, resolve = _dependencies()
        if not isinstance(assets, (tuple, list)) or len(assets) != 4:
            raise ValueError("Four slot-ordered assets required")
        assets = tuple(assets)
        groups = scene.cfg.num_envs
        if (type(groups) is not int or groups < 1 or scene.cfg.filter_collisions is not True
                or scene.cfg.replicate_physics is not True or "physx" not in scene.physics_backend
                or set(scene.articulations) != {f"robot_{slot}" for slot in range(4)}):
            raise ValueError("Caller must supply the initialized filtered four-slot G-env PhysX scene")
        for slot, asset in enumerate(assets):
            if scene.articulations[f"robot_{slot}"] is not asset:
                raise ValueError("Caller must supply the initialized filtered four-slot G-env PhysX scene")
        # Structural tensor seam permits stdlib mocks. SharedGroupWorld/bridge owns
        # actual Torch class validation; integer values/complete partition checked here.
        if (tuple(getattr(group_actor_rows, "shape", ())) != (groups, 4)
                or str(getattr(group_actor_rows, "dtype", "")) != "torch.int64"
                or str(getattr(group_actor_rows, "device", "")) != str(sim.device)
                or not callable(getattr(group_actor_rows, "tolist", None))):
            raise ValueError("Existing on-device int64[G,4] group rows required")
        rows = group_actor_rows.tolist()  # Once: explicit small host ID transfer, not force data.
        if (type(rows) is not list or len(rows) != groups or any(type(row) is not list or len(row) != 4 for row in rows)
                or any(type(actor) is not int for row in rows for actor in row)
                or sorted(actor for row in rows for actor in row) != list(range(groups * 4))):
            raise ValueError("Group rows must completely partition existing canonical actors0..N-1")
        env_paths = [f"/World/envs/env_{group}" for group in range(groups)]
        if list(scene.env_prim_paths) != env_paths:
            raise ValueError("Exact numerical group env paths/order required; do not assume lexical env10 ordering")
        binding = _Binding(sim, scene, assets, resolve)
        layouts, names_by_actor = [None] * (groups * 4), [None] * (groups * 4)
        all_paths = []
        for slot, (asset, root) in enumerate(zip(assets, binding.roots, strict=True)):
            names = tuple(asset.body_names)
            paths, roots = list(root.link_paths), list(root.prim_paths)
            if (asset.num_instances != groups or str(asset.device) != str(sim.device)
                    or len(names) != 24 or len(set(names)) != 24 or set(names) != set(canonical)
                    or len(paths) != groups or len(roots) != groups or not root.check()):
                raise ValueError("Native slot must have valid G articulations and exact named24 SMPL links")
            for group, (links, arti_path) in enumerate(zip(paths, roots, strict=True)):
                prefix = env_paths[group] + f"/Robot_{slot}/"
                links = list(links)
                if (not _path(arti_path) or not (arti_path == prefix[:-1] or arti_path.startswith(prefix)) or len(links) != 24
                        or any(not _path(path) or not path.startswith(prefix) for path in links)
                        or tuple(path.rsplit("/", 1)[-1] for path in links) != names):
                    raise ValueError("Actual native group/slot/body paths must match their named ordered rows")
                for path in links:
                    prim = binding.stage.GetPrimAtPath(path)
                    if not prim.IsValid() or "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
                        raise ValueError("Every actual body must already have ContactReportAPI; factory never authors it")
                actor = rows[group][slot]
                layouts[actor] = {"actor": actor, "sources": links}
                names_by_actor[actor] = names
                all_paths.extend(links)
        if len(set(all_paths)) != groups * 4 * 24:
            raise ValueError("Every actor/body path must be distinct")
        binding.check()
        views = []
        for layout in layouts:
            native = binding.view.create_rigid_contact_view(list(layout["sources"]), filter_patterns=[], max_contact_data_count=0)
            binding.check()
            if (native.sensor_count != 24 or list(native.sensor_paths) != layout["sources"]
                    or native.filter_count != 0 or list(native.filter_paths) not in ([], [[] for _ in range(24)]) or not native.check()):
                raise RuntimeError("Created native per-actor24 no-filter view differs from submitted path/order contract")
            views.append(_NormalView(native, binding))
        if sim.get_physics_step_count() != binding.baseline:
            raise RuntimeError("Creating normal views must not advance physics")
        bundle = tuple(views), tuple(layouts), tuple(names_by_actor)
        binding._bundle = bundle
        returned = True
        return bundle
    finally:
        if not returned:
            for wrapper in views:
                wrapper.native = wrapper.binding = None
            if binding is not None:
                _drop_binding_aliases(binding)
        views.clear()
        sim = scene = assets = group_actor_rows = binding = native = root = asset = prim = resolve = bundle = None
        wrapper = None


def release_normal_force_bundle(bundle):
    """Release this exact returned bundle once, without SDK calls or GC.

    Return native_weakrefs, native_view_count and module_owned_aliases_released.
    Weakrefs do not claim destruction: foreign/caller aliases may still keep
    views alive. Callers must drop their other owned aliases before context/App
    teardown and independently inspect these references after their own GC.
    """
    views = layouts = names = binding = wrapper = native = refs = value = None
    try:
        if type(bundle) is not tuple or len(bundle) != 3:
            raise ValueError("The original returned three-tuple bundle is required")
        views, layouts, names = bundle
        for value in bundle:
            if type(value) is not tuple:
                raise ValueError("Complete tuple views/layouts/names from this factory required")
        if (not views or len(views) % 4 or len(layouts) != len(views) or len(names) != len(views)):
            raise ValueError("Complete tuple views/layouts/names from this factory required")
        wrapper_ids = set()
        for wrapper in views:
            if type(wrapper) is not _NormalView:
                raise ValueError("Only this module's exact normal-view wrappers may be released")
            wrapper_ids.add(id(wrapper))
        binding = views[0].binding
        if (type(binding) is not _Binding or binding._released or binding._bundle is not bundle
                or len(wrapper_ids) != len(views)):
            raise ValueError("One unreleased original bundle with distinct wrappers is required")
        native_ids = set()
        for actor, (wrapper, layout, body_names) in enumerate(zip(views, layouts, names, strict=True)):
            if (wrapper.binding is not binding or wrapper.native is None
                    or type(layout) is not dict or set(layout) != {"actor", "sources"}
                    or type(layout["actor"]) is not int or layout["actor"] != actor
                    or type(layout["sources"]) is not list or len(layout["sources"]) != 24
                    or any(not _path(path) for path in layout["sources"])
                    or type(body_names) is not tuple or len(body_names) != 24):
                raise ValueError("Bundle ownership, actor layout or named24 contract differs")
            native_ids.add(id(wrapper.native))
        if len(native_ids) != len(views):
            raise ValueError("Distinct actor-native views required")
        # All validation and weakref creation precede the first ownership change.
        refs = []
        for wrapper in views:
            native = wrapper.native
            refs.append(weakref.ref(native))
        count = len(views)
        for wrapper in views:
            wrapper.native = wrapper.binding = None
        _drop_binding_aliases(binding)
        return {"native_weakrefs": tuple(refs), "native_view_count": count,
                "module_owned_aliases_released": True}
    finally:
        bundle = views = layouts = names = binding = wrapper = native = refs = value = None
