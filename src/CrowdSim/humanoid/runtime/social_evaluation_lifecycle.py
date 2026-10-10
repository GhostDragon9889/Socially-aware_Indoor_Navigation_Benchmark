"""Terminal lifecycle for the pinned, isolated AMPWorld tracking evaluator.

Import is stdlib only. The four-asset native helper remains sealed; a reversible
derivation narrows its owner count to one vectorized asset for this evaluator.
This module does not cover contact-enabled navigation or partial constructors.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import ast
import gc
import hashlib
import importlib.util
from pathlib import Path
import stat
import weakref

from .relocated_terminal import derive_relocated_checkout, overlay_kit_arguments

ROOT = repository_root(__file__)
TERMINAL = ROOT / "docs/style_improvement_20261004/social_native_overlay_entry_revision_v10/smoke_group_physics.py"
TERMINAL_SHA = "0e6b2ef97cc2f32612317df162ebba6107b1e276083f9912995badd5f416685e"


def record(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Require an ordinary independent file: " + str(path))
    content = path.read_bytes()
    after = path.lstat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("File changed while reading: " + str(path))
    return {"path": str(path.resolve()), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def derive_single_asset_release(source):
    """Change only five owner-count constants in one pinned terminal function."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_release_owned_asset_simulation_views")
    original = ast.get_source_segment(source, node)
    replacements = (
        ("len(assets) != 4", "len(assets) != 1"),
        ("len({id(asset) for asset in owned_assets}) == 4", "len({id(asset) for asset in owned_assets}) == 1"),
        ("len({id(data) for data in owned_data}) == 4", "len({id(data) for data in owned_data}) == 1"),
        ('{"articulation_fields": 4, "data_fields": 4}', '{"articulation_fields": 1, "data_fields": 1}'),
        ("All four assets and four data objects", "The single asset and its data object"),
        ("Exactly four caller-owned vectorized assets are required", "Exactly one caller-owned vectorized asset is required"),
        ("Require four distinct exact loaded Articulation instances", "Require one exact loaded Articulation instance"),
        ("All eight existing instance aliases", "Both existing instance aliases"),
    )
    derived = original
    for before, after in replacements:
        if derived.count(before) != 1:
            raise ValueError("Ambiguous single-asset derivation: " + before)
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(replacements):
        if inverse.count(after) != 1:
            raise ValueError("Ambiguous single-asset inverse")
        inverse = inverse.replace(after, before, 1)
    if inverse != original:
        raise RuntimeError("Single-asset derivation is not reversible")
    return derived, {"original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "owner_count_constants_changed": 5, "diagnostic_text_changes": 4, "exact_inverse": True,
                     "expected_articulation_fields": 1, "expected_data_fields": 1}


class IsolatedTrackingLifecycle:
    """One process, one App, one complete original N1 AMPWorld; terminal only."""

    def __init__(self, report):
        self.report = report
        identity = record(TERMINAL)
        if identity["sha256"] != TERMINAL_SHA or TERMINAL.stat().st_mode & 0o222:
            raise ValueError("Sealed native lifecycle helper changed")
        spec = importlib.util.spec_from_file_location("_isolated_tracking_terminal", TERMINAL)
        self.terminal = importlib.util.module_from_spec(spec)
        original_source = TERMINAL.read_text()
        relocated_source, path_derivation = derive_relocated_checkout(
            original_source, ROOT.parent / "ProtoMotions", ROOT)
        exec(compile(relocated_source, str(TERMINAL) + ":relocated_checkout", "exec"), self.terminal.__dict__)
        self.terminal.OVERLAY_KIT_ARGS = overlay_kit_arguments(
            ROOT.parent / "ProtoMotions", self.terminal.OVERLAY)
        source, derivation = derive_single_asset_release(TERMINAL.read_text())
        namespace = dict(self.terminal.__dict__)
        exec(compile(source, str(TERMINAL) + ":single_asset", "exec"), namespace)
        self.release_asset = namespace["_release_owned_asset_simulation_views"]
        report["evaluation_lifecycle"] = {"implementation": record(Path(__file__)),
            "sealed_terminal": identity, "single_asset_derivation": derivation,
            "checkout_path_derivation": path_derivation,
            "contact_enabled_navigation_supported": False, "partial_construction_verified": False}
        report.update(cleanup_succeeded=False, cleanup_errors=[], simulation_stop_returned=False,
                      simulation_clear_instance_returned=False, app_close_returned=False)
        self.terminal._verify_overlay_inputs(report)

    @property
    def kit_args(self):
        return "--/app/renderer/enabled=false --/renderer/enabled=false " + self.terminal.OVERLAY_KIT_ARGS

    def verify_loaded(self):
        self.terminal._verify_loaded_overlay(self.report)

    def _detach_world(self, world, producer):
        simulator = scene = asset = None
        try:
            if world is None or world.num_envs != 1 or world.shared_world or world.validation_scene is not None:
                raise RuntimeError("A complete original isolated N1 flat-world owner is required")
            simulator = world.simulator
            scene, asset = simulator._scene, simulator._robot
            if (set(scene.articulations) != {"robot"} or scene.articulations["robot"] is not asset
                    or scene.sensors or scene.rigid_objects or scene.rigid_object_collections
                    or scene.deformable_objects or simulator._contact_sensor_map
                    or simulator._object or simulator._projectile_objects
                    or getattr(simulator, "pedestrian_contacts", None) is not None):
                raise RuntimeError("Unreviewed additional native owners in isolated evaluation")
            self.release_asset(simulator._sim, (asset,), self.report)
            if producer is not None:
                if producer.world is not world or producer.sim is not simulator._sim:
                    raise RuntimeError("Observation producer is not owned by this evaluation World")
                producer.world = producer.sim = producer.sampler = None
            scene.articulations.clear()
            simulator._robot = simulator._scene = simulator._sim = simulator._simulation_app = None
            world.simulator = None
            self.report["evaluation_world_detached"] = True
        finally:
            simulator = scene = asset = world = producer = None

    def close(self, app, world, producer=None, *, native_guard=None):
        """Attempt every terminal phase; report cleanup errors without hiding failure."""
        report, sim, refs = self.report, None, ()
        errors = report["cleanup_errors"]

        def attempt(label, function):
            try:
                return function()
            except BaseException as error:
                errors.append(label + ": " + type(error).__name__ + ": " + str(error))
                return None

        try:
            if world is not None:
                sim = world.simulator._sim
            if sim is not None:
                def stop():
                    sim.stop()
                    report["simulation_stop_returned"] = True
                attempt("simulation stop", stop)
                attempt("single-asset release", lambda: self._detach_world(world, producer))
                world = producer = None
                gc.collect()
                attempt("owned manager normalization", lambda: self.terminal._release_terminal_timeline_caches(
                    sim, report, phase="before_clear"))

                def clear():
                    if type(sim).instance() is not sim:
                        raise RuntimeError("Simulation singleton is not owned by this evaluator")
                    sim.clear_instance()
                    if type(sim).instance() is not None:
                        raise RuntimeError("Simulation singleton survived clear_instance")
                    report["simulation_clear_instance_returned"] = True
                attempt("simulation clear", clear)
                refs = attempt("terminal SDK release", lambda: self.terminal._release_terminal_timeline_caches(
                    sim, report, phase="after_clear"))
                gc.collect()
                if (not isinstance(refs, tuple) or not refs
                        or any(type(ref) is not weakref.ReferenceType for ref in refs)):
                    errors.append("No complete backend weakref receipt")
                else:
                    report["scene_backend_views_live_after_post_context_gc"] = sum(ref() is not None for ref in refs)
                    if report["scene_backend_views_live_after_post_context_gc"] != 0:
                        errors.append("Backend native views survive before App shutdown")
                if native_guard is not None:
                    attempt("additional owned native release", native_guard)
            else:
                errors.append("Complete World was not returned; partial-constructor cleanup remains unverified")
            if app is not None:
                def close_app():
                    app.close()
                    report["app_close_returned"] = True
                attempt("App close", close_app)
            flags = ("simulation_stop_returned", "evaluation_world_detached", "owned_asset_simulation_views_release_returned",
                     "owned_manager_normalization_returned", "simulation_clear_instance_returned",
                     "terminal_scene_backend_release_returned", "terminal_timeline_cache_compat_returned",
                     "core_terminal_shutdown_returned", "app_close_returned")
            report["cleanup_succeeded"] = not errors and all(report.get(flag) is True for flag in flags)
        finally:
            sim = refs = world = producer = app = None
