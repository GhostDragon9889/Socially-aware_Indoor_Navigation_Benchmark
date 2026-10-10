"""CPU candidate: narrow admission of SDK-owned nonspatial /Render overlap.

The actual Warehouse source contributes only an empty metadata over. This
candidate does not accept source render products or change production code.
"""
from __future__ import annotations


def _authored_sdk_true(prim, name):
    from pxr import Sdf

    value = prim.GetMetadata(name)
    # Standalone OpenUSD does not register Kit metadata. Imported exact SDK
    # text exposes its authored boolean token as UnregisteredValue("true").
    # No default, textual substring, or truthiness admission is permitted.
    return value is True or (isinstance(value, Sdf.UnregisteredValue) and value.value == "true")


def admit_sdk_render_metadata_overlap(stage, source_layer):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root, session = stage.GetRootLayer(), stage.GetSessionLayer()
    if not root.anonymous or not session.anonymous:
        raise ValueError("Render overlap requires anonymous renderer root/session ownership")
    source_stage = Usd.Stage.Open(source_layer, load=Usd.Stage.LoadNone)
    source = source_stage.GetPrimAtPath("/Render")
    if not source or not source.IsActive():
        raise ValueError("Source /Render metadata over is missing/inactive")
    # The only admitted source form is the actual Warehouse no-content over.
    metadata = source.GetAllAuthoredMetadata()
    if (source.GetSpecifier() != Sdf.SpecifierOver or source.GetTypeName() or
            source.GetAllChildren() or source.GetAuthoredProperties() or source.GetAppliedSchemas() or
            source.HasAuthoredReferences() or source.HasAuthoredPayloads() or
            source.IsInstance() or source.IsInstanceable() or
            set(metadata) - {"specifier", "hide_in_stage_window"} or
            ("hide_in_stage_window" in metadata and not _authored_sdk_true(source, "hide_in_stage_window")) or
            any(spec.layer != source_layer for spec in source.GetPrimStack())):
        raise ValueError("Source /Render must remain the verified nonspatial metadata-only over; render products unsupported")
    render = stage.GetPrimAtPath("/Render")
    if not render:
        raise ValueError("Existing SDK /Render required for overlap admission")
    if (not _authored_sdk_true(render, "hide_in_stage_window") or not _authored_sdk_true(render, "no_delete") or
            {str(child.GetPath()) for child in render.GetAllChildren()} !=
            {"/Render/OmniverseKit", "/Render/OmniverseGlobalRenderSettings", "/Render/Vars"}):
        raise ValueError("Existing /Render does not match observed SDK-owned namespace")
    owned_paths = []
    allowed_types = {"", "Scope", "RenderProduct", "RenderSettings", "RenderVar"}
    allowed_metadata = {"specifier", "typeName", "apiSchemas", "hide_in_stage_window", "no_delete"}
    ordinary_properties = {"camera", "products", "orderedVars", "resolution", "overrideClipRange",
                           "sourceName", "sourceType", "viewPickingId", "viewportHandle"}
    for prim in Usd.PrimRange.AllPrims(render):
        path = str(prim.GetPath())
        stack = prim.GetPrimStack()
        if (not prim.IsActive() or prim.GetTypeName() not in allowed_types or not stack or
                any(spec.layer not in (root, session) for spec in stack) or
                bool(UsdGeom.Xformable(prim)) or prim.IsA(UsdGeom.Boundable) or
                prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or
                prim.HasAuthoredReferences() or prim.HasAuthoredPayloads() or
                prim.IsInstance() or prim.IsInstanceable() or
                set(prim.GetAllAuthoredMetadata()) - allowed_metadata or
                any(not name.startswith("OmniRtx") for name in prim.GetAppliedSchemas())):
            raise ValueError("Existing /Render has foreign, spatial, physics, arc, or unknown opinions: " + path)
        applied = prim.GetMetadata("apiSchemas")
        if applied is not None and any(not str(name).startswith("OmniRtx") for name in applied.GetAppliedItems()):
            raise ValueError("Unknown authored SDK /Render API: " + path)
        for prop in prim.GetAuthoredProperties():
            name = str(prop.GetName())
            if name not in ordinary_properties and not name.startswith("omni:rtx:"):
                raise ValueError("Unknown authored SDK /Render property: " + path + "." + name)
        owned_paths.append(path)
    static_paths = {"/Render", "/Render/OmniverseKit", "/Render/OmniverseKit/HydraTextures",
                    "/Render/OmniverseGlobalRenderSettings", "/Render/Vars", "/Render/Vars/LdrColor"}
    products = set(owned_paths) - static_paths
    prefix = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_"
    if (len(products) != 1 or not static_paths.issubset(owned_paths) or
            any(not path.startswith(prefix) or not path[len(prefix):].isdigit() for path in products)):
        raise ValueError("SDK /Render paths differ from observed viewport namespace")
    return {"schema": "native_visual.SDK_nonspatial_Render_overlap_candidate.v1",
        "root_path": "/Render", "SDK_owned_paths": owned_paths,
        "root_and_session_anonymous": True, "all_existing_prim_opinions_owned_by_root_or_session": True,
        "source_Render_metadata_only_over": True, "source_Render_products_or_geometry_added": [],
        "source_composition": "append original USD as weakest renderer root sublayer; session remains stronger",
        "other_source_roots_or_reference_floor_collision_allowed": False}
