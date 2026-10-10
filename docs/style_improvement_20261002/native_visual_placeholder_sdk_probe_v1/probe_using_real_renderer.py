"""Actual SDK initial visual-stage observation; wraps only the existing compositor."""
from pathlib import Path
import importlib.util
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime import native_visual_scene
original = native_visual_scene.compose_native_visual_scene

def observe(stage, contract):
    from pxr import UsdGeom
    rows = []
    for prim in stage.Traverse():
        transform = UsdGeom.Xformable(prim)
        rows.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                     "authored_metadata": {k: str(v) for k,v in prim.GetAllAuthoredMetadata().items()},
                     "authored_properties": [p.GetName() for p in prim.GetAuthoredProperties()],
                     "children": [str(p.GetPath()) for p in prim.GetChildren()],
                     "applied_schemas": list(prim.GetAppliedSchemas()),
                     "prim_stack_layers": [p.layer.identifier for p in prim.GetPrimStack()],
                     "has_authored_references": prim.HasAuthoredReferences(),
                     "has_authored_payloads": prim.HasAuthoredPayloads(),
                     "instanceable": prim.IsInstanceable(),
                     "local_matrix": [[float(v) for v in row] for row in transform.GetLocalTransformation()] if transform else None})
    report = {"scope": "actual production renderer initial Kit stage before native composition; original compositor still invoked",
              "root_layer_identifier": stage.GetRootLayer().identifier, "session_layer_identifier": stage.GetSessionLayer().identifier,
              "root_layer_text": stage.GetRootLayer().ExportToString(), "default_prim": str(stage.GetDefaultPrim().GetPath()),
              "prims": rows, "source_USD": contract["source_USD"], "SDK_execution_observed": True}
    (Path(__file__).parent / "actual_empty_stage.json").write_text(json.dumps(report,indent=2)+"\n")
    return original(stage, contract)

native_visual_scene.compose_native_visual_scene = observe
spec = importlib.util.spec_from_file_location("real_replay", ROOT / "scripts/replay_paper_pedestrians.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raise SystemExit(module.main(sys.argv[1:]))
