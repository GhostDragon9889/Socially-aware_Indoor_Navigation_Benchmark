#!/usr/bin/env python3
"""Validate official SMPL sample topology and install an aligned numeric UV archive.

This offline resource converter does not download, execute remote files, or modify
the licensed SMPL model. Run with the dedicated project Python from repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COMMIT = "c87d871c7498fdc98f3cabb7341b5c85c38debaa"
SOURCE = "https://github.com/Meshcapade/SMPL_texture_samples"
BLOBS = {
    "SMPL_male.obj": "a8fdcd5d4e5896b7804f3287e3b4bfb50ba1fdf3",
    "SMPL_female.obj": "c5e5cb7544d0ac5421961b605c3bba8c2108cfe8",
}


def file_evidence(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest(),
    }


def recover_uv(path: Path, model_faces: np.ndarray, vertex_count: int):
    evidence = file_evidence(path)
    assert evidence["git_blob_sha1"] == BLOBS[path.name], "Official source blob differs"
    vertices, uvs, faces, uv_faces = [], [], [], []
    for line in path.read_text().splitlines():
        items = line.split()
        if not items:
            continue
        if items[0] == "v":
            vertices.append([float(value) for value in items[1:4]])
        elif items[0] == "vt":
            uvs.append([float(value) for value in items[1:3]])
        elif items[0] == "f":
            assert len(items) == 5, "Source is expected to contain subdivision quads"
            corners = [item.split("/") for item in items[1:]]
            faces.append([int(corner[0]) - 1 for corner in corners])
            uv_faces.append([int(corner[1]) - 1 for corner in corners])
    faces = np.asarray(faces, dtype=np.int64)
    uv_faces = np.asarray(uv_faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)
    assert np.isfinite(uvs).all() and ((uvs >= 0) & (uvs <= 1)).all()
    assert uv_faces.min() >= 0 and uv_faces.max() < len(uvs)
    groups = defaultdict(list)
    for index, face in enumerate(faces):
        # The official OBJ retains the original 6890 vertex IDs, followed by
        # edge vertices and then face-center vertices. Check that structure.
        original = np.flatnonzero(face < vertex_count)
        assert len(original) == 1, "Expected exactly one original corner per quad"
        center = int(face.max())
        assert int(face[(int(original[0]) + 2) % 4]) == center
        groups[center].append(index)
    lookup = defaultdict(list)
    parents = {}
    for center, rows in groups.items():
        assert len(rows) in (3, 4), "Original face must be a triangle or quad"
        mapping, edge_corners, next_edge = {}, defaultdict(set), {}
        for row in rows:
            face = faces[row]
            corner = int(np.flatnonzero(face < vertex_count)[0])
            vertex = int(face[corner])
            assert vertex not in mapping, "Duplicate original corner"
            mapping[vertex] = int(uv_faces[row, corner])
            for edge in (int(face[(corner - 1) % 4]), int(face[(corner + 1) % 4])):
                assert edge >= vertex_count and edge < center
                edge_corners[edge].add(vertex)
            next_edge[vertex] = int(face[(corner + 1) % 4])
        assert len(mapping) == len(rows)
        assert len(edge_corners) == len(rows)
        assert all(len(corners) == 2 for corners in edge_corners.values())
        next_vertex = {
            vertex: next(iter(edge_corners[edge] - {vertex}))
            for vertex, edge in next_edge.items()
        }
        loop = [min(mapping)]
        while next_vertex[loop[-1]] != loop[0]:
            assert next_vertex[loop[-1]] not in loop, "Invalid source face ring"
            loop.append(next_vertex[loop[-1]])
        assert len(loop) == len(rows), "Disconnected source face ring"
        parents[center] = (mapping, loop)
        for triangle in itertools.combinations(sorted(mapping), 3):
            lookup[triangle].append(center)
    aligned = np.empty_like(model_faces)
    used = Counter()
    for index, face in enumerate(model_faces):
        candidates = lookup[tuple(sorted(int(vertex) for vertex in face))]
        assert len(candidates) == 1, f"Nonunique parent for SMPL face {index}"
        center = candidates[0]
        mapping, loop = parents[center]
        positions = [loop.index(int(vertex)) for vertex in face]
        n = len(loop)
        winding = sum((positions[(i + 1) % 3] - positions[i]) % n for i in range(3))
        assert winding == n, f"Source/model winding differs at face {index}"
        aligned[index] = [mapping[int(vertex)] for vertex in face]
        used[center] += 1
    assert len(used) == len(parents), "Unused source parent face"
    assert all(used[center] == len(loop) - 2 for center, (_, loop) in parents.items())
    evidence.update({
        "source_vertices": len(vertices), "source_uvs": len(uvs),
        "source_subdivision_quads": len(faces),
        "parent_faces": len(parents),
        "parent_face_sizes": dict(Counter(len(loop) for _, loop in parents.values())),
        "unique_model_triangles_mapped": len(aligned),
        "face_winding_matches": True,
        "missing_or_ambiguous_triangles": 0,
        "mapping_method": "shared face-center topology contraction; original vertex IDs and face-corner UVs preserved",
    })
    return uvs, aligned, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--write", action="store_true", help="Create UV archive without replacing existing files")
    args = parser.parse_args()
    repo = args.repo_root.absolute()
    import sys

    sys.path.insert(0, str(repo))
    from CrowdSim.tools.smpl_mesh_visualizer import load_smpl_model_data

    smpl = repo / "data/smpl"
    model = load_smpl_model_data(smpl / "SMPL_NEUTRAL.pkl")
    model_faces = np.asarray(model["f"], dtype=np.int64)
    vertex_count = len(model["v_template"])
    assert (vertex_count, model_faces.shape) == (6890, (13776, 3))
    male_uv, male_faces, male = recover_uv(smpl / "meshcapade_source/SMPL_male.obj", model_faces, vertex_count)
    female_uv, female_faces, female = recover_uv(smpl / "meshcapade_source/SMPL_female.obj", model_faces, vertex_count)
    # OBJ UV tables may be ordered differently; compare actual model corners.
    male_corners, female_corners = male_uv[male_faces], female_uv[female_faces]
    assert np.allclose(male_corners, female_corners, atol=1e-6, rtol=0), "Male/female UV layout differs"
    output = smpl / "smpl_uv.npz"
    if args.write:
        with output.open("xb") as stream:
            np.savez_compressed(stream, uvs=male_uv, uv_faces=male_faces)
    result = {
        "date": "2026-10-01", "source": SOURCE, "commit": COMMIT,
        "license": "SMPL-Body CC-BY-4.0",
        "license_url": "https://smpl.is.tue.mpg.de/bodylicense.html",
        "model_vertices": vertex_count, "model_faces": list(model_faces.shape),
        "model_faces_sha256": hashlib.sha256(model_faces.tobytes()).hexdigest(),
        "male": male, "female": female,
        "male_female_corner_uv_max_difference": float(np.abs(male_corners - female_corners).max()),
        "all_uv_finite": bool(np.isfinite(male_corners).all()),
        "uv_faces_shape": list(male_faces.shape),
        "derived_changes": "Contract subdivision topology to original SMPL triangles, reorder corner UV indices to existing model f; no geometry, neutral model, images or UV-coordinate values changed",
        "render_validation": "Not performed by this converter; root agent owns Kit validation",
        "write_requested": args.write,
        "output": file_evidence(output) if output.exists() else None,
    }
    report = repo / "docs/resource_completion_20261001/smpl_uv_geometry_verification.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
