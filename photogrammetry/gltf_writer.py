"""
Minimal glTF 2.0 (GLB) writer with skinning and morph targets
=============================================================
trimesh exports neither morph targets nor skins, so the animatable head (plan P2.7) is
written here: one mesh node with several primitives (skin with its texture and normal map,
eyes, teeth, tongue with flat materials), a skeleton (joint nodes + inverse bind matrices,
JOINTS_0 / WEIGHTS_0) and morph targets (POSITION deltas) named in mesh.extras.targetNames,
which Blender and three.js read as shape keys / morphTargetDictionary.

The GLB is not Draco-compressed: the encoder reorders vertices, which breaks the per-vertex
morph targets and skin weights (converters.draco_compressor skips such primitives anyway).
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass
class GLTFMaterial:
    name: str
    base_color: Sequence[float] = (1.0, 1.0, 1.0, 1.0)
    base_color_png: Optional[np.ndarray] = None      # RGB uint8
    normal_png: Optional[np.ndarray] = None          # RGB uint8 (tangent space, OpenGL)
    roughness: float = 0.6
    metallic: float = 0.0
    double_sided: bool = True


@dataclass
class GLTFPrimitive:
    positions: np.ndarray                            # (N, 3) float
    faces: np.ndarray                                # (F, 3) int
    material: int
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None                 # (N, 2), v up (converted to glTF's v down)
    joints: Optional[np.ndarray] = None              # (N, 4) uint16
    weights: Optional[np.ndarray] = None             # (N, 4) float
    morph_deltas: List[np.ndarray] = field(default_factory=list)   # each (N, 3)


@dataclass
class GLTFSkeleton:
    names: List[str]
    parents: Sequence[int]                           # -1 for the root
    positions: np.ndarray                            # (J, 3) world positions of the joints in the bind pose


class _Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views: List[dict] = []
        self.accessors: List[dict] = []

    def _view(self, raw: bytes, target: Optional[int]) -> int:
        self.data += b"\0" * (-len(self.data) % 4)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            view["target"] = target
        self.data += raw
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, ctype: int, kind: str, target: Optional[int] = ARRAY_BUFFER,
                 minmax: bool = False, normalized: bool = False) -> int:
        dtype = {FLOAT: np.float32, UBYTE: np.uint8, USHORT: np.uint16, UINT: np.uint32}[ctype]
        a = np.ascontiguousarray(arr, dtype=dtype)
        acc = {"bufferView": self._view(a.tobytes(), target), "componentType": ctype,
               "count": int(a.shape[0]), "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            flat = a.reshape(a.shape[0], -1)
            acc["min"] = flat.min(axis=0).tolist()
            acc["max"] = flat.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def image(self, rgb: np.ndarray) -> int:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return self._view(buf.getvalue(), None)


def write_glb(path: Path, primitives: List[GLTFPrimitive], materials: List[GLTFMaterial],
              skeleton: Optional[GLTFSkeleton] = None, target_names: Sequence[str] = (), name: str = "head") -> Path:
    b = _Buffer()
    js: Dict = {"asset": {"version": "2.0", "generator": "mediapipe_script face pipeline"},
                "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
                "textures": [], "images": [], "samplers": [{"magFilter": 9729, "minFilter": 9987}]}

    for m in materials:
        mat = {"name": m.name, "doubleSided": m.double_sided,
               "pbrMetallicRoughness": {"baseColorFactor": list(m.base_color), "metallicFactor": m.metallic,
                                        "roughnessFactor": m.roughness}}
        for key, img in (("baseColorTexture", m.base_color_png), ("normalTexture", m.normal_png)):
            if img is None:
                continue
            js["images"].append({"bufferView": b.image(img), "mimeType": "image/png"})
            js["textures"].append({"sampler": 0, "source": len(js["images"]) - 1})
            ref = {"index": len(js["textures"]) - 1}
            if key == "baseColorTexture":
                mat["pbrMetallicRoughness"][key] = ref
            else:
                mat[key] = ref
        js["materials"].append(mat)

    prims = []
    for p in primitives:
        attrs = {"POSITION": b.accessor(p.positions, FLOAT, "VEC3", minmax=True)}
        if p.normals is not None:
            attrs["NORMAL"] = b.accessor(p.normals / np.maximum(np.linalg.norm(p.normals, axis=1, keepdims=True), 1e-12),
                                         FLOAT, "VEC3")
        if p.uvs is not None:
            uv = np.asarray(p.uvs, dtype=np.float64).copy()
            uv[:, 1] = 1.0 - uv[:, 1]                    # glTF: v grows downwards
            attrs["TEXCOORD_0"] = b.accessor(uv, FLOAT, "VEC2")
        if p.joints is not None:
            w = np.asarray(p.weights, dtype=np.float64)
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
            attrs["JOINTS_0"] = b.accessor(p.joints, USHORT, "VEC4")
            attrs["WEIGHTS_0"] = b.accessor(w, FLOAT, "VEC4")
        prim = {"attributes": attrs, "material": p.material, "mode": 4,
                "indices": b.accessor(np.asarray(p.faces).reshape(-1), UINT, "SCALAR", target=ELEMENT_ARRAY_BUFFER)}
        if p.morph_deltas:
            prim["targets"] = [{"POSITION": b.accessor(d, FLOAT, "VEC3", minmax=True)} for d in p.morph_deltas]
        prims.append(prim)
    n_targets = {len(p.morph_deltas) for p in primitives}
    if len(n_targets) > 1:
        raise ValueError("every primitive of a glTF mesh must have the same number of morph targets")
    mesh = {"name": name, "primitives": prims}
    if prims and prims[0].get("targets"):
        mesh["weights"] = [0.0] * len(prims[0]["targets"])
        mesh["extras"] = {"targetNames": list(target_names)}
    js["meshes"].append(mesh)

    mesh_node = {"name": name, "mesh": 0}
    if skeleton is None:
        js["nodes"].append(mesh_node)
        js["scenes"][0]["nodes"].append(0)
    else:
        J = len(skeleton.names)
        first = 0
        for j in range(J):
            par = skeleton.parents[j]
            local = skeleton.positions[j] - (skeleton.positions[par] if par >= 0 else 0.0)
            js["nodes"].append({"name": skeleton.names[j], "translation": [float(v) for v in local]})
        for j in range(J):
            kids = [k for k in range(J) if skeleton.parents[k] == j]
            if kids:
                js["nodes"][first + j]["children"] = [first + k for k in kids]
        ibm = np.zeros((J, 4, 4), np.float32)
        for j in range(J):
            m = np.eye(4)
            m[:3, 3] = -skeleton.positions[j]
            ibm[j] = m.T                                     # glTF matrices are column-major
        js["skins"] = [{"joints": list(range(first, first + J)), "skeleton": first,
                        "inverseBindMatrices": b.accessor(ibm.reshape(J, 16), FLOAT, "MAT4", target=None)}]
        mesh_node["skin"] = 0
        js["nodes"].append(mesh_node)
        roots = [first + j for j in range(J) if skeleton.parents[j] < 0]
        js["scenes"][0]["nodes"] = roots + [len(js["nodes"]) - 1]

    for key in ("textures", "images"):
        if not js[key]:
            del js[key]
    if "textures" not in js:
        del js["samplers"]
    js["bufferViews"] = b.views
    js["accessors"] = b.accessors
    js["buffers"] = [{"byteLength": len(b.data) + (-len(b.data) % 4)}]
    data = bytes(b.data) + b"\0" * (-len(b.data) % 4)
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(data)))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(data), 0x004E4942) + data)
    return path
