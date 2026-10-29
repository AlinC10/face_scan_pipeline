"""
Draco Compressor: Geometry Compression for GLB
==============================================
Compresses the geometry of every triangle primitive of a GLB with Google Draco
(`KHR_draco_mesh_compression`), adapted from Turntable Studio (`pipeline/exporter.py`,
`draco_compress_glb`), where it is in production.

Geometry (positions, normals, UVs, indices) is most of a textured GLB; textures are untouched.
Position quantization of 14 bits over a ~0.3 m head is ~0.02 mm; UVs at 14 bits are 1/4 texel on a
4096 atlas. Viewers need a Draco decoder (Blender, three.js DRACOLoader, model-viewer, Babylon have one).

Primitives with morph targets or skinning (JOINTS_0 / WEIGHTS_0) are left uncompressed: the encoder
reorders the vertices, and those attributes would no longer match them.
"""

from __future__ import annotations

import json
import logging
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger("Converters.Draco")

_GLTF_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GLTF_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DRACO_EXT = "KHR_draco_mesh_compression"


def read_glb(path: Union[str, Path]) -> Tuple[dict, bytes]:
    """(JSON chunk, BIN chunk) of a GLB file."""
    data = Path(path).read_bytes()
    magic, _, total = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{Path(path).name} is not a GLB")
    off, js, binc = 12, None, b""
    while off < total:
        length, kind = struct.unpack("<II", data[off:off + 8])
        chunk = data[off + 8:off + 8 + length]
        if kind == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            binc = chunk
        off += 8 + length
    return js, binc


def write_glb(path: Union[str, Path], js: dict, binc: bytes) -> None:
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    binc += b"\0" * (-len(binc) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binc)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(binc), 0x004E4942) + binc)


def draco_compress_glb(glb_path: Union[str, Path], position_bits: int = 14, texcoord_bits: int = 14,
                       normal_bits: int = 10, level: int = 7) -> Optional[Dict[str, float]]:
    """
    Compresses the geometry of every eligible triangle primitive of a GLB in place.
    Returns {"before_mb", "after_mb"} or None when DracoPy is missing or nothing was compressed.
    """
    try:
        import DracoPy
    except ImportError:
        log.warning("DracoPy not installed: GLB left without Draco compression.")
        return None
    path = Path(glb_path)
    js, binc = read_glb(path)
    before = path.stat().st_size
    views = js.get("bufferViews", [])
    accessors = js.get("accessors", [])

    def read(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = views[acc["bufferView"]]
        dtype = _GLTF_COMPONENT[acc["componentType"]]
        width = _GLTF_WIDTH[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride")
        if stride and stride != np.dtype(dtype).itemsize * width:
            raise ValueError("interleaved GLB attributes are not supported")
        return np.frombuffer(binc, dtype, acc["count"] * width, start).reshape(acc["count"], width)

    draco_blobs: List[bytes] = []
    geometry_views = set()
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if prim.get("mode", 4) != 4 or "indices" not in prim or "POSITION" not in attrs:
                continue
            if DRACO_EXT in prim.get("extensions", {}) or prim.get("targets") or "JOINTS_0" in attrs:
                continue
            if set(attrs) - {"POSITION", "NORMAL", "TEXCOORD_0"}:
                continue          # other attributes (colours, tangents) would lose their vertex order
            pos = read(attrs["POSITION"]).astype(np.float64)
            faces = read(prim["indices"]).reshape(-1, 3).astype(np.int64)
            kwargs = {}
            if "TEXCOORD_0" in attrs:
                kwargs["tex_coord"] = read(attrs["TEXCOORD_0"]).astype(np.float64)
                kwargs["tex_coord_quantization_bits"] = texcoord_bits
            if "NORMAL" in attrs:
                kwargs["normals"] = read(attrs["NORMAL"]).astype(np.float64)
                kwargs["normal_quantization_bits"] = normal_bits
            blob = DracoPy.encode(pos, faces, quantization_bits=position_bits, compression_level=level,
                                  preserve_order=False, **kwargs)
            # Attribute ids and counts as the decoder will see them
            dec = DracoPy.decode(blob)
            ids = {a["attribute_type"]: a["unique_id"] for a in dec.attributes}
            draco_attrs = {"POSITION": ids[0]}
            if "NORMAL" in attrs:
                draco_attrs["NORMAL"] = ids[1]
            if "TEXCOORD_0" in attrs:
                draco_attrs["TEXCOORD_0"] = ids[3]
            n_points = len(dec.points)
            for key, acc_idx in [(k, attrs[k]) for k in draco_attrs] + [("indices", prim["indices"])]:
                acc = accessors[acc_idx]
                geometry_views.add(acc.pop("bufferView", None))
                acc.pop("byteOffset", None)
                acc["count"] = int(len(dec.faces) * 3 if key == "indices" else n_points)
                if key == "POSITION":
                    pts = np.asarray(dec.points, dtype=np.float64)
                    acc["min"], acc["max"] = pts.min(axis=0).tolist(), pts.max(axis=0).tolist()
                if key == "indices":
                    acc["componentType"] = 5125
            prim.setdefault("extensions", {})[DRACO_EXT] = {"bufferView": None, "attributes": draco_attrs}
            draco_blobs.append(blob)
            prim["_draco_index"] = len(draco_blobs) - 1
    if not draco_blobs:
        return None

    # New binary chunk: the views still in use (textures, skipped primitives) + one view per Draco blob
    keep = [i for i in range(len(views)) if i not in geometry_views]
    remap, new_views, chunks, offset = {}, [], [], 0
    for i in keep:
        bv = views[i]
        data = binc[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        pad = -offset % 4
        chunks.append(b"\0" * pad)
        offset += pad
        remap[i] = len(new_views)
        new_views.append({k: v for k, v in bv.items() if k not in ("byteOffset", "buffer")} |
                         {"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        chunks.append(data)
        offset += len(data)
    blob_view = []
    for blob in draco_blobs:
        pad = -offset % 4
        chunks.append(b"\0" * pad)
        offset += pad
        blob_view.append(len(new_views))
        new_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        chunks.append(blob)
        offset += len(blob)
    for img in js.get("images", []):
        if "bufferView" in img:
            img["bufferView"] = remap[img["bufferView"]]
    for acc in accessors:
        if "bufferView" in acc:
            acc["bufferView"] = remap[acc["bufferView"]]
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "_draco_index" in prim:
                prim["extensions"][DRACO_EXT]["bufferView"] = blob_view[prim.pop("_draco_index")]
    js["bufferViews"] = new_views
    js["buffers"] = [{"byteLength": offset}]
    for key in ("extensionsUsed", "extensionsRequired"):
        if DRACO_EXT not in js.setdefault(key, []):
            js[key].append(DRACO_EXT)
    write_glb(path, js, b"".join(chunks))
    after = path.stat().st_size
    log.info("Draco: %s %.1f MB -> %.1f MB", path.name, before / 1e6, after / 1e6)
    return {"before_mb": round(before / 1e6, 2), "after_mb": round(after / 1e6, 2)}


def apply_draco_compression(
    input_glb_path: Union[str, Path],
    output_glb_path: Union[str, Path],
    compression_level: int = 7,
) -> Path:
    """
    Draco-compresses `input_glb_path` into `output_glb_path` (the same path compresses in place).
    On any failure the uncompressed GLB is kept, so the deliverable is always valid.
    """
    in_path = Path(input_glb_path).resolve()
    out_path = Path(output_glb_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.is_file():
        raise FileNotFoundError(f"Source GLB not found for Draco compression: {in_path}")
    if in_path != out_path:
        shutil.copy2(in_path, out_path)
    original = out_path.read_bytes()
    try:
        draco_compress_glb(out_path, level=compression_level)
    except Exception as e:
        log.warning("Draco compression failed (%s); keeping the uncompressed GLB.", e)
        out_path.write_bytes(original)
    return out_path
