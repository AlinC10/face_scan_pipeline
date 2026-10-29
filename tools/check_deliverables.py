#!/usr/bin/env python3
"""
check_deliverables.py — Acceptance checks of a face job's output folder (local tool, not in the image).

  python tools/check_deliverables.py output/                 # prints a report, exit 1 when a check fails
  python tools/check_deliverables.py output/ --json report.json

Checks, for high and low:
  * OBJ / GLB / STL / 3MF exist and have the same size in export units (GLB decoded with DracoPy when
    Draco-compressed; STL / 3MF are in the print frame, Z up, so their axes are permuted);
  * the size matches object.json (mm x export_scale / 1000);
  * STL / 3MF are watertight;
  * the GLB is valid glTF (one mesh, every accessor inside the binary chunk) and how much Draco saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from converters.draco_compressor import DRACO_EXT, read_glb  # noqa: E402

TOL = 0.01      # relative size tolerance between formats


def glb_vertices(path: Path) -> tuple[np.ndarray, dict]:
    """Positions of every primitive of a GLB (Draco or not) and a validity summary."""
    js, binc = read_glb(path)
    info = {"meshes": len(js.get("meshes", [])), "nodes": len(js.get("nodes", [])),
            "draco": DRACO_EXT in js.get("extensionsUsed", []), "morph_targets": 0, "skins": len(js.get("skins", []))}
    views = js.get("bufferViews", [])
    for bv in views:
        if bv.get("byteOffset", 0) + bv["byteLength"] > len(binc):
            raise ValueError(f"{path.name}: bufferView outside the binary chunk")
    pts = []
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            info["morph_targets"] = max(info["morph_targets"], len(prim.get("targets", [])))
            ext = prim.get("extensions", {}).get(DRACO_EXT)
            if ext:
                import DracoPy
                bv = views[ext["bufferView"]]
                blob = binc[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
                pts.append(np.asarray(DracoPy.decode(blob).points, dtype=np.float64))
            else:
                acc = js["accessors"][prim["attributes"]["POSITION"]]
                bv = views[acc["bufferView"]]
                start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
                pts.append(np.frombuffer(binc, np.float32, acc["count"] * 3, start).reshape(-1, 3).astype(np.float64))
    return np.vstack(pts), info


def check(out_dir: Path) -> dict:
    report = {"ok": True, "checks": []}

    def add(name, ok, detail):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        report["ok"] &= bool(ok)

    obj_info = None
    oj = out_dir / "object.json"
    if oj.is_file():
        obj_info = json.loads(oj.read_text(encoding="utf-8"))
    add("object.json", obj_info is not None, str(oj))
    export_scale = (obj_info or {}).get("scale", {}).get("export_scale", 100.0)
    man = out_dir / "photogrammetry_manifest.json"
    head_mode = man.is_file() and bool(json.loads(man.read_text(encoding="utf-8")).get("gnm", {}).get("surface_rms_mm"))

    for level in ("high", "low"):
        sizes = {}
        obj = next((out_dir / "obj" / level).glob("*.obj"), None)
        glb = next((out_dir / "glb").glob(f"{level}_*.glb"), None)
        stl = next((out_dir / "stl").glob(f"{level}_*.stl"), None)
        thrf = next((out_dir / "3mf").glob(f"{level}_*.3mf"), None)
        for kind, path in (("obj", obj), ("glb", glb), ("stl", stl), ("3mf", thrf)):
            add(f"{level}.{kind} exists", path is not None and path.is_file(), str(path))
        if obj:
            m = trimesh.load(obj, force="mesh", process=False)
            sizes["obj"] = np.sort(m.extents)
        if glb:
            v, info = glb_vertices(glb)
            sizes["glb"] = np.sort(v.max(0) - v.min(0))
            add(f"{level}.glb single mesh", info["meshes"] == 1, info)
            report[f"{level}_glb"] = info | {"size_mb": round(glb.stat().st_size / 1e6, 2)}
        for kind, path in (("stl", stl), ("3mf", thrf)):
            if path:
                m = trimesh.load(path, force="mesh")
                sizes[kind] = np.sort(m.extents)
                add(f"{level}.{kind} watertight", m.is_watertight, f"{len(m.faces)} faces")
                if kind == "stl":
                    add(f"{level}.stl upright (base on Z = 0)", abs(m.bounds[0, 2]) < 1e-3 * m.extents.max(),
                        f"z min {m.bounds[0, 2]:.4f}")
        # Face surface (OBJ / GLB) vs solid (STL / 3MF): the face solid adds a back plate (compare the two largest
        # sizes); the whole-head solid is a bust cut below the chin (smaller, but the same order: units check).
        ref = sizes.get("obj")
        if ref is not None:
            for kind, sz in sizes.items():
                if head_mode and kind in ("stl", "3mf"):
                    ok = (sz.max() <= ref.max() * (1 + TOL)) and (sz.max() >= 0.5 * ref.max())
                else:
                    ok = (np.abs(sz[1:] - ref[1:]) / ref[1:]).max() < TOL
                add(f"{level}.{kind} same units as OBJ", ok, f"{np.round(sz, 3).tolist()} vs OBJ {np.round(ref, 3).tolist()}")
            if obj_info and level == "high":
                d = obj_info["dimensions_mm"]
                mm = np.sort([d["width"], d["height"], d["depth"]]) * export_scale / 1000.0
                rel = np.abs(mm - ref) / ref
                add("high OBJ size = object.json x export_scale", rel.max() < TOL,
                    f"{np.round(ref, 3).tolist()} vs {np.round(mm, 3).tolist()}")
        report[f"{level}_sizes"] = {k: np.round(v, 4).tolist() for k, v in sizes.items()}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rep = check(Path(args.out_dir))
    for c in rep["checks"]:
        print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    for key in ("high_glb", "low_glb"):
        if key in rep:
            print(f"{key}: {rep[key]}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("ALL OK" if rep["ok"] else "SOME CHECKS FAILED")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
