#!/usr/bin/env python3
"""
render_preview.py — Headless preview renders of a textured OBJ (no Blender, no GPU): flat texture colour
(like Blender's Material Preview without lights) from a few viewpoints, for comparing runs side by side.

  python tools/render_preview.py output/obj/high/high_model.obj -o output/debug/preview.jpg
  python tools/render_preview.py run_a/obj/high/high_model.obj run_b/obj/high/high_model.obj -o compare.jpg

Each input becomes one row: front, 35 degrees left, 35 degrees right, from below (nose / chin).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photogrammetry.face_texture import rasterize_mesh  # noqa: E402

VIEWS = {"front": (0.0, 0.0), "left 35": (35.0, 0.0), "right 35": (-35.0, 0.0), "below": (0.0, -40.0)}
BACK_VIEWS = {"left 110": (110.0, 0.0), "back": (180.0, 0.0), "right 110": (-110.0, 0.0), "above": (0.0, 60.0)}


def load_textured(path: Path):
    scene = trimesh.load(path, process=False)
    mesh = scene if isinstance(scene, trimesh.Trimesh) else trimesh.util.concatenate(list(scene.geometry.values()))
    mat = mesh.visual.material
    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    tex = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return mesh, np.asarray(mesh.visual.uv, dtype=np.float64), tex


def render(mesh, uv, tex, yaw_deg, pitch_deg, size=600) -> np.ndarray:
    V = mesh.vertices - mesh.bounds.mean(axis=0)
    R = (trimesh.transformations.rotation_matrix(np.radians(pitch_deg), [1, 0, 0])[:3, :3]
         @ trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0])[:3, :3])
    P = V @ R.T
    s = 0.9 * size / max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    px = np.stack([P[:, 0] * s + size / 2, -P[:, 1] * s + size / 2], axis=1)
    F = mesh.faces
    fid, bc = rasterize_mesh(px[F], (size, size), tri_depth=-P[F, 2])     # camera looks along -z
    out = np.full((size, size, 3), 60, np.uint8)
    ok = fid >= 0
    t_uv = (uv[F[fid[ok]]] * bc[ok][..., None]).sum(axis=1)
    h, w = tex.shape[:2]
    mx = (t_uv[:, 0] * (w - 1)).astype(np.float32)
    my = ((1.0 - t_uv[:, 1]) * (h - 1)).astype(np.float32)
    img = np.zeros((size, size, 3), np.uint8)
    map_x = np.zeros((size, size), np.float32)
    map_y = np.zeros((size, size), np.float32)
    map_x[ok], map_y[ok] = mx, my
    img = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR)
    out[ok] = img[ok]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("objs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--back", action="store_true", help="whole-head views: sides, back, above")
    args = ap.parse_args(argv)
    rows = []
    for obj in args.objs:
        mesh, uv, tex = load_textured(Path(obj))
        tiles = []
        for name, (yaw, pitch) in (BACK_VIEWS if args.back else VIEWS).items():
            img = render(mesh, uv, tex, yaw, pitch, args.size)
            cv2.putText(img, f"{name}  {Path(obj).parent.parent.parent.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows))
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
