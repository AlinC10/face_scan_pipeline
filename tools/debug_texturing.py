#!/usr/bin/env python3
"""
debug_texturing.py — Texture diagnostics of a face job (local tool, not in the image).

The pipeline writes, with save_debug (default), to <output>/debug/:
  texture_visibility_r_cam0_g_cam1.jpg   visibility per camera on the high atlas (yellow = both)
  texture_weight_share_cam0.jpg          share of camera 0 in the blend (cyan = cam 1, magenta = cam 0)
  texture_colour_diff_dE_over_20.jpg     colour difference between the cameras after balancing (dE / 20)
This tool prints the texture statistics of the manifest, checks that the maps exist and renders
preview images of the textured high mesh (tools/render_preview.py).

  python tools/debug_texturing.py output/
  python tools/debug_texturing.py run_a/ run_b/        # previews of both, one row each
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import render_preview  # noqa: E402

MAPS = ("texture_visibility_r_cam0_g_cam1.jpg", "texture_weight_share_cam0.jpg", "texture_colour_diff_dE_over_20.jpg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--preview", default=None, help="preview image (default <first out_dir>/debug/texture_preview.jpg)")
    args = ap.parse_args(argv)
    ok = True
    objs = []
    for d in map(Path, args.out_dirs):
        man = json.loads((d / "photogrammetry_manifest.json").read_text(encoding="utf-8"))
        tex = man.get("texture", {})
        print(f"== {d}")
        for key in ("camera_delta_e_median", "camera_delta_e_p90", "colour_gain_cam1_to_cam0_bgr",
                    "texels_seen_fraction", "texels_seen_by_both_fraction"):
            print(f"   {key}: {tex.get(key)}")
        for name in MAPS:
            present = (d / "debug" / name).is_file()
            ok &= present
            print(f"   [{'OK' if present else 'MISSING'}] debug/{name}")
        obj = d / "obj" / "high" / "high_model.obj"
        if obj.is_file():
            objs.append(str(obj))
    if objs:
        out = args.preview or str(Path(args.out_dirs[0]) / "debug" / "texture_preview.jpg")
        render_preview.main(objs + ["-o", out, "--size", "520"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
