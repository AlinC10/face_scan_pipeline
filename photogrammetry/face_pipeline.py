"""
Dense Two-View Face Reconstruction (Face Mode)
==============================================
End-to-end reconstruction of a metric, textured facial mesh from one
synchronised 12.3 MP stereo pair of the rig:

  1. Rig calibration (assets/rig_calibration.json, measured on the AprilTag disc).
  2. MediaPipe landmarks in both views (initialisation only).
  3. MediaPipe-guided dense matches -> refinement of the relative pose
     (rotation + baseline direction; the baseline length keeps the metric scale).
  4. Bouguet rectification at half resolution.
  5. Prior-guided disparity from both reference views (DIS flow, forward-backward and
     vertical checks, MediaPipe 3-D shape prior fitted to the measurement).
  6. Fusion into a height field in the head frame -> high-polygon grid mesh.
  7. QEM decimation -> low-polygon mesh; xatlas UV atlases for both.
  8. Two-view, visibility-aware, colour-balanced albedo from the original photographs;
     tangent-space normal map of the high mesh baked onto the low atlas.
  9. Deliverables: OBJ/GLB in metres, closed STL/3MF in millimetres.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from converters import convert_to_glb_draco, convert_to_stl, convert_to_3mf
from photogrammetry.face_correspondence import (
    detect_landmarks,
    estimate_disparity,
    guided_raw_matches,
    rectify_pair,
)
from photogrammetry.face_surface import fuse_face_surface, solidify_height_field
from photogrammetry.face_texture import (
    TextureCamera,
    bake_albedo,
    bake_normal_map,
    flat_normal_map,
    unwrap_uv,
)
from photogrammetry.rig_calibration import epipolar_errors_px, load_rig_calibration, refine_relative_pose

log = logging.getLogger("Photogrammetry.FacePipeline")


@dataclass
class FacePipelineConfig:
    calibration_path: Optional[Union[str, Path]] = None
    work_scale: float = 0.5               # rectified working resolution (0.5 -> 2028 x 1520)
    grid_resolution_mm: float = 0.5       # height-field sampling = high-polygon vertex spacing
    high_max_faces: int = 200_000
    low_faces: int = 20_000
    high_texture_size: int = 4096
    low_texture_size: int = 2048
    refine_pose: bool = True
    save_debug: bool = True
    name: str = "model"


@dataclass
class FacePipelineResult:
    deliverables: Dict[str, Path]
    manifest_path: Path
    manifest: Dict[str, Any] = field(default_factory=dict)


def reconstruct_face_pair(
    left_path: Union[str, Path],
    right_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[FacePipelineConfig] = None,
) -> FacePipelineResult:
    cfg = config or FacePipelineConfig()
    t_start = time.time()
    timings: Dict[str, float] = {}
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if cfg.save_debug else None
    pair_id = Path(left_path).stem

    def lap(name: str, t0: float) -> float:
        timings[name] = round(time.time() - t0, 2)
        return time.time()

    t = time.time()
    img0 = cv2.imread(str(left_path))
    img1 = cv2.imread(str(right_path))
    if img0 is None or img1 is None:
        raise FileNotFoundError(f"Cannot read the stereo pair: {left_path}, {right_path}")
    calib = load_rig_calibration(cfg.calibration_path)
    if (img0.shape[1], img0.shape[0]) != tuple(calib.image_size):
        calib = calib.scaled(img0.shape[1] / calib.image_size[0])

    # --- Landmarks and relative pose
    lm0, lm1 = detect_landmarks([img0, img1])
    if lm0 is None or lm1 is None:
        raise RuntimeError("MediaPipe did not find a face in both views (camera 0: %s, camera 1: %s)"
                           % (lm0 is not None, lm1 is not None))
    pts0, pts1 = guided_raw_matches(img0, img1, lm0, lm1)
    if cfg.refine_pose:
        calib = refine_relative_pose(calib, pts0, pts1)
    epi = epipolar_errors_px(calib, pts0, pts1)
    t = lap("calibration_and_matching_s", t)

    # --- Rectification and disparity from both reference views
    rect = rectify_pair(img0, img1, calib, cfg.work_scale)
    l0 = rect.rectify_points(lm0, 0)
    l1 = rect.rectify_points(lm1, 1)
    estimates = [
        estimate_disparity(rect, l0, l1, lm0[:, 2], reference="left"),
        estimate_disparity(rect, l1, l0, lm1[:, 2], reference="right"),
    ]
    t = lap("dense_disparity_s", t)

    # --- Surface
    surface = fuse_face_surface(rect, estimates, l0, grid_resolution_mm=cfg.grid_resolution_mm, debug_dir=debug_dir)
    high = _clean(surface.mesh)
    if len(high.faces) > cfg.high_max_faces:
        high = _decimate(high, cfg.high_max_faces)
    low = _decimate(high, cfg.low_faces)
    t = lap("surface_s", t)

    # --- Texture
    cameras = _texture_cameras(img0, img1, calib, rect, surface.head_to_rect)
    high_uv = unwrap_uv(high, cfg.high_texture_size)
    low_uv = unwrap_uv(low, cfg.low_texture_size)
    tex_high, tex_stats = bake_albedo(high_uv, cameras, size=cfg.high_texture_size)
    tex_low, _ = bake_albedo(low_uv, cameras, size=cfg.low_texture_size)
    normal_low = bake_normal_map(low_uv, high, size=cfg.low_texture_size)
    normal_high = flat_normal_map(cfg.high_texture_size)
    t = lap("texture_s", t)

    # --- Deliverables
    deliverables = export_face_deliverables(out_dir, cfg.name, high_uv, low_uv, tex_high, tex_low,
                                            normal_high, normal_low)
    t = lap("export_s", t)

    if debug_dir is not None:
        _save_debug_images(debug_dir, pair_id, rect, estimates)

    bbox = high.extents
    manifest = {
        "pair_id": pair_id,
        "reconstruction_mode": "dense_two_view",
        "inputs": {"left": str(left_path), "right": str(right_path)},
        "calibration": calib.to_dict(),
        "matching": {
            "guided_correspondences": int(len(pts0)),
            "epipolar_error_median_px": round(float(np.median(epi)), 3),
            "epipolar_error_p90_px": round(float(np.percentile(epi, 90)), 3),
        },
        "rectification": {
            "work_scale": cfg.work_scale,
            "focal_px": round(rect.focal_px, 2),
            "baseline_mm": round(calib.baseline_mm, 3),
        },
        "disparity": {
            e.reference: {
                "valid_fraction": round(float(e.valid.sum() / max(e.face_mask.sum(), 1)), 3),
                "shape_prior_fit": e.shape_fit,
            } for e in estimates
        },
        "surface": surface.stats,
        "meshes": {
            "high_faces": int(len(high.faces)), "high_vertices": int(len(high.vertices)),
            "low_faces": int(len(low.faces)), "low_vertices": int(len(low.vertices)),
            "bbox_mm": [round(float(v), 1) for v in bbox],
        },
        "texture": tex_stats,
        "timings": timings,
        "duration_s": round(time.time() - t_start, 2),
        "deliverables": {k: str(v) for k, v in deliverables.items()},
    }
    manifest_path = out_dir / "photogrammetry_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Face reconstruction finished in %.1f s: high %d faces, low %d faces, bbox %.0f x %.0f x %.0f mm",
             manifest["duration_s"], len(high.faces), len(low.faces), *bbox)
    return FacePipelineResult(deliverables, manifest_path, manifest)


# ---------------------------------------------------------------------------------------------

def _clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.remove_unreferenced_vertices()
    m.fix_normals()
    # Face normals towards the viewer (+z of the head frame).
    if np.mean(m.face_normals[:, 2]) < 0:
        m.invert()
    return m


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric error decimation (Open3D) with a strong boundary weight, so the outline stays put."""
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    om = om.simplify_quadric_decimation(target_number_of_triangles=int(target_faces), boundary_weight=10.0)
    om.remove_degenerate_triangles()
    om.remove_duplicated_triangles()
    om.remove_unreferenced_vertices()
    om.remove_non_manifold_edges()
    return _clean(trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=True))


def _texture_cameras(img0, img1, calib, rect, head_to_rect) -> list:
    """Both photographs posed in the head frame (mm): head -> rectified left -> camera 0 -> camera 1."""
    A, a = head_to_rect[:3, :3], head_to_rect[:3, 3]
    R0 = rect.R1.T @ A
    t0 = rect.R1.T @ a
    R1 = calib.R @ R0
    t1 = calib.R @ t0 + calib.T_mm
    return [
        TextureCamera(img0, calib.K0, calib.D0, R0, t0),
        TextureCamera(img1, calib.K1, calib.D1, R1, t1),
    ]


def _write_mtl(path: Path, obj_name: str, texture_name: str, normal_name: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# Material for {obj_name}\n"
            f"newmtl material_0\n"
            f"Ka 1.000000 1.000000 1.000000\n"
            f"Kd 1.000000 1.000000 1.000000\n"
            f"Ks 0.050000 0.050000 0.050000\n"
            f"Ns 10.000000\n"
            f"map_Kd {texture_name}\n"
            f"norm {normal_name}\n"
            f"map_Bump -bm 1.0 {normal_name}\n"
        )


def _textured(mesh_mm: trimesh.Trimesh, tex_bgr: np.ndarray, normal_bgr: np.ndarray) -> trimesh.Trimesh:
    """Copy in metres with a PBR material (OBJ / GLB deliverables)."""
    m = trimesh.Trimesh(mesh_mm.vertices * 1e-3, mesh_mm.faces, process=False)
    material = trimesh.visual.material.PBRMaterial(
        name="material_0",
        baseColorTexture=Image.fromarray(cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)),
        normalTexture=Image.fromarray(cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)),
        roughnessFactor=0.6,
        metallicFactor=0.0,
        doubleSided=True,
    )
    m.visual = trimesh.visual.TextureVisuals(uv=mesh_mm.visual.uv, material=material)
    return m


def export_face_deliverables(
    out_dir: Path,
    name: str,
    high_uv: trimesh.Trimesh,
    low_uv: trimesh.Trimesh,
    tex_high: np.ndarray,
    tex_low: np.ndarray,
    normal_high: np.ndarray,
    normal_low: np.ndarray,
) -> Dict[str, Path]:
    """The 8 standard deliverables (+ textures) in output/{obj/high, obj/low, glb, stl, 3mf}."""
    base = name or "model"
    paths: Dict[str, Path] = {}
    for level, mesh_uv, tex, nrm, tex_ext in (
        ("high", high_uv, tex_high, normal_high, "png"),
        ("low", low_uv, tex_low, normal_low, "jpg"),
    ):
        stem = f"{level}_{base}"
        obj_dir = out_dir / "obj" / level
        obj_dir.mkdir(parents=True, exist_ok=True)
        tex_path = obj_dir / f"{stem}_texture.{tex_ext}"
        nrm_path = obj_dir / f"{stem}_normal.png"
        if tex_ext == "jpg":
            cv2.imwrite(str(tex_path), tex, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            cv2.imwrite(str(tex_path), tex)
        cv2.imwrite(str(nrm_path), nrm)

        textured = _textured(mesh_uv, tex, nrm)
        obj_text = trimesh.exchange.obj.export_obj(
            textured, include_normals=True, include_texture=True, write_texture=False, mtl_name="material.mtl",
        )
        obj_path = obj_dir / f"{stem}.obj"
        obj_path.write_text(obj_text, encoding="utf-8")
        _write_mtl(obj_dir / "material.mtl", obj_path.name, tex_path.name, nrm_path.name)

        glb_path = out_dir / "glb" / f"{stem}.glb"
        convert_to_glb_draco(textured, glb_path)

        solid = solidify_height_field(trimesh.Trimesh(mesh_uv.vertices, mesh_uv.faces, process=True))
        stl_path = convert_to_stl(solid, out_dir / "stl" / f"{stem}.stl", enforce_watertight=False)
        thrf_path = convert_to_3mf(solid, out_dir / "3mf" / f"{stem}.3mf")
        paths.update({
            f"obj_{level}": obj_path, f"texture_{level}": tex_path, f"normal_{level}": nrm_path,
            f"glb_{level}": glb_path, f"stl_{level}": stl_path, f"3mf_{level}": thrf_path,
        })
        log.info("Exported %s level: %d faces (STL/3MF watertight: %s)", level, len(mesh_uv.faces), solid.is_watertight)
    return paths


def _save_debug_images(folder: Path, pair_id: str, rect, estimates) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    h, w = rect.left.shape[:2]
    vis = np.hstack([rect.left, rect.right])
    for y in range(0, h, h // 24):
        cv2.line(vis, (0, y), (2 * w, y), (0, 255, 0), 1)
    cv2.imwrite(str(folder / f"{pair_id}_epipolar_check.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5))
    tiles = []
    for e in estimates:
        d = e.disparity
        m = e.face_mask
        lo, hi = np.percentile(e.smooth[m], [1, 99])
        img = cv2.applyColorMap(np.clip((d - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
        img[~e.valid] = (img[~e.valid] * 0.25).astype(np.uint8)
        img[~m] = 0
        ys, xs = np.nonzero(m)
        tiles.append(cv2.resize(img[ys.min():ys.max(), xs.min():xs.max()], (600, 700)))
    cv2.imwrite(str(folder / f"{pair_id}_disparity_left_right.jpg"), np.hstack(tiles))
