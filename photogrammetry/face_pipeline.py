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
  9. Deliverables: OBJ / GLB / STL / 3MF, all in the same units (Turntable Studio convention):
     coordinate = metres x export_scale (100 -> centimetres); the tag-disc print scale (tag_scale)
     corrects the calibrated baseline once, so object.json reports real millimetres.
     STL / 3MF are closed solids standing upright (Z up, base on Z = 0); GLB is Draco-compressed.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import ExitStack
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
from photogrammetry.face_segmentation import inpaint_frames, segment_faces
from photogrammetry.gnm_assembly import assemble_head, fit_gnm, head_solid, split_by_corner_uv, write_animated_glb
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
    spectacles: bool = True               # mask spectacle frames, no measured detail behind the lenses (False: subject without glasses)
    tag_scale: float = 0.98               # print scale of the AprilTag disc (measured / designed size; this rig: ~98 %)
    export_scale: float = 100.0           # metres -> export units, in every format (100: centimetres, as Turntable Studio)
    glb_draco: bool = True                # Draco-compress the GLB geometry
    segment_face: bool = True             # face region from the multiclass segmenter (hairline, lateral cheeks)
    remove_frames: Optional[bool] = None  # inpaint spectacle frames out of the photographs (None: = spectacles)
    head_model: str = "gnm"               # "gnm": whole head (GNM Head v3 fitted + the measured detail); "none": face only
    expression: str = "capture"           # with GNM: "capture" (as photographed) or "neutral" (identity kept)
    max_gnm_rms_mm: float = 2.0           # GNM fit worse than this -> face-only deliverables (logged)
    save_debug: bool = True
    name: str = "model"
    calibration_info: Dict[str, Any] = field(default_factory=dict)   # how the job's calibration was obtained


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
    telemetry=None,
) -> FacePipelineResult:
    """
    One face reconstruction. With `telemetry` (photogrammetry.telemetry.SystemTelemetry) every stage is a
    measured step (time, peak RAM, CPU, disk) in stats.json.
    """
    cfg = config or FacePipelineConfig()
    steps = _StepChain(telemetry)
    try:
        return _reconstruct(left_path, right_path, output_dir, cfg, steps)
    finally:
        steps.close()


class _StepChain:
    """Consecutive telemetry steps: next() closes the running step and opens a new one (no-op without telemetry)."""

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.stack = ExitStack()

    def next(self, name: str) -> None:
        if self.telemetry is not None:
            self.stack.close()
            self.stack = ExitStack()
            self.stack.enter_context(self.telemetry.step(name))

    def close(self) -> None:
        self.stack.close()


def _reconstruct(left_path, right_path, output_dir, cfg: FacePipelineConfig, steps: _StepChain) -> FacePipelineResult:
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
    steps.next("Face 1: landmarks, guided matching, pose refinement")
    img0 = cv2.imread(str(left_path))
    img1 = cv2.imread(str(right_path))
    if img0 is None or img1 is None:
        raise FileNotFoundError(f"Cannot read the stereo pair: {left_path}, {right_path}")
    calib = load_rig_calibration(cfg.calibration_path)
    if abs(calib.tag_scale - cfg.tag_scale) > 1e-9:
        log.info("Tag-disc print scale %.4f (calibration made with %.4f): baseline %.2f -> %.2f mm",
                 cfg.tag_scale, calib.tag_scale, calib.baseline_mm, calib.baseline_mm * cfg.tag_scale / calib.tag_scale)
        calib = calib.with_tag_scale(cfg.tag_scale)
    if (img0.shape[1], img0.shape[0]) != tuple(calib.image_size):
        calib = calib.scaled(img0.shape[1] / calib.image_size[0])

    # --- Landmarks and relative pose
    lm0, lm1 = detect_landmarks([img0, img1])
    if lm0 is None or lm1 is None:
        raise RuntimeError("MediaPipe did not find a face in both views (camera 0: %s, camera 1: %s)"
                           % (lm0 is not None, lm1 is not None))
    segs = segment_faces([img0, img1], nose_px=[lm0[1, :2], lm1[1, :2]]) if cfg.segment_face else [None, None]
    pts0, pts1 = guided_raw_matches(img0, img1, lm0, lm1)
    if cfg.refine_pose:
        calib = refine_relative_pose(calib, pts0, pts1)
    epi = epipolar_errors_px(calib, pts0, pts1)
    t = lap("calibration_and_matching_s", t)

    # --- Rectification and disparity from both reference views
    steps.next("Face 2: rectification and dense disparity")
    rect = rectify_pair(img0, img1, calib, cfg.work_scale)
    l0 = rect.rectify_points(lm0, 0)
    l1 = rect.rectify_points(lm1, 1)
    regions = [rect.rectify_mask(sg.face, cam) if sg is not None else None for cam, sg in enumerate(segs)]
    estimates = [
        estimate_disparity(rect, l0, l1, lm0[:, 2], reference="left", mask_thin_occluders=cfg.spectacles,
                           region_mask=regions[0]),
        estimate_disparity(rect, l1, l0, lm1[:, 2], reference="right", mask_thin_occluders=cfg.spectacles,
                           region_mask=regions[1]),
    ]
    t = lap("dense_disparity_s", t)

    # --- Surface
    steps.next("Face 3: surface fusion and decimation")
    surface = fuse_face_surface(rect, estimates, l0, grid_resolution_mm=cfg.grid_resolution_mm,
                                spectacles=cfg.spectacles, debug_dir=debug_dir)
    head = None
    if cfg.head_model == "gnm":
        head = _gnm_head(surface, cfg)
    if head is not None:
        high = _clean(head.high)
        low = head.low
    else:
        high = _clean(surface.mesh)
        if len(high.faces) > cfg.high_max_faces:
            high = _decimate(high, cfg.high_max_faces)
        low = _decimate(high, cfg.low_faces)
    t = lap("surface_s", t)

    # --- Texture
    steps.next("Face 4: UV atlases and texture baking")
    remove_frames = cfg.spectacles if cfg.remove_frames is None else cfg.remove_frames
    tex0, tex1 = img0, img1
    if remove_frames and segs[0] is not None:
        tex0 = inpaint_frames(img0, segs[0].frames)
        tex1 = inpaint_frames(img1, segs[1].frames)
        log.info("Spectacle frames painted out of the photographs before texturing")
    cameras = _texture_cameras(tex0, tex1, calib, rect, surface.head_to_rect)
    if head is not None:
        # Whole head: the measured face keeps its texel density (the back of the head is only filled in),
        # the low mesh keeps the GNM UV layout (animatable GLB); occlusion by the whole head.
        visibility = head.low
        high_uv = unwrap_uv(high, cfg.high_texture_size, density=_face_density(surface, high))
        lv, lf, luv, low_src = split_by_corner_uv(head.low.vertices, head.low.faces, head.low_corner_uv)
        low_uv = trimesh.Trimesh(lv, lf, process=False)
        low_uv.visual = trimesh.visual.TextureVisuals(uv=luv)
    else:
        visibility = surface.smooth_mesh
        high_uv = unwrap_uv(high, cfg.high_texture_size)
        low_uv = unwrap_uv(low, cfg.low_texture_size)
    tex_high, tex_stats = bake_albedo(high_uv, cameras, size=cfg.high_texture_size,
                                      visibility_mesh=visibility, debug_dir=debug_dir)
    tex_low, _ = bake_albedo(low_uv, cameras, size=cfg.low_texture_size, visibility_mesh=visibility)
    normal_low = bake_normal_map(low_uv, high, size=cfg.low_texture_size)
    normal_high = flat_normal_map(cfg.high_texture_size)
    t = lap("texture_s", t)

    # --- Deliverables
    steps.next("Face 5: export (OBJ / GLB / STL / 3MF)")
    solid_fn = None
    if head is not None:
        chin_y = float(head.fit_lm68[8, 1])
        solid_fn = lambda m: head_solid(m, chin_y)
    deliverables = export_face_deliverables(out_dir, cfg.name, high_uv, low_uv, tex_high, tex_low,
                                            normal_high, normal_low, export_scale=cfg.export_scale,
                                            draco=cfg.glb_draco, solid_fn=solid_fn)
    if head is not None:
        # The low GLB becomes the animatable head (skeleton + semantic expressions), not Draco-compressed
        write_animated_glb(deliverables["glb_low"], head, head.fitter, low_uv, low_src, tex_low, normal_low,
                           factor=cfg.export_scale / 1000.0)
    ipd_mm = _interpupillary_distance_mm(rect, lm0, lm1)
    object_info = write_object_info(out_dir, cfg, high, calib, ipd_mm)
    deliverables["object_json"] = out_dir / OBJECT_INFO_FILE
    t = lap("export_s", t)

    if debug_dir is not None:
        _save_debug_images(debug_dir, pair_id, rect, estimates)
        _save_segmentation_debug(debug_dir, pair_id, [img0, img1], segs)

    bbox = high.extents
    manifest = {
        "pair_id": pair_id,
        "reconstruction_mode": "dense_two_view",
        "inputs": {"left": str(left_path), "right": str(right_path)},
        "calibration": {**calib.to_dict(), "job": cfg.calibration_info},
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
        "scale": {"tag_scale": calib.tag_scale, "export_scale": cfg.export_scale,
                  "export_units": _units_name(cfg.export_scale)},
        "object": object_info,
        "disparity": {
            e.reference: {
                "valid_fraction": round(float(e.valid.sum() / max(e.face_mask.sum(), 1)), 3),
                "shape_prior_fit": e.shape_fit,
            } for e in estimates
        },
        "surface": surface.stats,
        "gnm": ({**head.fit.stats, **head.stats} if head is not None else {"used": False}),
        "segmentation": {
            "used": segs[0] is not None,
            "face_fraction": [round(sg.face_fraction, 4) if sg else None for sg in segs],
            "frames_removed": bool(remove_frames and segs[0] is not None),
            "region_px": [int(e.face_mask.sum()) for e in estimates],
        },
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

def _gnm_head(surface, cfg: FacePipelineConfig):
    """GNM fitted to the measured face and the whole head assembled; None (face-only) when the fit is poor."""
    try:
        fitter, fit = fit_gnm(surface)
    except Exception as e:                       # model missing, too few points ...
        log.warning("GNM head model not used (%s): face-only deliverables.", e)
        return None
    if fit.stats["surface_rms_mm"] > cfg.max_gnm_rms_mm:
        log.warning("GNM fit RMS %.2f mm > %.2f mm: face-only deliverables.", fit.stats["surface_rms_mm"],
                    cfg.max_gnm_rms_mm)
        return None
    head = assemble_head(surface, fitter, fit, cfg.expression, cfg.high_max_faces, decimate=_decimate)
    head.fitter = fitter
    head.fit_lm68 = fitter.g.landmarks68(fit.vertices)
    return head


def _face_density(surface, mesh: trimesh.Trimesh) -> np.ndarray:
    """1 on the measured face (vertices on the height field), 0 on the rest of the head, smooth in between."""
    V = mesh.vertices
    z = surface.sample_height(V[:, :2])
    near = np.isfinite(z) & (np.abs(z - V[:, 2]) < 8.0) & (np.asarray(mesh.vertex_normals)[:, 2] > 0.2)
    d = near.astype(np.float64)
    from scipy.sparse import coo_matrix
    e = mesh.edges_unique
    n = len(V)
    A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
    for _ in range(8):
        d = (d + A @ d) / deg
    return d


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


def _textured(mesh_mm: trimesh.Trimesh, tex_bgr: np.ndarray, normal_bgr: np.ndarray, factor: float) -> trimesh.Trimesh:
    """Copy in export units (mm x factor) with a PBR material (OBJ / GLB deliverables)."""
    m = trimesh.Trimesh(mesh_mm.vertices * factor, mesh_mm.faces, process=False)
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


def to_print_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Copy of a Y-up mesh (head frame) standing upright for STL / 3MF: Z up, base on Z = 0 (as Turntable Studio)."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))   # Y -> Z
    out.apply_translation([0.0, 0.0, -out.bounds[0, 2]])
    return out


def _units_name(export_scale: float) -> str:
    return {1.0: "m", 100.0: "cm", 1000.0: "mm"}.get(float(export_scale), f"m x {export_scale:g}")


def export_face_deliverables(
    out_dir: Path,
    name: str,
    high_uv: trimesh.Trimesh,
    low_uv: trimesh.Trimesh,
    tex_high: np.ndarray,
    tex_low: np.ndarray,
    normal_high: np.ndarray,
    normal_low: np.ndarray,
    export_scale: float = 100.0,
    draco: bool = True,
    solid_fn=None,
) -> Dict[str, Path]:
    """
    The 8 standard deliverables (+ textures) in output/{obj/high, obj/low, glb, stl, 3mf}.
    The meshes are in millimetres (head frame); every format is written in the same units,
    metres x export_scale, with one explicit factor (no guessing from the mesh size).
    """
    base = name or "model"
    factor = float(export_scale) / 1000.0          # mm -> export units
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

        textured = _textured(mesh_uv, tex, nrm, factor)
        obj_text = trimesh.exchange.obj.export_obj(
            textured, include_normals=True, include_texture=True, write_texture=False, mtl_name="material.mtl",
        )
        obj_path = obj_dir / f"{stem}.obj"
        obj_path.write_text(obj_text, encoding="utf-8")
        _write_mtl(obj_dir / "material.mtl", obj_path.name, tex_path.name, nrm_path.name)

        glb_path = out_dir / "glb" / f"{stem}.glb"
        convert_to_glb_draco(textured, glb_path, draco=draco)

        surf = trimesh.Trimesh(mesh_uv.vertices, mesh_uv.faces, process=True)
        solid = to_print_frame(solid_fn(surf) if solid_fn is not None else solidify_height_field(surf))
        stl_path = convert_to_stl(solid, out_dir / "stl" / f"{stem}.stl", enforce_watertight=False, scale=factor)
        thrf_path = convert_to_3mf(solid, out_dir / "3mf" / f"{stem}.3mf", scale=factor)
        paths.update({
            f"obj_{level}": obj_path, f"texture_{level}": tex_path, f"normal_{level}": nrm_path,
            f"glb_{level}": glb_path, f"stl_{level}": stl_path, f"3mf_{level}": thrf_path,
        })
        log.info("Exported %s level: %d faces, %s units (STL/3MF watertight: %s)", level, len(mesh_uv.faces),
                 _units_name(export_scale), solid.is_watertight)
    return paths


def _interpupillary_distance_mm(rect, lm_raw0: np.ndarray, lm_raw1: np.ndarray) -> Optional[float]:
    """Distance between the iris centres (MediaPipe 468 / 473), triangulated in the rectified pair."""
    if len(lm_raw0) < 478 or len(lm_raw1) < 478:
        return None
    a = rect.rectify_points(lm_raw0[[468, 473]], 0)
    b = rect.rectify_points(lm_raw1[[468, 473]], 1)
    d = a[:, 0] - b[:, 0]
    hom = np.c_[a, d, np.ones(2)] @ rect.Q.T
    P = hom[:, :3] / hom[:, 3:4]
    ipd = float(np.linalg.norm(P[0] - P[1]))
    return round(ipd, 1) if np.isfinite(ipd) and 40.0 < ipd < 90.0 else None


OBJECT_INFO_FILE = "object.json"


def write_object_info(out_dir: Path, cfg: FacePipelineConfig, high_mm: trimesh.Trimesh, calib, ipd_mm) -> Dict[str, Any]:
    """
    object.json (as Turntable Studio's pipeline/object_info.py): the real size of the face in millimetres,
    measured on the mesh before the export scale, plus the scale and calibration used.
    """
    ext = high_mm.extents
    info = {
        "title": cfg.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dimensions_mm": {"width": round(float(ext[0]), 1), "height": round(float(ext[1]), 1),
                          "depth": round(float(ext[2]), 1)},
        "interpupillary_distance_mm": ipd_mm,
        "scale": {"tag_scale": calib.tag_scale, "export_scale": cfg.export_scale,
                  "export_units": _units_name(cfg.export_scale)},
        "calibration": {"source": calib.source, "baseline_mm": round(calib.baseline_mm, 2)},
        "units": "mm (dimensions): real size of the face; deliverables use export_units",
    }
    (out_dir / OBJECT_INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _save_segmentation_debug(folder: Path, pair_id: str, images, segs) -> None:
    """Face region (green), hair (red) and spectacle frames (magenta) on both photographs."""
    if segs[0] is None:
        return
    tiles = []
    for img, sg in zip(images, segs):
        vis = img.copy()
        for mask, colour in ((sg.face, (0, 200, 0)), (sg.hair, (0, 0, 220)), (sg.frames, (220, 0, 220))):
            vis[mask] = (0.55 * vis[mask] + 0.45 * np.array(colour)).astype(np.uint8)
        tiles.append(cv2.resize(vis, (1014, 760), interpolation=cv2.INTER_AREA))
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{pair_id}_segmentation.jpg"), np.hstack(tiles))


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
