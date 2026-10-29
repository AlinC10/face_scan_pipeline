"""
Metric scale and head frame for the MediaPipe face-mesh mode (reconstruct_3d.py).

The mesh of that mode comes from the 468 MediaPipe landmarks of the left photo, whose units are arbitrary. With the
calibrated stereo pair the same landmarks are triangulated in millimetres; a trimmed similarity fit of the
landmark mesh onto them gives the real size (`scale_mm_per_px`). A second fit of the canonical MediaPipe face onto the
metric mesh gives the pose of the head in the camera frame (camera pitch + head tilt), which is removed so the exported
face is upright: x right, y up, z towards the viewer, origin at the canonical face origin (same frame as the dense mode).

Depends only on numpy + OpenCV (RPi friendly). Calibration: `assets/rig_calibration.json`
(X_cam1 = R_1_0 * X_cam0 + T_1_0_mm); the baseline is rescaled to the tag-disc print scale (`tag_scale`, 0.98),
exactly as photogrammetry/rig_calibration.py does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import trimesh

DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "assets" / "rig_calibration.json"
DEFAULT_TAG_SCALE = 0.98          # print scale of the AprilTag disc the calibration was measured on
IPD_FALLBACK_MM = 63.0            # used only when no stereo pair is available (single photo / one camera without a face)
FACE_WIDTH_FALLBACK_MM = 150.0
IRIS_RIGHT, IRIS_LEFT = 468, 473
FLIP_YZ = np.diag([1.0, -1.0, -1.0])   # camera frame (y down, z forward) <-> viewer frame (y up, z towards the viewer)


@dataclass
class StereoCalibration:
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R: np.ndarray            # camera 0 -> camera 1
    T_mm: np.ndarray
    image_size: Tuple[int, int]
    tag_scale: float
    source: str

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T_mm))


def load_stereo_calibration(path: Optional[Path] = None, tag_scale: float = DEFAULT_TAG_SCALE) -> StereoCalibration:
    p = Path(path) if path else DEFAULT_CALIBRATION
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "cam0" not in data:
        raise ValueError(f"{p}: expected the own format (cam0 / cam1 / R_1_0 / T_1_0_mm), as assets/rig_calibration.json")
    stored = float(data.get("tag_scale", 1.0))
    T = np.asarray(data["T_1_0_mm"], dtype=np.float64).reshape(3) * (float(tag_scale) / stored)
    return StereoCalibration(
        K0=np.asarray(data["cam0"]["K"], dtype=np.float64), D0=np.asarray(data["cam0"]["dist"], dtype=np.float64),
        K1=np.asarray(data["cam1"]["K"], dtype=np.float64), D1=np.asarray(data["cam1"]["dist"], dtype=np.float64),
        R=np.asarray(data["R_1_0"], dtype=np.float64), T_mm=T,
        image_size=tuple(data.get("image_size", (4056, 3040))), tag_scale=float(tag_scale),
        source=data.get("source", str(p)),
    )


def _scaled_K(K: np.ndarray, calib_size: Tuple[int, int], image_size: Tuple[int, int]) -> np.ndarray:
    s = image_size[0] / float(calib_size[0])
    return np.diag([s, s, 1.0]) @ K


def triangulate_landmarks(pts0: np.ndarray, pts1: np.ndarray, calib: StereoCalibration,
                          size0: Tuple[int, int], size1: Tuple[int, int]) -> np.ndarray:
    """Pixel landmarks (N, 2) of both photos -> (N, 3) in millimetres, camera-0 frame (y down, z forward)."""
    K0, K1 = _scaled_K(calib.K0, calib.image_size, size0), _scaled_K(calib.K1, calib.image_size, size1)
    n0 = cv2.undistortPoints(np.asarray(pts0, np.float64).reshape(-1, 1, 2), K0, calib.D0).reshape(-1, 2)
    n1 = cv2.undistortPoints(np.asarray(pts1, np.float64).reshape(-1, 1, 2), K1, calib.D1).reshape(-1, 2)
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([calib.R, calib.T_mm.reshape(3, 1)])
    X = cv2.triangulatePoints(P0, P1, n0.T, n1.T)
    return (X[:3] / X[3]).T


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / A.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def similarity_fit(src: np.ndarray, dst: np.ndarray, with_scale: bool = True, keep: float = 0.7,
                   iterations: int = 5) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Trimmed Umeyama: dst ~ s * R @ src + t on the `keep` fraction of the points with the smallest residual."""
    idx = np.arange(len(src))
    for _ in range(iterations):
        s, R, t = _umeyama(src[idx], dst[idx], with_scale)
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        idx = np.argsort(res)[: max(12, int(keep * len(src)))]
    s, R, t = _umeyama(src[idx], dst[idx], with_scale)
    rms = float(np.sqrt(np.mean(np.sum((s * src[idx] @ R.T + t - dst[idx]) ** 2, axis=1))))
    return s, R, t, rms


def _euler_deg(R: np.ndarray) -> Dict[str, float]:
    return {
        "pitch": float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))),
        "yaw": float(np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))),
        "roll": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
    }


def metric_head_vertices(
    landmarks_px: np.ndarray,
    canonical: np.ndarray,
    triangulated_mm: Optional[np.ndarray] = None,
    level_head: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    landmarks_px: (>=468, 3) x, y, z in pixels of the primary (left) photo. canonical: (468, 3) MediaPipe canonical face.
    Returns the (468, 3) vertices in millimetres and a stats dict.
    """
    n = len(canonical)
    M = np.asarray(landmarks_px, np.float64)[:n, :3]
    stats: Dict[str, Any] = {}
    if triangulated_mm is not None:
        X = np.asarray(triangulated_mm, np.float64)
        s, R, t, rms = similarity_fit(M, X[:n])
        V = s * M @ R.T + t
        stats.update(scale_source="stereo_triangulation", scale_mm_per_px=s, landmark_fit_rms_mm=rms)
        if len(X) > IRIS_LEFT:
            stats["interpupillary_distance_mm"] = float(np.linalg.norm(X[IRIS_RIGHT] - X[IRIS_LEFT]))
    else:
        L = np.asarray(landmarks_px, np.float64)
        if len(L) > IRIS_LEFT and np.linalg.norm(L[IRIS_RIGHT, :2] - L[IRIS_LEFT, :2]) > 30.0:
            s = IPD_FALLBACK_MM / float(np.linalg.norm(L[IRIS_RIGHT, :2] - L[IRIS_LEFT, :2]))
            stats["scale_source"] = f"assumed_ipd_{IPD_FALLBACK_MM:.0f}mm"
        else:
            s = FACE_WIDTH_FALLBACK_MM / max(1.0, float(M[:, 0].max() - M[:, 0].min()))
            stats["scale_source"] = f"assumed_face_width_{FACE_WIDTH_FALLBACK_MM:.0f}mm"
        V = s * M
        stats["scale_mm_per_px"] = float(s)
    if level_head:
        s2, R2, t2, rms2 = similarity_fit(canonical, V, keep=0.6)
        V = (V - t2) @ R2                       # R2^T (V - t2): the canonical frame, metric
        stats.update(head_pose_removed_deg=_euler_deg(FLIP_YZ @ R2), canonical_fit_rms_mm=rms2,
                     size_vs_canonical=s2 / 10.0)   # the canonical model is in cm
    return V, stats


def solidify_height_field(mesh: trimesh.Trimesh, back_offset: float) -> trimesh.Trimesh:
    """
    Closed solid for printing (head frame: z towards the viewer): the facial surface, a flat back plate `back_offset`
    (mesh units) behind its deepest point and vertical walls along the outline. Same construction as the dense mode.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    n = len(V)
    back = V.copy()
    back[:, 2] = V[:, 2].min() - back_offset
    boundary = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    a, b = mesh.edges[boundary].T
    walls = np.concatenate([np.stack([b, a, a + n], axis=1), np.stack([b, a + n, b + n], axis=1)])
    solid = trimesh.Trimesh(np.vstack([V, back]), np.vstack([F, F[:, ::-1] + n, walls]), process=True)
    if solid.volume < 0:
        solid.invert()
    return solid


def textured_solid(mesh: trimesh.Trimesh, back_offset: float, back_uv: np.ndarray) -> trimesh.Trimesh:
    """
    The same closed solid as `solidify_height_field`, but textured (OBJ / GLB): the front keeps its UVs, the walls take the
    UV of the outline vertex they hang from, the flat back plate maps to the single UV `back_uv` (a skin-coloured texel).
    Vertices are not merged (coincident ones carry different UVs); the solid is closed by position.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    n = len(V)
    zb = V[:, 2].min() - back_offset
    boundary = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    a, b = mesh.edges[boundary].T
    ring = np.unique(np.concatenate([a, b]))
    ring_idx = np.full(n, -1, np.int64)
    ring_idx[ring] = 2 * n + np.arange(len(ring))                      # wall-bottom copies of the outline vertices
    back = V.copy()
    back[:, 2] = zb
    ring_v = V[ring].copy()
    ring_v[:, 2] = zb
    verts = np.vstack([V, back, ring_v])
    uvs = np.vstack([uv, np.tile(np.asarray(back_uv, np.float64), (n, 1)), uv[ring]])
    walls = np.concatenate([np.stack([b, a, ring_idx[a]], axis=1), np.stack([b, ring_idx[a], ring_idx[b]], axis=1)])
    faces = np.vstack([F, F[:, ::-1] + n, walls])
    solid = trimesh.Trimesh(verts, faces, process=False)
    if trimesh.Trimesh(verts, faces, process=True).volume < 0:           # outward normals
        solid = trimesh.Trimesh(verts, faces[:, ::-1], process=False)
    solid.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return solid


def skin_uv(mesh: trimesh.Trimesh, tex_bgr: np.ndarray) -> np.ndarray:
    """UV of the mesh vertex whose texel is closest to the median skin colour: the colour of the back plate."""
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    h, w = tex_bgr.shape[:2]
    px = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
    col = tex_bgr[py, px].astype(np.float64)
    i = int(np.argmin(np.linalg.norm(col - np.median(col, axis=0), axis=1)))
    return uv[i]


def to_print_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Copy of a Y-up mesh standing upright for STL / 3MF: Z up, base on Z = 0 (as Turntable Studio)."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))   # Y -> Z
    out.apply_translation([0.0, 0.0, -out.bounds[0, 2]])
    return out


def write_object_info(out_dir: Path, vertices_mm: np.ndarray, stats: Dict[str, Any], calib: Optional[StereoCalibration],
                      export_scale: float, title: str = "model", back_plate_mm: float = 0.0) -> Dict[str, Any]:
    """object.json as the dense mode writes it: the real size in millimetres (depth includes the back plate) + the scale used."""
    ext = vertices_mm.max(0) - vertices_mm.min(0)
    ext[2] += back_plate_mm
    units = {1.0: "m", 10.0: "dm", 100.0: "cm", 1000.0: "mm"}.get(float(export_scale), f"x{export_scale:g}")
    info = {
        "title": title,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "mediapipe_face_mesh",
        "dimensions_mm": {"width": round(float(ext[0]), 1), "height": round(float(ext[1]), 1), "depth": round(float(ext[2]), 1)},
        "back_plate_mm": back_plate_mm,
        "interpupillary_distance_mm": round(stats["interpupillary_distance_mm"], 1) if "interpupillary_distance_mm" in stats else None,
        "scale": {"tag_scale": calib.tag_scale if calib else None, "export_scale": export_scale, "export_units": units,
                  "source": stats.get("scale_source"), "mm_per_landmark_px": round(stats.get("scale_mm_per_px", 0.0), 5)},
        "calibration": {"source": calib.source, "baseline_mm": round(calib.baseline_mm, 2)} if calib else None,
        "head_pose_removed_deg": {k: round(v, 1) for k, v in stats.get("head_pose_removed_deg", {}).items()} or None,
        "fit_rms_mm": {"landmarks_to_stereo": round(stats.get("landmark_fit_rms_mm", 0.0), 2),
                       "canonical_to_mesh": round(stats.get("canonical_fit_rms_mm", 0.0), 2)},
        "units": f"mm (dimensions): real size of the face; deliverables use export_units ({units})",
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "object.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info
