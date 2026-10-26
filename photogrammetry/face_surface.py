"""
Two-View Facial Surface Fusion
==============================
Turns the two prior-guided disparity maps of the rectified pair into one metric
facial surface.

  1. Triangulation: every valid pixel of both reference views becomes a 3-D point
     (mm, rectified-left frame), weighted by how frontally its camera sees it.
  2. Head frame: the MediaPipe canonical face is aligned (similarity, trimmed
     Umeyama) to the triangulated landmarks; the surface is expressed in that
     frame (x to the subject's left, y up, z towards the viewer, millimetres).
  3. Height map: in the head frame the face is a height field z(x, y). The
     measurements are splatted on a regular grid and fused with the landmark
     prior as  z = z_prior + S + b·D,  where S is the robust low-frequency
     correction (σ = 6 mm), D the measured detail (σ = 0.6 mm) and b its
     support. Measurements that disagree with S by more than 2 mm (glasses
     frames, highlights, occlusion borders) are rejected; behind the lenses only S
     is used, because the lenses refract.
  4. Mesh: one vertex per grid cell, two triangles per cell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from photogrammetry.face_correspondence import (
    NUM_MESH_LANDMARKS,
    DisparityEstimate,
    RectifiedFacePair,
    fill_by_normalized_convolution,
)

log = logging.getLogger("Photogrammetry.FaceSurface")

CANONICAL_FACE_PATH = Path(__file__).resolve().parent.parent / "assets" / "canonical_face_model.obj"

# Eye contours of the MediaPipe face mesh (subject's right and left eye).
EYE_CONTOURS = (
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
)
LENS_REGION_SCALE = 1.7        # eye contour enlarged around its centre ~ area seen through a lens
MAX_GRAZING_DEG = 75.0         # measurements seen more obliquely are discarded
EDGE_TAPER_PX = 25.0           # weight ramp at the border of each view's face region (rectified px)
DETAIL_SIGMA_MM = 0.9          # spatial scale of the measured detail kept on top of the smooth surface


@dataclass
class FaceSurfaceResult:
    mesh: trimesh.Trimesh                  # head frame, millimetres
    head_to_rect: np.ndarray               # 4x4: head frame (mm) -> rectified-left camera frame (mm)
    landmarks_head: np.ndarray             # (468, 3) landmarks in the head frame
    grid_resolution_mm: float
    stats: Dict[str, float] = field(default_factory=dict)


def triangulate_disparity(
    rect: RectifiedFacePair,
    est: DisparityEstimate,
    disparity: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3-D points (h, w, 3) in the rectified-left frame and per-pixel weights for one reference
    view (measured disparity by default, or e.g. the smooth prior surface).
    """
    d = (est.disparity if disparity is None else disparity).astype(np.float64)
    valid = est.valid if valid is None else valid
    h, w = d.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x_left = xs if est.reference == "left" else xs + d
    hom = np.stack([x_left, ys, d, np.ones_like(d)], axis=-1) @ rect.Q.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = hom[..., :3] / hom[..., 3:4]

    # Grazing angle between the local surface normal (of the smooth surface) and the camera ray.
    hom_s = np.stack([xs if est.reference == "left" else xs + est.smooth, ys, est.smooth.astype(np.float64),
                      np.ones_like(d)], axis=-1) @ rect.Q.T
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth = (hom_s[..., :3] / hom_s[..., 3:4]).astype(np.float32)
    smooth = np.stack([cv2.GaussianBlur(smooth[..., i], (0, 0), 3.0) for i in range(3)], axis=-1)
    n = np.cross(np.gradient(smooth, axis=1), np.gradient(smooth, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    centre = np.zeros(3) if est.reference == "left" else np.array([-rect.P2[0, 3] / rect.P2[0, 0], 0.0, 0.0])
    ray = pts - centre
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-12
    cos = np.abs(np.sum(n * ray, axis=-1))
    # Weights fall smoothly to zero towards grazing views and towards the edge of the face
    # region of this view, so the contributions of the two views blend without seams.
    c_lo, c_hi = np.cos(np.radians(MAX_GRAZING_DEG)), np.cos(np.radians(MAX_GRAZING_DEG - 15.0))
    weight = cos ** 2 * np.clip((cos - c_lo) / (c_hi - c_lo), 0.0, 1.0)
    edge = cv2.distanceTransform(est.face_mask.astype(np.uint8), cv2.DIST_L2, 5)
    weight *= np.clip(edge / EDGE_TAPER_PX, 0.0, 1.0)
    ok = valid & est.face_mask & np.isfinite(pts).all(axis=-1) & (pts[..., 2] > 100) & (pts[..., 2] < 800)
    return pts, np.where(ok, weight, 0.0)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Similarity dst ≈ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def fit_head_frame(landmarks_rect3d: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Aligns the canonical MediaPipe face to the triangulated landmarks (trimmed Umeyama) and
    returns the 4x4 transform head frame (mm) -> rectified-left frame (mm).
    """
    canonical = np.asarray(trimesh.load(CANONICAL_FACE_PATH, process=False).vertices[:NUM_MESH_LANDMARKS]) * 10.0
    idx = np.where(valid)[0]
    for _ in range(3):
        s, R, t = _umeyama(canonical[idx], landmarks_rect3d[idx])
        res = np.linalg.norm((s * canonical[valid] @ R.T + t) - landmarks_rect3d[valid], axis=1)
        keep = res <= np.percentile(res, 80)
        idx = np.where(valid)[0][keep]
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    log.info("Head frame: canonical face scale %.3f (1.0 = MediaPipe average head), alignment RMS %.2f mm", s, rms)
    return M, {"canonical_scale": round(float(s), 4), "head_alignment_rms_mm": round(rms, 3)}


def _transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:3, :3].T + M[:3, 3]


def fuse_face_surface(
    rect: RectifiedFacePair,
    estimates: List[DisparityEstimate],
    lm_left_rect: np.ndarray,
    grid_resolution_mm: float = 0.5,
    margin_mm: float = 4.0,
    debug_dir: Optional[Path] = None,
) -> FaceSurfaceResult:
    """Fuses the disparity estimates of both reference views into a metric facial mesh."""
    left = next(e for e in estimates if e.reference == "left")
    h, w = left.disparity.shape

    # --- 3-D landmarks on the smooth surface of the left view
    lm = lm_left_rect[:NUM_MESH_LANDMARKS]
    inside = (lm[:, 0] >= 0) & (lm[:, 0] < w - 1) & (lm[:, 1] >= 0) & (lm[:, 1] < h - 1)
    lm_c = np.clip(lm, 0, [w - 1, h - 1]).astype(np.float32)
    d_lm = cv2.remap(left.smooth, lm_c[:, 0:1], lm_c[:, 1:2], cv2.INTER_LINEAR)[:, 0]
    inside &= cv2.remap(left.face_mask.astype(np.uint8), lm_c[:, 0:1], lm_c[:, 1:2], cv2.INTER_NEAREST)[:, 0] > 0
    hom = np.c_[lm, d_lm, np.ones(len(lm))] @ rect.Q.T
    lm3d = hom[:, :3] / hom[:, 3:4]

    head_to_rect, stats = fit_head_frame(lm3d, inside & np.isfinite(lm3d).all(1))
    rect_to_head = np.linalg.inv(head_to_rect)
    lm_head = _transform(rect_to_head, lm3d)

    # --- Grid over the landmark extent in the head frame
    res = grid_resolution_mm
    lo = lm_head[inside, :2].min(0) - margin_mm
    hi = lm_head[inside, :2].max(0) + margin_mm
    gw, gh = (np.ceil((hi - lo) / res).astype(int) + 1)

    def to_grid(p):                      # head (x, y) -> grid (col, row); rows grow downwards
        return np.stack([(p[:, 0] - lo[0]) / res, (hi[1] - p[:, 1]) / res], axis=1)

    def splat(points: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sel = weights > 0
        p = _transform(rect_to_head, points[sel])
        g = np.floor(to_grid(p) + 0.5).astype(int)
        ok = (g[:, 0] >= 0) & (g[:, 0] < gw) & (g[:, 1] >= 0) & (g[:, 1] < gh)
        flat = g[ok, 1] * gw + g[ok, 0]
        acc_w = np.bincount(flat, weights=weights[sel][ok], minlength=gh * gw).reshape(gh, gw)
        acc_z = np.bincount(flat, weights=(p[:, 2] * weights[sel])[ok], minlength=gh * gw).reshape(gh, gw)
        return acc_z, acc_w

    # --- Prior surface: the smooth (landmark prior + robust correction) surface of both views
    prior_z = np.zeros((gh, gw))
    prior_w = np.zeros((gh, gw))
    meas_z = np.zeros((gh, gw))
    meas_w = np.zeros((gh, gw))
    n_points = 0
    for est in estimates:
        pts, wgt = triangulate_disparity(rect, est, disparity=est.smooth, valid=est.face_mask)
        z_, w_ = splat(pts, wgt)
        prior_z += z_
        prior_w += w_
        pts, wgt = triangulate_disparity(rect, est)
        z_, w_ = splat(pts, wgt)
        meas_z += z_
        meas_w += w_
        n_points += int((wgt > 0).sum())

    covered = prior_w > 0.1 * np.median(prior_w[prior_w > 0])
    covered = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    mask = _fill_holes(_largest_component(covered))
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    z_prior = np.where(prior_w > 1e-3, prior_z / np.maximum(prior_w, 1e-9), np.nan)
    z_fill, _ = fill_by_normalized_convolution(np.nan_to_num(z_prior), np.isfinite(z_prior), sigma=1.5 / res)
    z_prior = np.where(np.isfinite(z_prior), z_prior, z_fill)
    z_prior = cv2.GaussianBlur(z_prior.astype(np.float32), (0, 0), 1.0 / res)

    # Regions seen through the spectacle lenses / eyes: only the low-frequency correction is used.
    lens = np.zeros((gh, gw), np.uint8)
    for contour in EYE_CONTOURS:
        g = to_grid(lm_head[contour])
        c = g.mean(0)
        poly = (c + LENS_REGION_SCALE * (g - c)).round().astype(np.int32)
        cv2.fillPoly(lens, [poly], 1)
    lens = lens.astype(bool)

    meas_valid = (meas_w > 1e-3) & mask
    z_meas = np.where(meas_valid, meas_z / np.maximum(meas_w, 1e-9), 0.0)

    # --- Robust fusion with the prior
    r = z_meas - z_prior
    sig_low = 6.0 / res
    inl = meas_valid.copy()
    for _ in range(3):
        S, sup = fill_by_normalized_convolution(r, inl, sigma=sig_low)
        S = np.where(sup > 0.02, S, np.median(r[inl]) if inl.any() else 0.0)
        inl = meas_valid & (np.abs(r - S) < 2.0)
    detail, dsup = fill_by_normalized_convolution(r - S, inl & ~lens, sigma=DETAIL_SIGMA_MM / res)
    b = np.clip(dsup / 0.35, 0.0, 1.0)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 2.0 / res) * (~lens)
    z = z_prior + S + b * detail
    z = np.where(mask, z, np.nan)

    coverage = float(inl.sum() / max(mask.sum(), 1))
    stats.update({
        "grid_resolution_mm": res,
        "measured_points": n_points,
        "measured_coverage": round(float(meas_valid.sum() / max(mask.sum(), 1)), 3),
        "inlier_coverage": round(coverage, 3),
        "prior_correction_median_mm": round(float(np.median(S[mask])), 2),
        "prior_correction_p95_abs_mm": round(float(np.percentile(np.abs(S[mask]), 95)), 2),
    })
    log.info("Surface fusion: %d points, %.0f%% of the face measured (%.0f%% inliers), "
             "median prior correction %.2f mm", n_points, 100 * stats["measured_coverage"], 100 * coverage,
             stats["prior_correction_median_mm"])

    if debug_dir is not None:
        _save_debug_reliefs(Path(debug_dir), res, mask, lens, inl, {
            "prior": z_prior, "prior_plus_correction": z_prior + S, "final": z,
        })

    mesh = _grid_mesh(z, lo, hi, res)
    stats.update({"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))})
    return FaceSurfaceResult(mesh, head_to_rect, lm_head, res, stats)


def _hillshade(z: np.ndarray, res: float) -> np.ndarray:
    zz = np.nan_to_num(z, nan=float(np.nanmin(z)))
    gy, gx = np.gradient(zz, res)
    n = np.dstack([-gx, gy, np.ones_like(zz)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.4, 0.5, 0.77])
    return (np.clip(n @ (light / np.linalg.norm(light)), 0, 1) * 255).astype(np.uint8)


def _save_debug_reliefs(folder: Path, res: float, mask, lens, inliers, surfaces: Dict[str, np.ndarray]) -> None:
    """Shaded reliefs of the fusion stages and the measurement masks (for inspection)."""
    folder.mkdir(parents=True, exist_ok=True)
    tiles = []
    for name, z in surfaces.items():
        img = cv2.cvtColor(_hillshade(np.where(mask, z, np.nan), res), cv2.COLOR_GRAY2BGR)
        img[~mask] = 0
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(img)
    m = np.zeros(mask.shape + (3,), np.uint8)
    m[mask] = (60, 60, 60)
    m[inliers] = (0, 200, 0)
    m[lens] = (0, 0, 200)
    cv2.putText(m, "measured inliers / lens", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(m)
    cv2.imwrite(str(folder / "face_surface_fusion.jpg"), np.hstack(tiles))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if n <= 1:
        return mask
    return labels == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def _grid_mesh(z: np.ndarray, lo: np.ndarray, hi: np.ndarray, res: float) -> trimesh.Trimesh:
    """Height field -> triangle mesh: one vertex per defined cell, two triangles per cell."""
    gh, gw = z.shape
    ok = np.isfinite(z)
    ok = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool) & ok
    idx = -np.ones((gh, gw), dtype=np.int64)
    idx[ok] = np.arange(ok.sum())
    rows, cols = np.nonzero(ok)
    verts = np.stack([lo[0] + cols * res, hi[1] - rows * res, z[ok]], axis=1)

    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = []
    for tri in ((a, c, b), (b, c, d)):
        t = np.stack([x.ravel() for x in tri], axis=1)
        faces.append(t[(t >= 0).all(axis=1)])
    mesh = trimesh.Trimesh(verts, np.concatenate(faces), process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    return mesh


def solidify_height_field(mesh: trimesh.Trimesh, back_offset_mm: float = 3.0) -> trimesh.Trimesh:
    """
    Closed solid for 3-D printing (head frame, mm): the facial surface, a flat back plate
    `back_offset_mm` behind its deepest point and vertical walls along the outline.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    n = len(V)
    back = V.copy()
    back[:, 2] = V[:, 2].min() - back_offset_mm

    # Boundary edges, directed as in their (single) face.
    boundary = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    a, b = mesh.edges[boundary].T
    walls = np.concatenate([np.stack([b, a, a + n], axis=1), np.stack([b, a + n, b + n], axis=1)])
    solid = trimesh.Trimesh(np.vstack([V, back]), np.vstack([F, F[:, ::-1] + n, walls]), process=True)
    if solid.volume < 0:
        solid.invert()
    return solid
