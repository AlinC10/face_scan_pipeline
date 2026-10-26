"""
MediaPipe-Guided Dense Facial Correspondence
============================================
Dense two-view matching for the convergent face pair.

At 12.3 MP the MediaPipe landmarks of the two views do not describe the same
surface points to better than 10-20 px (the face mesh is regressed at 256 px).
They are therefore used only as an initialisation:

  1. The 468 landmarks and the MediaPipe tessellation define a piecewise-affine
     warp between the two images, which removes the large perspective change
     of the 20° convergence.
  2. DIS optical flow measures the residual motion with sub-pixel precision on
     the skin texture; a forward-backward check removes occlusions, glasses
     frames and specular highlights.

The same scheme runs twice: on the raw images (correspondences for refining the
relative camera pose) and on the rectified pair, where the prior and the
measured disparity are 1-D and the vertical flow is a quality check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from photogrammetry.rig_calibration import RigCalibration

log = logging.getLogger("Photogrammetry.FaceCorrespondence")

NUM_MESH_LANDMARKS = 468
FACE_LANDMARKER_PATH = Path(__file__).resolve().parent.parent / "assets" / "face_landmarker.task"


def tessellation_triangles() -> np.ndarray:
    from face_scanner import TESSELLATION_TRIANGLES
    return np.asarray(TESSELLATION_TRIANGLES, dtype=np.int32)


def detect_landmarks(images_bgr) -> list:
    """
    Runs MediaPipe FaceLandmarker on each image; returns (478, 3) arrays (x, y in pixels,
    z = MediaPipe relative depth in pixel units, negative towards the camera) or None.
    """
    from face_scanner import FaceScanner

    out = []
    with FaceScanner(model_path=str(FACE_LANDMARKER_PATH)) as scanner:
        for img in images_bgr:
            res = scanner.scan_image(img)
            if res["num_faces_detected"] < 1:
                out.append(None)
                continue
            lm = res["faces"][0]["pixel_landmarks"]
            out.append(np.array([[p["x"], p["y"], p["z"]] for p in lm], dtype=np.float64))
    return out


def rasterize_triangles(
    pts2d: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
    depth_channel: Optional[int] = None,
) -> np.ndarray:
    """
    Linear (barycentric) interpolation of per-vertex `values` (N, C) over the 2-D
    triangles. Returns an (h, w, C) float32 map, NaN outside the triangles.
    With `depth_channel`, overlapping triangles are resolved by a z-buffer on that
    channel (smallest value wins, i.e. the surface closest to the camera).
    """
    h, w = shape
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    out = np.full((h, w, values.shape[1]), np.nan, dtype=np.float32)
    zbuf = np.full((h, w), np.inf, dtype=np.float64) if depth_channel is not None else None
    for tri in triangles:
        p = pts2d[tri]
        x0, y0 = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(p.max(axis=0)).astype(int), [w - 1, h - 1])
        if x1 < x0 or y1 < y0:
            continue
        M = np.array([[p[1, 0] - p[0, 0], p[2, 0] - p[0, 0]], [p[1, 1] - p[0, 1], p[2, 1] - p[0, 1]]])
        det = np.linalg.det(M)
        if abs(det) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        P = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        uv = (P - p[0]) @ np.linalg.inv(M).T
        inside = (uv[:, 0] >= -1e-9) & (uv[:, 1] >= -1e-9) & (uv.sum(axis=1) <= 1 + 1e-9)
        if not inside.any():
            continue
        uv = uv[inside]
        P = P[inside].astype(int)
        v = values[tri]
        val = v[0] + uv[:, :1] * (v[1] - v[0]) + uv[:, 1:] * (v[2] - v[0])
        if zbuf is not None:
            closer = val[:, depth_channel] < zbuf[P[:, 1], P[:, 0]]
            P, val = P[closer], val[closer]
            zbuf[P[:, 1], P[:, 0]] = val[:, depth_channel]
        out[P[:, 1], P[:, 0]] = val
    return out


def fill_by_normalized_convolution(values: np.ndarray, valid: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-weighted interpolation of `values` from the `valid` pixels; returns (filled, support)."""
    k = int(6 * sigma) | 1
    v = np.where(valid, values, 0).astype(np.float32)
    wgt = valid.astype(np.float32)
    num = cv2.GaussianBlur(v, (k, k), sigma)
    den = cv2.GaussianBlur(wgt, (k, k), sigma)
    return num / np.maximum(den, 1e-6), den


def _dis(patch_size: int = 9, stride: int = 2) -> "cv2.DISOpticalFlow":
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    dis.setPatchSize(patch_size)
    dis.setPatchStride(stride)
    dis.setGradientDescentIterations(25)
    dis.setVariationalRefinementIterations(20)
    return dis


def flow_with_consistency(ref_gray: np.ndarray, warped_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """DIS flow ref -> warped and its forward-backward error (pixels)."""
    dis = _dis()
    fwd = dis.calc(ref_gray, warped_gray, None)
    bwd = dis.calc(warped_gray, ref_gray, None)
    h, w = ref_gray.shape
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    bx = cv2.remap(bwd[..., 0], mx + fwd[..., 0], my + fwd[..., 1], cv2.INTER_LINEAR)
    by = cv2.remap(bwd[..., 1], mx + fwd[..., 0], my + fwd[..., 1], cv2.INTER_LINEAR)
    return fwd, np.hypot(fwd[..., 0] + bx, fwd[..., 1] + by)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def guided_raw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    lm0: np.ndarray,
    lm1: np.ndarray,
    work_scale: float = 0.5,
    step: int = 3,
    max_fb_error: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense correspondences between the raw images (full-resolution pixel coordinates),
    used to refine the relative pose. Image 1 is warped onto image 0 through the
    landmark tessellation, then the residual motion is measured by DIS flow.
    """
    tris = tessellation_triangles()
    a = lm0[:NUM_MESH_LANDMARKS, :2] * work_scale
    b = lm1[:NUM_MESH_LANDMARKS, :2] * work_scale
    g0 = _gray(cv2.resize(img0, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA))
    g1 = _gray(cv2.resize(img1, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA))
    h, w = g0.shape

    warp = rasterize_triangles(a, tris, b, (h, w))          # pixel of image 1 for each pixel of image 0
    covered = ~np.isnan(warp[..., 0])
    mapx = np.nan_to_num(warp[..., 0])
    mapy = np.nan_to_num(warp[..., 1])
    warped = cv2.remap(g1, mapx, mapy, cv2.INTER_LINEAR)
    warped[~covered] = 0

    flow, fb = flow_with_consistency(g0, warped)
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx = mx + flow[..., 0]
    fy = my + flow[..., 1]
    x1 = cv2.remap(mapx, fx, fy, cv2.INTER_LINEAR)
    y1 = cv2.remap(mapy, fx, fy, cv2.INTER_LINEAR)
    still_covered = cv2.remap(covered.astype(np.uint8), fx, fy, cv2.INTER_NEAREST).astype(bool)
    good = covered & still_covered & (fb < max_fb_error)

    ys, xs = np.mgrid[0:h:step, 0:w:step]
    sel = good[ys, xs]
    pts0 = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float64) / work_scale
    pts1 = np.stack([x1[ys, xs][sel], y1[ys, xs][sel]], axis=1).astype(np.float64) / work_scale
    log.info("Guided raw matches: %d correspondences (%.0f%% of the face passes the F-B check)",
             len(pts0), 100.0 * good.sum() / max(covered.sum(), 1))
    return pts0, pts1


@dataclass
class RectifiedFacePair:
    """Rectified pair at the working scale and the geometry needed for triangulation."""
    left: np.ndarray
    right: np.ndarray
    R1: np.ndarray            # rotation camera 0 -> rectified left
    R2: np.ndarray            # rotation camera 1 -> rectified right
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    calib: RigCalibration     # calibration at the working scale
    work_scale: float

    @property
    def focal_px(self) -> float:
        return float(self.P1[0, 0])

    def rectify_points(self, pts_raw_full: np.ndarray, cam: int) -> np.ndarray:
        """Maps raw full-resolution pixel coordinates of camera `cam` into the rectified image."""
        pts = np.asarray(pts_raw_full, dtype=np.float64)[:, :2].reshape(-1, 1, 2) * self.work_scale
        if cam == 0:
            out = cv2.undistortPoints(pts, self.calib.K0, self.calib.D0, R=self.R1, P=self.P1)
        else:
            out = cv2.undistortPoints(pts, self.calib.K1, self.calib.D1, R=self.R2, P=self.P2)
        return out.reshape(-1, 2)


def rectify_pair(img0: np.ndarray, img1: np.ndarray, calib_full: RigCalibration, work_scale: float = 0.5) -> RectifiedFacePair:
    """Bouguet rectification of the raw pair with the (refined) calibration, at `work_scale`."""
    calib = calib_full.scaled(work_scale)
    h, w = img0.shape[:2]
    size = (int(round(w * work_scale)), int(round(h * work_scale)))
    if work_scale != 1.0:
        img0 = cv2.resize(img0, size, interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib.K0, calib.D0, calib.K1, calib.D1, size, calib.R, calib.T_mm.reshape(3, 1), alpha=-1, flags=0,
    )
    m0 = cv2.initUndistortRectifyMap(calib.K0, calib.D0, R1, P1, size, cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(calib.K1, calib.D1, R2, P2, size, cv2.CV_32FC1)
    left = cv2.remap(img0, m0[0], m0[1], cv2.INTER_LINEAR)
    right = cv2.remap(img1, m1[0], m1[1], cv2.INTER_LINEAR)
    log.info("Rectified pair %dx%d: f' = %.1f px, B = %.2f mm", size[0], size[1], P1[0, 0], calib.baseline_mm)
    return RectifiedFacePair(left, right, R1, R2, P1, P2, Q, calib, work_scale)


@dataclass
class DisparityEstimate:
    """Disparity d = x_left - x_right seen from one reference view of the rectified pair."""
    reference: str            # "left" or "right"
    disparity: np.ndarray     # (h, w) float32, defined where `valid`
    valid: np.ndarray         # pixels that passed the forward-backward and vertical checks
    smooth: np.ndarray        # MediaPipe shape prior + robust low-frequency correction (whole face)
    prior: np.ndarray         # MediaPipe shape prior fitted to the measurement
    face_mask: np.ndarray     # face region of this view
    shape_fit: dict           # coefficients and residual of the shape-prior fit


def _robust_depth_fit(Z: np.ndarray, A: np.ndarray, scale_mm: float = 2.0, iterations: int = 8) -> Tuple[np.ndarray, float]:
    """IRLS (Huber) solution of A c ≈ Z; returns coefficients and the robust residual scale."""
    w = np.ones(len(Z))
    c = np.zeros(A.shape[1])
    for _ in range(iterations):
        sw = np.sqrt(w)
        c = np.linalg.lstsq(A * sw[:, None], Z * sw, rcond=None)[0]
        r = np.abs(A @ c - Z)
        w = np.where(r <= scale_mm, 1.0, scale_mm / np.maximum(r, 1e-9))
    return c, float(1.4826 * np.median(np.abs(A @ c - Z)))


def estimate_disparity(
    rect: RectifiedFacePair,
    lm_ref_rect: np.ndarray,
    lm_other_rect: np.ndarray,
    lm_ref_depth: np.ndarray,
    reference: str = "left",
    iterations: int = 3,
    max_fb_error: float = 0.75,
    max_vertical_px: float = 1.0,
) -> DisparityEstimate:
    """
    Prior-guided disparity from the `reference` view.

    Initialisation: the disparity of the landmark correspondences. Each pass warps the
    other view by the current estimate, measures the residual motion with DIS, and
    re-fits the shape prior: the MediaPipe 3-D face of the reference view (relative
    landmark depth `lm_ref_depth`, image aligned by construction) mapped to metric depth
        Z(x, y) = c0 + c1·z_mp(x, y) + c2·x + c3·y
    by robust least squares on the measured depth. The next pass starts from that prior
    plus a smooth correction of the measurement, so the final residual motion is small.
    """
    tris = tessellation_triangles()
    ref_img, other_img = (rect.left, rect.right) if reference == "left" else (rect.right, rect.left)
    sign = -1.0 if reference == "left" else 1.0     # other x = ref x + sign * d
    g_ref, g_other = _gray(ref_img), _gray(other_img)
    h, w = g_ref.shape
    fB = rect.focal_px * rect.calib.baseline_mm
    c12 = float(rect.P1[0, 2] - rect.P2[0, 2])       # Z = f·B / (d - c12)

    pts = lm_ref_rect[:NUM_MESH_LANDMARKS, :2]
    lm_d = sign * (lm_other_rect[:NUM_MESH_LANDMARKS, 0] - pts[:, 0])
    raster = rasterize_triangles(pts, tris, np.c_[lm_d, lm_ref_depth[:NUM_MESH_LANDMARKS]], (h, w), depth_channel=1)
    covered = ~np.isnan(raster[..., 0])
    face_mask = _fill_mask_holes(covered)           # closes the eye and mouth openings
    filled, _ = fill_by_normalized_convolution(np.nan_to_num(raster[..., 0]), covered, sigma=12.0)
    init = np.where(face_mask, np.where(covered, raster[..., 0], filled), 0).astype(np.float32)
    # MediaPipe relative depth: holes filled and the 468-vertex facets smoothed out.
    zmp, _ = fill_by_normalized_convolution(np.nan_to_num(raster[..., 1]), covered, sigma=6.0)
    zmp = np.where(covered, raster[..., 1], zmp)
    zmp, _ = fill_by_normalized_convolution(zmp, face_mask, sigma=4.0)

    occluders = thin_occluder_mask(g_ref, face_mask)
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xn = (mx - w / 2) / w
    yn = (my - h / 2) / h
    A_full = np.stack([np.ones_like(zmp), zmp, xn, yn], axis=-1).astype(np.float64)

    current = init
    disparity, valid, smooth_disp, prior = init, np.zeros((h, w), bool), init, init
    fit = {}
    for it in range(iterations):
        warped = cv2.remap(g_other, mx + sign * current, my, cv2.INTER_LINEAR)
        flow, fb = flow_with_consistency(g_ref, warped)
        disparity = current + sign * flow[..., 0]
        valid = face_mask & ~occluders & (fb < max_fb_error) & (np.abs(flow[..., 1]) < max_vertical_px)
        valid &= (disparity - c12) > 1.0

        # Shape prior: MediaPipe relative depth -> metric depth, fitted on the measurement.
        Z = fB / (disparity[valid] - c12)
        coef, sigma = _robust_depth_fit(Z, A_full[valid])
        Zp = A_full @ coef
        prior = np.where(face_mask, c12 + fB / np.maximum(Zp, 50.0), 0).astype(np.float32)

        # Smooth correction of the prior from the inlier measurements.
        resid = disparity - prior
        inl = valid.copy()
        for _ in range(2):
            corr, support = fill_by_normalized_convolution(resid, inl, sigma=20.0)
            inl = valid & (np.abs(resid - corr) < 3.0)
        # Two scales blended by support: no step where the measurements end.
        corr_wide, support_wide = fill_by_normalized_convolution(resid, inl, sigma=60.0)
        corr_wide *= np.clip(support_wide / 0.05, 0.0, 1.0)
        a = np.clip(support / 0.15, 0.0, 1.0)
        corr = a * corr + (1.0 - a) * corr_wide
        smooth_disp = np.where(face_mask, prior + corr, 0).astype(np.float32)
        current = smooth_disp
        fit = {"c0_mm": float(coef[0]), "depth_scale": float(coef[1]), "tilt_x_mm": float(coef[2]),
               "tilt_y_mm": float(coef[3]), "residual_sigma_mm": round(sigma, 3)}
        log.info(
            "Disparity (%s reference) pass %d: %.0f%% of the face valid, shape prior Z = %.1f mm + %.3f·z_mp, "
            "robust residual %.2f mm", reference, it + 1, 100.0 * valid.sum() / max(face_mask.sum(), 1),
            coef[0], coef[1], sigma,
        )
    return DisparityEstimate(reference, disparity.astype(np.float32), valid, smooth_disp, prior, face_mask, fit)


def thin_occluder_mask(gray: np.ndarray, face_mask: np.ndarray, min_area_px: int = 150, dilate_px: int = 7) -> np.ndarray:
    """
    Long, thin, dark structures in front of the face (spectacle frames, hair strands):
    black-hat response, large connected components only, dilated by the flow patch radius.
    Their disparity belongs to another surface and smears the flow around them.
    """
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cand = ((bh > 18) & face_mask).astype(np.uint8)
    n, labels, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area_px
    occ = keep[labels].astype(np.uint8)
    k = 2 * dilate_px + 1
    return cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)
