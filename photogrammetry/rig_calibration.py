"""
Stereo Rig Calibration for Face Mode
====================================
Loads the measured stereo calibration of the dual IMX477 rig and refines the
relative pose of the two cameras on the dense facial correspondences of the
current capture.

Calibration sources (in order of preference):
  1. `assets/rig_calibration.json` (own format, X_cam1 = R_1_0 * X_cam0 + T_1_0_mm).
  2. A `turntable_calib.json` written by turntable_poses.py (Turntable Studio):
     the relative pose is derived from the two locked camera poses on the disc.

Tag scale: the AprilTag disc of the rig came out of the printer at ~98 % (disc 255 mm instead of 260,
tags 0-6 196 mm instead of 200; Turntable Studio measurement). Every calibration records the print scale
it was computed with (`tag_scale`, 1.0 when the nominal layout was used); `with_tag_scale()` rescales the
baseline to another print scale. Only the metric scale changes (intrinsics and rotations do not), so the
correction is applied once, on the baseline, and every millimetre downstream is real.

Why refine: the turntable model is fitted on tags that only appear in the lower
part of the frame, so a residual rotation of a few tenths of a degree remains
(≈ 7 px of vertical parallax at 12.3 MP). Dense stereo needs < 1 px. The
rotation and the direction of the baseline are re-estimated by robust least
squares on thousands of sub-pixel correspondences, while the baseline length
(metric scale) stays the calibrated one.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger("Photogrammetry.RigCalibration")

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "assets" / "rig_calibration.json"

# Refinement guard rails: larger corrections mean the cameras were moved or the
# correspondences are wrong, so the calibrated pose is kept.
MAX_ROTATION_CORRECTION_DEG = 3.0
MAX_BASELINE_DIRECTION_CORRECTION_DEG = 6.0


@dataclass
class RigCalibration:
    """Intrinsics of both cameras and the pose of camera 1 relative to camera 0."""
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R: np.ndarray              # rotation camera 0 -> camera 1
    T_mm: np.ndarray           # translation camera 0 -> camera 1 (mm)
    image_size: Tuple[int, int] = (4056, 3040)
    source: str = ""
    refinement: Dict[str, float] = field(default_factory=dict)
    tag_scale: float = 1.0     # print scale of the tag disc this calibration was computed with

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T_mm))

    @property
    def relative_rotation_deg(self) -> float:
        return float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R)[0])))

    def scaled(self, scale: float) -> "RigCalibration":
        """Returns the calibration for images resized by `scale` (pinhole intrinsics only)."""
        S = np.diag([scale, scale, 1.0])
        return RigCalibration(
            K0=S @ self.K0, D0=self.D0.copy(), K1=S @ self.K1, D1=self.D1.copy(),
            R=self.R.copy(), T_mm=self.T_mm.copy(),
            image_size=(int(round(self.image_size[0] * scale)), int(round(self.image_size[1] * scale))),
            source=self.source, refinement=dict(self.refinement), tag_scale=self.tag_scale,
        )

    def with_pose(self, R: np.ndarray, T_mm: np.ndarray) -> "RigCalibration":
        return RigCalibration(
            K0=self.K0.copy(), D0=self.D0.copy(), K1=self.K1.copy(), D1=self.D1.copy(),
            R=np.asarray(R, dtype=np.float64), T_mm=np.asarray(T_mm, dtype=np.float64).reshape(3),
            image_size=self.image_size, source=self.source, refinement=dict(self.refinement),
            tag_scale=self.tag_scale,
        )

    def with_tag_scale(self, tag_scale: float) -> "RigCalibration":
        """
        The calibration for a tag disc printed at `tag_scale` (measured / designed size). The tag
        coordinates set the metric scale, so the baseline scales by tag_scale / self.tag_scale.
        """
        tag_scale = float(tag_scale)
        if tag_scale <= 0:
            raise ValueError(f"tag_scale must be positive, got {tag_scale}")
        out = self.with_pose(self.R, self.T_mm * (tag_scale / self.tag_scale))
        out.tag_scale = tag_scale
        return out

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "image_size": list(self.image_size),
            "cam0": {"K": self.K0.tolist(), "dist": self.D0.tolist()},
            "cam1": {"K": self.K1.tolist(), "dist": self.D1.tolist()},
            "R_1_0": self.R.tolist(),
            "T_1_0_mm": self.T_mm.tolist(),
            "baseline_mm": round(self.baseline_mm, 3),
            "tag_scale": self.tag_scale,
            "relative_rotation_deg": round(self.relative_rotation_deg, 3),
            "refinement": self.refinement,
        }


def _from_turntable_calib(data: Dict, path: Path) -> RigCalibration:
    """Relative stereo pose from the locked camera poses of turntable_poses.py (disc frame, metres)."""
    p = data["params"]
    Ks, Ds, Rs, Cs = [], [], [], []
    for cam in data.get("cams", ["0", "1"])[:2]:
        f, cx, cy = p["K_%s" % cam]
        Ks.append(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]))
        k = p["D_%s" % cam]
        # AliceVision radialk3 == OpenCV (k1, k2, p1=0, p2=0, k3)
        Ds.append(np.array([k[0], k[1], 0.0, 0.0, k[2]]))
        Rs.append(cv2.Rodrigues(np.asarray(p["R_%s" % cam], dtype=np.float64))[0])
        Cs.append(np.asarray(p["C_%s" % cam], dtype=np.float64))
    R = Rs[1] @ Rs[0].T
    T = (-Rs[1] @ Cs[1]) - R @ (-Rs[0] @ Cs[0])
    w = int(data.get("intrinsics", {}).get("0", {}).get("width", 4056))
    h = int(data.get("intrinsics", {}).get("0", {}).get("height", 3040))
    # turntable_poses.py does not record --tagScale in its output; turntable_calibration.py adds it.
    return RigCalibration(Ks[0], Ds[0], Ks[1], Ds[1], R, T * 1000.0, (w, h), source=f"turntable_poses: {path}",
                          tag_scale=float(data.get("tag_scale", 1.0)))


def load_rig_calibration(path: Optional[Union[str, Path]] = None) -> RigCalibration:
    """Loads `assets/rig_calibration.json` or any compatible file (own format or turntable_calib.json)."""
    p = Path(path) if path else DEFAULT_CALIBRATION_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "params" in data and "K_0" in data["params"]:
        calib = _from_turntable_calib(data, p)
    else:
        calib = RigCalibration(
            K0=np.asarray(data["cam0"]["K"], dtype=np.float64),
            D0=np.asarray(data["cam0"]["dist"], dtype=np.float64),
            K1=np.asarray(data["cam1"]["K"], dtype=np.float64),
            D1=np.asarray(data["cam1"]["dist"], dtype=np.float64),
            R=np.asarray(data["R_1_0"], dtype=np.float64),
            T_mm=np.asarray(data["T_1_0_mm"], dtype=np.float64).reshape(3),
            image_size=tuple(data.get("image_size", (4056, 3040))),
            source=data.get("source", str(p)),
            tag_scale=float(data.get("tag_scale", 1.0)),
        )
    log.info(
        "Rig calibration loaded (%s): B = %.2f mm (tag scale %.3f), relative rotation = %.2f deg, f = %.1f / %.1f px",
        calib.source, calib.baseline_mm, calib.tag_scale, calib.relative_rotation_deg, calib.K0[0, 0], calib.K1[0, 0],
    )
    return calib


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def epipolar_errors_px(calib: RigCalibration, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """Symmetric point-to-epipolar-line distances (pixels) of raw image correspondences."""
    n0 = cv2.undistortPoints(pts0.reshape(-1, 1, 2).astype(np.float64), calib.K0, calib.D0).reshape(-1, 2)
    n1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2).astype(np.float64), calib.K1, calib.D1).reshape(-1, 2)
    h0 = np.c_[n0, np.ones(len(n0))]
    h1 = np.c_[n1, np.ones(len(n1))]
    E = _skew(calib.T_mm / np.linalg.norm(calib.T_mm)) @ calib.R
    l1 = h0 @ E.T
    l0 = h1 @ E
    e1 = np.abs(np.sum(l1 * h1, axis=1)) / np.hypot(l1[:, 0], l1[:, 1]) * calib.K1[0, 0]
    e0 = np.abs(np.sum(l0 * h0, axis=1)) / np.hypot(l0[:, 0], l0[:, 1]) * calib.K0[0, 0]
    return 0.5 * (e0 + e1)


def refine_relative_pose(
    calib: RigCalibration,
    pts0: np.ndarray,
    pts1: np.ndarray,
    robust_scale_px: float = 1.0,
) -> RigCalibration:
    """
    Refines R and the baseline direction on correspondences (raw pixel coordinates of
    the full-resolution images). The baseline length is kept, so the scale stays metric.
    Returns the calibration unchanged when the correction is implausibly large.
    """
    pts0 = np.asarray(pts0, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64)
    n0 = cv2.undistortPoints(pts0.reshape(-1, 1, 2), calib.K0, calib.D0).reshape(-1, 2)
    n1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), calib.K1, calib.D1).reshape(-1, 2)
    h0 = np.c_[n0, np.ones(len(n0))]
    h1 = np.c_[n1, np.ones(len(n1))]
    f = 0.5 * (calib.K0[0, 0] + calib.K1[0, 0])
    t_dir = calib.T_mm / np.linalg.norm(calib.T_mm)

    # The baseline direction is perturbed by a rotation about the two axes orthogonal to it.
    a1 = np.cross(t_dir, [0.0, 1.0, 0.0])
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(t_dir, a1)

    def pose(x):
        R = cv2.Rodrigues(x[:3])[0] @ calib.R
        t = cv2.Rodrigues(x[3] * a1 + x[4] * a2)[0] @ t_dir
        return R, t

    def residuals(x):
        R, t = pose(x)
        E = _skew(t) @ R
        l1 = h0 @ E.T
        return np.sum(l1 * h1, axis=1) / np.hypot(l1[:, 0], l1[:, 1]) * f

    before = np.abs(residuals(np.zeros(5)))
    sol = least_squares(residuals, np.zeros(5), loss="cauchy", f_scale=robust_scale_px)
    after = np.abs(residuals(sol.x))
    rot_deg = math.degrees(np.linalg.norm(sol.x[:3]))
    dir_deg = math.degrees(np.linalg.norm(sol.x[3:]))
    stats = {
        "correspondences": int(len(pts0)),
        "epipolar_median_before_px": round(float(np.median(before)), 3),
        "epipolar_median_after_px": round(float(np.median(after)), 3),
        "inliers_1px_after": round(float(np.mean(after < 1.0)), 3),
        "rotation_correction_deg": round(rot_deg, 3),
        "baseline_direction_correction_deg": round(dir_deg, 3),
    }
    if rot_deg > MAX_ROTATION_CORRECTION_DEG or dir_deg > MAX_BASELINE_DIRECTION_CORRECTION_DEG:
        log.warning(
            "Pose refinement rejected (rotation %.2f deg, baseline direction %.2f deg): "
            "the rig was probably moved; keeping the calibrated pose.", rot_deg, dir_deg,
        )
        stats["accepted"] = False
        calib.refinement = stats
        return calib

    R, t = pose(sol.x)
    refined = calib.with_pose(R, t * calib.baseline_mm)
    stats["accepted"] = True
    refined.refinement = stats
    log.info(
        "Relative pose refined on %d correspondences: epipolar median %.2f -> %.2f px "
        "(rotation %.3f deg, baseline direction %.3f deg)",
        len(pts0), stats["epipolar_median_before_px"], stats["epipolar_median_after_px"], rot_deg, dir_deg,
    )
    return refined
