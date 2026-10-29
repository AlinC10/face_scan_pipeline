"""
Rig Calibration from the Face Pair (AprilTag disc visible under the subject)
============================================================================
When the face photographs show (part of) the AprilTag disc, the stereo calibration of
this capture is estimated from those two images alone, so a focus or lens change since
the last full calibration is caught.

turntable_poses.py cannot be used on one pair: a full fit needs several turntable
positions, and --calibIn needs the turntable angle and the carriage height of the photos,
which a face capture does not know (tested: the refinement diverges, RMS ~580 px). The
disc angle does not matter for the stereo pair, so the model here has no turntable:

    x_0 = proj(K_0(f_0), D_0, R_d X + t_d)              camera 0 <- disc
    x_1 = proj(K_1(f_1), D_1, R (R_d X + t_d) + T)      camera 1 <- camera 0

with X the tag corners of the printed layout (mm, x tag_scale). Unknowns: disc pose (6),
relative pose (6), the two focal lengths (2). Initialisation: PnP per camera with the stored
intrinsics; a weak prior keeps f, R, T near the stored calibration (few corners, one plane).
The result is accepted only within 2 % of the stored focal length, 3 mm of its baseline and
1 deg of its relative rotation; otherwise the stored calibration is used (and the log says why).
Measured on single pairs of the disc sweep (arbitrary turntable angles): see the unit tests and
PLAN_IMPLEMENTARE_FATA.md, P1.7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import least_squares

from photogrammetry.rig_calibration import RigCalibration
from photogrammetry.turntable_calibration import TAG_LAYOUT_PATH

log = logging.getLogger("Photogrammetry.PairCalibration")

MIN_TAGS_PER_CAMERA = 4
MAX_FOCAL_CHANGE = 0.02          # relative, vs the stored calibration
MAX_BASELINE_CHANGE_MM = 3.0
MAX_ROTATION_CHANGE_DEG = 1.0
PRIOR_FOCAL_SIGMA = 0.01         # relative
PRIOR_ROTATION_SIGMA_DEG = 2.0
PRIOR_TRANSLATION_SIGMA_MM = 5.0


@dataclass
class PairCalibrationResult:
    calibration: Optional[RigCalibration]       # None when rejected or not enough tags
    accepted: bool
    reason: str
    stats: Dict[str, float] = field(default_factory=dict)


def load_tag_layout_mm(layout_path: Union[str, Path] = TAG_LAYOUT_PATH, tag_scale: float = 1.0) -> Dict[int, np.ndarray]:
    """{tag id: (4, 3) corners in mm} of the printed disc (layout in metres, OpenCV corner order)."""
    from photogrammetry import turntable_poses
    return {i: c * 1000.0 for i, c in turntable_poses.load_layout(str(layout_path), tag_scale).items()}


def detect_tags(image_path: Union[str, Path]) -> Dict[int, np.ndarray]:
    """{tag id: (4, 2) corners in pixels}, with the detector settings of turntable_poses.py."""
    from photogrammetry import turntable_poses
    _, det = turntable_poses.detect_tags(str(image_path))
    return det or {}


def _correspondences(det: Dict[int, np.ndarray], layout: Dict[int, np.ndarray]):
    ids = sorted(i for i in det if i in layout)
    if not ids:
        return ids, np.zeros((0, 3)), np.zeros((0, 2))
    return ids, np.concatenate([layout[i] for i in ids]), np.concatenate([det[i] for i in ids])


def calibrate_pair_from_tags(
    det0: Dict[int, np.ndarray],
    det1: Dict[int, np.ndarray],
    base: RigCalibration,
    layout_mm: Dict[int, np.ndarray],
) -> PairCalibrationResult:
    """Stereo calibration of one pair from the disc tags of both images; `base` = stored calibration (same tag scale)."""
    ids0, X0, u0 = _correspondences(det0, layout_mm)
    ids1, X1, u1 = _correspondences(det1, layout_mm)
    stats = {"tags_cam0": len(ids0), "tags_cam1": len(ids1)}
    if len(ids0) < MIN_TAGS_PER_CAMERA or len(ids1) < MIN_TAGS_PER_CAMERA:
        return PairCalibrationResult(None, False, f"not enough disc tags in the face pair "
                                     f"({len(ids0)} / {len(ids1)}, need {MIN_TAGS_PER_CAMERA} per camera)", stats)

    # Initialisation: PnP per camera with the stored intrinsics
    ok0, r0, t0 = cv2.solvePnP(X0, u0, base.K0, base.D0, flags=cv2.SOLVEPNP_IPPE)
    ok1, r1, t1 = cv2.solvePnP(X1, u1, base.K1, base.D1, flags=cv2.SOLVEPNP_IPPE)
    if not (ok0 and ok1):
        return PairCalibrationResult(None, False, "PnP on the disc tags failed", stats)
    R0, R1 = cv2.Rodrigues(r0)[0], cv2.Rodrigues(r1)[0]
    R_init = R1 @ R0.T
    T_init = t1.ravel() - R_init @ t0.ravel()
    # The relative pose starts from the stored one (more reliable than two single-plane PnPs); PnP sets the disc
    x_init = np.concatenate([r0.ravel(), t0.ravel(), np.zeros(3), np.zeros(3), np.zeros(2)])
    R_base, T_base = base.R, base.T_mm

    def unpack(x):
        Rd = cv2.Rodrigues(x[0:3])[0]
        td = x[3:6]
        R = cv2.Rodrigues(x[6:9])[0] @ R_base
        T = T_base + x[9:12]
        K0 = base.K0.copy()
        K1 = base.K1.copy()
        K0[0, 0] *= np.exp(x[12])
        K0[1, 1] *= np.exp(x[12])
        K1[0, 0] *= np.exp(x[13])
        K1[1, 1] *= np.exp(x[13])
        return Rd, td, R, T, K0, K1

    def residuals(x):
        Rd, td, R, T, K0, K1 = unpack(x)
        p0, _ = cv2.projectPoints(X0, cv2.Rodrigues(Rd)[0], td, K0, base.D0)
        Rc1 = R @ Rd
        tc1 = R @ td + T
        p1, _ = cv2.projectPoints(X1, cv2.Rodrigues(Rc1)[0], tc1, K1, base.D1)
        prior = np.concatenate([
            x[12:14] / PRIOR_FOCAL_SIGMA,
            x[6:9] / np.radians(PRIOR_ROTATION_SIGMA_DEG),
            x[9:12] / PRIOR_TRANSLATION_SIGMA_MM,
        ])
        return np.concatenate([(p0.reshape(-1, 2) - u0).ravel(), (p1.reshape(-1, 2) - u1).ravel(), prior])

    sol = least_squares(residuals, x_init, loss="huber", f_scale=2.0, x_scale="jac")
    Rd, td, R, T, K0, K1 = unpack(sol.x)
    n_obs = 2 * (len(u0) + len(u1))
    reproj = np.linalg.norm(residuals(sol.x)[:n_obs].reshape(-1, 2), axis=1)
    calib = RigCalibration(K0, base.D0.copy(), K1, base.D1.copy(), R, T, base.image_size,
                           source=f"face pair tags ({len(u0) + len(u1)} corners)", tag_scale=base.tag_scale)
    d_rot = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R @ R_base.T)[0])))
    stats.update({
        "corners": int(len(u0) + len(u1)),
        "reprojection_median_px": round(float(np.median(reproj)), 3),
        "reprojection_rms_px": round(float(np.sqrt(np.mean(reproj ** 2))), 3),
        "focal_change": [round(float(np.expm1(sol.x[12])), 4), round(float(np.expm1(sol.x[13])), 4)],
        "baseline_mm": round(calib.baseline_mm, 2),
        "baseline_change_mm": round(calib.baseline_mm - base.baseline_mm, 2),
        "rotation_change_deg": round(d_rot, 3),
        "pnp_relative_rotation_deg": round(float(np.degrees(np.linalg.norm(cv2.Rodrigues(R_init)[0]))), 2),
        "pnp_baseline_mm": round(float(np.linalg.norm(T_init)), 2),
    })
    problems = []
    if max(abs(v) for v in stats["focal_change"]) > MAX_FOCAL_CHANGE:
        problems.append(f"focal length changed by {stats['focal_change']} (> {MAX_FOCAL_CHANGE:.0%})")
    if abs(stats["baseline_change_mm"]) > MAX_BASELINE_CHANGE_MM:
        problems.append(f"baseline changed by {stats['baseline_change_mm']} mm (> {MAX_BASELINE_CHANGE_MM} mm)")
    if d_rot > MAX_ROTATION_CHANGE_DEG:
        problems.append(f"relative rotation changed by {d_rot:.2f} deg (> {MAX_ROTATION_CHANGE_DEG} deg)")
    calib.source += f", RMS {stats['reprojection_rms_px']} px"
    if problems:
        return PairCalibrationResult(None, False, "; ".join(problems), stats)
    return PairCalibrationResult(calib, True, "ok", stats)


def calibrate_face_pair(
    left_path: Union[str, Path],
    right_path: Union[str, Path],
    base: RigCalibration,
    tag_scale: float,
    layout_path: Union[str, Path] = TAG_LAYOUT_PATH,
) -> PairCalibrationResult:
    """Detects the disc tags in both photographs and calibrates the pair (base rescaled to `tag_scale` first)."""
    base = base.with_tag_scale(tag_scale)
    layout = load_tag_layout_mm(layout_path, tag_scale)
    res = calibrate_pair_from_tags(detect_tags(left_path), detect_tags(right_path), base, layout)
    log.info("Face-pair tag calibration: %s (%s)", "accepted" if res.accepted else "not used", res.reason)
    return res


def save_calibration(calib: RigCalibration, path: Union[str, Path], stats: Dict) -> Path:
    """Writes the calibration in the own format of assets/rig_calibration.json (readable by load_rig_calibration)."""
    import json
    d = calib.to_dict()
    out = {"description": "Stereo calibration of this capture from the AprilTag disc in the face pair",
           "source": calib.source, "tag_scale": calib.tag_scale, "image_size": d["image_size"],
           "cam0": d["cam0"], "cam1": d["cam1"], "R_1_0": d["R_1_0"], "T_1_0_mm": d["T_1_0_mm"], "stats": stats}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return p
