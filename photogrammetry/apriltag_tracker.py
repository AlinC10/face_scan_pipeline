"""
AprilTag Metric Tracker & Stereo Rig Calibrator
===============================================
Extracts sub-pixel metric AprilTag coordinates (family tag16h5) from dual-camera
photogrammetry captures, calculates camera extrinsics via PnP in meters, and
computes Euclidean metric stereo rectification and baseline ground truth.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger("Photogrammetry.AprilTag")

# Default optical constants for Arducam IMX477 + M23390H08 lens
DEFAULT_FOCAL_PX = 2516.13  # f = 3.9 mm / 1.55 um pixel pitch
DEFAULT_IMG_W = 4056
DEFAULT_IMG_H = 3040


@dataclass
class AprilTagPoseResult:
    """Metric camera extrinsics and stereo geometry derived from AprilTags."""
    tags_cam0: List[int]
    tags_cam1: List[int]
    common_tags: List[int]
    
    # Camera 0 Extrinsics (world in meters, Y=up)
    rvec_0: np.ndarray
    tvec_0: np.ndarray
    R_0: np.ndarray
    C_0: np.ndarray  # Camera 0 center in world (meters)
    
    # Camera 1 Extrinsics
    rvec_1: np.ndarray
    tvec_1: np.ndarray
    R_1: np.ndarray
    C_1: np.ndarray  # Camera 1 center in world (meters)
    
    # Relative Stereo Extrinsics
    R_rel: np.ndarray      # R1 @ R0.T
    T_rel: np.ndarray      # t1 - R_rel @ t0 (meters)
    baseline_m: float      # Euclidean distance between camera optical centers (m)
    baseline_mm: float     # Baseline in millimeters
    convergence_deg: float # Angle between optical axes (degrees)
    
    # Distance to turntable origin (meters)
    dist_cam0_m: float
    dist_cam1_m: float
    
    # Reprojection errors (pixels)
    rms_cam0: float
    rms_cam1: float
    
    # Stereo Rectification Maps (Bouguet algorithm)
    rect_R0: Optional[np.ndarray] = None
    rect_R1: Optional[np.ndarray] = None
    rect_P0: Optional[np.ndarray] = None
    rect_P1: Optional[np.ndarray] = None
    rect_Q: Optional[np.ndarray] = None
    roi_0: Optional[Tuple[int, int, int, int]] = None
    roi_1: Optional[Tuple[int, int, int, int]] = None


class AprilTagTracker:
    """
    Detector and metric PnP solver for AprilTag fiducial markers.
    Default uses tag16h5 geometry from turntable / rig plate.
    """

    def __init__(
        self,
        layout_path: Optional[Union[str, Path]] = None,
        tag_scale: float = 1.0,
        focal_px: float = DEFAULT_FOCAL_PX,
        img_width: int = DEFAULT_IMG_W,
        img_height: int = DEFAULT_IMG_H,
    ):
        self.tag_scale = float(tag_scale)
        self.focal_px = float(focal_px)
        self.img_w = int(img_width)
        self.img_h = int(img_height)

        # Build default intrinsic matrix K
        self.K_default = np.array([
            [self.focal_px, 0.0, self.img_w / 2.0],
            [0.0, self.focal_px, self.img_h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Load layout definition
        self.layout_corners = self._load_layout(layout_path)

        # Initialize OpenCV ArUco AprilTag detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        params = cv2.aruco.DetectorParameters()
        params.errorCorrectionRate = 0.0
        params.minMarkerPerimeterRate = 0.005  # Detect markers down to ~15-20 px
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def _load_layout(self, path: Optional[Union[str, Path]]) -> Dict[int, np.ndarray]:
        """Loads 3D corner coordinates (in meters) for each tag ID."""
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.append(Path(__file__).parent.parent / "assets" / "platan_260mm_layout.json")
        candidates.append(Path(__file__).parent / "platan_260mm_layout.json")

        for cand in candidates:
            if cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    corners_dict = {}
                    for t in data["tags"]:
                        corners_dict[int(t["id"])] = np.asarray(t["corners"], dtype=np.float64) * self.tag_scale
                    log.info("Loaded AprilTag layout with %d markers from: %s", len(corners_dict), cand.name)
                    return corners_dict
                except Exception as e:
                    log.warning("Failed reading layout from %s: %s", cand, e)

        # Fallback procedural ring: 12 tags, radius 100mm, side 30mm
        log.info("Generating fallback circular AprilTag layout (12 tags, r=100mm, size=30mm)...")
        return self._procedural_ring_layout(n=12, radius_m=0.1, size_m=0.03, start_deg=15.0)

    def _procedural_ring_layout(
        self, n: int = 12, radius_m: float = 0.1, size_m: float = 0.03, start_deg: float = 15.0
    ) -> Dict[int, np.ndarray]:
        tags = {}
        h = (size_m / 2.0) * self.tag_scale
        r = radius_m * self.tag_scale
        for i in range(n):
            a = math.radians(start_deg + i * 360.0 / n)
            b = a - math.pi / 2.0
            ca, sa = math.cos(b), math.sin(b)
            cx, cy = r * math.cos(a), r * math.sin(a)
            corners = []
            for lx, ly in ((-h, h), (h, h), (h, -h), (-h, -h)):
                px = ca * lx - sa * ly + cx
                py = sa * lx + ca * ly + cy
                corners.append([px, 0.0, -py])  # World: Y is UP, X is right, Z is front
            tags[i] = np.array(corners, dtype=np.float64)
        return tags

    def detect_tags(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        search_bottom_half: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        Detects AprilTags in the image and returns a dict mapping tag_id -> 4x2 array of pixel corners.
        Corners order matches OpenCV ArUco: [top_left, top_right, bottom_right, bottom_left].
        """
        if isinstance(image_or_path, (str, Path)):
            img = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {image_or_path}")
        else:
            if len(image_or_path.shape) == 3:
                img = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
            else:
                img = image_or_path

        y_offset = 0
        if search_bottom_half:
            # Optimize search on the bottom 45% where the turntable floor rests
            h = img.shape[0]
            y_offset = int(h * 0.55)
            img = img[y_offset:, :]

        corners, ids, _ = self.detector.detectMarkers(img)
        detections: Dict[int, np.ndarray] = {}

        if ids is not None and len(ids) > 0:
            for c, i in zip(corners, ids.flatten()):
                tag_id = int(i)
                c_pts = c.reshape(4, 2).astype(np.float64)
                if y_offset > 0:
                    c_pts[:, 1] += y_offset
                detections[tag_id] = c_pts

        return detections

    def estimate_single_camera_pose(
        self,
        detected_tags: Dict[int, np.ndarray],
        K: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Solves PnP to find camera extrinsics [R | t] in world coordinate frame (meters).
        Returns (rvec, tvec, R, C_camera_center, rms_reproj_err) or None if insufficient tags.
        """
        if K is None:
            K = self.K_default
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5, dtype=np.float64)

        common_ids = [i for i in detected_tags if i in self.layout_corners]
        if len(common_ids) == 0:
            return None

        # Gather corresponding 3D world points and 2D image coordinates
        obj_pts = np.concatenate([self.layout_corners[i] for i in common_ids]).astype(np.float64)
        img_pts = np.concatenate([detected_tags[i] for i in common_ids]).astype(np.float64)

        if len(obj_pts) < 8:
            log.warning("PnP requires at least 2 detected tags (8 coplanar corners) to eliminate planar flip ambiguity (found %d points).", len(obj_pts))
            return None

        # Solve PnP: use SQPNP (or IPPE for planar) which is robust across OpenCV 4.x and 5.x
        pnp_flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            dist_coeffs,
            flags=pnp_flag
        )

        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        # Optical center in world coordinates: C = -R^T * t
        C = (-R.T @ tvec).ravel()

        # Calculate reprojection RMS error
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
        proj_pts = proj_pts.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((img_pts - proj_pts) ** 2, axis=1))))

        return rvec, tvec, R, C, rms

    def calibrate_stereo_pair(
        self,
        img_left: Union[str, Path, np.ndarray],
        img_right: Union[str, Path, np.ndarray],
        K_left: Optional[np.ndarray] = None,
        K_right: Optional[np.ndarray] = None,
        dist_left: Optional[np.ndarray] = None,
        dist_right: Optional[np.ndarray] = None,
    ) -> Optional[AprilTagPoseResult]:
        """
        Detects AprilTags in both stereo views and computes the physical metric baseline B,
        relative pose, convergence angle, and calibrated Euclidean stereo rectification maps.
        """
        tags_l = self.detect_tags(img_left)
        tags_r = self.detect_tags(img_right)

        log.info("AprilTag detection: Cam 0 found %d markers, Cam 1 found %d markers.", len(tags_l), len(tags_r))

        # Determine effective image dimensions
        img_size = (self.img_w, self.img_h)
        if isinstance(img_left, np.ndarray):
            h, w = img_left.shape[:2]
            img_size = (w, h)
        elif isinstance(img_left, (str, Path)) and Path(img_left).is_file():
            hdr = cv2.imread(str(img_left))
            if hdr is not None:
                h, w = hdr.shape[:2]
                img_size = (w, h)

        if K_left is None:
            K_left = self.K_default.copy()
        else:
            K_left = K_left.copy()

        if K_right is None:
            K_right = self.K_default.copy()
        else:
            K_right = K_right.copy()

        # Adapt intrinsics K if resolution differs from native 12.3MP
        scale_x = img_size[0] / self.img_w
        scale_y = img_size[1] / self.img_h
        if abs(scale_x - 1.0) > 1e-4 or abs(scale_y - 1.0) > 1e-4:
            K_left[0, 0] *= scale_x
            K_left[0, 2] *= scale_x
            K_left[1, 1] *= scale_y
            K_left[1, 2] *= scale_y

            K_right[0, 0] *= scale_x
            K_right[0, 2] *= scale_x
            K_right[1, 1] *= scale_y
            K_right[1, 2] *= scale_y

        if dist_left is None:
            dist_left = np.zeros(5, dtype=np.float64)
        if dist_right is None:
            dist_right = np.zeros(5, dtype=np.float64)

        pose_0 = self.estimate_single_camera_pose(tags_l, K_left, dist_left)
        pose_1 = self.estimate_single_camera_pose(tags_r, K_right, dist_right)

        if pose_0 is None or pose_1 is None:
            log.warning("Insufficient AprilTags in one or both views to perform metric PnP calibration.")
            return None

        rvec0, tvec0, R0, C0, rms0 = pose_0
        rvec1, tvec1, R1, C1, rms1 = pose_1

        # Relative transformation: Cam 0 -> Cam 1
        # X_cam1 = R_rel * X_cam0 + T_rel
        R_rel = R1 @ R0.T
        T_rel = tvec1 - R_rel @ tvec0

        # Physical baseline between centers
        baseline_m = float(np.linalg.norm(C1 - C0))
        baseline_mm = baseline_m * 1000.0

        # Convergence angle between optical axes: z_cam = R^T * [0, 0, 1]
        z_axis_0 = R0.T @ np.array([0.0, 0.0, 1.0])
        z_axis_1 = R1.T @ np.array([0.0, 0.0, 1.0])
        cos_conv = float(np.dot(z_axis_0, z_axis_1) / (np.linalg.norm(z_axis_0) * np.linalg.norm(z_axis_1)))
        cos_conv = max(-1.0, min(1.0, cos_conv))
        convergence_deg = float(math.degrees(math.acos(cos_conv)))

        dist_c0 = float(np.linalg.norm(C0))
        dist_c1 = float(np.linalg.norm(C1))

        # Common tag IDs
        common_ids = sorted(set(tags_l.keys()) & set(tags_r.keys()))

        log.info("Metric Stereo Rig Calibrated via AprilTags:")
        log.info("  • True Physical Baseline B: %.2f mm (%.4f m)", baseline_mm, baseline_m)
        log.info("  • Optical Convergence Angle: %.2f°", convergence_deg)
        log.info("  • Cam 0 Distance: %.3f m (RMS: %.2f px)", dist_c0, rms0)
        log.info("  • Cam 1 Distance: %.3f m (RMS: %.2f px)", dist_c1, rms1)

        # Compute Euclidean stereo rectification using Bouguet algorithm
        img_size = (self.img_w, self.img_h)
        R_rect0, R_rect1, P_rect0, P_rect1, Q_mat, roi0, roi1 = cv2.stereoRectify(
            cameraMatrix1=K_left,
            distCoeffs1=dist_left,
            cameraMatrix2=K_right,
            distCoeffs2=dist_right,
            imageSize=img_size,
            R=R_rel,
            T=T_rel,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,  # Crop black borders
        )

        return AprilTagPoseResult(
            tags_cam0=sorted(tags_l.keys()),
            tags_cam1=sorted(tags_r.keys()),
            common_tags=common_ids,
            rvec_0=rvec0,
            tvec_0=tvec0,
            R_0=R0,
            C_0=C0,
            rvec_1=rvec1,
            tvec_1=tvec1,
            R_1=R1,
            C_1=C1,
            R_rel=R_rel,
            T_rel=T_rel,
            baseline_m=baseline_m,
            baseline_mm=baseline_mm,
            convergence_deg=convergence_deg,
            dist_cam0_m=dist_c0,
            dist_cam1_m=dist_c1,
            rms_cam0=rms0,
            rms_cam1=rms1,
            rect_R0=R_rect0,
            rect_R1=R_rect1,
            rect_P0=P_rect0,
            rect_P1=P_rect1,
            rect_Q=Q_mat,
            roi_0=roi0,
            roi_1=roi1
        )
