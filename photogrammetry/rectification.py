"""
Bouguet Calibrated Stereo Rectification Module
==============================================
Transforms converged dual-camera stereo images (27° convergence) into strictly
parallel epipolar geometry where corresponding epipolar scanlines are horizontal.

Supports:
  1. Ground-truth AprilTag Euclidean calibration matrices (from Etapa 1).
  2. Nominal physical rig geometry prior (B = 163.4 mm, Convergence = 27°, Pitch = 15°).
  3. Feature-based epipolar refinement when AprilTags are absent from the view.
  4. Dual-scale rectification (2K for fast dense neural matching, 12.3MP for texture baking).
"""

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np

from photogrammetry.apriltag_tracker import DEFAULT_FOCAL_PX
from photogrammetry.ingestion import StereoIngestionResult

log = logging.getLogger("Photogrammetry.Rectification")

# Optical & Geometric Rig Defaults
NOMINAL_BASELINE_MM = 163.4
NOMINAL_CONVERGENCE_DEG = 27.0
NOMINAL_PITCH_DEG = 15.0
NOMINAL_WORKING_DIST_M = 0.35

NATIVE_WIDTH = 4056
NATIVE_HEIGHT = 3040
WORKING_2K_WIDTH = 2048
WORKING_2K_HEIGHT = 1536


@dataclass
class StereoRectificationResult:
    """Contains rectified images, epipolar transformation matrices, and re-projection Q."""
    pair_id: str
    rect_left_2k: np.ndarray          # 2048 x 1536 rectified BGR image
    rect_right_2k: np.ndarray         # 2048 x 1536 rectified BGR image
    Q_2k: np.ndarray                  # 4x4 Disparity-to-Depth mapping matrix
    P_rect0: np.ndarray               # 3x4 projection matrix Cam 0
    P_rect1: np.ndarray               # 3x4 projection matrix Cam 1
    R_rect0: np.ndarray               # 3x3 rectification rotation Cam 0
    R_rect1: np.ndarray               # 3x3 rectification rotation Cam 1
    baseline_mm: float
    convergence_deg: float
    effective_focal_2k: float
    is_apriltag_calibrated: bool
    scale_factor_2k: float = WORKING_2K_WIDTH / NATIVE_WIDTH
    rect_left_native: Optional[np.ndarray] = None   # Optional 4056 x 3040 for texture baking
    rect_right_native: Optional[np.ndarray] = None
    Q_native: Optional[np.ndarray] = None

    def save_rectified_pair(self, output_dir: Union[str, Path], save_native: bool = False) -> Dict[str, Path]:
        """Saves rectified images to disk."""
        out_d = Path(output_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)
        paths = {}

        p_l_2k = out_d / f"{self.pair_id}_rect_0_2k.jpg"
        p_r_2k = out_d / f"{self.pair_id}_rect_1_2k.jpg"
        cv2.imwrite(str(p_l_2k), self.rect_left_2k, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(p_r_2k), self.rect_right_2k, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths["rect_left_2k"] = p_l_2k
        paths["rect_right_2k"] = p_r_2k

        if save_native and self.rect_left_native is not None:
            p_l_nat = out_d / f"{self.pair_id}_rect_0_12mp.jpg"
            p_r_nat = out_d / f"{self.pair_id}_rect_1_12mp.jpg"
            cv2.imwrite(str(p_l_nat), self.rect_left_native, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(p_r_nat), self.rect_right_native, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths["rect_left_native"] = p_l_nat
            paths["rect_right_native"] = p_r_nat

        log.info("Saved rectified stereo pair to: %s", out_d)
        return paths

    def generate_epipolar_inspection_image(self, num_scanlines: int = 24) -> np.ndarray:
        """
        Creates a side-by-side composite with horizontal colored scanlines to visually
        verify that epipolar rectification has zero vertical parallax.
        """
        h, w = self.rect_left_2k.shape[:2]
        composite = np.hstack([self.rect_left_2k, self.rect_right_2k])

        # Draw alternating horizontal epipolar inspection lines across the entire pair
        step = h // (num_scanlines + 1)
        colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 200)]
        for idx in range(1, num_scanlines + 1):
            y = idx * step
            col = colors[idx % len(colors)]
            cv2.line(composite, (0, y), (2 * w, y), col, 1, cv2.LINE_AA)

        # Labels
        cv2.putText(composite, "CAM 0 (LEFT RECTIFIED)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(composite, "CAM 1 (RIGHT RECTIFIED)", (w + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return composite


class StereoRectifier:
    """
    High-precision epipolar stereo rectifier implementing Bouguet's algorithm.
    """

    def __init__(
        self,
        nominal_baseline_mm: float = NOMINAL_BASELINE_MM,
        nominal_convergence_deg: float = NOMINAL_CONVERGENCE_DEG,
        focal_length_px: float = DEFAULT_FOCAL_PX,
        working_width: int = WORKING_2K_WIDTH,
        working_height: int = WORKING_2K_HEIGHT,
    ):
        self.nominal_baseline_mm = nominal_baseline_mm
        self.nominal_convergence_deg = nominal_convergence_deg
        self.focal_length_px = focal_length_px
        self.working_w = working_width
        self.working_h = working_height

        # Precompute nominal relative extrinsics (R_rel, T_rel)
        self.R_rel_nominal, self.T_rel_nominal = self._compute_nominal_extrinsics()

    def _compute_nominal_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the theoretical relative rotation and translation between
        Camera 0 (-13.5° yaw) and Camera 1 (+13.5° yaw) with physical baseline B = 163.4 mm.
        """
        half_conv = math.radians(self.nominal_convergence_deg / 2.0)
        B_m = self.nominal_baseline_mm / 1000.0

        # Camera 0 (Left): Yaw = +13.5° (turns right towards subject at origin)
        yaw0 = half_conv
        R0 = np.array([
            [math.cos(yaw0), 0, -math.sin(yaw0)],
            [0, 1, 0],
            [math.sin(yaw0), 0, math.cos(yaw0)],
        ], dtype=np.float64)
        C0 = np.array([-B_m / 2.0, 0.0, NOMINAL_WORKING_DIST_M], dtype=np.float64)
        t0 = -R0 @ C0

        # Camera 1 (Right): Yaw = -13.5° (turns left towards subject at origin)
        yaw1 = -half_conv
        R1 = np.array([
            [math.cos(yaw1), 0, -math.sin(yaw1)],
            [0, 1, 0],
            [math.sin(yaw1), 0, math.cos(yaw1)],
        ], dtype=np.float64)
        C1 = np.array([+B_m / 2.0, 0.0, NOMINAL_WORKING_DIST_M], dtype=np.float64)
        t1 = -R1 @ C1

        R_rel = R1 @ R0.T
        T_rel = (t1 - R_rel @ t0).reshape(3, 1)
        return R_rel, T_rel

    def _get_intrinsic_matrix(self, width: int, height: int) -> np.ndarray:
        """Constructs camera intrinsic matrix K scaled for (width, height)."""
        scale_x = width / float(NATIVE_WIDTH)
        scale_y = height / float(NATIVE_HEIGHT)
        fx = self.focal_length_px * scale_x
        fy = self.focal_length_px * scale_y
        cx = width / 2.0
        cy = height / 2.0
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

    def rectify_pair(
        self,
        img_left: Union[str, Path, np.ndarray],
        img_right: Union[str, Path, np.ndarray],
        ingestion_result: Optional[StereoIngestionResult] = None,
        rectify_native: bool = False,
        refine_with_features: bool = True,
    ) -> StereoRectificationResult:
        """
        Executes calibrated Bouguet stereo rectification on an input pair.
        """
        # 1. Load images
        if isinstance(img_left, (str, Path)):
            p_l = Path(img_left)
            pair_id = p_l.stem
            mat_l = cv2.imread(str(p_l))
            if mat_l is None:
                raise FileNotFoundError(f"Cannot load left image: {p_l}")
        else:
            mat_l = img_left
            pair_id = ingestion_result.pair_id if ingestion_result else "stereo_pair"

        if isinstance(img_right, (str, Path)):
            p_r = Path(img_right)
            mat_r = cv2.imread(str(p_r))
            if mat_r is None:
                raise FileNotFoundError(f"Cannot load right image: {p_r}")
        else:
            mat_r = img_right

        h_nat, w_nat = mat_l.shape[:2]

        # 2. Extract Extrinsics (AprilTag ground truth or Nominal Priors)
        is_apriltag = False
        if ingestion_result and ingestion_result.has_apriltag_metric_ground and ingestion_result.apriltag_calib:
            calib = ingestion_result.apriltag_calib
            if "R_rel" in calib and "T_rel" in calib:
                R_rel = np.array(calib["R_rel"], dtype=np.float64)
                T_rel = np.array(calib["T_rel"], dtype=np.float64).reshape(3, 1)
                baseline_mm = float(calib.get("baseline_mm", self.nominal_baseline_mm))
                convergence_deg = float(calib.get("convergence_deg", self.nominal_convergence_deg))
                is_apriltag = True
                log.info("Applying Ground-Truth AprilTag Extrinsics (B = %.2f mm, Conv = %.2f°)", baseline_mm, convergence_deg)
            else:
                R_rel, T_rel = self.R_rel_nominal, self.T_rel_nominal
                baseline_mm, convergence_deg = self.nominal_baseline_mm, self.nominal_convergence_deg
        else:
            R_rel, T_rel = self.R_rel_nominal, self.T_rel_nominal
            baseline_mm, convergence_deg = self.nominal_baseline_mm, self.nominal_convergence_deg
            log.info("Applying Nominal Rig Geometry Prior (B = %.2f mm, Conv = %.2f°)", baseline_mm, convergence_deg)

        # 3. Optional Feature Refinement if AprilTags not available and requested
        if not is_apriltag and refine_with_features:
            R_rel_refined, T_rel_refined = self._refine_extrinsics_with_sift(mat_l, mat_r, R_rel, T_rel, w_nat, h_nat)
            if R_rel_refined is not None:
                R_rel = R_rel_refined
                T_rel = T_rel_refined
                log.info("Refined relative extrinsics using robust epipolar feature matching.")

        # 4. Compute Bouguet Rectification for Working 2K Resolution
        K_2k = self._get_intrinsic_matrix(self.working_w, self.working_h)
        dist_coeffs = np.zeros(5, dtype=np.float64)

        # alpha=-1 preserves the true focal length without aggressive zooming
        R_rect0, R_rect1, P_rect0, P_rect1, Q_2k, _, _ = cv2.stereoRectify(
            cameraMatrix1=K_2k,
            distCoeffs1=dist_coeffs,
            cameraMatrix2=K_2k,
            distCoeffs2=dist_coeffs,
            imageSize=(self.working_w, self.working_h),
            R=R_rel,
            T=T_rel,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
        )

        # 5. Compute Remap Maps and Remap to 2K
        map0_x, map0_y = cv2.initUndistortRectifyMap(
            K_2k, dist_coeffs, R_rect0, P_rect0, (self.working_w, self.working_h), cv2.CV_32FC1
        )
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            K_2k, dist_coeffs, R_rect1, P_rect1, (self.working_w, self.working_h), cv2.CV_32FC1
        )

        img_l_2k = cv2.resize(mat_l, (self.working_w, self.working_h), interpolation=cv2.INTER_AREA)
        img_r_2k = cv2.resize(mat_r, (self.working_w, self.working_h), interpolation=cv2.INTER_AREA)

        rect_l_2k = cv2.remap(img_l_2k, map0_x, map0_y, cv2.INTER_LINEAR)
        rect_r_2k = cv2.remap(img_r_2k, map1_x, map1_y, cv2.INTER_LINEAR)

        # 6. Optional Rectification at Native 12.3MP for Master Texture Baking
        rect_l_nat, rect_r_nat, Q_nat = None, None, None
        if rectify_native:
            log.info("Computing native 12.3MP rectification maps for albedo texture baking...")
            K_nat = self._get_intrinsic_matrix(w_nat, h_nat)
            R_rect0_n, R_rect1_n, P_rect0_n, P_rect1_n, Q_nat, _, _ = cv2.stereoRectify(
                cameraMatrix1=K_nat,
                distCoeffs1=dist_coeffs,
                cameraMatrix2=K_nat,
                distCoeffs2=dist_coeffs,
                imageSize=(w_nat, h_nat),
                R=R_rel,
                T=T_rel,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=-1,
            )
            map0_x_n, map0_y_n = cv2.initUndistortRectifyMap(
                K_nat, dist_coeffs, R_rect0_n, P_rect0_n, (w_nat, h_nat), cv2.CV_32FC1
            )
            map1_x_n, map1_y_n = cv2.initUndistortRectifyMap(
                K_nat, dist_coeffs, R_rect1_n, P_rect1_n, (w_nat, h_nat), cv2.CV_32FC1
            )
            rect_l_nat = cv2.remap(mat_l, map0_x_n, map0_y_n, cv2.INTER_LINEAR)
            rect_r_nat = cv2.remap(mat_r, map1_x_n, map1_y_n, cv2.INTER_LINEAR)

        effective_focal_2k = float(P_rect0[0, 0])
        log.info(
            "Rectification Complete: 2K resolution (%dx%d), Effective Focal: %.1f px, Baseline: %.2f mm",
            self.working_w, self.working_h, effective_focal_2k, baseline_mm
        )

        return StereoRectificationResult(
            pair_id=pair_id,
            rect_left_2k=rect_l_2k,
            rect_right_2k=rect_r_2k,
            Q_2k=Q_2k,
            P_rect0=P_rect0,
            P_rect1=P_rect1,
            R_rect0=R_rect0,
            R_rect1=R_rect1,
            baseline_mm=baseline_mm,
            convergence_deg=convergence_deg,
            effective_focal_2k=effective_focal_2k,
            is_apriltag_calibrated=is_apriltag,
            scale_factor_2k=self.working_w / float(w_nat),
            rect_left_native=rect_l_nat,
            rect_right_native=rect_r_nat,
            Q_native=Q_nat,
        )

    def _refine_extrinsics_with_sift(
        self,
        img_l: np.ndarray,
        img_r: np.ndarray,
        R_initial: np.ndarray,
        T_initial: np.ndarray,
        w_nat: int,
        h_nat: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Uses SIFT features with RANSAC Essential Matrix to refine small pitch/roll tolerances.
        Preserves metric scale by enforcing ||T|| = baseline_m.
        """
        try:
            # Downscale for fast feature detection
            scale = 0.5
            gray_l = cv2.cvtColor(cv2.resize(img_l, (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(cv2.resize(img_r, (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create(nfeatures=2000)
            kp1, des1 = sift.detectAndCompute(gray_l, None)
            kp2, des2 = sift.detectAndCompute(gray_r, None)

            if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
                return None, None

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good_matches) < 30:
                return None, None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]) / scale
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]) / scale

            K_nat = self._get_intrinsic_matrix(w_nat, h_nat)
            E, inlier_mask = cv2.findEssentialMat(
                pts1, pts2, K_nat, method=cv2.RANSAC, prob=0.999, threshold=1.5
            )
            if E is None:
                return None, None

            _, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1, pts2, K_nat, mask=inlier_mask)
            inliers = int(np.sum(mask_pose > 0))

            if inliers < 25:
                return None, None

            # Enforce camera 1 to the right: T_x must be negative in OpenCV convention
            B_m = self.nominal_baseline_mm / 1000.0
            T_est = -t_est * B_m if t_est[0] > 0 else t_est * B_m

            # Sanity check: Ensure refined rotation does not deviate by > 5° from nominal convergence
            dot_rot = (np.trace(R_est @ R_initial.T) - 1.0) / 2.0
            angle_diff_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot_rot))))
            if angle_diff_deg > 6.0:
                log.warning("SIFT refined extrinsics deviated excessively (%.1f°) from nominal prior; keeping nominal.", angle_diff_deg)
                return None, None

            return R_est, T_est.reshape(3, 1)

        except Exception as e:
            log.warning("Feature refinement failed (%s); proceeding with nominal extrinsics.", e)
            return None, None
