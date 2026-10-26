"""
Facial Masking & Disparity Pre-Shift Module
===========================================
Extracts high-fidelity facial contours using MediaPipe FaceLandmarker,
creates a dilated facial boundary mask to isolate the face volume from background,
and calculates the median facial disparity to initialize the CREStereo Pre-Shift d0.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import cv2
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from face_scanner import FaceScanner

log = logging.getLogger("Photogrammetry.FaceMask")

# Canonical Face Oval Connections in MediaPipe
FACE_OVAL_CONNECTIONS = list(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)


@dataclass
class FacialMaskResult:
    """Contains binary facial mask, disparity pre-shift parameters, and epipolar diagnostics."""
    mask_2k: np.ndarray                    # uint8 (H, W) in {0, 255}
    d_median: float                        # Median disparity on face (pixels)
    d0_shift: int                          # Horizontal pre-shift d0 for residual search
    residual_window: int                   # Residual search window (typically 256 or 272 px)
    face_detected_cam0: bool
    face_detected_cam1: bool
    epipolar_rms_error_px: float           # Vertical parallax error |y0 - y1| (should be < 3 px)
    bbox_2k: Tuple[float, float, float, float]  # [xmin, ymin, xmax, ymax]
    nose_tip_coord: Tuple[float, float]    # (x, y) in Cam 0
    confidence_score: float = 1.0

    def save_mask(self, output_path: Union[str, Path], save_overlay: bool = True) -> Dict[str, Path]:
        """Saves binary mask and debug overlay."""
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), self.mask_2k)
        paths = {"mask": out_p}

        if save_overlay:
            overlay_p = out_p.parent / f"{out_p.stem}_overlay.jpg"
            # Visual blend: mask in green
            color_mask = np.zeros((*self.mask_2k.shape, 3), dtype=np.uint8)
            color_mask[self.mask_2k > 0] = [0, 255, 0]
            cv2.imwrite(str(overlay_p), color_mask)
            paths["overlay"] = overlay_p

        log.info("Saved facial mask to: %s", out_p)
        return paths


class FacialMaskExtractor:
    """
    Extracts binary face masks and calculates disparity pre-shift d0.
    """

    def __init__(
        self,
        dilation_radius_px: int = 21,
        residual_window_px: int = 320,
        nominal_disparity_px: float = 593.0,
    ):
        self.dilation_radius = dilation_radius_px
        self.residual_window = residual_window_px
        self.nominal_disparity = nominal_disparity_px
        self._scanner: Optional[FaceScanner] = None

    def _get_scanner(self) -> FaceScanner:
        if self._scanner is None:
            self._scanner = FaceScanner()
        return self._scanner

    def close(self):
        if self._scanner is not None:
            self._scanner.close()
            self._scanner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_rectified_pair(
        self,
        rect_left_2k: np.ndarray,
        rect_right_2k: np.ndarray,
    ) -> FacialMaskResult:
        """
        Runs MediaPipe inference on rectified stereo images, extracts the facial mask,
        and computes the optimal pre-shift d0.
        """
        scanner = self._get_scanner()
        h, w = rect_left_2k.shape[:2]

        scan_0 = scanner.scan_image(rect_left_2k)
        scan_1 = scanner.scan_image(rect_right_2k)

        det_0 = scan_0["num_faces_detected"] > 0
        det_1 = scan_1["num_faces_detected"] > 0

        if not det_0:
            log.warning("No face detected in rectified Camera 0 image. Falling back to center elliptic mask and nominal d0.")
            return self._build_fallback_result(h, w)

        face_0 = scan_0["faces"][0]
        lm_0 = face_0["pixel_landmarks"]
        bbox_dict = face_0["bounding_box"]
        bbox = (bbox_dict["xmin"], bbox_dict["ymin"], bbox_dict["xmax"], bbox_dict["ymax"])
        nose_coord = (float(lm_0[1]["x"]), float(lm_0[1]["y"]))

        # 1. Build Face Oval Mask
        mask_2k = self._create_face_oval_mask(lm_0, w, h)

        # 2. Compute Disparity Pre-Shift d0 and Epipolar Parallax
        if det_1:
            face_1 = scan_1["faces"][0]
            lm_1 = face_1["pixel_landmarks"]

            n_lms = min(len(lm_0), len(lm_1), 468)
            dxs = []
            dys = []
            for i in range(n_lms):
                x0, y0 = lm_0[i]["x"], lm_0[i]["y"]
                x1, y1 = lm_1[i]["x"], lm_1[i]["y"]
                dxs.append(x0 - x1)
                dys.append(abs(y0 - y1))

            d_median = float(np.median(dxs))
            d_p5 = float(np.percentile(dxs, 5))
            d_p95 = float(np.percentile(dxs, 95))
            rms_y = float(np.sqrt(np.mean(np.array(dys) ** 2)))
            log.info(
                "Facial Landmarks Disparity: Median d = %.1f px (p5=%.1f, p95=%.1f) | Epipolar RMS Parallax = %.2f px",
                d_median, d_p5, d_p95, rms_y
            )

            # Robust pre-shift: guarantees that even the furthest facial features
            # (cheeks/temples with minimum disparity) have non-negative residual disparity Δd >= 0
            d0_shift = max(0, int(d_p5 - 32.0))
            # Ensure the closest features (nose tip with max disparity) stay inside the search window
            if (d_p95 - d0_shift) > (self.residual_window - 24):
                d0_shift = max(0, int(d_p95 - (self.residual_window - 24)))
        else:
            log.warning("Face detected in Camera 0 but not Camera 1; using nominal rig disparity prior (%.1f px).", self.nominal_disparity)
            d_median = self.nominal_disparity
            rms_y = 0.0
            d0_shift = max(0, int(d_median - self.residual_window // 2))

        log.info("Initialized CREStereo Pre-Shift: d0 = %d px (Residual Search Window: [0, %d] px)", d0_shift, self.residual_window)

        return FacialMaskResult(
            mask_2k=mask_2k,
            d_median=d_median,
            d0_shift=d0_shift,
            residual_window=self.residual_window,
            face_detected_cam0=True,
            face_detected_cam1=det_1,
            epipolar_rms_error_px=rms_y,
            bbox_2k=bbox,
            nose_tip_coord=nose_coord,
            confidence_score=1.0,
        )

    def _create_face_oval_mask(
        self,
        landmarks: List[Dict[str, float]],
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Builds a smooth, dilated binary mask from the FACE_OVAL contour.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # Extract ordered or convex polygon from Face Oval connections
        oval_indices = set()
        for conn in FACE_OVAL_CONNECTIONS:
            oval_indices.add(conn.start)
            oval_indices.add(conn.end)

        pts = []
        for idx in oval_indices:
            if idx < len(landmarks):
                pts.append([landmarks[idx]["x"], landmarks[idx]["y"]])

        if len(pts) < 3:
            return mask

        pts_np = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts_np)

        # Fill convex polygon
        cv2.fillPoly(mask, [hull], 255)

        # Morphological dilation with elliptical kernel to include jawline, temples, and chin edge
        if self.dilation_radius > 0:
            k_size = 2 * self.dilation_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Subtle Gaussian blur at the boundary for smooth falloff
        mask = cv2.GaussianBlur(mask, (7, 7), 2)
        # Re-binarize at threshold
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _build_fallback_result(self, h: int, w: int) -> FacialMaskResult:
        """Constructs an elliptical center mask when face detection fails."""
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.22), int(h * 0.30))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        d_median = self.nominal_disparity
        d0_shift = max(0, int(d_median - self.residual_window // 2))
        return FacialMaskResult(
            mask_2k=mask,
            d_median=d_median,
            d0_shift=d0_shift,
            residual_window=self.residual_window,
            face_detected_cam0=False,
            face_detected_cam1=False,
            epipolar_rms_error_px=0.0,
            bbox_2k=(float(center[0] - axes[0]), float(center[1] - axes[1]), float(center[0] + axes[0]), float(center[1] + axes[1])),
            nose_tip_coord=(float(center[0]), float(center[1])),
            confidence_score=0.5,
        )
