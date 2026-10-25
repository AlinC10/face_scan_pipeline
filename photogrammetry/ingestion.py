"""
Stereo Rig Data Ingestion & EXIF Validation Module
=================================================
Discovers, validates and prepares stereo photo pairs from physical rig directories (0/ and 1/),
extracts metric carriage height Z_rig from filename convention (f_<height>), inspects 12.3MP
native resolution and EXIF telemetry, and initiates AprilTag metric ground calibration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import piexif
except ImportError:
    piexif = None

from .apriltag_tracker import AprilTagPoseResult, AprilTagTracker, DEFAULT_FOCAL_PX

log = logging.getLogger("Photogrammetry.Ingestion")

# Standard rig image dimension for 12.3MP Arducam IMX477
EXPECTED_WIDTH = 4056
EXPECTED_HEIGHT = 3040
REGEX_RIG_HEIGHT = re.compile(r"f_(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class ImageMetadata:
    path: str
    filename: str
    camera_id: int  # 0 for left, 1 for right
    width: int
    height: int
    aspect_ratio: float
    is_native_12mp: bool
    serial_number: Optional[str] = None
    unique_id: Optional[str] = None
    focal_length_mm: Optional[float] = None
    exposure_time_s: Optional[float] = None
    iso: Optional[int] = None


@dataclass
class StereoIngestionResult:
    """Consolidated ingestion output representing a validated stereo capture pair."""
    pair_id: str
    cam0_meta: ImageMetadata
    cam1_meta: ImageMetadata
    z_rig_mm: Optional[float] = None
    has_metric_z_prior: bool = False
    apriltag_calib: Optional[Dict[str, Any]] = None
    has_apriltag_metric_ground: bool = False
    nominal_baseline_mm: float = 163.4  # Nominal baseline for 27° convergence at 350mm
    nominal_focal_px: float = DEFAULT_FOCAL_PX
    status: str = "success"
    warnings: List[str] = field(default_factory=list)
    pose_result: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("pose_result", None)
        return d

    def save_manifest(self, output_path: Union[str, Path]) -> Path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("Saved ingestion manifest: %s", out_p)
        return out_p


def extract_exif_telemetry(img_path: Union[str, Path]) -> Dict[str, Any]:
    """Reads EXIF tags including BodySerialNumber, ImageUniqueID and camera settings."""
    meta: Dict[str, Any] = {
        "serial_number": None,
        "unique_id": None,
        "focal_length_mm": None,
        "exposure_time_s": None,
        "iso": None,
    }
    p = Path(img_path)
    if not p.is_file() or piexif is None:
        return meta

    try:
        exif_dict = piexif.load(str(p))
        exif_sub = exif_dict.get("Exif", {})

        # Serial number (tag 0xA431 = 42033)
        if 42033 in exif_sub:
            val = exif_sub[42033]
            meta["serial_number"] = val.decode("utf-8", "ignore").strip("\x00") if isinstance(val, bytes) else str(val)

        # Unique ID (tag 0xA420 = 42016)
        if 42016 in exif_sub:
            val = exif_sub[42016]
            meta["unique_id"] = val.decode("utf-8", "ignore").strip("\x00") if isinstance(val, bytes) else str(val)

        # Focal length (tag 0x920A = 37386)
        if 37386 in exif_sub:
            num, den = exif_sub[37386]
            meta["focal_length_mm"] = float(num) / float(den) if den != 0 else None

        # Exposure time (tag 0x829A = 33434)
        if 33434 in exif_sub:
            num, den = exif_sub[33434]
            meta["exposure_time_s"] = float(num) / float(den) if den != 0 else None

        # ISO (tag 0x8827 = 34855)
        if 34855 in exif_sub:
            meta["iso"] = int(exif_sub[34855])

    except Exception as e:
        log.debug("EXIF extraction notice for %s: %s", p.name, e)

    return meta


class StereoRigIngestor:
    """Discovers, validates, and prepares stereo pairs from the hardware rig."""

    def __init__(
        self,
        apriltag_tracker: Optional[AprilTagTracker] = None,
        nominal_baseline_mm: float = 163.4,
        check_apriltags: bool = True
    ):
        self.nominal_baseline_mm = nominal_baseline_mm
        self.check_apriltags = check_apriltags
        self.tracker = apriltag_tracker or (AprilTagTracker() if check_apriltags else None)

    def ingest_pair(
        self,
        left_path: Union[str, Path],
        right_path: Union[str, Path],
        pair_id: Optional[str] = None
    ) -> StereoIngestionResult:
        """Validates a concrete pair of left and right images."""
        p_l = Path(left_path).resolve()
        p_r = Path(right_path).resolve()

        if not p_l.is_file():
            raise FileNotFoundError(f"Left image (Cam 0) not found: {p_l}")
        if not p_r.is_file():
            raise FileNotFoundError(f"Right image (Cam 1) not found: {p_r}")

        # 1. Read image dimensions
        img_l = cv2.imread(str(p_l))
        img_r = cv2.imread(str(p_r))
        if img_l is None:
            raise ValueError(f"Could not decode Left image: {p_l}")
        if img_r is None:
            raise ValueError(f"Could not decode Right image: {p_r}")

        h_l, w_l = img_l.shape[:2]
        h_r, w_r = img_r.shape[:2]

        warnings: List[str] = []
        is_l_12mp = (w_l == EXPECTED_WIDTH and h_l == EXPECTED_HEIGHT)
        is_r_12mp = (w_r == EXPECTED_WIDTH and h_r == EXPECTED_HEIGHT)

        if not is_l_12mp:
            warnings.append(f"Camera 0 resolution is {w_l}x{h_l}, expected native 12.3MP ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT}).")
        if not is_r_12mp:
            warnings.append(f"Camera 1 resolution is {w_r}x{h_r}, expected native 12.3MP ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT}).")
        if (w_l, h_l) != (w_r, h_r):
            warnings.append(f"Resolution mismatch between cameras: Left is {w_l}x{h_l}, Right is {w_r}x{h_r}.")

        # 2. Extract EXIF telemetry
        exif_l = extract_exif_telemetry(p_l)
        exif_r = extract_exif_telemetry(p_r)

        meta_l = ImageMetadata(
            path=str(p_l),
            filename=p_l.name,
            camera_id=0,
            width=w_l,
            height=h_l,
            aspect_ratio=round(w_l / h_l, 4),
            is_native_12mp=is_l_12mp,
            serial_number=exif_l["serial_number"],
            unique_id=exif_l["unique_id"],
            focal_length_mm=exif_l["focal_length_mm"],
            exposure_time_s=exif_l["exposure_time_s"],
            iso=exif_l["iso"]
        )

        meta_r = ImageMetadata(
            path=str(p_r),
            filename=p_r.name,
            camera_id=1,
            width=w_r,
            height=h_r,
            aspect_ratio=round(w_r / h_r, 4),
            is_native_12mp=is_r_12mp,
            serial_number=exif_r["serial_number"],
            unique_id=exif_r["unique_id"],
            focal_length_mm=exif_r["focal_length_mm"],
            exposure_time_s=exif_r["exposure_time_s"],
            iso=exif_r["iso"]
        )

        # 3. Parse carriage height Z_rig from filename convention (f_<height>)
        z_rig_mm = None
        m_l = REGEX_RIG_HEIGHT.search(p_l.stem)
        m_r = REGEX_RIG_HEIGHT.search(p_r.stem)
        if m_l:
            z_rig_mm = float(m_l.group(1))
        elif m_r:
            z_rig_mm = float(m_r.group(1))

        # 4. Optional AprilTag Metric Ground Calibration
        apriltag_dict = None
        has_apriltag = False
        pose_res = None
        if self.check_apriltags and self.tracker:
            try:
                pose_res = self.tracker.calibrate_stereo_pair(img_l, img_r)
                if pose_res is not None:
                    has_apriltag = True
                    apriltag_dict = {
                        "tags_cam0": pose_res.tags_cam0,
                        "tags_cam1": pose_res.tags_cam1,
                        "common_tags": pose_res.common_tags,
                        "baseline_mm": round(pose_res.baseline_mm, 2),
                        "baseline_m": round(pose_res.baseline_m, 4),
                        "convergence_deg": round(pose_res.convergence_deg, 2),
                        "dist_cam0_m": round(pose_res.dist_cam0_m, 3),
                        "dist_cam1_m": round(pose_res.dist_cam1_m, 3),
                        "rms_cam0_px": round(pose_res.rms_cam0, 3),
                        "rms_cam1_px": round(pose_res.rms_cam1, 3),
                        "camera_0_center_m": [round(float(v), 4) for v in pose_res.C_0],
                        "camera_1_center_m": [round(float(v), 4) for v in pose_res.C_1],
                        "R_rel": pose_res.R_rel.tolist(),
                        "T_rel": pose_res.T_rel.tolist(),
                        "rect_R0": pose_res.rect_R0.tolist(),
                        "rect_R1": pose_res.rect_R1.tolist(),
                        "rect_P0": pose_res.rect_P0.tolist(),
                        "rect_P1": pose_res.rect_P1.tolist(),
                        "Q_mat": pose_res.Q_mat.tolist(),
                    }
                    log.info("AprilTag ground truth confirmed: B = %.2f mm, Conv = %.2f°", pose_res.baseline_mm, pose_res.convergence_deg)
                else:
                    log.info("No AprilTag ground markers detected in this pair; continuing with nominal metric priors.")
            except Exception as e:
                warnings.append(f"AprilTag detection error: {e}")
                log.warning("AprilTag calibration attempt encountered: %s", e)

        eff_id = pair_id or (p_l.stem if p_l.stem == p_r.stem else f"{p_l.stem}_{p_r.stem}")

        return StereoIngestionResult(
            pair_id=eff_id,
            cam0_meta=meta_l,
            cam1_meta=meta_r,
            z_rig_mm=z_rig_mm,
            has_metric_z_prior=(z_rig_mm is not None),
            apriltag_calib=apriltag_dict,
            has_apriltag_metric_ground=has_apriltag,
            nominal_baseline_mm=self.nominal_baseline_mm,
            nominal_focal_px=DEFAULT_FOCAL_PX,
            status="success",
            warnings=warnings,
            pose_result=pose_res
        )

    def ingest_rig_directory(self, rig_dir: Union[str, Path]) -> List[StereoIngestionResult]:
        """Scans rig subdirectories 0/ (Camera 0) and 1/ (Camera 1) for matching pairs."""
        r_dir = Path(rig_dir).resolve()
        dir_0 = r_dir / "0"
        dir_1 = r_dir / "1"

        if not dir_0.is_dir() or not dir_1.is_dir():
            raise FileNotFoundError(f"Rig directory '{r_dir}' must contain '0' and '1' subdirectories.")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files_0 = {p.stem: p for p in dir_0.iterdir() if p.suffix.lower() in valid_exts}
        files_1 = {p.stem: p for p in dir_1.iterdir() if p.suffix.lower() in valid_exts}

        common_stems = sorted(set(files_0.keys()) & set(files_1.keys()))
        if not common_stems:
            # Fallback: check matching by numerical order if names differ
            f0_list = sorted(list(files_0.values()))
            f1_list = sorted(list(files_1.values()))
            if len(f0_list) == len(f1_list) and len(f0_list) > 0:
                common_stems = [f"pair_{idx}" for idx in range(len(f0_list))]
                files_0 = {f"pair_{idx}": p for idx, p in enumerate(f0_list)}
                files_1 = {f"pair_{idx}": p for idx, p in enumerate(f1_list)}
            else:
                raise FileNotFoundError(f"No matching stereo image pairs found between {dir_0} ({len(files_0)}) and {dir_1} ({len(files_1)}).")

        results: List[StereoIngestionResult] = []
        for stem in common_stems:
            res = self.ingest_pair(files_0[stem], files_1[stem], pair_id=stem)
            results.append(res)

        return results


def load_stereo_rig_pair(
    rig_dir: Optional[Union[str, Path]] = None,
    left_path: Optional[Union[str, Path]] = None,
    right_path: Optional[Union[str, Path]] = None,
    check_apriltags: bool = True
) -> StereoIngestionResult:
    """Convenience helper to load a stereo pair from either explicit paths or rig directory."""
    ingestor = StereoRigIngestor(check_apriltags=check_apriltags)

    if left_path and right_path:
        return ingestor.ingest_pair(left_path, right_path)

    if rig_dir:
        pairs = ingestor.ingest_rig_directory(rig_dir)
        if not pairs:
            raise FileNotFoundError(f"No valid stereo pairs in rig directory: {rig_dir}")
        return pairs[0]

    raise ValueError("Must provide either (left_path and right_path) or rig_dir.")
