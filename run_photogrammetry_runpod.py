#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Dense Stereo Photogrammetry CLI (Etapele 1, 2, 3)
=========================================================
Orchestrates the photogrammetry pipeline for high-fidelity 3D facial reconstruction
from dual Arducam IMX477 camera rigs (27° convergence, 15° pitch, 35 cm distance).

Default mode (--mode dense): after Etapa 1 the pair goes to photogrammetry/face_pipeline.py
(measured rig calibration, MediaPipe-guided dense matching with pose refinement, two-view
disparity, height-field fusion, two-view texture baking and the 8 standard deliverables).
The stages below describe the classical modes (--mode poisson / hybrid).

Etapa 1:
  1. Discovery & ingestion of 12.3MP (4056x3040) stereo pairs from rig/0 and rig/1.
  2. Extraction of carriage metric height Z_rig from filename convention (f_<height>.jpg).
  3. Reading EXIF telemetry (BodySerialNumber, ImageUniqueID).
  4. Automatic AprilTag (tag16h5) detection and metric PnP calibration (baseline B, convergence, rect).
  5. Manifest serialization into output directory.

Etapa 2:
  1. Calibrated Bouguet stereo rectification at 2K working resolution (2048x1536).
  2. Construction of re-projection matrix Q for 3D metric triangulation.
  3. MediaPipe FaceOval contour extraction with morphological dilation (masking out background).
  4. Calculation of median facial disparity and initialization of CREStereo Pre-Shift d0.
  5. Export of rectified pair, face mask, and horizontal epipolar inspection composite.

Etapa 3:
  1. Pre-shifted dense stereo disparity matching (residual search in [0, 256 px]).
  2. High-precision SGBM with Left-Right Consistency Filtering (|dL - dR| <= 1.0 px) and bilateral WLS.
  3. Integration of CREStereo neural network when weights are provided.
  4. Metric disparity reconstitution: d_total(x, y) = Δd(x, y) + d0.
  5. Export of 16-bit metric disparity PNG (disp * 256) and colorized TURBO visualization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("Photogrammetry.RunPod")

# Import photogrammetry modules
try:
    from photogrammetry.ingestion import StereoRigIngestor, StereoIngestionResult
    from photogrammetry.apriltag_tracker import AprilTagTracker, DEFAULT_FOCAL_PX
    from photogrammetry.rectification import (
        StereoRectifier,
        StereoRectificationResult,
        WORKING_2K_WIDTH,
        WORKING_2K_HEIGHT,
    )
    from photogrammetry.face_mask import FacialMaskExtractor, FacialMaskResult
    from photogrammetry.dense_stereo import DenseStereoMatcher, DenseDisparityResult
    from photogrammetry.triangulation_poisson import (
        MetricPoissonReconstructor,
        PoissonReconstructionResult,
    )
    from photogrammetry.decimation_uv import (
        MeshDecimatorUV,
        DecimationUVResult,
    )
    from photogrammetry.texture_baker import (
        TextureBaker,
        TextureBakingResult,
    )
    from photogrammetry.export_manager import (
        StandardDeliverablesExporter,
        ExportResult,
    )
except ImportError:
    # Ensure current dir is in sys.path
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from photogrammetry.ingestion import StereoRigIngestor, StereoIngestionResult
    from photogrammetry.apriltag_tracker import AprilTagTracker, DEFAULT_FOCAL_PX
    from photogrammetry.rectification import (
        StereoRectifier,
        StereoRectificationResult,
        WORKING_2K_WIDTH,
        WORKING_2K_HEIGHT,
    )
    from photogrammetry.face_mask import FacialMaskExtractor, FacialMaskResult
    from photogrammetry.dense_stereo import DenseStereoMatcher, DenseDisparityResult
    from photogrammetry.triangulation_poisson import (
        MetricPoissonReconstructor,
        PoissonReconstructionResult,
    )
    from photogrammetry.decimation_uv import (
        MeshDecimatorUV,
        DecimationUVResult,
    )
    from photogrammetry.texture_baker import (
        TextureBaker,
        TextureBakingResult,
    )
    from photogrammetry.export_manager import (
        StandardDeliverablesExporter,
        ExportResult,
    )


@dataclass
class JobConfig:
    """
    One reconstruction job (as Turntable Studio's job dict): the keys of run_pipeline(job), of the CLI,
    of runpod_job.py (JOB_JSON) and of tools/cloud_scan.py --set. The defaults below are the only ones.
    """
    rig_dir: Optional[str] = None
    left_path: Optional[str] = None
    right_path: Optional[str] = None
    output_dir: str = "./output"
    calib_dir: Optional[str] = None
    calibration_path: Optional[str] = None
    layout_path: Optional[str] = None
    name: str = "model"
    reconstruction_mode: str = "dense"
    spectacles: bool = True
    tag_scale: float = 0.98
    export_scale: float = 100.0
    glb_draco: bool = True
    texture_size: int = 4096
    segment_face: bool = True
    remove_frames: Optional[bool] = None
    head_model: str = "gnm"
    expression: str = "capture"
    check_apriltags: bool = True
    stop_after_stage: int = 7
    rectify_native: bool = False
    refine_with_features: bool = True

    @classmethod
    def from_dict(cls, job: Dict[str, Any]) -> "JobConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(job) - known)
        if unknown:
            raise ValueError(f"Unknown job option(s) {unknown}; known: {sorted(known)}")
        return cls(**job)


# Options that are paths on the machine running the job (set by the pod / CLI, never sent by a cloud client)
PATH_OPTIONS = {"rig_dir", "left_path", "right_path", "output_dir", "calib_dir", "layout_path"}


def run_pipeline(job: Dict[str, Any]) -> Dict[str, Any]:
    """Runs one job given as a dict of JobConfig keys (unknown keys are refused, not ignored)."""
    return run_photogrammetry_pipeline(**asdict(JobConfig.from_dict(job)))


def _latest_pair(pairs: list) -> "StereoIngestionResult":
    """
    The most recent capture of a rig folder. The pairs are sorted by name as strings
    (f_150 < f_95), so with several captures the first one is not the latest.
    """
    def mtime(res) -> float:
        return max(Path(res.cam0_meta.path).stat().st_mtime, Path(res.cam1_meta.path).stat().st_mtime)

    latest = max(pairs, key=mtime)
    if len(pairs) > 1:
        log.warning("%d stereo pairs in the rig folder (%s); using the most recent one: '%s'. "
                    "Use -l / -r to choose another.", len(pairs), ", ".join(r.pair_id for r in pairs), latest.pair_id)
    return latest


def _job_rig_calibration(
    rig_dir: str | Path | None,
    left_path: str | Path | None,
    calib_dir: str | Path | None,
    calibration_path: str | Path | None,
    out_dir: Path,
    tag_scale: float,
    face_pair: tuple | None = None,
) -> tuple:
    """
    Stereo calibration of this job, in order of preference:
      1. photographs of the free AprilTag disc (calib/0 and calib/1 next to rig/, or --calib-dir):
         turntable_poses.py runs on them and its turntable_calib.json is used;
      2. the disc tags visible in the face pair itself (photogrammetry/pair_calibration.py), accepted only
         close to the stored calibration (focal 2 %, baseline 3 mm, rotation 1 deg);
      3. the stored calibration (--calib, otherwise assets/rig_calibration.json).
    Returns (calibration file or None for the default, description for the manifest).
    """
    from photogrammetry.rig_calibration import DEFAULT_CALIBRATION_PATH
    from photogrammetry.turntable_calibration import (
        TurntableCalibrationConfig,
        find_calibration_images,
        run_turntable_calibration,
    )

    if calib_dir is None:
        # <job>/rig -> <job>/calib, or <job>/rig/0/f_150.jpg -> <job>/calib (paths as given: no symlink resolution)
        if rig_dir:
            candidate = Path(rig_dir).absolute().parent / "calib"
        elif left_path:
            candidate = Path(left_path).absolute().parent.parent.parent / "calib"
        else:
            candidate = Path("input_images/calib").absolute()
        if find_calibration_images(candidate):
            calib_dir = candidate
    if calib_dir is not None:
        log.info("=" * 70)
        log.info("CALIBRARE RIG: turntable_poses.py pe pozele discului din %s", calib_dir)
        log.info("=" * 70)
        res = run_turntable_calibration(calib_dir, out_dir / "calibration", TurntableCalibrationConfig(tag_scale=tag_scale))
        if calibration_path:
            log.warning("--calib %s ignored: this job has its own disc photographs.", calibration_path)
        return res.calib_path, {"source": "turntable_poses", **res.to_dict()}

    stored = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
    pair_reason = None
    if face_pair is not None:
        from photogrammetry.pair_calibration import calibrate_face_pair, save_calibration
        from photogrammetry.rig_calibration import load_rig_calibration
        res = calibrate_face_pair(face_pair[0], face_pair[1], load_rig_calibration(stored), tag_scale)
        if res.accepted:
            path = save_calibration(res.calibration, out_dir / "calibration" / "pair_calibration.json", res.stats)
            log.info("Calibration from the disc tags of the face pair: %s", res.stats)
            return path, {"source": "face_pair_tags", "calib": str(path), "base": str(stored), **res.stats}
        pair_reason = res.reason
    log.warning("No disc photographs for this job (calib/0, calib/1)%s: using the stored calibration %s.",
                f" and the face pair cannot calibrate the rig ({pair_reason})" if pair_reason else "", stored)
    info = {"source": "stored", "calib": str(stored)}
    if pair_reason:
        info["face_pair_tags"] = pair_reason
    return calibration_path, info


def _run_dense(ingestion_res, summary_report, out_dir, rig_dir, left_path, calib_dir, calibration_path,
               name, spectacles, tag_scale, export_scale, glb_draco, texture_size,
               segment_face=True, remove_frames=None, head_model="gnm", expression="capture") -> Dict[str, Any]:
    """
    Dense two-view face reconstruction with telemetry (as Turntable Studio): RAM check first, every stage a
    measured step, stats.json (schema v2) with the hardware, the steps, the face quality metrics and the
    exported meshes, written also when the run fails (outcome + failed step).
    """
    from photogrammetry import run_metrics
    from photogrammetry.face_pipeline import FacePipelineConfig, reconstruct_face_pair
    from photogrammetry.resources import check_ram, describe_resources
    from photogrammetry.telemetry import SystemTelemetry

    telemetry = SystemTelemetry(output_dir=str(out_dir))
    telemetry.start_sampling(interval_sec=0.25)
    telemetry.watch(str(out_dir))
    telemetry.set_info(
        version=run_metrics.code_version(),
        hardware={"cpu": run_metrics.cpu_info(), "host": run_metrics.host_info()},
        resources=describe_resources(),
        inputs={"left": ingestion_res.cam0_meta.path, "right": ingestion_res.cam1_meta.path,
                "image_size": [ingestion_res.cam0_meta.width, ingestion_res.cam0_meta.height],
                "z_rig_mm": ingestion_res.z_rig_mm if ingestion_res.has_metric_z_prior else None},
        options={"spectacles": spectacles, "tag_scale": tag_scale, "export_scale": export_scale,
                 "glb_draco": glb_draco, "texture_size": texture_size, "segment_face": segment_face,
                 "remove_frames": remove_frames, "head_model": head_model, "expression": expression},
    )
    stats_path = out_dir / "stats.json"
    try:
        # A pod that is too small answers in seconds (runpod_job reports ram_insufficient, cloud/job retries)
        ram = check_ram(ingestion_res.cam0_meta.width, ingestion_res.cam0_meta.height, texture_size,
                        margin=float(os.environ.get("RAM_MARGIN", "1.0")))
        telemetry.set_info(ram_check=ram)
        log.info("RAM check: %.1f GB needed, %.1f GB available (%s)", ram["required_gb"], ram["available_gb"],
                 ram["status"])

        with telemetry.step("Calibration (turntable_poses, face-pair tags or stored)"):
            job_calibration, calibration_info = _job_rig_calibration(
                rig_dir, left_path, calib_dir, calibration_path, out_dir, tag_scale,
                face_pair=(ingestion_res.cam0_meta.path, ingestion_res.cam1_meta.path),
            )
        summary_report["calibration"] = calibration_info
        if calibration_info.get("report"):
            telemetry.set_info(pose_quality=run_metrics.pose_quality(calibration_info["report"]))

        log.info("=" * 70)
        log.info("ETAPA 2-7: RECONSTRUCȚIE DENSĂ DIN DOUĂ VEDERI (CALIBRARE RIG + MEDIAPIPE + DIS)")
        log.info("=" * 70)
        face_res = reconstruct_face_pair(
            ingestion_res.cam0_meta.path,
            ingestion_res.cam1_meta.path,
            out_dir,
            FacePipelineConfig(calibration_path=job_calibration, name=name, spectacles=spectacles,
                               calibration_info=calibration_info, tag_scale=tag_scale,
                               export_scale=export_scale, glb_draco=glb_draco,
                               high_texture_size=int(texture_size),
                               low_texture_size=int(min(2048, texture_size)),
                               segment_face=segment_face, remove_frames=remove_frames,
                               head_model=head_model, expression=expression),
            telemetry=telemetry,
        )
        telemetry.set_info(face=run_metrics.face_metrics(face_res.manifest))
        for key, path in face_res.deliverables.items():
            if key.split("_")[0] in ("obj", "stl", "3mf"):
                telemetry.record_mesh(str(path))
        telemetry.set_outcome("ok")
    except Exception as e:
        telemetry.set_outcome("error", error_type=type(e).__name__, error=str(e)[:500],
                              failed_step=telemetry.current_step)
        raise
    finally:
        telemetry.stop_sampling()
        telemetry.save_json(str(stats_path))

    summary_report["stages_completed"].append("stage_2_7_dense_two_view_face")
    summary_report["artifacts"].update({k: str(v) for k, v in face_res.deliverables.items()})
    summary_report["artifacts"]["stats"] = str(stats_path)
    summary_report["status"] = "success"
    summary_report["manifest"] = str(face_res.manifest_path)
    return summary_report


def run_photogrammetry_pipeline(
    rig_dir: str | Path | None = None,
    left_path: str | Path | None = None,
    right_path: str | Path | None = None,
    output_dir: str | Path = "./output",
    check_apriltags: bool = True,
    layout_path: str | Path | None = None,
    tag_scale: float = 0.98,
    name: str = "model",
    stop_after_stage: int = 7,
    rectify_native: bool = False,
    refine_with_features: bool = True,
    reconstruction_mode: str = "dense",
    calibration_path: str | Path | None = None,
    spectacles: bool = True,
    calib_dir: str | Path | None = None,
    export_scale: float = 100.0,
    glb_draco: bool = True,
    texture_size: int = 4096,
    segment_face: bool = True,
    remove_frames: Optional[bool] = None,
    head_model: str = "gnm",
    expression: str = "capture",
) -> Dict[str, Any]:
    """
    Executes the multi-stage photogrammetry pipeline.
    """
    t_pipeline_start = time.time()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_report: Dict[str, Any] = {
        "status": "in_progress",
        "stages_completed": [],
        "artifacts": {}
    }

    # =========================================================================
    # ETAPA 1: INGESTION DATE, EXIF & CALIBRARE METRICĂ APRILTAGS
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 1: INGESTION DATE, EXIF & CALIBRARE METRICĂ")
    log.info("=" * 70)

    tracker = None
    if check_apriltags:
        tracker = AprilTagTracker(layout_path=layout_path, tag_scale=tag_scale)

    ingestor = StereoRigIngestor(apriltag_tracker=tracker, check_apriltags=check_apriltags)

    if left_path and right_path:
        ingestion_res = ingestor.ingest_pair(left_path, right_path, pair_id=name)
    elif rig_dir:
        pairs = ingestor.ingest_rig_directory(rig_dir)
        if not pairs:
            raise FileNotFoundError(f"No valid stereo pairs found in rig directory: {rig_dir}")
        ingestion_res = _latest_pair(pairs)
    else:
        default_rig = Path("input_images/rig")
        if default_rig.is_dir() and (default_rig / "0").is_dir() and (default_rig / "1").is_dir():
            pairs = ingestor.ingest_rig_directory(default_rig)
            if not pairs:
                raise FileNotFoundError(f"No stereo pairs found in {default_rig}")
            ingestion_res = _latest_pair(pairs)
        else:
            raise ValueError("Please provide either --rig-dir or -l / --left and -r / --right.")

    log.info("Stereo Pair Ingested: '%s'", ingestion_res.pair_id)
    log.info("  • Camera 0 (Left):  %s (%dx%d, Native 12MP: %s)",
             ingestion_res.cam0_meta.filename, ingestion_res.cam0_meta.width,
             ingestion_res.cam0_meta.height, ingestion_res.cam0_meta.is_native_12mp)
    log.info("  • Camera 1 (Right): %s (%dx%d, Native 12MP: %s)",
             ingestion_res.cam1_meta.filename, ingestion_res.cam1_meta.width,
             ingestion_res.cam1_meta.height, ingestion_res.cam1_meta.is_native_12mp)

    if ingestion_res.has_metric_z_prior:
        log.info("  • Rig Carriage Height Prior: Z_rig = %.1f mm (derived from filename)", ingestion_res.z_rig_mm)

    if ingestion_res.has_apriltag_metric_ground and ingestion_res.apriltag_calib:
        calib = ingestion_res.apriltag_calib
        log.info("  • AprilTag Metric Ground Calibration: DETECTED & VERIFIED")
        log.info("    └─ True Baseline B: %.2f mm", calib["baseline_mm"])
        log.info("    └─ Convergence: %.2f°", calib["convergence_deg"])
        log.info("    └─ Distance Cam 0: %.3f m | Cam 1: %.3f m", calib["dist_cam0_m"], calib["dist_cam1_m"])
    else:
        log.info("  • AprilTag Metric Ground Calibration: NOT DETECTED (Using Nominal Priors)")
        log.info("    └─ Nominal Baseline B: %.2f mm", ingestion_res.nominal_baseline_mm)
        log.info("    └─ Nominal Convergence: 27.00°")
        log.info("    └─ Nominal Focal Length: %.2f px (f = 3.9 mm)", ingestion_res.nominal_focal_px)

    manifest_path = out_dir / "ingestion_manifest.json"
    ingestion_res.save_manifest(manifest_path)
    summary_report["stages_completed"].append("stage_1_ingestion")
    summary_report["artifacts"]["ingestion_manifest"] = str(manifest_path)

    if stop_after_stage == 1:
        log.info("Pipeline terminated after Etapa 1 as requested.")
        return summary_report

    # =========================================================================
    # MOD DENSE (implicit): calibrare rig + potriviri dense ghidate MediaPipe,
    # disparitate din ambele vederi, fuziune metrică, textură din pozele 12.3MP
    # =========================================================================
    if reconstruction_mode == "dense":
        return _run_dense(ingestion_res, summary_report, out_dir, rig_dir, left_path, calib_dir, calibration_path,
                          name, spectacles, tag_scale, export_scale, glb_draco, texture_size,
                          segment_face, remove_frames, head_model, expression)

    # =========================================================================
    # ETAPA 2: RECTIFICARE STEREO BOUGUET & MASCARE FACIALĂ
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 2: RECTIFICARE STEREO BOUGUET & MASCARE FACIALĂ")
    log.info("=" * 70)

    rectifier = StereoRectifier()
    rect_res = rectifier.rectify_pair(
        img_left=ingestion_res.cam0_meta.path,
        img_right=ingestion_res.cam1_meta.path,
        ingestion_result=ingestion_res,
        rectify_native=rectify_native,
        refine_with_features=refine_with_features,
    )

    rect_paths = rect_res.save_rectified_pair(out_dir, save_native=rectify_native)
    summary_report["artifacts"].update({k: str(v) for k, v in rect_paths.items()})

    # Save horizontal epipolar inspection composite
    epipolar_composite = rect_res.generate_epipolar_inspection_image(num_scanlines=24)
    epipolar_comp_path = out_dir / f"{ingestion_res.pair_id}_epipolar_check.jpg"
    cv2.imwrite(str(epipolar_comp_path), epipolar_composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
    summary_report["artifacts"]["epipolar_check"] = str(epipolar_comp_path)
    log.info("Generated epipolar verification composite: %s", epipolar_comp_path)

    # Facial Masking & Pre-shift d0 Extraction
    mask_extractor = FacialMaskExtractor()
    mask_res = mask_extractor.process_rectified_pair(rect_res.rect_left_2k, rect_res.rect_right_2k)

    mask_path = out_dir / f"{ingestion_res.pair_id}_face_mask.png"
    mask_saved = mask_res.save_mask(mask_path, save_overlay=True)
    summary_report["artifacts"].update({k: str(v) for k, v in mask_saved.items()})

    summary_report["stages_completed"].append("stage_2_rectification_and_mask")
    summary_report["d0_shift"] = mask_res.d0_shift
    summary_report["d_median"] = mask_res.d_median

    if stop_after_stage == 2:
        log.info("Pipeline terminated after Etapa 2 as requested.")
        return summary_report

    # =========================================================================
    # ETAPA 3: DISPARITATE DENSĂ NEURALĂ 2K CU PRE-SHIFT D0
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 3: DISPARITATE DENSĂ 2K CU PRE-SHIFT D0")
    log.info("=" * 70)

    matcher = DenseStereoMatcher(num_residual_disparities=320, block_size=5)
    disp_res = matcher.compute_disparity(
        rect_left_2k=rect_res.rect_left_2k,
        rect_right_2k=rect_res.rect_right_2k,
        d0_shift=mask_res.d0_shift,
        mask_2k=mask_res.mask_2k,
        pair_id=ingestion_res.pair_id,
    )

    disp_16u_path = out_dir / f"{ingestion_res.pair_id}_disparity_16bit.png"
    disp_res.save_disparity_16bit(disp_16u_path)

    disp_turbo_path = out_dir / f"{ingestion_res.pair_id}_disparity_turbo.jpg"
    disp_res.save_colored_visualization(disp_turbo_path, colormap=cv2.COLORMAP_TURBO)

    summary_report["artifacts"]["disparity_16bit"] = str(disp_16u_path)
    summary_report["artifacts"]["disparity_colored"] = str(disp_turbo_path)
    summary_report["stages_completed"].append("stage_3_dense_disparity")

    if stop_after_stage == 3:
        total_dt = time.time() - t_pipeline_start
        consolidated_manifest = {
            "pair_id": ingestion_res.pair_id,
            "stages_completed": summary_report["stages_completed"],
            "pipeline_duration_s": round(total_dt, 3),
            "optical_parameters": {
                "focal_length_px_2k": round(rect_res.effective_focal_2k, 2),
                "baseline_mm": round(rect_res.baseline_mm, 2),
                "convergence_deg": round(rect_res.convergence_deg, 2),
                "z_rig_mm": ingestion_res.z_rig_mm,
                "is_apriltag_calibrated": rect_res.is_apriltag_calibrated,
            },
            "rectification": {
                "resolution": [WORKING_2K_WIDTH, WORKING_2K_HEIGHT],
                "epipolar_rms_error_px": round(mask_res.epipolar_rms_error_px, 3),
                "Q_matrix_2k": rect_res.Q_2k.tolist(),
            },
            "facial_masking": {
                "face_detected_cam0": mask_res.face_detected_cam0,
                "face_detected_cam1": mask_res.face_detected_cam1,
                "d_median_px": round(mask_res.d_median, 2),
                "d0_shift_px": mask_res.d0_shift,
                "bbox_2k": [round(float(v), 1) for v in mask_res.bbox_2k],
            },
            "dense_disparity": {
                "matcher_engine": disp_res.matcher_engine,
                "valid_pixels_count": disp_res.valid_pixels_count,
                "min_disparity_px": round(disp_res.min_disparity, 2),
                "max_disparity_px": round(disp_res.max_disparity, 2),
                "median_disparity_px": round(disp_res.median_disparity, 2),
                "matcher_time_s": round(disp_res.execution_time_s, 3),
            },
            "artifacts": summary_report["artifacts"],
        }
        consolidated_path = out_dir / "photogrammetry_manifest.json"
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_manifest, f, indent=2)
        log.info("Pipeline terminated after Etapa 3 as requested.")
        summary_report["manifest"] = str(consolidated_path)
        return summary_report

    # =========================================================================
    # ETAPA 4 - 7: RECONSTRUCȚIE (HIBRIDĂ sau POISSON CLASIC)
    # =========================================================================
    if reconstruction_mode == "hybrid":
        log.info("=" * 70)
        log.info("ETAPA 4-7: FUZIUNE HIBRIDĂ (MEDIAPIPE PRIOR + DENSE STEREO DEFORMATION)")
        log.info("=" * 70)

        from reconstruction.hybrid_fuser import HybridFaceFuser
        fuser = HybridFaceFuser(
            canonical_mesh_path="assets/canonical_face_model.obj",
            tex_size=2048,
        )

        cx = float(rect_res.Q_2k[0, 3] * -1.0) if rect_res.Q_2k is not None else float(WORKING_2K_WIDTH // 2)
        cy = float(rect_res.Q_2k[1, 3] * -1.0) if rect_res.Q_2k is not None else float(WORKING_2K_HEIGHT // 2)

        hybrid_res = fuser.fuse_and_export(
            rect_left_2k=rect_res.rect_left_2k,
            rect_right_2k=rect_res.rect_right_2k,
            disparity_map_2k=disp_res.disparity_2k,
            landmarks_0=mask_res.landmarks_cam0 if mask_res.landmarks_cam0 else [],
            landmarks_1=mask_res.landmarks_cam1 if mask_res.landmarks_cam1 else (mask_res.landmarks_cam0 or []),
            focal_px_2k=rect_res.effective_focal_2k,
            baseline_mm=rect_res.baseline_mm,
            cx=cx,
            cy=cy,
            output_dir=out_dir,
        )

        summary_report["artifacts"].update({k: str(v) for k, v in hybrid_res.deliverables.items()})
        summary_report["stages_completed"].extend([
            "stage_4_hybrid_landmark_alignment",
            "stage_5_laplacian_depth_deformation",
            "stage_6_winner_takes_all_texture_baking",
            "stage_7_export_deliverables",
        ])

        total_dt = time.time() - t_pipeline_start
        consolidated_manifest = {
            "pair_id": ingestion_res.pair_id,
            "reconstruction_mode": "hybrid",
            "stages_completed": summary_report["stages_completed"],
            "pipeline_duration_s": round(total_dt, 3),
            "optical_parameters": {
                "focal_length_px_2k": round(rect_res.effective_focal_2k, 2),
                "baseline_mm": round(rect_res.baseline_mm, 2),
                "convergence_deg": round(rect_res.convergence_deg, 2),
                "z_rig_mm": ingestion_res.z_rig_mm,
                "is_apriltag_calibrated": rect_res.is_apriltag_calibrated,
            },
            "rectification": {
                "resolution": [WORKING_2K_WIDTH, WORKING_2K_HEIGHT],
                "epipolar_rms_error_px": round(mask_res.epipolar_rms_error_px, 3),
                "Q_matrix_2k": rect_res.Q_2k.tolist(),
            },
            "facial_masking": {
                "face_detected_cam0": mask_res.face_detected_cam0,
                "face_detected_cam1": mask_res.face_detected_cam1,
                "d_median_px": round(mask_res.d_median, 2),
                "d0_shift_px": mask_res.d0_shift,
                "bbox_2k": [round(float(v), 1) for v in mask_res.bbox_2k],
            },
            "dense_disparity": {
                "matcher_engine": disp_res.matcher_engine,
                "valid_pixels_count": disp_res.valid_pixels_count,
                "min_disparity_px": round(disp_res.min_disparity, 2),
                "max_disparity_px": round(disp_res.max_disparity, 2),
                "median_disparity_px": round(disp_res.median_disparity, 2),
                "matcher_time_s": round(disp_res.execution_time_s, 3),
            },
            "hybrid_reconstruction": {
                "ipd_metric_mm": round(hybrid_res.ipd_metric_mm, 1),
                "high_poly_faces": hybrid_res.high_poly_faces,
                "low_poly_faces": hybrid_res.low_poly_faces,
                "bbox_dimensions_mm": [round(float(v), 1) for v in hybrid_res.bbox_dimensions_mm],
                "reconstruction_time_s": round(hybrid_res.execution_time_s, 3),
                "features": [
                    "canonical_topology",
                    "welded_ears",
                    "laplacian_depth_fitting",
                    "anti_ghosting_winner_takes_all",
                ],
            },
            "artifacts": summary_report["artifacts"],
        }
        consolidated_path = out_dir / "photogrammetry_manifest.json"
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_manifest, f, indent=2)

        log.info("=" * 70)
        log.info("PIPELINE HIBRID FINALIZAT CU SUCCES în %.2f secunde.", total_dt)
        log.info("  • High-Poly Model: %s (%d faces)", hybrid_res.high_mesh_path, hybrid_res.high_poly_faces)
        log.info("  • Low-Poly Model: %s (%d faces, canonical UV)", hybrid_res.low_mesh_path, hybrid_res.low_poly_faces)
        log.info(
            "  • BBox Real Metric: %.1f x %.1f x %.1f mm",
            hybrid_res.bbox_dimensions_mm[0],
            hybrid_res.bbox_dimensions_mm[1],
            hybrid_res.bbox_dimensions_mm[2],
        )
        log.info("  • 8 Deliverables: OBJ (High/Low), GLB Draco, STL, 3MF în %s", out_dir)
        log.info("Manifest consolidat salvat în: %s", consolidated_path)
        log.info("=" * 70)

        summary_report["status"] = "success"
        summary_report["manifest"] = str(consolidated_path)
        return summary_report

    # =========================================================================
    # ETAPA 4: TRIANGULARE METRICĂ & SCREENED POISSON (DEPTH 9-10) [MOD CLASIC]
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 4: TRIANGULARE METRICĂ & SCREENED POISSON CU TRIMMING")
    log.info("=" * 70)

    reconstructor = MetricPoissonReconstructor(octree_depth=9, density_trim_percentile=6.0)
    poisson_res = reconstructor.triangulate_and_reconstruct(
        disparity_2k=disp_res.disparity_2k,
        rect_left_2k=rect_res.rect_left_2k,
        focal_px_2k=rect_res.effective_focal_2k,
        baseline_mm=rect_res.baseline_mm,
        mask_2k=mask_res.mask_2k,
        d0_shift=mask_res.d0_shift,
    )

    # Save to standard deliverables structure: output/obj/high/high_model.obj
    obj_high_dir = out_dir / "obj" / "high"
    obj_high_dir.mkdir(parents=True, exist_ok=True)
    high_obj_path = obj_high_dir / "high_model.obj"
    poisson_res.save_mesh(high_obj_path)
    summary_report["artifacts"]["high_model_obj"] = str(high_obj_path)
    summary_report["stages_completed"].append("stage_4_poisson_reconstruction")

    if stop_after_stage == 4:
        log.info("Pipeline terminated after Etapa 4 as requested.")
        return summary_report

    # =========================================================================
    # ETAPA 5: DECIMARE QEM (15k-30k) & DESPACHETARE UV XATLAS
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 5: DECIMARE QEM & DESPACHETARE UV XATLAS")
    log.info("=" * 70)

    decimator = MeshDecimatorUV(target_faces=20000)
    uv_res = decimator.decimate_and_unwrap(poisson_res.mesh, target_faces=20000)

    # Save to standard deliverables structure: output/obj/low/low_model.obj
    obj_low_dir = out_dir / "obj" / "low"
    obj_low_dir.mkdir(parents=True, exist_ok=True)
    low_obj_path = obj_low_dir / "low_model.obj"
    uv_res.save_mesh(low_obj_path)
    summary_report["artifacts"]["low_model_obj"] = str(low_obj_path)
    summary_report["stages_completed"].append("stage_5_decimation_uv")

    if stop_after_stage == 5:
        total_dt = time.time() - t_pipeline_start
        consolidated_manifest = {
            "pair_id": ingestion_res.pair_id,
            "stages_completed": summary_report["stages_completed"],
            "pipeline_duration_s": round(total_dt, 3),
            "optical_parameters": {
                "focal_length_px_2k": round(rect_res.effective_focal_2k, 2),
                "baseline_mm": round(rect_res.baseline_mm, 2),
                "convergence_deg": round(rect_res.convergence_deg, 2),
                "z_rig_mm": ingestion_res.z_rig_mm,
                "is_apriltag_calibrated": rect_res.is_apriltag_calibrated,
            },
            "rectification": {
                "resolution": [WORKING_2K_WIDTH, WORKING_2K_HEIGHT],
                "epipolar_rms_error_px": round(mask_res.epipolar_rms_error_px, 3),
                "Q_matrix_2k": rect_res.Q_2k.tolist(),
            },
            "facial_masking": {
                "face_detected_cam0": mask_res.face_detected_cam0,
                "face_detected_cam1": mask_res.face_detected_cam1,
                "d_median_px": round(mask_res.d_median, 2),
                "d0_shift_px": mask_res.d0_shift,
                "bbox_2k": [round(float(v), 1) for v in mask_res.bbox_2k],
            },
            "dense_disparity": {
                "matcher_engine": disp_res.matcher_engine,
                "valid_pixels_count": disp_res.valid_pixels_count,
                "min_disparity_px": round(disp_res.min_disparity, 2),
                "max_disparity_px": round(disp_res.max_disparity, 2),
                "median_disparity_px": round(disp_res.median_disparity, 2),
                "matcher_time_s": round(disp_res.execution_time_s, 3),
            },
            "poisson_reconstruction": {
                "octree_depth": poisson_res.octree_depth,
                "high_poly_faces": poisson_res.num_faces,
                "high_poly_vertices": poisson_res.num_vertices,
                "point_cloud_size": poisson_res.point_cloud_size,
                "density_trim_percentile": poisson_res.density_trim_percentile,
                "bbox_dimensions_mm": [round(float(v), 1) for v in poisson_res.bbox_dimensions_mm],
                "reconstruction_time_s": round(poisson_res.execution_time_s, 3),
            },
            "decimation_uv": {
                "low_poly_faces": uv_res.low_face_count,
                "low_poly_vertices": uv_res.low_vertex_count,
                "target_faces": uv_res.target_faces,
                "uv_unwrapped": True,
                "decimation_time_s": round(uv_res.execution_time_s, 3),
            },
            "artifacts": summary_report["artifacts"],
        }
        consolidated_path = out_dir / "photogrammetry_manifest.json"
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_manifest, f, indent=2)
        log.info("Pipeline terminated after Etapa 5 as requested.")
        summary_report["manifest"] = str(consolidated_path)
        return summary_report

    # =========================================================================
    # ETAPA 6: COACERE TEXTURĂ ALBEDO & HĂRȚI DE NORMALE TANGENT-SPACE
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 6: COACERE TEXTURĂ ALBEDO & HĂRȚI DE NORMALE TANGENT-SPACE")
    log.info("=" * 70)

    baker = TextureBaker(texture_size=2048)
    baking_res = baker.bake_textures(
        low_mesh=uv_res.low_mesh,
        high_mesh=poisson_res.mesh,
        rect_left_2k=rect_res.rect_left_2k,
        rect_right_2k=rect_res.rect_right_2k,
        focal_px_2k=rect_res.effective_focal_2k,
        centroid_mm=poisson_res.centroid_mm,
        output_dir=out_dir,
        baseline_mm=rect_res.baseline_mm,
    )

    summary_report["artifacts"].update({k: str(v) for k, v in baking_res.saved_paths.items()})
    summary_report["stages_completed"].append("stage_6_texture_baking")

    if stop_after_stage == 6:
        log.info("Pipeline terminated after Etapa 6 as requested.")
        return summary_report

    # =========================================================================
    # ETAPA 7: EXPORTUL STANDARD ÎN 8 LIVRABILE (OBJ, GLB, STL, 3MF)
    # =========================================================================
    log.info("=" * 70)
    log.info("ETAPA 7: EXPORTUL STANDARD ÎN 8 LIVRABILE (OBJ, GLB, STL, 3MF)")
    log.info("=" * 70)

    exporter = StandardDeliverablesExporter(output_dir=out_dir)
    export_res = exporter.export_all_deliverables(
        high_mesh=poisson_res.mesh,
        low_mesh=uv_res.low_mesh,
    )

    summary_report["artifacts"].update({k: str(v) for k, v in export_res.deliverables.items()})
    summary_report["stages_completed"].append("stage_7_export_deliverables")

    # Serialize Complete Master Photogrammetry Manifest (All 7 Stages)
    total_dt = time.time() - t_pipeline_start
    consolidated_manifest = {
        "pair_id": ingestion_res.pair_id,
        "stages_completed": summary_report["stages_completed"],
        "pipeline_duration_s": round(total_dt, 3),
        "optical_parameters": {
            "focal_length_px_2k": round(rect_res.effective_focal_2k, 2),
            "baseline_mm": round(rect_res.baseline_mm, 2),
            "convergence_deg": round(rect_res.convergence_deg, 2),
            "z_rig_mm": ingestion_res.z_rig_mm,
            "is_apriltag_calibrated": rect_res.is_apriltag_calibrated,
        },
        "rectification": {
            "resolution": [WORKING_2K_WIDTH, WORKING_2K_HEIGHT],
            "epipolar_rms_error_px": round(mask_res.epipolar_rms_error_px, 3),
            "Q_matrix_2k": rect_res.Q_2k.tolist(),
        },
        "facial_masking": {
            "face_detected_cam0": mask_res.face_detected_cam0,
            "face_detected_cam1": mask_res.face_detected_cam1,
            "d_median_px": round(mask_res.d_median, 2),
            "d0_shift_px": mask_res.d0_shift,
            "bbox_2k": [round(float(v), 1) for v in mask_res.bbox_2k],
        },
        "dense_disparity": {
            "matcher_engine": disp_res.matcher_engine,
            "valid_pixels_count": disp_res.valid_pixels_count,
            "min_disparity_px": round(disp_res.min_disparity, 2),
            "max_disparity_px": round(disp_res.max_disparity, 2),
            "median_disparity_px": round(disp_res.median_disparity, 2),
            "matcher_time_s": round(disp_res.execution_time_s, 3),
        },
        "poisson_reconstruction": {
            "octree_depth": poisson_res.octree_depth,
            "high_poly_faces": poisson_res.num_faces,
            "high_poly_vertices": poisson_res.num_vertices,
            "point_cloud_size": poisson_res.point_cloud_size,
            "density_trim_percentile": poisson_res.density_trim_percentile,
            "bbox_dimensions_mm": [round(float(v), 1) for v in poisson_res.bbox_dimensions_mm],
            "reconstruction_time_s": round(poisson_res.execution_time_s, 3),
        },
        "decimation_uv": {
            "low_poly_faces": uv_res.low_face_count,
            "low_poly_vertices": uv_res.low_vertex_count,
            "target_faces": uv_res.target_faces,
            "uv_unwrapped": True,
            "decimation_time_s": round(uv_res.execution_time_s, 3),
        },
        "texture_baking": {
            "texture_resolution": baker.texture_size,
            "baking_time_s": round(baking_res.execution_time_s, 3),
        },
        "export_deliverables": {
            "total_deliverable_files": export_res.total_files_count,
            "is_complete": export_res.is_complete,
            "export_time_s": round(export_res.execution_time_s, 3),
        },
        "artifacts": summary_report["artifacts"],
    }

    consolidated_path = out_dir / "photogrammetry_manifest.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_manifest, f, indent=2)

    log.info("=" * 70)
    log.info("PIPELINE COMPLET (ETAPELE 1 - 7) FINALIZAT CU SUCCES în %.2f secunde.", total_dt)
    log.info("  • High-Poly Model: %s (%d faces)", high_obj_path, poisson_res.num_faces)
    log.info("  • Low-Poly Model: %s (%d faces, UV unwrap)", low_obj_path, uv_res.low_face_count)
    log.info("  • 8 Deliverables: OBJ (High/Low), GLB Draco, STL, 3MF în %s", out_dir)
    log.info("Manifest consolidat salvat în: %s", consolidated_path)
    log.info("=" * 70)

    summary_report["status"] = "success"
    summary_report["manifest"] = str(consolidated_path)
    return summary_report


def main():
    D = JobConfig()
    parser = argparse.ArgumentParser(
        description="RunPod Photogrammetry Pipeline (Etapele 1, 2, 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Complete pipeline on standard rig directory (Etapele 1, 2, 3):
     python run_photogrammetry_runpod.py --rig-dir input_images/rig -o output/

  2. Explicit stereo photos:
     python run_photogrammetry_runpod.py -l input_images/rig/0/f_150.jpg -r input_images/rig/1/f_150.jpg -o output/

  3. Run only Etapa 1 & 2:
     python run_photogrammetry_runpod.py --rig-dir input_images/rig --stop-after-stage 2 -o output/

  4. Dense face reconstruction with another rig calibration (e.g. straight from Turntable Studio):
     python run_photogrammetry_runpod.py --rig-dir input_images/rig --calib path/to/turntable_calib.json -o output/
        """
    )
    parser.add_argument(
        "--rig-dir",
        default=None,
        help="Path to rig directory containing '0' (Left/Cam0) and '1' (Right/Cam1) subdirectories",
    )
    parser.add_argument(
        "-l",
        "--left",
        default=None,
        help="Path to Left camera photo (Camera 0)",
    )
    parser.add_argument(
        "-r",
        "--right",
        default=None,
        help="Path to Right camera photo (Camera 1)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=D.output_dir,
        help="Directory to save photogrammetry deliverables (default: ./output)",
    )
    parser.add_argument(
        "--name",
        default=D.name,
        help="Prefix name for the model and output files (default: model)",
    )
    parser.add_argument(
        "--no-apriltags",
        action="store_true",
        help="Disable AprilTag marker detection and use nominal rig priors exclusively",
    )
    parser.add_argument(
        "--layout",
        default=None,
        help="Path to custom AprilTag layout JSON (default: assets/platan_260mm_layout.json)",
    )
    parser.add_argument(
        "--tag-scale",
        type=float,
        default=D.tag_scale,
        help="Print scale of the AprilTag disc: measured / designed size (default 0.98: this rig's disc came out at "
             "~98 %%, disc 255 mm, tags 0-6 196 mm, as in Turntable Studio). Corrects the metric scale once",
    )
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=D.stop_after_stage,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Stage after which to terminate execution (1 to 7; default: 7)",
    )
    parser.add_argument(
        "--mode",
        dest="reconstruction_mode",
        default=D.reconstruction_mode,
        choices=["dense", "hybrid", "poisson"],
        help="Reconstruction algorithm: 'dense' (calibrated rig + MediaPipe-guided dense two-view stereo, "
             "recommended), 'hybrid' (MediaPipe prior + Laplacian fitting) or 'poisson' (classical SGBM + Poisson)",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="Stereo rig calibration for --mode dense: assets/rig_calibration.json (default) "
             "or a turntable_calib.json from turntable_poses.py",
    )
    parser.add_argument(
        "--export-scale",
        type=float,
        default=D.export_scale,
        help="--mode dense: metres -> export units, the same in OBJ / GLB / STL / 3MF (default 100: centimetres, "
             "as Turntable Studio; 1000 = millimetres)",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=D.texture_size,
        help="--mode dense: high atlas size in px (default 4096; the low atlas is min(2048, this)). "
             "RAM scales with its square: ~9 GB at 4096, ~3 GB at 2048",
    )
    parser.add_argument(
        "--head-model",
        default=D.head_model,
        choices=["gnm", "none"],
        help="--mode dense: 'gnm' = whole head (GNM Head v3 fitted to the measured face + its detail; animatable "
             "low GLB), 'none' = the measured face only",
    )
    parser.add_argument(
        "--neutral-expression",
        action="store_true",
        help="--mode dense with --head-model gnm: neutral expression in the deliverables (identity and detail kept)",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="--mode dense: face region = MediaPipe mesh outline (instead of the segmented face skin up to the "
             "hairline and the lateral cheeks)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="--mode dense: keep the spectacle frames in the texture (by default they are painted out when "
             "spectacles are expected)",
    )
    parser.add_argument(
        "--no-glb-draco",
        action="store_true",
        help="--mode dense: do not Draco-compress the GLB geometry",
    )
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="--mode dense: photographs of the free AprilTag disc (subfolders 0/ and 1/, f_<level><step>.jpg) on "
             "which turntable_poses.py calibrates the rig for this job (default: a 'calib' folder next to the rig "
             "folder, if present; otherwise the stored calibration is used)",
    )
    parser.add_argument(
        "--no-spectacles",
        dest="spectacles",
        action="store_false",
        help="--mode dense: the subject wears no glasses; the spectacle-frame mask and the lens regions are turned "
             "off, so eyebrows, lash lines and eyelids are measured as surface",
    )
    parser.add_argument(
        "--rectify-native",
        action="store_true",
        help="Also compute and export native 12.3MP rectified images for master texture baking",
    )

    args = parser.parse_args()
    job = {
        "rig_dir": args.rig_dir,
        "left_path": args.left,
        "right_path": args.right,
        "output_dir": args.output_dir,
        "check_apriltags": not args.no_apriltags,
        "layout_path": args.layout,
        "tag_scale": args.tag_scale,
        "name": args.name,
        "stop_after_stage": args.stop_after_stage,
        "rectify_native": args.rectify_native,
        "reconstruction_mode": args.reconstruction_mode,
        "calibration_path": args.calib,
        "spectacles": args.spectacles,
        "calib_dir": args.calib_dir,
        "export_scale": args.export_scale,
        "glb_draco": not args.no_glb_draco,
        "texture_size": args.texture_size,
        "segment_face": not args.no_segmentation,
        "remove_frames": False if args.keep_frames else None,
        "head_model": args.head_model,
        "expression": "neutral" if args.neutral_expression else "capture",
    }
    try:
        run_pipeline(job)
    except Exception as e:
        log.error("Pipeline failure: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
