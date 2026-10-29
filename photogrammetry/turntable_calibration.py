"""
Rig Calibration with turntable_poses.py (Turntable Studio)
==========================================================
The face pair alone cannot calibrate the rig: the person covers the AprilTag disc
and the carriage stands at face height, so no tag is visible. The calibration comes
from photographs of the free disc taken by the same two cameras while the turntable
rotates (the same images an object scan starts from), stored next to the face pair:

    <job>/rig/0/f_150.jpg, rig/1/f_150.jpg              face pair (carriage height in mm)
    <job>/calib/0/f_1000.jpg ... f_1019.jpg, calib/1/...  disc sweep (level 1, steps 000-019)

turntable_poses.py is copied unchanged from Turntable Studio and runs as a separate
process (its rule: never modified, only called through its command line). It fits
the turntable model (axis, step, both cameras' intrinsics and poses) on the tag
corners and writes turntable_calib.json, which load_rig_calibration() reads directly:
the relative pose of the two cameras does not depend on the carriage height, because
both cameras ride the same carriage.

CPU only (OpenCV tag detection in parallel processes + SciPy least squares):
a few seconds for 2 x 20 photographs.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import ExifTags, Image

log = logging.getLogger("Photogrammetry.TurntableCalibration")

TURNTABLE_POSES_SCRIPT = Path(__file__).resolve().parent / "turntable_poses.py"
TAG_LAYOUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "platan_260mm_layout.json"

# Same conventions as Turntable Studio (pipeline/config.py): f_<level><step:03d>.jpg, 18 deg per step,
# 100 mm carriage lift per level, IMX477 + M23390H08 lens.
NAME_REGEX = r"(?P<level>\d)[_]?(?P<index>\d{1,4})$"
DEFAULT_FOCAL_MM = 3.9
DEFAULT_SENSOR_WIDTH_MM = 6.287
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class TurntableCalibrationConfig:
    step_deg: float = 18.0
    lift_mm: float = 100.0
    estimate_lift: bool = True
    tag_refine: bool = True         # per-tag height / tilt (paper unevenness): RMS 4.09 -> 2.48 px in Turntable Studio
    tag_scale: float = 0.98         # measured / designed size of the printed tags (this rig's disc: ~98 %)
    layout_path: Path = TAG_LAYOUT_PATH
    python_exe: Optional[str] = None


@dataclass
class TurntableCalibrationResult:
    calib_path: Path                # turntable_calib.json (input of load_rig_calibration)
    report_path: Path
    sfm_path: Path                  # cameras_known.sfm (poses of the disc photographs)
    n_images: Dict[str, int]
    duration_s: float
    stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "calib": str(self.calib_path), "report": str(self.report_path), "cameras_known_sfm": str(self.sfm_path),
            "images_per_camera": self.n_images, "duration_s": round(self.duration_s, 2), **self.stats,
        }


def find_calibration_images(calib_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """{"0": [...], "1": [...]}: disc photographs of each camera; empty if the folder has no 0/ and 1/."""
    root = Path(calib_dir)
    out: Dict[str, List[Path]] = {}
    for cam in ("0", "1"):
        d = root / cam
        if d.is_dir():
            out[cam] = sorted(p for p in d.iterdir()
                              if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.endswith(".mask.png"))
    if len(out) < 2 or not all(out.values()):
        return {}
    return out


def _image_metadata(path: Path) -> Dict:
    """Size and EXIF focal length / serial of one photograph (IMX477 defaults when EXIF is missing)."""
    focal = DEFAULT_FOCAL_MM
    serial = ""
    with Image.open(path) as img:
        width, height = img.size
        exif = img.getexif()
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()} if exif else {}
        try:
            sub = exif.get_ifd(0x8769) if exif else {}
            tags.update({ExifTags.TAGS.get(k, str(k)): v for k, v in sub.items()})
        except Exception:
            pass
    try:
        focal = float(tags.get("FocalLength", focal)) or DEFAULT_FOCAL_MM
    except (TypeError, ValueError):
        pass
    serial = str(tags.get("BodySerialNumber", "") or "").strip()
    return {"width": width, "height": height, "focalLength": round(focal, 4), "serialNumber": serial}


def write_camera_init_sfm(images: Dict[str, List[Path]], out_path: Union[str, Path]) -> Path:
    """AliceVision CameraInit JSON (the input of turntable_poses.py): one intrinsic per camera, one view per photo."""
    views, intrinsics = [], []
    view_id = 1000
    for cam, files in images.items():
        meta = _image_metadata(files[0])
        intrinsics.append({
            "intrinsicId": cam,
            "width": meta["width"],
            "height": meta["height"],
            "sensorWidth": DEFAULT_SENSOR_WIDTH_MM,
            "sensorHeight": round(DEFAULT_SENSOR_WIDTH_MM * meta["height"] / meta["width"], 4),
            "focalLength": meta["focalLength"],
            "initialFocalLength": meta["focalLength"],
            "pixelRatio": 1.0,
            "serialNumber": meta["serialNumber"],
        })
        for f in files:
            if not re.search(NAME_REGEX, f.stem):
                raise ValueError(f"Calibration photo '{f.name}' does not follow f_<level><step:03d> (e.g. f_1000.jpg)")
            views.append({"viewId": str(view_id), "poseId": str(view_id), "intrinsicId": cam,
                          "path": str(f.resolve()), "width": meta["width"], "height": meta["height"]})
            view_id += 1
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": ["1", "0", "0"], "featuresFolders": [], "matchesFolders": [],
                               "views": views, "intrinsics": intrinsics}, indent=2), encoding="utf-8")
    return out


def build_turntable_poses_cmd(camera_init: Path, out_dir: Path, config: TurntableCalibrationConfig) -> List[str]:
    """Command line of turntable_poses.py with the options Turntable Studio uses (main.py / camera_init.py)."""
    return [
        config.python_exe or sys.executable, str(TURNTABLE_POSES_SCRIPT),
        "--input", str(camera_init),
        "--output", str(out_dir / "cameras_known.sfm"),
        "--calibOut", str(out_dir / "turntable_calib.json"),
        "--report", str(out_dir / "turntable_report.txt"),
        "--layout", str(Path(config.layout_path).resolve()),
        "--stepDeg", str(config.step_deg),
        "--tagScale", str(config.tag_scale),
        "--nameRegex", NAME_REGEX,
        "--liftMm", str(config.lift_mm),
        "--estimateLift", "True" if config.estimate_lift else "False",
        "--tagRefine", "True" if config.tag_refine else "False",
    ]


def _report_stats(report: str) -> Dict[str, float]:
    """Reprojection error of the last optimisation pass in the turntable_poses report."""
    stats: Dict[str, float] = {}
    for m in re.finditer(r"(\d+) colțuri, eroare mediană ([\d.]+) px, RMS ([\d.]+) px", report):
        stats = {"tag_corners": int(m.group(1)), "reprojection_median_px": float(m.group(2)),
                 "reprojection_rms_px": float(m.group(3))}
    return stats


def run_turntable_calibration(
    calib_dir: Union[str, Path],
    out_dir: Union[str, Path],
    config: Optional[TurntableCalibrationConfig] = None,
) -> TurntableCalibrationResult:
    """Runs turntable_poses.py on the disc photographs of `calib_dir`; outputs go to `out_dir`."""
    cfg = config or TurntableCalibrationConfig()
    images = find_calibration_images(calib_dir)
    if not images:
        raise FileNotFoundError(f"No calibration photographs in {calib_dir}/0 and {calib_dir}/1")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    camera_init = write_camera_init_sfm(images, out / "cameraInit.sfm")
    cmd = build_turntable_poses_cmd(camera_init, out, cfg)
    log.info("turntable_poses: %d + %d disc photographs -> %s", len(images["0"]), len(images["1"]), out)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    duration = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"turntable_poses.py failed (exit {proc.returncode}): {tail}")
    calib_path = out / "turntable_calib.json"
    if not calib_path.is_file():
        raise FileNotFoundError(f"turntable_poses.py finished without writing {calib_path}")
    # turntable_poses.py (unchanged) does not record --tagScale: add it, so load_rig_calibration() knows the
    # metric scale is already corrected and the pipeline does not apply it a second time.
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    calib["tag_scale"] = float(cfg.tag_scale)
    calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    report_path = out / "turntable_report.txt"
    stats = _report_stats(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    log.info("turntable_poses finished in %.1f s (reprojection RMS %s px)", duration,
             stats.get("reprojection_rms_px", "?"))
    stats["tag_scale"] = float(cfg.tag_scale)
    return TurntableCalibrationResult(calib_path, report_path, out / "cameras_known.sfm",
                                      {k: len(v) for k, v in images.items()}, duration, stats)
