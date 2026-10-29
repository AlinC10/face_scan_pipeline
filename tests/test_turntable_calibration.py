"""
Unit Tests for the turntable_poses.py Integration
=================================================
Tests:
  1. Disc photographs are found only when both camera folders have images.
  2. cameraInit.sfm: one intrinsic per camera (EXIF focal length and serial), one view per photograph.
  3. Photographs that do not follow f_<level><step> are refused before turntable_poses.py runs.
  4. Command line and report parsing.
  5. The runner finds calib/ next to rig/ without resolving symbolic links, and falls back to the stored
     calibration when a job has no disc photographs.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from photogrammetry.turntable_calibration import (
    NAME_REGEX,
    TurntableCalibrationConfig,
    _report_stats,
    build_turntable_poses_cmd,
    find_calibration_images,
    write_camera_init_sfm,
)


def _photo(path: Path, serial: str = "", focal: float = 3.9) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.full((304, 405, 3), 128, np.uint8))
    exif = Image.Exif()
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x920A] = focal            # FocalLength
    if serial:
        exif_ifd[0xA431] = serial       # BodySerialNumber
    img.save(path, exif=exif)
    return path


def test_find_calibration_images_needs_both_cameras():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert find_calibration_images(root) == {}
        _photo(root / "0" / "f_1000.jpg")
        assert find_calibration_images(root) == {}
        _photo(root / "1" / "f_1000.jpg")
        (root / "1" / "f_1000.mask.png").write_bytes(b"")
        found = find_calibration_images(root)
        assert sorted(found) == ["0", "1"]
        assert [p.name for p in found["1"]] == ["f_1000.jpg"]


def test_camera_init_sfm_from_exif():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for cam, serial in (("0", "SN0"), ("1", "SN1")):
            for k in range(3):
                _photo(root / cam / f"f_100{k}.jpg", serial=serial, focal=3.85)
        sfm = json.loads(write_camera_init_sfm(find_calibration_images(root), root / "out" / "cameraInit.sfm").read_text())
    assert len(sfm["views"]) == 6
    assert len({v["viewId"] for v in sfm["views"]}) == 6
    intr = {i["intrinsicId"]: i for i in sfm["intrinsics"]}
    assert (intr["0"]["width"], intr["0"]["height"], intr["0"]["focalLength"]) == (405, 304, 3.85)
    assert (intr["0"]["serialNumber"], intr["1"]["serialNumber"]) == ("SN0", "SN1")
    assert all(v["intrinsicId"] in ("0", "1") for v in sfm["views"])


def test_badly_named_photo_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _photo(root / "0" / "f_1000.jpg")
        _photo(root / "1" / "left.jpg")
        try:
            write_camera_init_sfm(find_calibration_images(root), root / "cameraInit.sfm")
        except ValueError as e:
            assert "left.jpg" in str(e)
        else:
            raise AssertionError("a photo without level/step in its name must be refused")


def test_command_line_and_report():
    cmd = build_turntable_poses_cmd(Path("/j/cameraInit.sfm"), Path("/j/out"), TurntableCalibrationConfig(tag_scale=1.002))
    opts = dict(zip(cmd[2::2], cmd[3::2]))
    assert cmd[1].endswith("turntable_poses.py")
    assert opts["--nameRegex"] == NAME_REGEX
    assert (opts["--stepDeg"], opts["--liftMm"], opts["--tagScale"]) == ("18.0", "100.0", "1.002")
    assert (opts["--estimateLift"], opts["--tagRefine"]) == ("True", "True")
    assert opts["--calibOut"].endswith("turntable_calib.json")
    report = ("Optimizare 1: 1580 colțuri, eroare mediană 1.863 px, RMS 2.703 px, excluse peste 7.45 px: 18 (x)\n"
              "Optimizare 2: 1562 colțuri, eroare mediană 1.850 px, RMS 2.480 px, excluse peste 7.40 px: 0 (x)\n")
    assert _report_stats(report) == {"tag_corners": 1562, "reprojection_median_px": 1.85, "reprojection_rms_px": 2.48}


def test_runner_finds_calib_next_to_rig_and_falls_back():
    from run_photogrammetry_runpod import _job_rig_calibration
    import photogrammetry.turntable_calibration as tc

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        elsewhere = _photo(root / "store" / "f_150.jpg")
        job = root / "job"
        (job / "rig" / "0").mkdir(parents=True)
        (job / "rig" / "0" / "f_150.jpg").symlink_to(elsewhere)   # a symlinked photo must not move the job root

        path, info = _job_rig_calibration(job / "rig", None, None, None, root / "out", 1.0)
        assert path is None and info["source"] == "stored"

        for cam in ("0", "1"):
            _photo(job / "calib" / cam / "f_1000.jpg")
        seen = {}

        def fake_run(calib_dir, out_dir, config):
            seen["calib_dir"] = Path(calib_dir)
            return tc.TurntableCalibrationResult(Path(out_dir) / "turntable_calib.json", Path(out_dir) / "r.txt",
                                                 Path(out_dir) / "c.sfm", {"0": 1, "1": 1}, 0.1)

        real = tc.run_turntable_calibration
        tc.run_turntable_calibration = fake_run
        try:
            path, info = _job_rig_calibration(None, job / "rig" / "0" / "f_150.jpg", None, "x.json", root / "out", 1.0)
        finally:
            tc.run_turntable_calibration = real
        assert seen["calib_dir"] == (job / "calib").absolute()
        assert info["source"] == "turntable_poses"
        assert path.name == "turntable_calib.json"
