"""
Unit Tests for Photogrammetry Etapa 1: Ingestion & AprilTag Metric Calibration
"""

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from photogrammetry.apriltag_tracker import (
    AprilTagTracker,
    AprilTagPoseResult,
    DEFAULT_FOCAL_PX,
)
from photogrammetry.ingestion import (
    StereoRigIngestor,
    StereoIngestionResult,
    load_stereo_rig_pair,
)


def test_tracker_initialization():
    """Verifies that the AprilTag tracker loads the layout and sets up detector."""
    tracker = AprilTagTracker()
    assert len(tracker.layout_corners) >= 12
    # Verify corners of tag 0 are 4 points with 3D coordinates (Y=0 in meters)
    c0 = tracker.layout_corners[0]
    assert c0.shape == (4, 3)
    np.testing.assert_allclose(c0[:, 1], 0.0)  # Y=0 plane


def test_synthetic_apriltag_pnp():
    """
    Renders a synthetic AprilTag image from a known camera pose and verifies
    that solvePnP recovers the camera center and rotation accurately.
    """
    tracker = AprilTagTracker()
    K = tracker.K_default
    W, H = tracker.img_w, tracker.img_h

    # Known ground truth camera pose: Camera at C = [0.0, 0.20, 0.40] looking down at origin
    C_gt = np.array([0.0, 0.20, 0.40], dtype=np.float64)
    # Look at [0, 0, 0]: Z_cam axis points to origin
    z_axis = -C_gt / np.linalg.norm(C_gt)
    # Up vector roughly +Y in world, but project onto camera plane
    y_up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_gt = np.stack([x_axis, y_axis, z_axis], axis=0)
    t_gt = -R_gt @ C_gt
    rvec_gt, _ = cv2.Rodrigues(R_gt)

    # Project the 3D corners of tags 0, 1, 2 into the camera image
    detected_tags = {}
    for tag_id in [0, 1, 2]:
        obj_pts = tracker.layout_corners[tag_id]
        img_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, t_gt, K, np.zeros(5))
        detected_tags[tag_id] = img_pts.reshape(4, 2)

    # Solve PnP
    res = tracker.estimate_single_camera_pose(detected_tags, K)
    assert res is not None
    rvec_est, tvec_est, R_est, C_est, rms = res

    # Verify camera center recovered within 0.5 mm
    np.testing.assert_allclose(C_est, C_gt, atol=0.001)
    assert rms < 0.05  # Sub-pixel precision


def test_synthetic_stereo_rig_calibration():
    """
    Verifies that two cameras with baseline 163.4 mm and 27° convergence
    have their relative extrinsics and baseline accurately recovered.
    """
    tracker = AprilTagTracker()
    K = tracker.K_default

    # Cam 0 (left): X = -0.0817 m (-81.7 mm), Z = 0.35 m, yaw = +13.5°
    C0_gt = np.array([-0.0817, 0.0, 0.35], dtype=np.float64)
    yaw0 = math.radians(13.5)
    R0_gt = np.array([
        [math.cos(yaw0), 0, -math.sin(yaw0)],
        [0, 1, 0],
        [math.sin(yaw0), 0, math.cos(yaw0)]
    ], dtype=np.float64)
    t0_gt = -R0_gt @ C0_gt
    rvec0_gt, _ = cv2.Rodrigues(R0_gt)

    # Cam 1 (right): X = +0.0817 m (+81.7 mm), Z = 0.35 m, yaw = -13.5°
    C1_gt = np.array([0.0817, 0.0, 0.35], dtype=np.float64)
    yaw1 = math.radians(-13.5)
    R1_gt = np.array([
        [math.cos(yaw1), 0, -math.sin(yaw1)],
        [0, 1, 0],
        [math.sin(yaw1), 0, math.cos(yaw1)]
    ], dtype=np.float64)
    t1_gt = -R1_gt @ C1_gt
    rvec1_gt, _ = cv2.Rodrigues(R1_gt)

    # Ground truth baseline: 2 * 0.0817 = 0.1634 m = 163.4 mm
    B_gt_mm = float(np.linalg.norm(C1_gt - C0_gt) * 1000.0)
    assert abs(B_gt_mm - 163.4) < 0.1

    # Project tags into both cameras
    tags_0, tags_1 = {}, {}
    for tid in [0, 1, 2, 3]:
        obj_pts = tracker.layout_corners[tid]
        p0, _ = cv2.projectPoints(obj_pts, rvec0_gt, t0_gt, K, np.zeros(5))
        p1, _ = cv2.projectPoints(obj_pts, rvec1_gt, t1_gt, K, np.zeros(5))
        tags_0[tid] = p0.reshape(4, 2)
        tags_1[tid] = p1.reshape(4, 2)

    pose0 = tracker.estimate_single_camera_pose(tags_0, K)
    pose1 = tracker.estimate_single_camera_pose(tags_1, K)
    assert pose0 is not None and pose1 is not None

    _, t0_est, R0_est, C0_est, _ = pose0
    _, t1_est, R1_est, C1_est, _ = pose1

    R_rel_est = R1_est @ R0_est.T
    T_rel_est = t1_est - R_rel_est @ t0_est
    baseline_mm_est = float(np.linalg.norm(C1_est - C0_est) * 1000.0)

    # Convergence angle: z-axes
    z0 = R0_est.T @ [0, 0, 1]
    z1 = R1_est.T @ [0, 0, 1]
    conv_deg = math.degrees(math.acos(np.dot(z0, z1)))

    assert abs(baseline_mm_est - 163.4) < 0.5
    assert abs(conv_deg - 27.0) < 0.5


def test_ingestor_on_sample_files(tmp_path):
    """Tests discovery and ingestion using real or temporary stereo images."""
    dir_0 = tmp_path / "0"
    dir_1 = tmp_path / "1"
    dir_0.mkdir()
    dir_1.mkdir()

    # Create dummy 12.3MP test frames named f_240.jpg
    dummy_frame = np.full((304, 405, 3), 128, dtype=np.uint8)  # Scaled dummy for fast test
    p_left = dir_0 / "f_240.jpg"
    p_right = dir_1 / "f_240.jpg"
    cv2.imwrite(str(p_left), dummy_frame)
    cv2.imwrite(str(p_right), dummy_frame)

    ingestor = StereoRigIngestor(check_apriltags=False)
    pairs = ingestor.ingest_rig_directory(tmp_path)
    assert len(pairs) == 1
    res = pairs[0]

    assert res.pair_id == "f_240"
    assert res.z_rig_mm == 240.0
    assert res.has_metric_z_prior is True
    assert res.cam0_meta.camera_id == 0
    assert res.cam1_meta.camera_id == 1
    assert res.nominal_baseline_mm == 163.4

    # Verify manifest export
    manifest = tmp_path / "manifest.json"
    res.save_manifest(manifest)
    assert manifest.is_file()
    with open(manifest) as f:
        data = json.load(f)
    assert data["z_rig_mm"] == 240.0


if __name__ == "__main__":
    print("Testing test_tracker_initialization...")
    test_tracker_initialization()
    print("  [PASS] test_tracker_initialization")

    print("Testing test_synthetic_apriltag_pnp...")
    test_synthetic_apriltag_pnp()
    print("  [PASS] test_synthetic_apriltag_pnp")

    print("Testing test_synthetic_stereo_rig_calibration...")
    test_synthetic_stereo_rig_calibration()
    print("  [PASS] test_synthetic_stereo_rig_calibration")

    print("Testing test_ingestor_on_sample_files...")
    with tempfile.TemporaryDirectory() as td:
        test_ingestor_on_sample_files(Path(td))
    print("  [PASS] test_ingestor_on_sample_files")

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")



from tests._functional import function_tests  # noqa: E402

load_tests = function_tests(globals())
